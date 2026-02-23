"""
Common functions for _solve.py and _solve_alternate.py
"""

from __future__ import (
    annotations,
)  # https://docs.python.org/3/library/typing.html#constant

from typing import TYPE_CHECKING, Callable
from functools import partial
import jax
from jax import jit
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import PyTree, Float, Array, PRNGKeyArray
import optax


from jinns.nn._hyperpinn import _get_param_nb

from jinns.data._utils import append_param_batch, append_obs_batch
from jinns.utils._utils import _check_nan_in_pytree
from jinns.data._DataGeneratorODE import DataGeneratorODE
from jinns.data._CubicMeshPDEStatio import CubicMeshPDEStatio
from jinns.data._CubicMeshPDENonStatio import CubicMeshPDENonStatio
from jinns.data._DataGeneratorParameter import DataGeneratorParameter
from jinns.parameters._params import Params
from jinns.utils._containers import (
    LossContainer,
    StoredObjectContainer,
)

if TYPE_CHECKING:
    from jinns.utils._types import AnyBatch, SolveCarry, SolveAlternateCarry
    from jinns.loss._abstract_loss import AbstractLoss
    from jinns.data._DataGeneratorObservations import DataGeneratorObservations
    from jinns.data._AbstractDataGenerator import AbstractDataGenerator


def _init_stored_weights_terms(loss, n_iter):
    return eqx.tree_at(
        lambda pt: jax.tree.leaves(
            pt, is_leaf=lambda x: x is not None and eqx.is_inexact_array(x)
        ),
        loss.loss_weights,
        tuple(
            jnp.zeros((n_iter))
            for n in range(
                len(
                    jax.tree.leaves(
                        loss.loss_weights,
                        is_leaf=lambda x: x is not None and eqx.is_inexact_array(x),
                    )
                )
            )
        ),
    )


def _init_stored_params(tracked_params, params, n_iter):
    return jax.tree_util.tree_map(
        lambda tracked_param, param: (
            jnp.zeros((n_iter,) + jnp.asarray(param).shape)
            if tracked_param is not None
            else None
        ),
        tracked_params,
        params,
        is_leaf=lambda x: x is None,  # None values in tracked_params will not
        # be traversed. Thus the user can provide something like
        # ```
        #  tracked_params = jinns.parameters.Params(
        #   nn_params=None,
        #   eq_params={"nu": True})
        # ```
        # even when init_params.nn_params is a complex data structure.
    )


@partial(jit, static_argnames=["optimizer", "params_mask", "with_loss_weight_update"])
def _loss_evaluate_and_gradient_step(
    i,
    batch: AnyBatch,
    loss: AbstractLoss,
    params: Params[Array],
    last_non_nan_params: Params[Array],
    state: optax.OptState,
    optimizer: optax.GradientTransformation,
    loss_container: LossContainer,
    key: PRNGKeyArray,
    params_mask: Params[bool] | None = None,
    opt_state_field_for_acceleration: str | None = None,
    with_loss_weight_update: bool = True,
):
    """
    # The crux of our new approach is partitioning and recombining the parameters and optimization state according to params_mask.

    params_mask:
        A jinns.parameters.Params object with boolean as leaves, specifying
        over which parameters optimization is enabled. This usually implies
        important computational gains. Internally, it is used as the
        filter_spec of a eqx.partition function on the parameters. Note that this
        differs from (and complement) DerivativeKeys, as the latter allows
        for more granularity by freezing some gradients with respect to
        different loss terms, but do not subset the optimized parameters globally.

    NOTE: in this function body, we change naming convention for concision:
     * `state` refers to the general optimizer state
     * `opt_state` refers to the unmasked optimizer state, i.e. which are
     really involved in the parameter update as defined by `params_mask`.
     * `non_opt_state` refers to the the optimizer state for non-optimized
     params.
    """

    (
        opt_params,
        opt_params_accel,
        non_opt_params,
        opt_state,
        non_opt_state,
    ) = _get_masked_optimization_stuff(
        params, state, opt_state_field_for_acceleration, params_mask
    )

    # The following part is the equivalent of a
    # > train_loss_value, grads = jax.values_and_grad(total_loss.evaluate)(params, ...)
    # but it is decomposed on individual loss terms so that we can use it
    # if needed for updating loss weights.
    # Since the total loss is a weighted sum of individual loss terms, so
    # are its total gradients.

    # 1. Compute individual losses and individual gradients
    loss_terms, grad_terms = loss.evaluate(
        opt_params_accel
        if opt_state_field_for_acceleration is not None
        else opt_params,
        batch,
        non_opt_params=non_opt_params,
        ret_std_grad_terms=True,
    )

    if loss.update_weight_method is not None and with_loss_weight_update:
        key, subkey = jax.random.split(key)  # type: ignore because key can
        # still be None currently
        # avoid computations of tree_at if no updates
        loss = loss.update_weights(
            i, loss_terms, loss_container.stored_loss_terms, grad_terms, subkey
        )

    # 2. total grad
    grads = loss.ponderate_and_sum_gradient(grad_terms)

    # total loss
    train_loss_value = loss.ponderate_and_sum_loss(loss_terms)

    opt_grads, _ = grads.partition(
        params_mask
    )  # because the update cannot be made otherwise

    # Here, we only use the gradient step of the Optax optimizer on the
    # parameters specified by params_mask. , no dummy state with filled with zero entries
    # all other entries of the pytrees are None thanks to params_mask)
    opt_params, opt_state = _gradient_step(
        opt_grads,
        optimizer,
        opt_params,  # NOTE that we never give the accelerated
        # params here, this would be a wrong procedure
        opt_state,
    )

    params, state = _get_unmasked_optimization_stuff(
        opt_params,
        non_opt_params,
        state,
        opt_state,
        non_opt_state,
        params_mask,
    )

    # check if any of the parameters is NaN
    last_non_nan_params = jax.lax.cond(
        _check_nan_in_pytree(params),
        lambda _: last_non_nan_params,
        lambda _: params,
        None,
    )
    return train_loss_value, params, last_non_nan_params, state, loss, loss_terms


@partial(jit, static_argnames=["optimizer", "params_mask", "with_loss_weight_update"])
def _loss_evaluate_and_natural_gradient_step(
    i,
    batch: AnyBatch,
    loss: AbstractLoss,
    params: Params[Array],
    last_non_nan_params: Params[Array],
    state: optax.OptState,
    optimizer: optax.GradientTransformation,
    loss_container: LossContainer,
    key: PRNGKeyArray,
    params_mask: Params[bool] | None = None,
    opt_state_field_for_acceleration: str | None = None,
    with_loss_weight_update: bool = True,
):
    """
    # The crux of our new approach is partitioning and recombining the parameters and optimization state according to params_mask.

    params_mask:
        A jinns.parameters.Params object with boolean as leaves, specifying
        over which parameters optimization is enabled. This usually implies
        important computational gains. Internally, it is used as the
        filter_spec of a eqx.partition function on the parameters. Note that this
        differs from (and complement) DerivativeKeys, as the latter allows
        for more granularity by freezing some gradients with respect to
        different loss terms, but do not subset the optimized parameters globally.

    NOTE: in this function body, we change naming convention for concision:
     * `state` refers to the general optimizer state
     * `opt_state` refers to the masked optimizer state, i.e. which are
     really involved in the parameter update as defined by `params_mask`.
     * `non_opt_state` refers to the the optimizer state for non-optimized
     params.
    """
    (
        opt_params,
        opt_params_accel,
        non_opt_params,
        opt_state,
        non_opt_state,
    ) = _get_masked_optimization_stuff(
        params, state, opt_state_field_for_acceleration, params_mask
    )
    # 1. Get the unreduced residuals and their gradient (for each sample)
    # for each loss term
    r, g = loss.evaluate_natural_gradient(
        opt_params_accel
        if opt_state_field_for_acceleration is not None
        else opt_params,
        batch,
        non_opt_params=non_opt_params,
        ret_nat_grad_terms=True,
    )

    def post_process_pytree_of_grad(y):
        # TODO: document to describe steps
        # TODO: take loss_weights into account ?

        l = jax.tree.map(
            lambda pt: jax.tree.leaves(
                pt.nn_params, is_leaf=lambda x: eqx.is_inexact_array(x)
            ),
            y,
            is_leaf=lambda x: isinstance(x, Params),
        )

        l2 = jax.tree.map(
            lambda l1: [a.reshape((a.shape[0], -1)) for a in l1],
            l,
            is_leaf=lambda x: isinstance(x, list),
        )

        l3 = jax.tree.map(
            lambda leaf: jnp.concatenate(leaf, axis=1),
            l2,
            is_leaf=lambda x: isinstance(x, list),
        )

        return jnp.concatenate(jax.tree.leaves(l3), axis=0)

    # Flatten the pytree of params as a big (n, p) matrix
    M = post_process_pytree_of_grad(g)
    R = jnp.concatenate(jax.tree.leaves(r), axis=0)

    # Form euclidean grad
    # NOTE: beware that euclidean gradient (might) differs from jax.grad(loss.evaluate) here. Indeed jinns takes the sum(mean(loss_type)) while here we compute mean(sum(all_loss_types). These might differs when different number of samples are used.
    # Equality can be matched by changing the jinns reduction function internally.
    euclidean_grad_array = jnp.mean(R * M, axis=0)

    # Assemble Gram Matrix
    n = M.shape[0]
    gram_mat = (1 / n) * M.T @ M

    # Solve the linear system Gx = eucl_grad
    reg = 1e-5
    n_param = gram_mat.shape[0]
    natural_grad_array = jax.scipy.linalg.solve(
        gram_mat + reg * jnp.eye(n_param), euclidean_grad_array, assume_a="sym"
    )

    # Final step : restructure the natural gradient as a nn_params PyTree
    def nn_params_array_to_pytree(nn_params_array, _eq_params=None):
        _, params_cumsum = _get_param_nb(opt_params.nn_params)
        ng_flat = eqx.tree_at(
            jax.tree.leaves,
            opt_params.nn_params,
            jnp.split(nn_params_array, params_cumsum[:-1]),
        )

        nn_params_pt = jax.tree.map(
            lambda a, b: a.reshape(b.shape),
            ng_flat,
            opt_params.nn_params,
            is_leaf=lambda x: isinstance(x, jnp.ndarray),
        )
        # Wrap everything in a Params() object
        # by default eq_params is filled with Zeros so that additive updates
        # leave them unchanged.
        if _eq_params is None:
            _eq_params = optax.tree.zeros_like(params.eq_params)
        return Params(
            nn_params=nn_params_pt,
            eq_params=_eq_params,
        )

    natural_grads = nn_params_array_to_pytree(natural_grad_array)
    euclidean_grads = nn_params_array_to_pytree(euclidean_grad_array)

    # For now, NGD is not compatible with weight renormalization

    # total loss
    # TODO: build loss_terms by a reduction on `r`
    loss_terms = jax.tree.map(
        lambda _: 0.0, r, is_leaf=lambda x: isinstance(x, tuple)
    )  # leaf check necessary to handle multi-faceted boundaries
    # TODO: remove computation of loss_terms below (only for debugging)
    loss_terms, _ = loss.evaluate(
        opt_params_accel
        if opt_state_field_for_acceleration is not None
        else opt_params,
        batch,
        non_opt_params=non_opt_params,
        ret_std_grad_terms=True,
    )

    train_loss_value = jnp.mean(
        jnp.concatenate(jax.tree.leaves(jax.tree.map(jnp.square, r)), axis=0)
    )

    opt_natural_grads, _ = natural_grads.partition(params_mask)
    opt_euclidean_grads, _ = euclidean_grads.partition(
        params_mask
    )  # useful for backtracking

    # Here, we only use the gradient step of the Optax optimizer on the
    # parameters specified by params_mask. , no dummy state with filled with zero entries
    # all other entries of the pytrees are None thanks to params_mask)

    # Here we force optax vanilla additive gradient update
    # since it makes no sense to use other optimizers
    # TODO: check sgd only
    # assert isinstance(optimizer, optax.sgd)

    # NOTE: we don't use `_gradient_step` here because of its @jit decorator.
    # Indeed, it cannot be modified to accept extra kwargs for
    # optimizer.update(). The latter is necessary for backtracking line search
    # (or any other optax.BaseTransformationExtraArgs).
    def ngd_value_fn(params):
        # Not using loss.evaluate here cause of the mean(sum()) vs sum(mean)
        # remark. This fn computes the loss we are truly minimizing with NGD.
        r = loss.evaluate_natural_gradient(
            params,  # it should be ok to pass `params` here, no need to use
            # _get_masked_optimization_stuff
            batch,
            non_opt_params=None,
            ret_nat_grad_terms=False,
        )
        total_loss = jnp.mean(
            jnp.concatenate(jax.tree.leaves(jax.tree.map(jnp.square, r)), axis=0)
        )
        return total_loss

    if not isinstance(optimizer, optax.GradientTransformationExtraArgs):
        # TODO: maybe just modify type hint instead of raising error ?
        raise TypeError(
            f"Natural gradient `update` need an"
            f"`optax.GradientTransformationExtraArgs`."
            f"You passed an {type(optimizer)}."
        )

    updates, opt_state = optimizer.update(
        opt_natural_grads,
        opt_state,
        opt_params,
        # extra kwargs passed to backtracking line search `update()` method
        value=train_loss_value,
        grad=opt_euclidean_grads,
        value_fn=ngd_value_fn,
    )

    opt_params = optax.apply_updates(opt_params, updates)  # type: ignore

    params, state = _get_unmasked_optimization_stuff(
        opt_params,
        non_opt_params,
        state,
        opt_state,
        non_opt_state,
        params_mask,
    )

    # check if any of the parameters is NaN
    last_non_nan_params = jax.lax.cond(
        _check_nan_in_pytree(params),
        lambda _: last_non_nan_params,
        lambda _: params,
        None,
    )
    return train_loss_value, params, last_non_nan_params, state, loss, loss_terms


@partial(
    jit,
    static_argnames=["optimizer"],
)
def _gradient_step(
    grads: Params[Array],
    optimizer: optax.GradientTransformation,
    params: Params[Array],
    state: optax.OptState,
) -> tuple[
    Params[Array],
    optax.OptState,
]:
    """
    optimizer cannot be jit-ted.

    a plain old gradient step that is compatible with the new masked update
    stuff.

    All kwargs are passed to `optimizer.update()` in case user provided a
    optax.GradientTransformationExtraArgs
    """

    updates, state = optimizer.update(
        grads,  # type: ignore
        state,
        params,  # type: ignore
    )  # Also see optimizer.init for explanation of type ignore
    params = optax.apply_updates(params, updates)  # type: ignore

    return (
        params,
        state,
    )


@partial(jit, static_argnames=["params_mask"])
def _get_masked_optimization_stuff(
    params, state, state_field_for_acceleration, params_mask
):
    """
    From the parameters `params`, the optimizer state `state`, we use the
    parameter mask `params_mask` to retrieve the partitioned version of those
    two objects, `opt_params` for the parameters that are optimized and
    `non_opt_params` for those that are not optimized. Same for `state`.

    The argument `state_field_for_acceleration` can correspond to a field
    inside the `state` module. If it is not None, a `opt_params_accel` object
    is created that is different of `opt_params`. See
    `opt_state_field_for_acceleration` in `jinns.solve` docstring for more
    details.

    The opposite of `eqx.partition` ie, `eqx.combine` is made in the loss
    `evaluate_by_terms()` method for the computations and in
    `_get_unmasked_optimization_stuff` to reconstruct the object after the
    gradient step
    """
    opt_params, non_opt_params = params.partition(params_mask)
    opt_state = jax.tree.map(
        lambda l: l.partition(params_mask)[0] if isinstance(l, Params) else l,
        state,
        is_leaf=lambda x: isinstance(x, Params),
    )
    non_opt_state = jax.tree.map(
        lambda l: l.partition(params_mask)[1] if isinstance(l, Params) else l,
        state,
        is_leaf=lambda x: isinstance(x, Params),
    )

    # NOTE to enable optimization procedures with acceleration
    if state_field_for_acceleration is not None:
        opt_params_accel = getattr(opt_state, state_field_for_acceleration)
    else:
        opt_params_accel = opt_params

    return (
        opt_params,
        opt_params_accel,
        non_opt_params,
        opt_state,
        non_opt_state,
    )


@partial(jit, static_argnames=["params_mask"])
def _get_unmasked_optimization_stuff(
    opt_params, non_opt_params, state, opt_state, non_opt_state, params_mask
):
    """
    Reverse operations of `_get_masked_optimization_stuff`
    """
    # NOTE the combine which closes the partitioned chunck
    if params_mask is not None:
        params = eqx.combine(opt_params, non_opt_params)
        state = jax.tree.map(
            lambda a, b, c: eqx.combine(b, c) if isinstance(a, Params) else b,
            # NOTE else b in order to take all non Params stuff from
            # opt_state that may have been updated too
            state,
            opt_state,
            non_opt_state,
            is_leaf=lambda x: isinstance(x, Params),
        )
    else:
        params = opt_params
        state = opt_state

    return params, state


@partial(jit, static_argnames=["prefix"])
def _print_fn(i: int, loss_val: Float, print_loss_every: int, prefix: str = ""):
    # note that if the following is not jitted in the main for loop, it is
    # super slow
    _ = jax.lax.cond(
        i % print_loss_every == 0,
        lambda _: jax.debug.print(
            prefix + "Iteration {i}: loss value = {loss_val}",
            i=i,
            loss_val=loss_val,
        ),
        lambda _: None,
        (None,),
    )


@jit
def _store_loss_and_params(
    i: int,
    params: Params[Array],
    stored_params: Params[Array | None],
    loss_container: LossContainer,
    train_loss_val: float,
    loss_terms: PyTree[Array],
    weight_terms: PyTree[Array],
    tracked_params: Params,
) -> tuple[StoredObjectContainer, LossContainer]:
    stored_params = jax.tree_util.tree_map(
        lambda stored_value, param, tracked_param: (
            None
            if stored_value is None
            else jax.lax.cond(
                tracked_param,
                lambda ope: ope[0].at[i].set(ope[1]),
                lambda ope: ope[0],
                (stored_value, param),
            )
        ),
        stored_params,
        params,
        tracked_params,
        is_leaf=lambda x: x is None,
    )
    stored_loss_terms = jax.tree_util.tree_map(
        lambda stored_term, loss_term: stored_term.at[i].set(loss_term),
        loss_container.stored_loss_terms,
        loss_terms,
    )

    if loss_container.stored_weights_terms is not None:
        stored_weights_terms = jax.tree_util.tree_map(
            lambda stored_term, weight_term: stored_term.at[i].set(weight_term),
            jax.tree.leaves(
                loss_container.stored_weights_terms,
                is_leaf=lambda x: x is not None and eqx.is_inexact_array(x),
            ),
            jax.tree.leaves(
                weight_terms,
                is_leaf=lambda x: x is not None and eqx.is_inexact_array(x),
            ),
        )
        stored_weights_terms = eqx.tree_at(
            lambda pt: jax.tree.leaves(
                pt, is_leaf=lambda x: x is not None and eqx.is_inexact_array(x)
            ),
            loss_container.stored_weights_terms,
            stored_weights_terms,
        )
    else:
        stored_weights_terms = None

    train_loss_values = loss_container.train_loss_values.at[i].set(train_loss_val)
    loss_container = LossContainer(
        stored_loss_terms, stored_weights_terms, train_loss_values
    )
    stored_objects = StoredObjectContainer(stored_params)
    return stored_objects, loss_container


def _get_break_fun(
    n_iter: int,
    verbose: bool,
    conditions_str: tuple[str, ...] = (
        "bool_max_iter",
        "bool_nan_in_params",
        "bool_early_stopping",
    ),
) -> Callable[[SolveCarry | SolveAlternateCarry], bool]:
    """
    Wrapper to get the break_fun with appropriate `n_iter`.
    The verbose argument is here to control printing (or not) when exiting
    the optimisation loop. It can be convenient is jinns.solve is itself
    called in a loop and user want to avoid std output.
    """

    @jit
    def break_fun(carry: tuple):
        """
        Function to break from the main optimization loop whe the following
        conditions are met : maximum number of iterations, NaN
        appearing in the parameters, and early stopping criterion.
        """

        def stop_while_loop(msg):
            """
            Note that the message is wrapped in the jax.lax.cond because a
            string is not a valid JAX type that can be fed into the operands
            """
            if verbose:
                jax.debug.print(f"\nStopping main optimization loop, cause: {msg}")
            return False

        def continue_while_loop(_):
            return True

        i = carry[0]
        optimization = carry[2]
        optimization_extra = carry[3]

        conditions_bool = ()
        if "bool_max_iter" in conditions_str:
            # Condition 1
            bool_max_iter = jax.lax.cond(
                i >= n_iter,
                lambda _: stop_while_loop("max iteration is reached"),
                continue_while_loop,
                None,
            )
            conditions_bool += (bool_max_iter,)
        if "bool_nan_in_params" in conditions_str:
            # Condition 2
            bool_nan_in_params = jax.lax.cond(
                _check_nan_in_pytree(optimization.params),
                lambda _: stop_while_loop(
                    "NaN values in parameters (returning last non NaN values)"
                ),
                continue_while_loop,
                None,
            )
            conditions_bool += (bool_nan_in_params,)
        if "bool_early_stopping" in conditions_str:
            # Condition 3
            bool_early_stopping = jax.lax.cond(
                optimization_extra.early_stopping,
                lambda _: stop_while_loop("early stopping"),
                continue_while_loop,
                None,
            )
            conditions_bool += (bool_early_stopping,)

        # stop when one of the cond to continue is False
        return jax.tree_util.tree_reduce(
            lambda x, y: jnp.logical_and(jnp.array(x), jnp.array(y)),
            conditions_bool,
        )

    return break_fun


def _build_get_batch(
    obs_batch_sharding: jax.sharding.Sharding | None,
) -> Callable[
    [
        AbstractDataGenerator,
        DataGeneratorParameter | None,
        DataGeneratorObservations | None,
    ],
    tuple[
        AnyBatch,
        AbstractDataGenerator,
        DataGeneratorParameter | None,
        DataGeneratorObservations | None,
    ],
]:
    """
    Return the get_batch function that will be used either the jittable one or
    the non-jittable one with sharding using jax.device.put()
    """

    def get_batch_sharding(data, param_data, obs_data):
        """
        This function is used at each loop but it cannot be jitted because of
        device_put
        """
        data, batch = data.get_batch()
        if param_data is not None:
            param_data, param_batch = param_data.get_batch()
            batch = append_param_batch(batch, param_batch)
        if obs_data is not None:
            # This is the part that motivated the transition from scan to for loop
            # Indeed we need to be transit obs_batch from CPU to GPU when we have
            # huge observations that cannot fit on GPU. Such transfer wasn't meant
            # to be jitted, i.e. in a scan loop
            obs_data, obs_batch = obs_data.get_batch()
            obs_batch = jax.device_put(obs_batch, obs_batch_sharding)
            batch = append_obs_batch(batch, obs_batch)
        return batch, data, param_data, obs_data

    @jit
    def get_batch(data, param_data, obs_data):
        """
        Original get_batch with no sharding
        """
        data, batch = data.get_batch()
        if param_data is not None:
            param_data, param_batch = param_data.get_batch()
            batch = append_param_batch(batch, param_batch)
        if obs_data is not None:
            obs_data, obs_batch = obs_data.get_batch()
            batch = append_obs_batch(batch, obs_batch)
        return batch, data, param_data, obs_data

    if obs_batch_sharding is not None:
        return get_batch_sharding
    return get_batch


def _check_batch_size(other_data, main_data, attr_name):
    if isinstance(main_data, DataGeneratorODE):
        if main_data.temporal_batch_size is not None:
            if getattr(other_data, attr_name) != main_data.temporal_batch_size:
                raise ValueError(
                    f"{other_data.__class__}.{attr_name} must be equal"
                    f" to {main_data.__class__}.temporal_batch_size for correct"
                    " vectorization"
                )
        else:
            if main_data.nt is not None:
                if getattr(other_data, attr_name) != main_data.nt:
                    raise ValueError(
                        f"{other_data.__class__}.{attr_name} must be equal"
                        f" to {main_data.__class__}.nt for correct"
                        " vectorization"
                    )
    if isinstance(main_data, CubicMeshPDEStatio) and not isinstance(
        main_data, CubicMeshPDENonStatio
    ):
        if main_data.omega_batch_size is not None:
            if getattr(other_data, attr_name) != main_data.omega_batch_size:
                raise ValueError(
                    f"{other_data.__class__}.{attr_name} must be equal"
                    f" to {main_data.__class__}.omega_batch_size for correct"
                    " vectorization"
                )
        else:
            if main_data.n is not None:
                if getattr(other_data, attr_name) != main_data.n:
                    raise ValueError(
                        f"{other_data.__class__}.{attr_name} must be equal"
                        f" to {main_data.__class__}.n for correct"
                        " vectorization"
                    )
        if main_data.omega_border_batch_size is not None:
            if getattr(other_data, attr_name) != main_data.omega_border_batch_size:
                raise ValueError(
                    f"{other_data.__class__}.{attr_name} must be equal"
                    f" to {main_data.__class__}.omega_border_batch_size for correct"
                    " vectorization"
                )
        else:
            if main_data.nb is not None:
                if getattr(other_data, attr_name) != main_data.nb:
                    raise ValueError(
                        f"{other_data.__class__}.{attr_name} must be equal"
                        f" to {main_data.__class__}.nb for correct"
                        " vectorization"
                    )
    if isinstance(main_data, CubicMeshPDENonStatio):
        if main_data.domain_batch_size is not None:
            if getattr(other_data, attr_name) != main_data.domain_batch_size:
                raise ValueError(
                    f"{other_data.__class__}.{attr_name} must be equal"
                    f" to {main_data.__class__}.domain_batch_size for correct"
                    " vectorization"
                )
        else:
            if main_data.n is not None:
                if getattr(other_data, attr_name) != main_data.n:
                    raise ValueError(
                        f"{other_data.__class__}.{attr_name} must be equal"
                        f" to {main_data.__class__}.n for correct"
                        " vectorization"
                    )
        if main_data.border_batch_size is not None:
            if getattr(other_data, attr_name) != main_data.border_batch_size:
                raise ValueError(
                    f"{other_data.__class__}.{attr_name} must be equal"
                    f" to {main_data.__class__}.border_batch_size for correct"
                    " vectorization"
                )
        else:
            if main_data.nb is not None:
                if main_data.dim > 1 and getattr(other_data, attr_name) != (
                    main_data.nb // 2**main_data.dim
                ):
                    raise ValueError(
                        f"{other_data.__class__}.{attr_name} must be equal"
                        f" to ({main_data.__class__}.nb // 2**{main_data.__class__}.dim)"
                        " for correct vectorization"
                    )
        if main_data.initial_batch_size is not None:
            if getattr(other_data, attr_name) != main_data.initial_batch_size:
                raise ValueError(
                    f"{other_data.__class__}.{attr_name} must be equal"
                    f" to {main_data.__class__}.initial_batch_size for correct"
                    " vectorization"
                )
        else:
            if main_data.ni is not None:
                if getattr(other_data, attr_name) != main_data.ni:
                    raise ValueError(
                        f"{other_data.__class__}.{attr_name} must be equal"
                        f" to {main_data.__class__}.ni for correct"
                        " vectorization"
                    )
    if isinstance(main_data, DataGeneratorParameter):
        if main_data.param_batch_size is not None:
            if getattr(other_data, attr_name) != main_data.param_batch_size:
                raise ValueError(
                    f"{other_data.__class__}.{attr_name} must be equal"
                    f" to {main_data.__class__}.param_batch_size for correct"
                    " vectorization"
                )
        else:
            if main_data.n is not None:
                if getattr(other_data, attr_name) != main_data.n:
                    raise ValueError(
                        f"{other_data.__class__}.{attr_name} must be equal"
                        f" to {main_data.__class__}.n for correct"
                        " vectorization"
                    )
