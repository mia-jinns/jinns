"""
Common functions for _solve.py and _solve_alternate.py
"""

from __future__ import (
    annotations,
)  # https://docs.python.org/3/library/typing.html#constant

from typing import TYPE_CHECKING, Callable
import jax
from jax import jit
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import PyTree, Float, Array, PRNGKeyArray
import optax


from jinns.loss._loss_components import ODEComponents
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
from jinns.optimizers._natural_gradient import NGDState
from jinns.optimizers._utils_ngd import (
    _get_sqrt_weights_per_sample,
    _reweight_pytree,
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


def _loss_evaluate_and_gradient_step(**kwargs):
    """
    Intermediate function where the bifurcation is made
    This seems to be better practice and solve several pyright issues
    """
    if isinstance(kwargs["state"], NGDState):
        return _loss_evaluate_and_natural_gradient_step(**kwargs)
    else:
        return _loss_evaluate_and_euclidean_gradient_step(**kwargs)


def _loss_evaluate_and_euclidean_gradient_step(
    *,
    i: int,
    batch: AnyBatch,
    loss: AbstractLoss,
    params: Params[Array],
    last_non_nan_params: Params[Array],
    state: optax.OptState,
    optimizer: optax.GradientTransformation,
    loss_container: LossContainer,
    key: PRNGKeyArray | None,
    params_mask: Params[bool] | None = None,
    state_field_for_acceleration: str | None = None,
    with_loss_weight_update: bool = True,
    **__,  # ignore supplementary kwargs that are for
    # _loss_evaluate_and_natural_gradient_step
):
    """Computes loss values and (euclidean) gradient, then performs the update
    according to the given optimizer.

    NOTE: Jinns allows partitioning and recombining the parameters
    and optimization state according to `params_mask`.

    NOTE: in this function body, we change naming convention for concision:
     * `state` refers to the general optimizer state
     * `opt_state` refers to the unmasked optimizer state, i.e. which are
     really involved in the parameter update as defined by `params_mask`.
     * `non_opt_state` refers to the the optimizer state for non-optimized
     params.

    Parameters
    ----------



    Parameters
    ----------
    i : int
        iteration number
    batch : AnyBatch
        the current batch
    loss : AbstractLoss
        the jinns.loss to minimize
    params : Params[Array]
        the current parameters
    last_non_nan_params : Params[Array]
        the last non NaN parameters.
    state : optax.OptState
        the optimizer state
    optimizer : optax.GradientTransformation
        the optimizer
    loss_container : LossContainer
        the loss values throughout optimization.
    key : PRNGKeyArray
        the curren PRNG key
    params_mask : Params[bool] | None, optional
        A jinns.parameters.Params object with boolean as leaves, specifying
        over which parameters optimization is enabled. This usually implies
        important computational gains. Internally, it is used as the
        `filter_spec` of an `eqx.partition` on the parameters. Note that this
        differs from (and complement) DerivativeKeys, as the latter allows
        for more granularity by freezing some gradients with respect to
        different loss terms, but do not subset the optimized parameters globally.
        by default None
    state_field_for_acceleration : str | None, optional
        The argument `state_field_for_acceleration` can correspond to a field
        inside the `state` module. If it is not None, a `opt_params_accel` object
        is created that is different of `opt_params`. See
        `opt_state_field_for_acceleration` in `jinns.solve` docstring for more
        details. By default None
    with_loss_weight_update : bool, optional
        should we update loss_weights for next iteration according to current loss_values
        by default True

    Returns
    -------
    tuple
        a tuple with curent loss value, params, last non nan params, optimizer state, loss object, loss per term
    """

    ## NOTE to enable optimization procedures with acceleration
    if state_field_for_acceleration is not None:
        params_accel = getattr(state, state_field_for_acceleration)
    else:
        params_accel = None
        # opt_params_accel = opt_params

    # The following part is the equivalent of a
    # > train_loss_value, grads = jax.values_and_grad(total_loss.evaluate)(params, ...)
    # but it is decomposed on individual loss terms so that we can use it
    # if needed for updating loss weights.
    # Since the total loss is a weighted sum of individual loss terms, so
    # are its total gradients.

    # 1. Compute individual losses and individual gradients
    loss_terms, grad_terms = loss.values_and_grads(
        params_accel if params_accel is not None else params,
        batch,
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

    # 3. total loss after possible weight update
    train_loss_value = loss.ponderate_and_sum_loss(loss_terms)

    # extra_args_optax_keys = {} # the extra_args dict that is used in optax
    # extra_args_keys = []
    # extra_args_to_jinns_locals = {
    #    "value": "train_loss_value",
    #    "value_fn": ("loss", "evaluate"),
    #    "grad": "params",
    #    "batch": "batch"
    # }
    # if 'extra_args' in inspect.signature(optimizer.update).parameters.keys():
    #    # possible cases either a single optax transform with extra args
    #    # or a chain of optax transforms with or without extra args
    #    extra_args_keys = []
    #    if 'update_fns' in inspect.getclosurevars(optimizer.update).nonlocals.keys():
    #        # this is a chain optax transform
    #        # get the arguments of each update_fn
    #        for update_fn_inside_chain in inspect.getclosurevars(optimizer.update).nonlocals['update_fns']:
    #            # first three args are always 'updates', 'state', 'params'
    #            # and they don't interest us
    #            extra_args_keys += list(
    #                inspect.signature(update_fn_inside_chain).parameters.keys()
    #            )[3:]
    #    # custom rule for jinns: if value_fn is require (ie loss.evaluate) then
    #    # batch is necesarily an extra_arg since we always call
    #    # loss.evaluate(params, batch)
    # else:
    #    # this can be optax transform with extra args without extra_args named!
    #    extra_args_for_update_fn =

    # Below are the stuff passed by the user when they see they have an optax
    # transform with extra args. The example below enables using optax.lbfgs
    # whose update extra args contain in order: value, grad, value_fn,
    # extra_kwargs (for value_fn)
    # the user should then look for the corresponding variable in jinns
    # loss_evaluate_and_standard_gradient and pass the variable name as a
    # string
    extra_optax_args_and_kwargs_in_jinns_locals = {
        "value": "train_loss_value",
        "grad": "params",
        "value_fn": "lambda params, batch: loss.evaluate(params, batch)[0]",
        "batch": "batch",
    }
    # extra_kwargs_optax_in_jinns_locals = {"batch": "batch"}

    # TODO use the above code to ensure user's dictionary match what's expected
    # by optax

    extra_args_and_kwargs_for_update_fn = {}
    for kw, variable_name in extra_optax_args_and_kwargs_in_jinns_locals.items():
        jinns_local_var = eval(variable_name, locals())
        # var_name_components = variable_name.split(".")
        # try:
        #    jinns_local_var = locals()[var_name_components[0]]
        #    # we must access attributes if len(var_name_component) > 1
        #    for var_name_component in var_name_components[1:]:
        #        jinns_local_var = getattr(jinns_local_var, var_name_component)
        # except KeyError as e:
        #    raise ValueError(
        #        f"No local variable corresponding to" f"{variable_name}"
        #    ) from e
        extra_args_and_kwargs_for_update_fn[kw] = jinns_local_var

    # extra_kwargs_for_update_fn = {}
    # for kw, variable_name in extra_kwargs_optax_in_jinns_locals.items():
    #    var_name_components = variable_name.split(".")
    #    try:
    #        jinns_local_var = locals()[var_name_components[0]]
    #        # we must access attributes if len(var_name_component) > 1
    #        for var_name_component in var_name_components[1:]:
    #            jinns_local_var = getattr(jinns_local_var, var_name_component)
    #    except KeyError as e:
    #        raise ValueError(
    #            f"No local variable corresponding to" f"{variable_name}"
    #        ) from e
    #    extra_kwargs_for_update_fn[kw] = jinns_local_var

    params, state = _gradient_step(
        grads,
        optimizer,
        params,  # NOTE that we never give the accelerated
        # params here, this would be a wrong procedure
        state,
        params_mask,
        **extra_args_and_kwargs_for_update_fn,
    )

    # check if any of the parameters is NaN
    last_non_nan_params = jax.lax.cond(
        _check_nan_in_pytree(params),
        lambda _: last_non_nan_params,
        lambda _: params,
        None,
    )
    return train_loss_value, params, last_non_nan_params, state, loss, loss_terms


def _loss_evaluate_and_natural_gradient_step(
    *,
    batch: AnyBatch,
    loss: AbstractLoss,
    params: Params[Array],
    last_non_nan_params: Params[Array],
    state: NGDState,
    optimizer: optax.GradientTransformation,
    params_mask: Params[bool] | None = None,
    state_field_for_acceleration: str | None = None,
    **__,  # to ignore more arguments
):
    """Similar to ` _loss_evaluate_and_euclidean_gradient_step` but for natural gradient.

    This step function
        1. Computes the loss on the current `batch` and current `params`
        2. Computes the natural gradient descent (NGD) updates
        3. Feed these NGD updates to the optimizer provided by user.
           NOTE this optimizer needs to use an `NGDState`.

    For more details see docstring in jinns/optimizers/_natural_gradient.py

    All arguments are similar to `_loss_evaluate_and_gradient_step` above except for
    one extra argument

    Parameters
    ----------

    Returns
    -------
    tuple
        a tuple with curent loss value, params, last non nan params, optimizer state, loss object, loss per term
    """

    if not isinstance(optimizer, optax.GradientTransformationExtraArgs):
        # TODO: maybe just modify type hint instead of raising error ?
        raise TypeError(
            f"Natural gradient `update` need an"
            f"`optax.GradientTransformationExtraArgs`."
            f"You passed an {type(optimizer)}."
        )

    ## NOTE to enable optimization procedures with acceleration
    if state_field_for_acceleration is not None:
        params_accel = getattr(state, state_field_for_acceleration)
    else:
        params_accel = None
        # opt_params_accel = opt_params

    # --
    # Get the unreduced residuals and their gradient (for each sample)
    # for each loss term
    r, g = loss.values_and_grad_per_sample(
        params_accel if params_accel is not None else params,
        batch,
    )

    # --
    # Preprocess loss_weights if needed: we want to have tuples at field boundary_loss in d>1
    # because of multiple facets.
    if not isinstance(r, ODEComponents) and r.boundary_loss is not None:
        lw_ = eqx.tree_at(
            lambda pt: pt.boundary_loss,
            loss.loss_weights,
            tuple(loss.loss_weights.boundary_loss for _ in range(len(r.boundary_loss))),
        )
    else:
        lw_ = loss.loss_weights

    # --
    # We can also assemble the total loss and loss_terms from reweighted residuals `r`
    # NOTE: subtility here, we don't reweight by * 1/sqrt{n} because the averaging is done
    # inside loss._reduction_functions so we don't want to account twice for it.
    sqrt_weights_per_sample = _get_sqrt_weights_per_sample(lw=lw_, r=r, batch_norm=True)
    sqrt_weights_per_sample_no_avg = _get_sqrt_weights_per_sample(
        lw=lw_, r=r, batch_norm=False
    )
    loss_terms = jax.tree.map(
        lambda red_fun, r_: red_fun(r_),
        loss._reduction_functions,
        _reweight_pytree(pt=r, lw=sqrt_weights_per_sample_no_avg),
    )

    train_loss_value = jnp.sum(
        jnp.concatenate(
            jax.tree.leaves(
                jax.tree.map(
                    lambda arr: jnp.sum(arr**2, axis=-1),
                    _reweight_pytree(pt=r, lw=sqrt_weights_per_sample),
                ),
            ),
            axis=0,
        )
    )

    params, state = _gradient_step(
        (r, g, sqrt_weights_per_sample),
        optimizer,
        params,
        state,  # type: ignore
        params_mask,
        # extra kwargs passed to the NGD optimizer
        loss=loss,
        batch=batch,
        loss_value=train_loss_value,
    )

    # check if any of the parameters is NaN
    last_non_nan_params = jax.lax.cond(
        _check_nan_in_pytree(params),
        lambda _: last_non_nan_params,
        lambda _: params,
        None,
    )
    return train_loss_value, params, last_non_nan_params, state, loss, loss_terms


def _gradient_step(
    grads: Params[Array | None] | tuple,
    optimizer: optax.GradientTransformation,
    params: Params[Array],
    state: optax.OptState,
    params_mask: Params[bool] | None = None,
    *args,
    **kwargs,
) -> tuple[
    Params[Array],
    optax.OptState,
]:
    """

    A plain old gradient step that takes into account parameter masking.

    **When doing an alternate optimization** with jinns.solve_alternate(), the
    params_mask will be True for the optimized parameters and False for the
    others. This enables to use eqx.partition, to set None at non optimized
    leaves (of the gradient pytree, of the
    parameters pytree, ...) in order to save computations (and avoid the
    default "fill with 0" behaviour of optax alternate. NOTE that the masking
    for `opt_state`, is already done at an outer level (during `carry`
    manipulations before and after a local while loop (on nn_params or
    eq_params is started) for efficiency.

    All kwargs are passed to `optimizer.update()` in case user provided a
    optax.GradientTransformationExtraArgs
    """

    # NOTE, if params_mask is not None, this means we are doing
    # solve_alternate, this means state has already been masked and so grads
    # must be masked too to avoid errors of the type:
    # TypeError: unsupported operand type(s) for *: 'float' and 'NoneType'
    # Same for params (see below)
    if isinstance(state, NGDState):
        assert isinstance(grads, tuple)  # for type checking
        # in this case the grads object if different
        opt_grads = jax.tree.map(
            lambda l: l.partition(params_mask)[0],
            grads[1],
            is_leaf=lambda x: isinstance(x, Params),
        )
        opt_grads = (grads[0], opt_grads, grads[2])
    else:
        assert isinstance(grads, Params)  # for type checking
        opt_grads, _ = grads.partition(params_mask)

    # NOTE, here we need to pass the full unmasked params for more complex
    # optimizer updates to be done (updates which rely on params or more).
    # NOTE also that, while compute gradients beforehand, we have computed
    # gradients for ALL parameters, even the non optimized ones: we hope that
    # JAX interal machinery will discard those computations from the
    # computational graph since the gradients for the non optimized parameters
    # are finally not used.
    # Making sure gradients are only computed for optimized (unmasked)
    # parameters seem like a big jinns refactorization (future TODO)
    updates, state = optimizer.update(
        opt_grads,  # type: ignore
        state,
        params,  # type: ignore
        *args,
        **kwargs,
    )  # Also see optimizer.init for explanation of type ignore

    # NOTE, if params_mask is not None, this means we are doing
    # solve_alternate, this means state has already been masked and so grads
    # and thus updates contain None. Hence params
    # must be masked too to avoid errors of the type:
    # TypeError: unsupported operand type(s) for *: 'float' and 'NoneType'
    opt_params, non_opt_params = params.partition(params_mask)
    opt_params = optax.apply_updates(opt_params, updates)  # type: ignore

    if params_mask is not None:
        params = eqx.combine(opt_params, non_opt_params)  # type: ignore
        # (bad cohabitaiton with PyTree)
    else:
        params = opt_params  # type: ignore (bad cohabitaiton with PyTree)

    return (
        params,
        state,
    )


def _get_masked_state(state, params_mask):
    """
    From the optimizer state, we use the
    parameter mask `params_mask` to retrieve the partitioned version of the
    objects, `opt_state` for the parameters that are optimized and
    `non_opt_state` for those that are not optimized.
    """
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

    return opt_state, non_opt_state


def _get_unmasked_state(state, opt_state, non_opt_state):
    """
    Reverse operations of `_get_masked_state`
    """
    state = jax.tree.map(
        lambda a, b, c: eqx.combine(b, c) if isinstance(a, Params) else b,
        # NOTE else b in order to take all non Params stuff from
        # opt_state that may have been updated too
        state,
        opt_state,
        non_opt_state,
        is_leaf=lambda x: isinstance(x, Params),
    )

    return state


@jit(static_argnames=["prefix"])  # new in jax 0.8.1
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


def _store_loss_and_params(
    i: int,
    params: Params[Array],
    stored_params: Params[Array | None],
    loss_container: LossContainer,
    train_loss_val: Float[Array, " "],
    loss_terms: PyTree[Array],
    weight_terms: PyTree[Array],
    tracked_params: Params | None,
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
        batch_size = getattr(other_data, attr_name)  # this can be a tuple with
        # DataGeneratorObservations
        if not isinstance(batch_size, tuple):
            batch_size = (batch_size,)

        for bs in batch_size:
            if main_data.param_batch_size is not None:
                if bs != main_data.param_batch_size:
                    raise ValueError(
                        f"{other_data.__class__}.{attr_name} must be equal"
                        f" to {main_data.__class__}.param_batch_size for correct"
                        " vectorization"
                    )
            else:
                if main_data.n is not None:
                    if bs != main_data.n:
                        raise ValueError(
                            f"{other_data.__class__}.{attr_name} must be equal"
                            f" to {main_data.__class__}.n for correct"
                            " vectorization"
                        )
