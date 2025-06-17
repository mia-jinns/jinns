"""
This modules implements the main `solve()` function of jinns which
handles the optimization process
"""

from __future__ import (
    annotations,
)  # https://docs.python.org/3/library/typing.html#constant

import time
from typing import TYPE_CHECKING, Any, TypeAlias, Callable
from functools import partial
import optax
import jax
from jax import jit
import jax.numpy as jnp
from jaxtyping import Float, Array, PyTree, Key
import equinox as eqx
from jinns.solver._rar import init_rar, trigger_rar
from jinns.utils._utils import _check_nan_in_pytree
from jinns.solver._utils import _check_batch_size
from jinns.utils._containers import (
    DataGeneratorContainer,
    OptimizationContainer,
    OptimizationExtraContainer,
    LossContainer,
    StoredObjectContainer,
)
from jinns.data._utils import append_param_batch, append_obs_batch

if TYPE_CHECKING:
    from jinns.parameters._params import Params
    from jinns.utils._types import AnyBatch
    from jinns.loss._abstract_loss import AbstractLoss
    from jinns.validation._validation import AbstractValidationModule
    from jinns.data._DataGeneratorParameter import DataGeneratorParameter
    from jinns.data._DataGeneratorObservations import DataGeneratorObservations
    from jinns.data._AbstractDataGenerator import AbstractDataGenerator

    main_carry: TypeAlias = tuple[
        int,
        AbstractLoss,
        OptimizationContainer,
        OptimizationExtraContainer,
        DataGeneratorContainer,
        AbstractValidationModule | None,
        LossContainer,
        StoredObjectContainer,
        Float[Array, " n_iter"] | None,
        Key | None,
    ]


def solve(
    n_iter: int,
    init_params: Params[Array],
    data: AbstractDataGenerator,
    loss: AbstractLoss,
    optimizer: optax.GradientTransformation,
    print_loss_every: int = 1000,
    opt_state: optax.OptState | None = None,
    tracked_params: Params[Any | None] | None = None,
    param_data: DataGeneratorParameter | None = None,
    obs_data: DataGeneratorObservations | None = None,
    validation: AbstractValidationModule | None = None,
    obs_batch_sharding: jax.sharding.Sharding | None = None,
    verbose: bool = True,
    ahead_of_time: bool = True,
    key: Key = None,
) -> tuple[
    Params[Array],
    Float[Array, " n_iter"],
    PyTree,
    AbstractDataGenerator,
    AbstractLoss,
    optax.OptState,
    Params[Array | None],
    PyTree,
    Float[Array, " n_iter"] | None,
    Params[Array],
]:
    """
    Performs the optimization process via stochastic gradient descent
    algorithm. We minimize the function defined `loss.evaluate()` with
    respect to the learnable parameters of the problem whose initial values
    are given in `init_params`.


    Parameters
    ----------
    n_iter
        The maximum number of iterations in the optimization.
    init_params
        The initial jinns.parameters.Params object.
    data
        A DataGenerator object to retrieve batches of collocation points.
    loss
        The loss function to minimize.
    optimizer
        An optax optimizer.
    print_loss_every
        Default 1000. The rate at which we print the loss value in the
        gradient step loop.
    opt_state
        Provide an optional initial state to the optimizer.
    tracked_params
        Default None. An eqx.Module of type Params with non-None values for
        parameters that needs to be tracked along the iterations.
        None values in tracked_params will not be traversed. Thus
        the user can provide something like `tracked_params = jinns.parameters.Params(
        nn_params=None, eq_params={"nu": True})` while init_params.nn_params
        being a complex data structure.
    param_data
        Default None. A DataGeneratorParameter object which can be used to
        sample equation parameters.
    obs_data
        Default None. A DataGeneratorObservations
        object which can be used to sample minibatches of observations.
    validation
        Default None. Otherwise, a callable ``eqx.Module`` which implements a
        validation strategy. See documentation of `jinns.validation.
        _validation.AbstractValidationModule` for the general interface, and
        `jinns.validation._validation.ValidationLoss` for a practical
        implementation of a vanilla validation stategy on a validation set of
        collocation points.

        **Note**: The ``__call__(self, params)`` method should have
        the latter prescribed signature and return ``(validation [eqx.Module],
        early_stop [bool], validation_criterion [Array])``. It is called every
        ``validation.call_every`` iteration. Users are free to design any
        validation strategy of their choice, and to decide on the early
        stopping criterion.
    obs_batch_sharding
        Default None. An optional sharding object to constraint the obs_batch.
        Typically, a SingleDeviceSharding(gpu_device) when obs_data has been
        created with sharding_device=SingleDeviceSharding(cpu_device) to avoid
        loading on GPU huge datasets of observations.
    verbose
        Default True. If False, no std output (loss or cause of
        exiting the optimization loop) will be produced.
    ahead_of_time
        Default True. Separate the compilation of the main training loop from
        the execution to get both timings. You might need to avoid this
        behaviour if you need to perform JAX transforms over chunks of code
        containing `jinns.solve()` since AOT-compiled functions cannot be JAX
        transformed (see https://jax.readthedocs.io/en/latest/aot.html#aot-compiled-functions-cannot-be-transformed).
        When False, jinns does not provide any timing information (which would
        be nonsense in a JIT transformed `solve()` function).
    key
        Default None. A JAX random key that can be used for diverse purpose in
        the main iteration loop.

    Returns
    -------
    params
        The last non NaN value of the params at then end of the
        optimization process
    total_loss_values
        An array of the total loss term along the gradient steps
    stored_loss_terms
        A PyTree with attributes being arrays of all the values for each loss
        term
    data
        The input data object
    loss
        The input loss object
    opt_state
        The final optimized state
    stored_params
        A Params objects with the stored values of the desired parameters (as
        signified in tracked_params argument)
    stored_weights_terms
        A PyTree with attributes being arrays of all the values for each loss
        weight. Note that if Loss.update_weight_method is None, we return None,
        because loss weights are never updated and we can then save some
        computations
    validation_crit_values
        An array containing the validation criterion values of the training
    best_val_params
        The best parameters according to the validation criterion
    """
    if n_iter < 1:
        raise ValueError("Cannot run jinns.solve for n_iter<1")

    if param_data is not None:
        if param_data.param_batch_size is not None:
            # We need to check that batch sizes will all be compliant for
            # correct vectorization
            _check_batch_size(param_data, data, "param_batch_size")
        else:
            # If DataGeneratorParameter does not have a batch size we will
            # vectorization using `n`, and the same checks must be done
            _check_batch_size(param_data, data, "n")

    if obs_data is not None and param_data is not None:
        # obs_data batch dimensions need only to be aligned with param_data
        # batch dimensions if the latter exist
        if obs_data.obs_batch_size is not None:
            _check_batch_size(obs_data, param_data, "obs_batch_size")
        else:
            _check_batch_size(obs_data, param_data, "n")

    if opt_state is None:
        opt_state = optimizer.init(init_params)  # type: ignore
        # our Params are eqx.Module (dataclass + PyTree), PyTree is
        # compatible with optax transform but not dataclass, this leads to a
        # type hint error: we could prevent this by ensuring with the eqx.filter that
        # we have only floating points optimizable params given to optax
        # see https://docs.kidger.site/equinox/faq/#optax-throwing-a-typeerror
        # opt_state = optimizer.init(
        #    eqx.filter(init_params, eqx.is_inexact_array)
        # )
        # but this seems like a hack and there is no better way
        # https://github.com/google-deepmind/optax/issues/384

    # RAR sampling init (ouside scanned function to avoid dynamic slice error)
    # If RAR is not used the _rar_step_*() are juste None and data is unchanged
    data, _rar_step_true, _rar_step_false = init_rar(data)  # type: ignore

    # Seq2seq
    curr_seq = 0

    train_loss_values = jnp.zeros((n_iter))
    # depending on obs_batch_sharding we will get the simple get_batch or the
    # get_batch with device_put, the latter is not jittable
    get_batch = _get_get_batch(obs_batch_sharding)

    # initialize the dict for stored parameter values
    # we need to get a loss_term to init stuff
    batch_ini, data, param_data, obs_data = get_batch(data, param_data, obs_data)
    _, loss_terms = loss(init_params, batch_ini)

    # initialize parameter tracking
    if tracked_params is None:
        tracked_params = jax.tree.map(lambda p: None, init_params)
    stored_params = jax.tree_util.tree_map(
        lambda tracked_param, param: (
            jnp.zeros((n_iter,) + jnp.asarray(param).shape)
            if tracked_param is not None
            else None
        ),
        tracked_params,
        init_params,
        is_leaf=lambda x: x is None,  # None values in tracked_params will not
        # be traversed. Thus the user can provide something like `tracked_params = jinns.parameters.Params(
        # nn_params=None, eq_params={"nu": True})` while init_params.nn_params
        # being a complex data structure
    )

    # initialize the PyTree for stored loss values
    stored_loss_terms = jax.tree_util.tree_map(
        lambda _: jnp.zeros((n_iter)), loss_terms
    )

    # initialize the PyTree for stored loss weights values
    if loss.update_weight_method is not None:
        stored_weights_terms = eqx.tree_at(
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
    else:
        stored_weights_terms = None
    if loss.update_weight_method is not None and key is None:
        raise ValueError(
            "`key` argument must be passed to jinns.solve when"
            " `loss.update_weight_method` is not None"
        )

    train_data = DataGeneratorContainer(
        data=data, param_data=param_data, obs_data=obs_data
    )
    optimization = OptimizationContainer(
        params=init_params,
        last_non_nan_params=init_params,
        opt_state=opt_state,
    )
    optimization_extra = OptimizationExtraContainer(
        curr_seq=curr_seq,
        best_iter_id=0,
        best_val_criterion=jnp.nan,
        best_val_params=init_params,
    )
    loss_container = LossContainer(
        stored_loss_terms=stored_loss_terms,
        stored_weights_terms=stored_weights_terms,
        train_loss_values=train_loss_values,
    )
    stored_objects = StoredObjectContainer(
        stored_params=stored_params,
    )

    if validation is not None:
        validation_crit_values = jnp.zeros(n_iter)
    else:
        validation_crit_values = None

    break_fun = _get_break_fun(n_iter, verbose)

    iteration = 0
    carry = (
        iteration,
        loss,
        optimization,
        optimization_extra,
        train_data,
        validation,
        loss_container,
        stored_objects,
        validation_crit_values,
        key,
    )

    def _one_iteration(carry: main_carry) -> main_carry:
        (
            i,
            loss,
            optimization,
            optimization_extra,
            train_data,
            validation,
            loss_container,
            stored_objects,
            validation_crit_values,
            key,
        ) = carry

        batch, data, param_data, obs_data = get_batch(
            train_data.data, train_data.param_data, train_data.obs_data
        )

        # ---------------------------------------------------------------------
        # The following part is the equivalent of a
        # > train_loss_value, grads = jax.values_and_grad(total_loss.evaluate)(params, ...)
        # but it is decomposed on individual loss terms so that we can use it
        # if needed for updating loss weights.
        # Since the total loss is a weighted sum of individual loss terms, so
        # are its total gradients.

        # Compute individual losses and individual gradients
        loss_terms, grad_terms = loss.evaluate_by_terms(optimization.params, batch)

        if loss.update_weight_method is not None:
            key, subkey = jax.random.split(key)  # type: ignore because key can
            # still be None currently
            # avoid computations of tree_at if no updates
            loss = loss.update_weights(
                i, loss_terms, loss_container.stored_loss_terms, grad_terms, subkey
            )

        # total grad
        grads = loss.ponderate_and_sum_gradient(grad_terms)

        # total loss
        train_loss_value = loss.ponderate_and_sum_loss(loss_terms)
        # ---------------------------------------------------------------------

        # gradient step
        (
            params,
            opt_state,
            last_non_nan_params,
        ) = _gradient_step(
            grads,
            optimizer,
            optimization.params,
            optimization.opt_state,
            optimization.last_non_nan_params,
        )

        # Print train loss value during optimization
        if verbose:
            _print_fn(i, train_loss_value, print_loss_every, prefix="[train] ")

        if validation is not None and validation_crit_values is not None:
            # there is a jax.lax.cond because we do not necesarily call the
            # validation step every iteration
            (
                validation,  # always return `validation` for in-place mutation
                early_stopping,
                validation_criterion,
                update_best_params,
            ) = jax.lax.cond(
                i % validation.call_every == 0,
                lambda operands: operands[0](*operands[1:]),  # validation.__call__()
                lambda operands: (
                    operands[0],
                    False,
                    validation_crit_values[i - 1],  # type: ignore don't know why it can still be None
                    False,
                ),
                (
                    validation,  # validation must be in operands
                    params,
                ),
            )
            # Print validation loss value during optimization
            if verbose:
                _print_fn(
                    i, validation_criterion, print_loss_every, prefix="[validation] "
                )
            validation_crit_values = validation_crit_values.at[i].set(
                validation_criterion
            )

            # update best_val_params and best_val_criterion w.r.t val_loss if needed
            (best_val_params, best_val_criterion, best_iter_id) = jax.lax.cond(
                update_best_params,
                lambda operands: (
                    params,
                    validation_criterion,
                    i,
                ),  # update with current value
                lambda operands: (
                    operands[0].best_val_params,
                    operands[0].best_val_criterion,
                    operands[0].best_iter_id,
                ),  # unchanged
                (optimization_extra,),
            )
        else:
            early_stopping = False
            best_iter_id = 0
            best_val_params = params
            best_val_criterion = jnp.nan

        # Trigger RAR
        loss, params, data = trigger_rar(
            i, loss, params, data, _rar_step_true, _rar_step_false
        )

        # save loss value and selected parameters
        stored_objects, loss_container = _store_loss_and_params(
            i,
            params,
            stored_objects.stored_params,
            loss_container,
            train_loss_value,
            loss_terms,
            loss.loss_weights,
            tracked_params,
        )

        # increment iteration number
        i += 1

        return (
            i,
            loss,
            OptimizationContainer(params, last_non_nan_params, opt_state),
            OptimizationExtraContainer(
                curr_seq,
                best_iter_id,
                best_val_criterion,
                best_val_params,
                early_stopping,
            ),
            DataGeneratorContainer(data, param_data, obs_data),
            validation,
            loss_container,
            stored_objects,
            validation_crit_values,
            key,
        )

    # Main optimization loop. We use the LAX while loop (fully jitted) version
    # if no mixing devices. Otherwise we use the standard while loop. Here devices only
    # concern obs_batch, but it could lead to more complex scheme in the future
    if obs_batch_sharding is not None:
        while break_fun(carry):
            carry = _one_iteration(carry)
    else:

        def train_fun(carry):
            return jax.lax.while_loop(break_fun, _one_iteration, carry)

        if ahead_of_time:
            start = time.time()
            compiled_train_fun = jax.jit(train_fun).lower(carry).compile()
            end = time.time()
            if verbose:
                print("\nCompilation took\n", end - start, "\n")

            start = time.time()
            carry = compiled_train_fun(carry)
            jax.block_until_ready(carry)
            end = time.time()
            if verbose:
                print("\nTraining took\n", end - start, "\n")
        else:
            carry = train_fun(carry)

    (
        i,
        loss,
        optimization,
        optimization_extra,
        train_data,
        validation,
        loss_container,
        stored_objects,
        validation_crit_values,
        key,
    ) = carry

    if verbose:
        jax.debug.print(
            "\nFinal iteration {i}: train loss value = {train_loss_val}",
            i=i,
            train_loss_val=loss_container.train_loss_values[i - 1],
        )

    # get ready to return the parameters at last iteration...
    # (by default arbitrary choice, this could be None)
    validation_parameters = optimization.last_non_nan_params
    if validation is not None and validation_crit_values is not None:
        jax.debug.print(
            "validation loss value = {validation_loss_val}",
            validation_loss_val=validation_crit_values[i - 1],
        )
        if optimization_extra.early_stopping:
            jax.debug.print(
                "\n Returning a set of best parameters from early stopping"
                " as last argument!\n"
                " Best parameters from iteration {best_iter_id}"
                " with validation loss criterion = {best_val_criterion}",
                best_iter_id=optimization_extra.best_iter_id,
                best_val_criterion=optimization_extra.best_val_criterion,
            )
            # ...but if early stopping, return the parameters at the best_iter_id
            validation_parameters = optimization_extra.best_val_params

    return (
        optimization.last_non_nan_params,
        loss_container.train_loss_values,
        loss_container.stored_loss_terms,
        train_data.data,  # return the DataGenerator if needed (no in-place modif)
        loss,  # return the Loss if needed (no-inplace modif)
        optimization.opt_state,
        stored_objects.stored_params,
        loss_container.stored_weights_terms,
        validation_crit_values if validation is not None else None,
        validation_parameters,
    )


@partial(jit, static_argnames=["optimizer"])
def _gradient_step(
    grads: Params[Array],
    optimizer: optax.GradientTransformation,
    params: Params[Array],
    opt_state: optax.OptState,
    last_non_nan_params: Params[Array],
) -> tuple[
    Params[Array],
    optax.OptState,
    Params[Array],
]:
    """
    optimizer cannot be jit-ted.
    """

    updates, opt_state = optimizer.update(
        grads,  # type: ignore
        opt_state,
        params,  # type: ignore
    )  # see optimizer.init for explaination for the ignore(s) here
    params = optax.apply_updates(params, updates)  # type: ignore

    # check if any of the parameters is NaN
    last_non_nan_params = jax.lax.cond(
        _check_nan_in_pytree(params),
        lambda _: last_non_nan_params,
        lambda _: params,
        None,
    )

    return (
        params,
        opt_state,
        last_non_nan_params,
    )


@partial(jit, static_argnames=["prefix"])
def _print_fn(i: int, loss_val: Float, print_loss_every: int, prefix: str = ""):
    # note that if the following is not jitted in the main lor loop, it is
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


def _get_break_fun(n_iter: int, verbose: bool) -> Callable[[main_carry], bool]:
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

        (i, _, optimization, optimization_extra, _, _, _, _, _, _) = carry

        # Condition 1
        bool_max_iter = jax.lax.cond(
            i >= n_iter,
            lambda _: stop_while_loop("max iteration is reached"),
            continue_while_loop,
            None,
        )
        # Condition 2
        bool_nan_in_params = jax.lax.cond(
            _check_nan_in_pytree(optimization.params),
            lambda _: stop_while_loop(
                "NaN values in parameters (returning last non NaN values)"
            ),
            continue_while_loop,
            None,
        )
        # Condition 3
        bool_early_stopping = jax.lax.cond(
            optimization_extra.early_stopping,
            lambda _: stop_while_loop("early stopping"),
            continue_while_loop,
            _,
        )

        # stop when one of the cond to continue is False
        return jax.tree_util.tree_reduce(
            lambda x, y: jnp.logical_and(jnp.array(x), jnp.array(y)),
            (bool_max_iter, bool_nan_in_params, bool_early_stopping),
        )

    return break_fun


def _get_get_batch(
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
