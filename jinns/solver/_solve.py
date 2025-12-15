"""
This modules implements the main `solve()` function of jinns which
handles the optimization process
"""

from __future__ import (
    annotations,
)  # https://docs.python.org/3/library/typing.html#constant

import time
from typing import TYPE_CHECKING, Any
import optax
import jax
import jax.numpy as jnp
from jaxtyping import Float, Array, PRNGKeyArray
from jinns.solver._rar import init_rar, trigger_rar
from jinns.solver._utils import (
    _check_batch_size,
    _init_stored_weights_terms,
    _init_stored_params,
    _get_break_fun,
    _loss_evaluate_and_gradient_step,
    _build_get_batch,
    _store_loss_and_params,
    _print_fn,
)
from jinns.parameters._params import Params
from jinns.utils._containers import (
    DataGeneratorContainer,
    OptimizationContainer,
    OptimizationExtraContainer,
    LossContainer,
    StoredObjectContainer,
)

if TYPE_CHECKING:
    from jinns.utils._types import AnyLossComponents, SolveCarry
    from jinns.loss._abstract_loss import AbstractLoss
    from jinns.validation._validation import AbstractValidationModule
    from jinns.data._DataGeneratorParameter import DataGeneratorParameter
    from jinns.data._DataGeneratorObservations import DataGeneratorObservations
    from jinns.data._AbstractDataGenerator import AbstractDataGenerator


def solve(
    *,
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
    opt_state_field_for_acceleration: str | None = None,
    verbose: bool = True,
    ahead_of_time: bool = True,
    key: PRNGKeyArray | None = None,
) -> tuple[
    Params[Array],
    Float[Array, " n_iter"],
    AnyLossComponents[Float[Array, " n_iter"]],
    AbstractDataGenerator,
    AbstractLoss,
    optax.OptState,
    Params[Array | None],
    AnyLossComponents[Float[Array, " n_iter"]],
    DataGeneratorObservations | None,
    DataGeneratorParameter | None,
    Float[Array, " n_iter"] | None,
    Params[Array] | None,
]:
    """
    Performs the optimization process via stochastic gradient descent
    algorithm. We minimize the function defined in `loss.evaluate()` with
    respect to the learnable parameters of the problem whose initial values
    are given in `init_params`.


    Parameters
    ----------
    n_iter
        The maximum number of iterations in the optimization.
    init_params
        The initial `jinns.parameters.Params` object.
    data
        A `jinns.data.AbstractDataGenerator` object to retrieve batches of collocation points.
    loss
        The loss function to minimize.
    optimizer
        An optax optimizer.
    print_loss_every
        Default 1000. The rate at which we print the loss value in the
        gradient step loop.
    opt_state
        Default `None`. Provides an optional initial state to the optimizer.
    tracked_params
        Default `None`. A `jinns.parameters.Params` object with non-`None` values for
        parameters that needs to be tracked along the iterations.
        The user can provide something like `tracked_params = jinns.parameters.Params(
        nn_params=None, eq_params={"nu": True})` while `init_params.nn_params`
        being a complex data structure.
    param_data
        Default `None`. A `jinns.data.DataGeneratorParameter` object which can be used to
        sample equation parameters.
    obs_data
        Default `None`. A `jinns.data.DataGeneratorObservations`
        object which can be used to sample minibatches of observations.
    validation
        Default `None`. Otherwise, a callable `eqx.Module` which implements a
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
        Default `None`. An optional sharding object to constraint the
        `obs_batch`.
        Typically, a `SingleDeviceSharding(gpu_device)` when `obs_data` has been
        created with `sharding_device=SingleDeviceSharding(cpu_device)` to avoid
        loading on GPU huge datasets of observations.
    opt_state_field_for_acceleration
        A string. Default `None`, i.e. the optimizer without acceleration.
        Because in some optimization scheme one can have what is called
        acceleration where the loss is computed at some accelerated parameter
        values, different from the actual parameter values. These accelerated
        parameter can be stored in the optimizer state as a field. If this
        field name is passed to `opt_state_field_for_acceleration` then the
        gradient step will be done by evaluate gradients at parameter value
        `opt_state.opt_state_field_for_acceleration`.
    verbose
        Default `True`. If `False`, no output (loss or cause of
        exiting the optimization loop) will be produced.
    ahead_of_time
        Default `True`. Separate the compilation of the main training loop from
        the execution to get both timings. You might need to avoid this
        behaviour if you need to perform JAX transforms over chunks of code
        containing `jinns.solve()` since AOT-compiled functions cannot be JAX
        transformed (see https://jax.readthedocs.io/en/latest/aot.html#aot-compiled-functions-cannot-be-transformed).
        When `False`, jinns does not provide any timing information (which would
        be nonsense in a JIT transformed `solve()` function).
    key
        Default `None`. A JAX random key that can be used for diverse purpose in
        the main iteration loop.

    Returns
    -------
    params
        The last non-NaN value of the params at then end of the
        optimization process.
    total_loss_values
        An array of the total loss term along the gradient steps.
    stored_loss_terms
        A PyTree with attributes being arrays of all the values for each loss
        term.
    data
        The data generator object passed as input, possibly modified.
    loss
        The loss object passed as input, possibly modified.
    opt_state
        The final optimized state.
    stored_params
        A object with the stored values of the desired parameters (as
        signified in `tracked_params` argument).
    stored_weights_terms
        A PyTree with leaves being arrays of all the values for each loss
        weight. Note that if `Loss.update_weight_method is None`, we return
        `None`,
        because loss weights are never updated and we can then save some
        computations.
    param_data
        The `jinns.data.DataGeneratorParameter` object passed as input or
        `None`.
    obs_data
        The `jinns.data.DataGeneratorObservations` object passed as input or
        `None`.
    validation_crit_values
        An array containing the validation criterion values of the training.
    best_val_params
        The best parameters according to the validation criterion.
    """
    initialization_time = time.time()
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
    get_batch = _build_get_batch(obs_batch_sharding)

    # initialize parameter tracking
    if tracked_params is None:
        tracked_params = jax.tree.map(lambda p: None, init_params)
    stored_params = _init_stored_params(tracked_params, init_params, n_iter)

    # initialize the dict for stored parameter values
    # we need to get a loss_term to init stuff
    # NOTE: we use jax.eval_shape to avoid FLOPS since we only need the tree
    # structure
    batch_ini, data, param_data, obs_data = get_batch(data, param_data, obs_data)
    _, loss_terms = jax.eval_shape(loss, init_params, batch_ini)

    # initialize the PyTree for stored loss values
    stored_loss_terms = jax.tree_util.tree_map(
        lambda _: jnp.zeros((n_iter)), loss_terms
    )

    # initialize the PyTree for stored loss weights values
    if loss.update_weight_method is not None:
        stored_weights_terms = _init_stored_weights_terms(loss, n_iter)
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

    def _one_iteration(carry: SolveCarry) -> SolveCarry:
        # Note that optimizer are not part of the carry since
        # the former is not tractable and the latter (while it could be
        # hashable) must be static because of the equinox `filter_spec` (https://github.com/patrick-kidger/equinox/issues/1036)

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

        if key is not None:
            key, subkey = jax.random.split(key)
        else:
            subkey = None
        (train_loss_value, params, last_non_nan_params, opt_state, loss, loss_terms) = (
            _loss_evaluate_and_gradient_step(
                i,
                batch,
                loss,
                optimization.params,
                optimization.last_non_nan_params,
                optimization.opt_state,
                optimizer,
                loss_container,
                subkey,
                None,
                opt_state_field_for_acceleration,
            )
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

    if verbose:
        print("Initialization time:", time.time() - initialization_time)

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
        train_data.data,
        loss,
        optimization.opt_state,
        stored_objects.stored_params,
        loss_container.stored_weights_terms,
        train_data.obs_data,
        train_data.param_data,
        validation_crit_values if validation is not None else None,
        validation_parameters,
    )
