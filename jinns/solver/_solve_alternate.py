"""
`jinns.solve_alternate()` to efficiently resolve inverse problems
"""

from __future__ import annotations

import time
import operator
from dataclasses import fields
from typing import TYPE_CHECKING
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, PRNGKeyArray, Float
import equinox as eqx

from jinns.parameters._params import Params
from jinns.solver._utils import (
    _init_stored_weights_terms,
    _init_stored_params,
    _get_break_fun,
    _loss_evaluate_and_gradient_step,
    _build_get_batch,
    _store_loss_and_params,
    _print_fn,
)
from jinns.utils._containers import (
    DataGeneratorContainer,
    OptimizationContainer,
    OptimizationExtraContainer,
    LossContainer,
    StoredObjectContainer,
)

if TYPE_CHECKING:
    from typing import Any
    from jinns.utils._types import AnyLossComponents
    from jinns.loss._abstract_loss import AbstractLoss
    from jinns.data._AbstractDataGenerator import AbstractDataGenerator
    from jinns.data._DataGeneratorObservations import DataGeneratorObservations
    from jinns.data._DataGeneratorParameter import DataGeneratorParameter


def solve_alternate(
    *,
    n_iter: int,
    optimizers: Params[optax.GradientTransformation],
    n_iter_by_solver: Params[int],
    init_params: Params[Array],
    data: AbstractDataGenerator,
    loss: AbstractLoss,
    print_loss_every: int = 10,
    tracked_params: Params[Any | None] | None = None,
    verbose: bool = True,
    obs_data: DataGeneratorObservations | None = None,
    param_data: DataGeneratorParameter | None = None,
    opt_state_fields_for_acceleration: Params[str] | None = None,
    key: PRNGKeyArray | None = None,
) -> tuple[
    Params[Array],
    Float[Array, " n_iter_total"],
    AnyLossComponents[Float[Array, " n_iter_total"]],
    AbstractDataGenerator,
    AbstractLoss,
    optax.OptState,
    Params[Array | None],
    AnyLossComponents[Float[Array, " n_iter_total"]],
    DataGeneratorObservations | None,
    DataGeneratorParameter | None,
]:
    """
    Efficient implementation of the alternate minimization scheme between
    `Params.nn_params` and `Params.eq_params`. This function is recommended for inverse problems where `Params.nn_params` is arbitrarily big, but
    `Params.eq_params` prepresents only a few physical parameters.


    In this functions both type of parameters (`eq` and `nn`) are handled
    separately, as well as all related quantities such as gradient updates,
    opt_states, etc. This approach becomes more efficient than solely
    relying on optax masked transforms and `jinns.parameters.DerivativeKeys`
    when `Params.nn_params` is big while `Params.eq_params` is much smaller,
    which is often the case. Indeed, `DerivativeKeys` only prevents some
    gradients computations but a major computational bottleneck comes from
    passing huge optax states filled with dummy zeros udpdates (for frozen
    parameters) at each iteration, [see the `optax` issue that we raised](https://www.github.com/google-deepmind/optax/issues/993)).

    Using `solve_alternate` improves this situation by handling Optax
    optimization states separately for `nn` and `eq` params. This allows to
    pass `None` instead of huge dummy zero updates for "frozen" parameters in
    the optimization states. Internally, this is done thanks to the
    `params_mask` PyTree of booleans used for `eqx.partition` and `eqx.combine`.


    Parameters
    ----------
    n_iter
        The maximum number of cyles of alternate iterations.
    optimizers
        A `jinns.parameters.Params` object, where each leave is an optax
        optimizer. Note that when using an `optax.chain` with a schedular for a
        certain parameter, the iteration count considered is the one of this
        precise parameter. That is, for parameter `theta`, the scheduler is
        spread over `n_iter_by_solver.eq_params.theta * n_iter` steps.
    n_iter_by_optimizer
        A Params object, where each leaves gives the number of iteration of the
        corresponding optimizer, within one alternate cycle.
    init_params
        The initial `jinns.parameters.Params` object.
    data
        A `jinns.data.AbstractDataGenerator` object to retrieve batches of collocation points.
    loss
        The loss function to minimize.
    print_loss_every
        Default 10. The rate at which we print the loss value in the
        gradient step loop.
    tracked_params
        Default `None`. A `jinns.parameters.Params` object with non-`None` values for
        parameters that needs to be tracked along the iterations.
        The user can provide something like `tracked_params = jinns.parameters.Params(
        nn_params=None, eq_params={"nu": True})` while `init_params.nn_params`
        being a complex data structure.
    verbose
        Default `True`. If `False`, no output (loss or cause of
        exiting the optimization loop) will be produced.
    obs_data
        Default `None`. A `jinns.data.DataGeneratorObservations`
        object which can be used to sample minibatches of observations.
    param_data
        Default `None`. A `jinns.data.DataGeneratorParameter` object which can be used to
        sample equation parameters.
    opt_state_fields_for_acceleration
        A `jinns.parameters.Params` object, where leave
        is an `opt_state_field_for_acceleration` as
        described in `jinns.solve`.
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
        The final `jinns.parameters.Params` PyTree with opt_state as leaves.
    stored_params
        A object with the stored values of the desired parameters (as
        signified in `tracked_params` argument).
    stored_weights_terms
        A PyTree with leaves being arrays of all the values for each loss
        weight. Note that if `Loss.update_weight_method is None`, we return
        `None`,
        because loss weights are never updated and we can then save some
        computations.
    obs_data
        The `jinns.data.DataGeneratorObservations` object passed as input or
        `None`.
    param_data
        The `jinns.data.DataGeneratorParameter` object passed as input or
        `None`.
    """
    # The key functions that perform the partitions are
    # `_get_masked_optimization_stuff` and `_get_unmasked_optimization_stuff` in
    # `jinns/solver/_utils.py`.

    # The `solve_alternate()` main loop efficiently alternates between a local
    # optimization on `nn_params` and local optimizations on all `eq_params`.
    # There is then a main `jax.while_loop` with a main carry, and several
    # local `jax.while_loop` for each local optimizations, with local carry
    # structures. Local optimizations (local loops and carrys) are defined
    # in AOT jitted functions
    # (`nn_params_train_fun_compiled` and the elements of the dict
    # `eq_params_train_fun_compiled`). Those AOT jitted functions comprise the
    # body of the local loop (`_nn_params_one_iteration` and
    # `_eq_params_one_iteration`) as well as 3 steps:

    # 1) Step 1. Prepare the local carry. Make the junction with the main carry
    # and make the appropriate initializations. See the function
    # `_init_before_local_optimization`.
    # 2) Step 2. Perfom the local gradient steps (local `jax.while_loop`)
    # 3) Step 3. Extract the needed elements from the local carry at the end of
    # the local loop to the main carry. See the function
    # `_get_loss_and_objects_container`.

    initialization_time = time.time()
    if n_iter < 1:
        raise ValueError("Cannot run jinns.solve for n_iter<1")

    main_break_fun = _get_break_fun(
        n_iter, verbose, conditions_str=("bool_max_iter", "bool_nan_in_params")
    )
    get_batch = _build_get_batch(None)

    nn_n_iter = n_iter_by_solver.nn_params
    eq_n_iters = n_iter_by_solver.eq_params

    nn_optimizer = optimizers.nn_params
    eq_optimizers = optimizers.eq_params

    # NOTE below we have opt_states that are shaped as Params
    # but this seems OK since the real gain is to not differentiate
    # wrt to unwanted params
    nn_opt_state = nn_optimizer.init(init_params)

    if opt_state_fields_for_acceleration is None:
        nn_opt_state_field_for_acceleration = None
        eq_params_opt_state_field_for_accel = jax.tree.map(
            lambda l: None,
            eq_optimizers,
            is_leaf=lambda x: isinstance(x, optax.GradientTransformation),
        )
    else:
        nn_opt_state_field_for_acceleration = (
            opt_state_fields_for_acceleration.nn_params
        )
        eq_params_opt_state_field_for_accel = (
            opt_state_fields_for_acceleration.eq_params
        )

    eq_opt_states = jax.tree.map(
        lambda opt_: opt_.init(init_params),
        eq_optimizers,
        is_leaf=lambda x: isinstance(x, optax.GradientTransformation),
        # do not traverse further
    )

    # params mask to be able to optimize only on nn_params
    # NOTE we can imagine that later on, params mask is given as user input and
    # we could then have more refined scheme than just nn_params and eq_params.
    nn_params_mask = Params(
        nn_params=True, eq_params=jax.tree.map(lambda ll: False, init_params.eq_params)
    )
    # derivative keys with only nn_params updates for the gradient steps over nn_params
    # this is a standard derivative key, with True for nn_params and False to
    # all leaves of eq_params
    nn_gd_steps_derivative_keys = jax.tree.map(
        lambda l: nn_params_mask,
        loss.derivative_keys,
        is_leaf=lambda x: isinstance(x, Params),
    )

    # and get the negative to optimize only on eq_params FOR EACH EQ_PARAMS
    # Hence the PyTree we need to construct to tree.map over is a little more
    # complex since we need to keep the overall dict structure

    eq_params_masks, eq_gd_steps_derivative_keys = (
        _get_eq_param_masks_and_derivative_keys(eq_optimizers, init_params, loss)
    )

    #######################################
    # SOME INITIALIZATIONS FOR CONTAINERS #
    #######################################

    # initialize the PyTree for stored loss values
    total_iter_all_solvers = jax.tree.reduce(operator.add, n_iter_by_solver, 0)

    # initialize parameter tracking
    if tracked_params is None:
        tracked_params = jax.tree.map(lambda p: None, init_params)
    stored_params = _init_stored_params(
        tracked_params, init_params, n_iter * total_iter_all_solvers
    )

    # initialize the dict for stored parameter values
    # we need to get a loss_term to init stuff
    # NOTE: we use jax.eval_shape to avoid FLOPS since we only need the tree
    # structure
    batch_ini, data, param_data, obs_data = get_batch(data, param_data, obs_data)
    _, loss_terms = jax.eval_shape(loss, init_params, batch_ini)

    stored_loss_terms = jax.tree_util.tree_map(
        lambda _: jnp.zeros((n_iter * total_iter_all_solvers)), loss_terms
    )
    n_iter_list_eq_params = jax.tree.leaves(n_iter_by_solver.eq_params)
    train_loss_values = jnp.zeros((n_iter * total_iter_all_solvers))

    # initialize the PyTree for stored loss weights values
    if loss.update_weight_method is not None:
        stored_weights_terms = _init_stored_weights_terms(
            loss, n_iter * total_iter_all_solvers
        )
    else:
        stored_weights_terms = None

    train_data = DataGeneratorContainer(
        data=data, param_data=param_data, obs_data=obs_data
    )
    optimization = OptimizationContainer(
        params=init_params,
        last_non_nan_params=init_params,
        opt_state=(nn_opt_state, eq_opt_states),  # NOTE that this field changes
        # between the outer while loop and inner loops
    )
    optimization_extra = OptimizationExtraContainer(
        curr_seq=None,
        best_iter_id=None,
        best_val_criterion=None,
        best_val_params=None,
    )
    loss_container = LossContainer(
        stored_loss_terms=stored_loss_terms,
        train_loss_values=train_loss_values,
        stored_weights_terms=stored_weights_terms,
    )
    stored_objects = StoredObjectContainer(
        stored_params=stored_params,
    )

    # Main carry defined here
    carry = (
        0,
        loss,
        optimization,
        optimization_extra,
        train_data,
        loss_container,
        stored_objects,
        key,
    )
    ###

    # NOTE we precompile the eq_n_iters[eq_params]-iterations over eq_params
    # that we will repeat many times. This gets the compilation cost out of the
    # loop. This is done for each equation parameters, those functions are
    # stored in a dictionary.

    eq_param_eq_optim = tuple(
        (f.name, getattr(eq_optimizers, f.name)) for f in fields(eq_optimizers)
    )

    eq_params_train_fun_compiled = {}
    for idx_params, (eq_param, eq_optim) in enumerate(eq_param_eq_optim):
        n_iter_for_params = getattr(eq_n_iters, eq_param)

        def eq_train_fun(_, carry):
            i = carry[0]
            loss_container = carry[5]
            stored_objects = carry[6]

            def _eq_params_one_iteration(carry):
                (
                    i,
                    loss,
                    optimization,
                    _,
                    train_data,
                    loss_container,
                    stored_objects,
                    key,
                ) = carry

                (nn_opt_state, eq_opt_states) = optimization.opt_state

                batch, data, param_data, obs_data = get_batch(
                    train_data.data, train_data.param_data, train_data.obs_data
                )

                if key is not None:
                    key, subkey = jax.random.split(key)
                else:
                    subkey = None
                # Gradient step
                (
                    train_loss_value,
                    params,
                    last_non_nan_params,
                    eq_opt_state,
                    loss,
                    loss_terms,
                ) = _loss_evaluate_and_gradient_step(
                    i,
                    batch,
                    loss,
                    optimization.params,
                    optimization.last_non_nan_params,
                    getattr(eq_opt_states, eq_param),
                    eq_optim,
                    loss_container,
                    subkey,
                    getattr(eq_params_masks, eq_param),
                    getattr(eq_params_opt_state_field_for_accel, eq_param),
                    with_loss_weight_update=True,
                )

                # save loss value and selected parameters
                stored_objects_, loss_container_ = _store_loss_and_params(
                    i,
                    params,
                    stored_objects.stored_params,
                    loss_container,
                    train_loss_value,
                    loss_terms,
                    loss.loss_weights,
                    tracked_params,
                )

                carry = (
                    i + 1,
                    loss,
                    OptimizationContainer(
                        params,
                        last_non_nan_params,
                        (
                            nn_opt_state,
                            eqx.tree_at(
                                lambda pt: (getattr(pt, eq_param),),
                                eq_opt_states,
                                (eq_opt_state,),
                            ),
                        ),
                    ),
                    carry[3],
                    DataGeneratorContainer(
                        data=data, param_data=param_data, obs_data=obs_data
                    ),
                    loss_container_,
                    stored_objects_,
                    carry[7],
                )

                return carry

            break_fun_ = _get_break_fun(
                n_iter_for_params,
                verbose=False,
                conditions_str=("bool_max_iter", "bool_nan_in_params"),
            )

            # STEP 1 (see main docstring)
            start_idx = i * (sum(n_iter_list_eq_params) + nn_n_iter) + sum(
                n_iter_list_eq_params[:idx_params]
            )

            loss_, loss_container_, stored_objects_ = _init_before_local_optimization(
                eq_gd_steps_derivative_keys[eq_param],
                n_iter_for_params,
                loss_terms,
                carry[1],
                loss_container,
                start_idx,
                tracked_params,
                init_params,
            )

            carry_ = (
                0,
                loss_,
                carry[2],
                carry[3],
                carry[4],
                loss_container_,
                stored_objects_,
                carry[7],
            )
            # STEP 2 (see main docstring)
            carry_ = jax.lax.while_loop(break_fun_, _eq_params_one_iteration, carry_)

            # STEP 3 (see main docstring)
            loss_container, stored_objects = _get_loss_and_objects_container(
                loss_container, carry_[5], stored_objects, carry_[6], start_idx
            )

            carry = (
                i,
                carry_[1],
                carry_[2],
                carry_[3],
                carry_[4],
                loss_container,
                stored_objects,
                carry_[7],
            )
            return carry

        eq_params_train_fun_compiled[eq_param] = (
            jax.jit(eq_train_fun, static_argnums=0)
            .trace(n_iter_for_params, jax.eval_shape(lambda _: carry, (None,)))
            .lower()
            .compile()
        )

    # NOTE we precompile the local optimization loop on the nn params
    # In the plain while loop, the compilation is costly each time
    # In the jax lax while loop, the compilation is better but AOT is
    # disallowed there
    nn_break_fun_ = _get_break_fun(
        nn_n_iter, verbose=False, conditions_str=("bool_max_iter", "bool_nan_in_params")
    )

    def nn_train_fun(carry):
        i = carry[0]
        loss_container = carry[5]
        stored_objects = carry[6]

        def _nn_params_one_iteration(carry):
            (
                i,
                loss,
                optimization,
                _,
                train_data,
                loss_container,
                stored_objects,
                key,
            ) = carry

            #
            (nn_opt_state, eq_opt_states) = optimization.opt_state

            batch, data, param_data, obs_data = get_batch(
                train_data.data, train_data.param_data, train_data.obs_data
            )

            # Gradient step
            if key is not None:
                key, subkey = jax.random.split(key)
            else:
                subkey = None
            (
                train_loss_value,
                params,
                last_non_nan_params,
                nn_opt_state,
                loss,
                loss_terms,
            ) = _loss_evaluate_and_gradient_step(
                i,
                batch,
                loss,
                optimization.params,
                optimization.last_non_nan_params,
                nn_opt_state,
                nn_optimizer,
                loss_container,
                subkey,
                nn_params_mask,
                nn_opt_state_field_for_acceleration,
                with_loss_weight_update=True,
            )

            # save loss value and selected parameters
            stored_objects_, loss_container_ = _store_loss_and_params(
                i,
                params,
                stored_objects.stored_params,
                loss_container,
                train_loss_value,
                loss_terms,
                loss.loss_weights,
                tracked_params,
            )

            carry = (
                i + 1,
                loss,
                OptimizationContainer(
                    params, last_non_nan_params, (nn_opt_state, eq_opt_states)
                ),
                carry[3],
                DataGeneratorContainer(
                    data=data, param_data=param_data, obs_data=obs_data
                ),
                loss_container_,
                stored_objects_,
                carry[7],
            )

            return carry

        # STEP 1 (see main docstring)
        start_idx = i * (sum(n_iter_list_eq_params) + nn_n_iter) + sum(
            n_iter_list_eq_params
        )
        loss_, loss_container_, stored_objects_ = _init_before_local_optimization(
            nn_gd_steps_derivative_keys,
            nn_n_iter,
            loss_terms,
            carry[1],
            loss_container,
            start_idx,
            tracked_params,
            init_params,
        )
        carry_ = (
            0,
            loss_,
            carry[2],
            carry[3],
            carry[4],
            loss_container_,
            stored_objects_,
            carry[7],
        )
        # STEP 2 (see main docstring)
        carry_ = jax.lax.while_loop(nn_break_fun_, _nn_params_one_iteration, carry_)

        # Now we prepare back the main carry
        # STEP 3 (see main docstring)
        loss_container, stored_objects = _get_loss_and_objects_container(
            loss_container, carry_[5], stored_objects, carry_[6], start_idx
        )

        carry = (
            i,
            carry_[1],
            carry_[2],
            carry_[3],
            carry_[4],
            loss_container,
            stored_objects,
            carry_[7],
        )
        return carry

    nn_params_train_fun_compiled = (
        jax.jit(nn_train_fun)
        .trace(jax.eval_shape(lambda _: carry, (None,)))
        .lower()
        .compile()
    )

    if verbose:
        print("Initialization time:", time.time() - initialization_time)

    def _one_alternate_iteration(carry):
        (
            i,
            loss,
            optimization,
            optimization_extra,
            train_data,
            loss_container,
            stored_objects,
            key,
        ) = carry

        ###### OPTIMIZATION ON EQ_PARAMS ###########

        for eq_param, _ in eq_param_eq_optim:
            carry = eq_params_train_fun_compiled[eq_param](carry)

        ###### OPTIMIZATION ON NN_PARAMS ###########

        carry = nn_params_train_fun_compiled(carry)

        ############################################

        if verbose:
            n_iter_total = (
                i * (sum(n_iter_list_eq_params) + nn_n_iter)
                + sum(n_iter_list_eq_params)
                + nn_n_iter
            )
            _print_fn(
                i,
                carry[5].train_loss_values[n_iter_total - 1],
                print_loss_every,
                prefix="[train alternate]",
            )

        i += 1
        return (i, carry[1], carry[2], carry[3], carry[4], carry[5], carry[6], carry[7])

    start = time.time()
    # jax.lax.while_loop jits its content so cannot be used when we try to
    # precompile what is inside. JAX tranformations are not compatible with AOT
    while main_break_fun(carry):
        carry = _one_alternate_iteration(carry)
    jax.block_until_ready(carry)
    end = time.time()

    if verbose:
        n_iter_total = (carry[0]) * (sum(n_iter_list_eq_params) + nn_n_iter)
        jax.debug.print(
            "\nFinal alternate iteration {i}: loss value = {train_loss_val}",
            i=carry[0],
            train_loss_val=carry[5].train_loss_values[n_iter_total - 1],
        )

    if verbose:
        print("\nTraining took\n", end - start, "\n")

    return (
        carry[2].params,
        carry[5].train_loss_values,
        carry[5].stored_loss_terms,
        carry[4].data,
        carry[1],  # loss
        carry[2].opt_state,
        carry[6].stored_params,
        carry[5].stored_weights_terms,
        carry[4].obs_data,
        carry[4].param_data,
    )


def _get_loss_and_objects_container(
    loss_container, loss_container_, stored_objects, stored_objects_, start_idx
):
    """
    This functions contains what needs to be done at the end of a local
    optimization on `nn_params` or on one of the `eq_params`. This mainly
    consists in extracting from the local carry what needs to be transferred to
    the global carry:

    - loss_container content (to get the continuity of loss values, etc.)
    - stored_objects content (to get the continuity of stored params etc.)
    """
    loss_container = LossContainer(
        stored_loss_terms=jax.tree.map(
            lambda s, l: jax.lax.dynamic_update_slice(s, l, (start_idx,)),
            loss_container.stored_loss_terms,
            loss_container_.stored_loss_terms,
        ),
        train_loss_values=jax.lax.dynamic_update_slice(
            loss_container.train_loss_values,
            loss_container_.train_loss_values,
            (start_idx,),
        ),
        stored_weights_terms=jax.tree.map(
            lambda s, l: jax.lax.dynamic_update_slice(s, l, (start_idx,)),
            loss_container.stored_weights_terms,
            loss_container_.stored_weights_terms,
        ),
    )
    stored_objects = StoredObjectContainer(
        stored_params=jax.tree.map(
            lambda s, l: jax.lax.dynamic_update_slice(s, l, (start_idx,) + s[0].shape),
            stored_objects.stored_params,
            stored_objects_.stored_params,
        )
    )
    return loss_container, stored_objects


def _init_before_local_optimization(
    derivative_keys,
    n_iter_local,
    loss_terms,
    loss,
    loss_container,
    start_idx,
    tracked_params,
    init_params,
):
    """
    This functions contains what needs to be done at the beginning of a local
    optimization on `nn_params` or on one of the `eq_params`. This maily
    consists in initializating the local carry with the object having the
    correct shape for the incoming local while loop.
    This also
    consists in extracting from the global carry what needs to be transferred to
    the local carry:

    - loss weight values to get the continuity of loss_weight updates methods
    """
    loss_ = eqx.tree_at(
        lambda pt: (pt.derivative_keys,),
        loss,
        (derivative_keys,),
    )
    # Reinit a loss container for this inner loop
    stored_loss_terms_ = jax.tree_util.tree_map(
        lambda _: jnp.zeros((n_iter_local)), loss_terms
    )
    train_loss_values_ = jnp.zeros((n_iter_local,))
    if loss_.update_weight_method is not None:
        stored_weights_terms_ = _init_stored_weights_terms(loss_, n_iter_local)
        # ensure continuity between steps for loss weights
        # this is important for update weight methods which requires
        # previous weight values
        stored_weights_terms_ = jax.tree_util.tree_map(
            lambda st_, st: st_.at[-1].set(st[start_idx - 1]),
            stored_weights_terms_,
            loss_container.stored_weights_terms,
        )
    else:
        stored_weights_terms_ = None
    loss_container_ = LossContainer(
        stored_loss_terms=stored_loss_terms_,
        train_loss_values=train_loss_values_,
        stored_weights_terms=stored_weights_terms_,
    )

    # Reinit a stored_objects for this inner loop
    stored_params_ = _init_stored_params(tracked_params, init_params, n_iter_local)
    stored_objects_ = StoredObjectContainer(stored_params=stored_params_)
    return loss_, loss_container_, stored_objects_


def _get_eq_param_masks_and_derivative_keys(eq_optimizers, init_params, loss):
    nb_eq_params = len(
        jax.tree.leaves(
            eq_optimizers, is_leaf=lambda x: isinstance(x, optax.GradientTransformation)
        )
    )
    # masks_ is a sort of one hot encoding for each eq_param
    masks_ = tuple(jnp.eye(nb_eq_params)[i] for i in range(nb_eq_params))
    # eq_params_masks_ is a EqParams with each leaf getting its one hot
    # encoding of the eq_param it represents
    eq_params_masks_ = jax.tree.unflatten(
        jax.tree.structure(
            eq_optimizers, is_leaf=lambda x: isinstance(x, optax.GradientTransformation)
        ),
        masks_,
    )
    # if you forget about the broadcast below
    # eq_params_masks is a EqParams where each leaf is a Params
    # where we have a 1 where the subleaf of Params is the same as the upper
    # leaf of the EqParams
    # now add the broadcast: it is needed because eg ll=[0, 0, 1] has just been
    # unflattened into 3 eq_params (from eq_optimizers structure). The problem
    # is that here, a float (0 or 0 or 1) has been assigned, all with struct
    # (). This is problematic since it will not match struct of
    # Params.eq_params that are tuple for eg. Then if
    # Params.eq_params=(alpha=(0., 0.), beta=(1.,), gamma=(4., 4.,
    # jnp.array([4., 4.]))) then the result of the unflatten will be
    # modified into the correct structures ie,
    # (alpha=(0, 0), beta=(0,), gamma=(1, 1, 1)) instead of
    # (alpha=0, beta=0, gamma=1)
    # the tree.broadcast has been added to prevent a bug in the tree.map of
    # `_set_derivatives` of jinns DerivativeKeys

    eq_params_masks = jax.tree.map(
        lambda l, ll, p: Params(
            nn_params=False,
            eq_params=jax.tree.broadcast(
                jax.tree.unflatten(
                    jax.tree.structure(
                        eq_optimizers,
                        is_leaf=lambda x: isinstance(x, optax.GradientTransformation),
                    ),
                    ll,
                ),
                init_params.eq_params,
            ),
        ),
        eq_optimizers,
        eq_params_masks_,
        init_params.eq_params,
        is_leaf=lambda x: isinstance(x, optax.GradientTransformation),
    )

    def replace_float(leaf):
        if isinstance(leaf, bool):
            return leaf
        elif leaf == 1:
            return True
        elif leaf == 0:
            return False
        else:
            raise ValueError

    # Note that we need to replace with plain bool:
    # 1. filter_spec does not even accept onp.array
    # 2. filter_spec does not accept non static arguments. So any jnp array is
    # non hashable and we will not be able to make it static
    # params_mask cannot be inside the carry of course, just like the
    # optimizer
    eq_params_masks = jax.tree.map(lambda l: replace_float(l), eq_params_masks)

    # derivative keys with only eq_params updates for the gradient steps over eq_params
    # Here we make a dict for simplicity
    # A key=a eq_param=the content to form the jinns DerivativeKeys for each eq_param
    # There is then True for where needed
    eq_gd_steps_derivative_keys = {
        f.name: jax.tree.map(
            lambda l: getattr(eq_params_masks, f.name),
            loss.derivative_keys,
            is_leaf=lambda x: isinstance(x, Params),
        )
        for f in fields(eq_params_masks)
    }

    return eq_params_masks, eq_gd_steps_derivative_keys
