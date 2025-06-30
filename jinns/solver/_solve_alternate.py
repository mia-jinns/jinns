"""
`jinns.solve_alternate()` to efficiently resolve inverse problems
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Key
import equinox as eqx

from jinns.parameters._params import Params
from jinns.solver._solve import (
    _get_break_fun,
    _loss_evaluate_and_gradient_step,
    _get_get_batch,
)
from jinns.utils._containers import (
    DataGeneratorContainer,
    OptimizationContainer,
    OptimizationExtraContainer,
    LossContainer,
    StoredObjectContainer,
)

if TYPE_CHECKING:
    from jinns.loss._abstract_loss import AbstractLoss
    from jinns.data._AbstractDataGenerator import AbstractDataGenerator
    from jinns.data._DataGeneratorObservations import DataGeneratorObservations
    from jinns.data._DataGeneratorParameter import DataGeneratorParameter


def solve_alternate(
    n_iter: int,
    n_iter_by_solver: Params[int],
    init_params: Params[Array],
    data: AbstractDataGenerator,
    loss: AbstractLoss,
    optimizers: Params[optax.GradientTransformation],
    verbose: bool = True,
    param_data: DataGeneratorParameter | None = None,
    obs_data: DataGeneratorObservations | None = None,
    opt_state_fields_for_acceleration: Params[str] | None = None,
    key: Key = None,
):
    """
    Solve alternatively between nn_params and eq_params. In this functions both
    set of parameters, all gradient updates, all opt_states, etc. are
    explicitly handled separately. This approach becomes more efficient than
    relying on optax masked transforms and jinns DerivativeKeys when nn_params
    is big while eq_params is very small. The former do not prevent all
    computations from being done (for example, we still have a big opt_state when updating
    only eq_params, jinns DerivativeKeys only consists in putting
    stop gradients, etc. -- see for example Inverse problem tutorial on Burgers
    in jinns documentation) and hence is suboptimal.

    With `jinns.solve_alternate` we want to address efficiently inverse
    problems where nn_optimizer is arbitrarily big, but eq_params prepresents
    only a few physical parameters.

    Parameters
    ----------
    n_iter
        The maximum number of cyles of alternate iterations.
    n_iter_by_solver
        A Params object, where for each leaves (`nn_params` and each
        `eq_params` keys) we find the number of iterations
        inside one alternate cycle.
    init_params
        The initial jinns.parameters.Params object.
    data
        A DataGenerator object to retrieve batches of collocation points.
    loss
        The loss function to minimize.
    optimizers
        A Params object, where for each leaves (`nn_params` and each
        `eq_params` keys) we find an optimizer
    verbose
        Default True. If False, no std output (loss or cause of
        exiting the optimization loop) will be produced.
    param_data
        Default None. A DataGeneratorParameter object which can be used to
        sample equation parameters.
    obs_data
        Default None. A DataGeneratorObservations
        object which can be used to sample minibatches of observations.
    opt_state_fields_for_acceleration
        A Params object, where for each leaves (`nn_params` and each
        `eq_params` keys) we find an `opt_state_field_for_acceleration` as
        described in `jinns.solve`.
    key
        Default None. A JAX random key that can be used for diverse purpose in
        the main iteration loop.

    Returns
    -------
    """
    main_break_fun = _get_break_fun(
        n_iter, verbose, conditions_str=("bool_max_iter", "bool_nan_in_params")
    )
    get_batch = _get_get_batch(None)

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
    nn_gd_steps_derivative_keys = jax.tree.map(
        lambda l: nn_params_mask,
        loss.derivative_keys,
        is_leaf=lambda x: isinstance(x, Params),
    )

    # and get the negative to optimize only on eq_params FOR EACH EQ_PARAMS
    # Hence the PyTree we need to construct to tree.map over is a little more
    # complex since we need to keep the overall dict structure
    nb_eq_params = len(
        jax.tree.leaves(
            eq_optimizers, is_leaf=lambda x: isinstance(x, optax.GradientTransformation)
        )
    )
    masks_ = tuple(jnp.eye(nb_eq_params)[i] for i in range(nb_eq_params))
    eq_params_masks_ = jax.tree.unflatten(
        jax.tree.structure(
            eq_optimizers, is_leaf=lambda x: isinstance(x, optax.GradientTransformation)
        ),
        masks_,
    )
    eq_params_masks = jax.tree.map(
        lambda l, ll: Params(
            nn_params=False,
            eq_params=jax.tree.unflatten(
                jax.tree.structure(
                    eq_optimizers,
                    is_leaf=lambda x: isinstance(x, optax.GradientTransformation),
                ),
                ll,
            ),
        ),
        eq_optimizers,
        eq_params_masks_,
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
    # Again this is a dict for each eq_params
    eq_gd_steps_derivative_keys = {
        eq_param: jax.tree.map(
            lambda l: eq_params_mask,
            loss.derivative_keys,
            is_leaf=lambda x: isinstance(x, Params),
        )
        for eq_param, eq_params_mask in eq_params_masks.items()
    }

    # we checked that it was OK up to there but TODO is to introduce unit tests
    # for the pytree manipulations above

    # NOTE much more containers than what's currently used
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
        stored_loss_terms=None, train_loss_values=None, stored_weights_terms=None
    )
    stored_objects = StoredObjectContainer(
        stored_params=None,
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
    # loop

    eq_params_train_fun_compiled = {}
    for eq_param, eq_optim in eq_optimizers.items():
        n_iter_for_params = eq_n_iters[eq_param]

        def eq_train_fun(n_iter, carry):
            def _eq_params_one_iteration(carry):
                (i, loss, optimization, _, train_data, _, _, key) = carry

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
                    eq_opt_states[eq_param],
                    eq_optim,
                    loss_container,
                    subkey,
                    eq_params_masks[eq_param],
                    eq_params_opt_state_field_for_accel[eq_param],
                    with_loss_weight_update=False,
                )

                carry = (
                    i + 1,
                    loss,
                    OptimizationContainer(
                        params,
                        last_non_nan_params,
                        (
                            nn_opt_state,
                            eq_opt_states | {eq_param: eq_opt_state},
                        ),
                    ),
                    carry[3],
                    carry[4],
                    carry[5],
                    carry[6],
                    carry[7],
                )

                return carry

            break_fun_ = _get_break_fun(
                n_iter_for_params,
                verbose,
                conditions_str=("bool_max_iter", "bool_nan_in_params"),
            )

            # 1 - some init
            loss = eqx.tree_at(
                lambda pt: (pt.derivative_keys),
                carry[1],
                (eq_gd_steps_derivative_keys[eq_param]),
            )

            carry = (
                0,
                loss,
                carry[2],
                carry[3],
                DataGeneratorContainer(
                    data=data, param_data=param_data, obs_data=obs_data
                ),
                carry[5],
                carry[6],
                carry[7],
            )
            # 2 - go for gradient steps on this eq_params
            return jax.lax.while_loop(break_fun_, _eq_params_one_iteration, carry)

        eq_params_train_fun_compiled[eq_param] = (
            jax.jit(eq_train_fun, static_argnums=0)
            .trace(n_iter_for_params, jax.eval_shape(lambda _: carry, (None,)))
            .lower()
            .compile()
        )

    # NOTE we precompile the repetitive call to jinns.solve()
    # In the plain while loop, the compilation is costly each time
    # In the jax lax while loop, the compilation is better but AOT is
    # disallowed there
    nn_break_fun = _get_break_fun(
        nn_n_iter, verbose, conditions_str=("bool_max_iter", "bool_nan_in_params")
    )

    def nn_train_fun(carry):
        def _nn_params_one_iteration(carry):
            (
                i,
                loss,
                optimization,
                _,
                train_data,
                _,
                _,
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
                carry[5],
                carry[6],
                carry[7],
            )

            return carry

        # 1 - some init
        loss = eqx.tree_at(
            lambda pt: pt.derivative_keys, carry[1], nn_gd_steps_derivative_keys
        )
        carry = (
            0,
            loss,
            carry[2],
            carry[3],
            carry[4],
            carry[5],
            carry[6],
            carry[7],
        )
        # 2 - go for gradient steps on nn_params
        return jax.lax.while_loop(nn_break_fun, _nn_params_one_iteration, carry)

    nn_params_train_fun_compiled = jax.jit(nn_train_fun).lower(carry).compile()

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

        jax.debug.print("jinns alternate solver iteration {x}", x=i)

        ###### OPTIMIZATION ON EQ_PARAMS ###########

        for eq_param, eq_optim in eq_optimizers.items():
            carry = eq_params_train_fun_compiled[eq_param](carry)

        ###### OPTIMIZATION ON NN_PARAMS ###########

        carry = nn_params_train_fun_compiled(carry)

        ############################################

        i += 1
        return (i, carry[1], carry[2], carry[3], carry[4], carry[5], carry[6], carry[7])

    # jax.lax.while_loop jits its content so cannot be used when we try to
    # precompile what is inside. JAX tranformations are not compatible with AOT
    while main_break_fun(carry):
        carry = _one_alternate_iteration(carry)

    return carry[2].params
