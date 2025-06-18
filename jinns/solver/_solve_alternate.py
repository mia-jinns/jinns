"""
`jinns.solve_alternate()` to efficiently resolve inverse problems
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array
import equinox as eqx

from jinns.parameters._params import Params
from jinns.solver._solve import _get_break_fun, solve, _gradient_step
from jinns.data._Batchs import ODEBatch
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


def solve_alternate(
    n_iter: int,
    n_iter_by_solver: Params[int],
    init_params: Params[Array],
    data: AbstractDataGenerator,
    loss: AbstractLoss,
    optimizers: Params[optax.GradientTransformation],
    verbose: bool = True,
    obs_data: DataGeneratorObservations | None = None,
):
    """
    Solve alternatively between nn_params and eq_params. In this functions both
    set of parameters, all gradient updates, all opt_states, etc. are
    explicitly handled separately. This approach becomes more efficient than
    relying on optax masked transforms and jinns DerivativeKeys when nn_params
    is big while eq_params is very small. The former do not prevent all
    computations from begin done (for example, we still have a big opt_state when updating
    only eq_params, jinns DerivativeKeys only consists in putting
    stop gradients, etc. -- see for example Inverse problem tutorial on Burgers
    in jinns documentation) and hence is suboptimal.

    With `jinns.solve_alternate` we want to address efficiently inverse
    problems where nn_optimizer is arbitrarily big, but eq_params prepresents
    only a few physical parameters.

    About the argument in this function: https://stackoverflow.com/a/58018872
    """
    main_break_fun = _get_break_fun(
        n_iter, verbose, conditions_str=("bool_max_iter", "bool_nan_in_params")
    )

    # get_batch = _get_get_batch(None)
    # Even if provided by the user we get rid of the following keys because we
    # need to control their value for the alternate optimization
    # for k in ("opt_state", "ahead_of_time", "params_mask"):
    #    kwargs_solve.pop(k, None)

    nn_n_iter = n_iter_by_solver.nn_params
    eq_n_iters = n_iter_by_solver.eq_params

    nn_optimizer = optimizers.nn_params
    eq_optimizers = optimizers.eq_params

    # NOTE NOTE NOTE below we have opt_states that are shaped as Params
    # maybe this is OK if the real gain is not to differentiate
    # wrt to unwanted params
    # OR MAYBE option 2 (see jinns solve)
    nn_opt_state = nn_optimizer.init(init_params)  # .nn_params)

    eq_opt_states = jax.tree.map(
        lambda opt_: opt_.init(init_params),
        eq_optimizers,
        # init_params.eq_params,
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

    # NOTE that we need to replace will plain bool:
    # 1. filter_spec does not even accept onp.array
    # 2. filter_spec does not accept non static arguments. So any jnp array is
    # non hashable and we will not be able to make it static
    # params_mask cannot be inside the carry of course, just like the
    # optimizer
    eq_params_masks = jax.tree.map(lambda l: replace_float(l), eq_params_masks)

    # derivative keys with only eq_params updates for the gradient steps over eq_params
    # Again this is a dict for eqch eq_params
    # eq_gd_steps_derivative_keys = jax.tree.map(
    #    lambda l: jax.tree.map(lambda ll: l, loss.derivative_keys),
    #    eq_params_masks,
    # )
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
    train_data = DataGeneratorContainer(data=data, param_data=None, obs_data=obs_data)
    optimization = OptimizationContainer(
        params=init_params,
        last_non_nan_params=init_params,
        opt_state=(nn_opt_state, eq_opt_states),
        # params_mask=(nn_params_mask, eq_params_masks),
    )
    optimization_extra = OptimizationExtraContainer(
        curr_seq=None,
        best_iter_id=None,
        best_val_criterion=None,
        best_val_params=None,
    )
    loss_container = LossContainer(
        stored_loss_terms=None,
        train_loss_values=None,
    )
    stored_objects = StoredObjectContainer(
        stored_params=None,
    )

    def _one_alternate_iteration(carry):
        (
            i,
            loss,
            optimization,
            optimization_extra,
            train_data,
            loss_container,
            stored_objects,
        ) = carry

        (nn_opt_state, eq_opt_states) = optimization.opt_state

        ###### OPTIMIZATION ON EQ_PARAMS ###########

        eq_opt_states_ = {}
        for eq_param, eq_optim in eq_optimizers.items():
            break_fun_ = _get_break_fun(
                eq_n_iters[eq_param],
                verbose,
                conditions_str=("bool_max_iter", "bool_nan_in_params"),
            )

            def _eq_params_one_iteration(carry):
                (
                    i,
                    loss,
                    optimization,
                    _,
                    train_data,
                    _,
                    _,
                ) = carry

                # batch, data, param_data, obs_data = get_batch(
                #    train_data.data, train_data.param_data, train_data.obs_data
                # )
                batch = ODEBatch(train_data.data.times)  # This is what Yanis does

                # Gradient step
                (
                    loss,
                    train_loss_value,
                    loss_terms,
                    params,
                    eq_opt_state,
                    last_non_nan_params,
                ) = _gradient_step(
                    loss,
                    eq_optim,
                    batch,
                    optimization.params,
                    optimization.opt_state,
                    optimization.last_non_nan_params,
                    eq_params_masks[eq_param],
                )

                carry = (
                    i + 1,
                    loss,
                    OptimizationContainer(params, last_non_nan_params, eq_opt_state),
                    carry[3],
                    carry[4],
                    carry[5],
                    carry[6],
                )

                return carry

            # 1 - some init
            loss = eqx.tree_at(
                lambda pt: (pt.derivative_keys),  # , pt.loss_weights.initial_condition,
                # pt.loss_weights.observations),
                carry[1],
                (eq_gd_steps_derivative_keys[eq_param]),  # , 0., 0.),
            )

            carry = (
                0,
                loss,
                OptimizationContainer(
                    carry[2].params,
                    carry[2].last_non_nan_params,
                    eq_opt_states[eq_param],
                ),
                carry[3],
                carry[4],
                carry[5],
                carry[6],
            )
            # 2 - go for gradient steps on this eq_params
            carry = jax.lax.while_loop(break_fun_, _eq_params_one_iteration, carry)
            eq_opt_states_ = eq_opt_states_ | {eq_param: carry[2].opt_state}
        eq_opt_states = eq_opt_states_

        ############################################

        # DEBUG: early get out of the loop
        # i += 1
        # return (
        #    i,
        #    carry[1],
        #    OptimizationContainer(
        #        carry[2].params,
        #        carry[2].params,
        #        (nn_opt_state, eq_opt_states),
        #    ),
        #    optimization_extra,
        #    DataGeneratorContainer(carry[4].data, train_data.param_data, train_data.obs_data),
        #    loss_container,
        #    stored_objects,
        # )

        ###### OPTIMIZATION ON NN_PARAMS ###########

        loss_ = eqx.tree_at(
            lambda pt: pt.derivative_keys, carry[1], nn_gd_steps_derivative_keys
        )
        # TODO jinns solve should return the modified obs
        # TODO below this should not call jinns solve
        # but directly loop over _gradient_step
        params, loss_values, _, data, loss, nn_opt_state, _, _, _ = solve(
            n_iter=nn_n_iter,
            init_params=carry[2].params,
            data=carry[4].data,
            loss=loss_,
            optimizer=nn_optimizer,
            opt_state=nn_opt_state,
            ahead_of_time=False,
            params_mask=nn_params_mask,
            verbose=verbose,
            print_loss_every=1000,
            obs_data=obs_data,
        )

        ############################################

        i += 1
        return (
            i,
            loss,
            OptimizationContainer(
                params,
                params,
                (nn_opt_state, eq_opt_states),
            ),
            optimization_extra,
            DataGeneratorContainer(data, train_data.param_data, train_data.obs_data),
            loss_container,
            stored_objects,
        )

    carry = (
        0,
        loss,
        optimization,
        optimization_extra,
        train_data,
        loss_container,
        stored_objects,
    )

    carry = jax.lax.while_loop(main_break_fun, _one_alternate_iteration, carry)
    return carry[2].params
