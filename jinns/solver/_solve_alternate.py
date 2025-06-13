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
from jinns.solver._solve import _get_break_fun, solve

if TYPE_CHECKING:
    from jinns.loss._abstract_loss import AbstractLoss
    from jinns.data._AbstractDataGenerator import AbstractDataGenerator


def solve_alternate(
    n_iter: Params[int],
    init_params: Params[Array],
    data: AbstractDataGenerator,
    loss: AbstractLoss,
    optimizers: Params[optax.GradientTransformation],
    verbose: bool = True,
    **kwargs_solve,
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
    """
    main_break_fun = _get_break_fun(
        n_iter, verbose, conditions_str=("bool_max_iter", "bool_nan_in_params")
    )
    # Even if provided by the user we get rid of the following keys because we
    # need to control their value for the alternate optimization
    for k in ("opt_state", "ahead_of_time", "params_mask"):
        kwargs_solve.pop(k, None)

    nn_n_iter = n_iter.nn_params
    eq_n_iters = n_iter.eq_params

    nn_optimizer = optimizers.nn_params
    eq_optimizers = optimizers.eq_params

    nn_opt_state = nn_optimizer.init(init_params.nn_params)

    eq_opt_states = jax.tree.map(
        lambda opt_, params_: opt_.init(params_),
        eq_optimizers,
        init_params.eq_params,
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
    )

    # and get the negative to optimize only on eq_params FOR EACH EQ_PARAMS
    # Hence the PyTree we need to construct to tree.map over is a little more
    # complex since we need to keep the overall dict structure
    nb_eq_params = len(
        jax.tree.leaves(
            eq_optimizers, is_leaf=lambda x: isinstance(x, optax.GradientTransformation)
        )
    )
    masks_ = tuple(jnp.eye(nb_eq_params)[i].astype(bool) for i in range(nb_eq_params))
    eq_params_masks_ = jax.tree.unflatten(
        jax.tree.structure(
            eq_optimizers, is_leaf=lambda x: isinstance(x, optax.GradientTransformation)
        ),
        masks_,
    )
    eq_params_masks = jax.tree.map(
        lambda l, ll: Params(nn_params=False, eq_params=ll),
        eq_optimizers,
        eq_params_masks_,
        is_leaf=lambda x: isinstance(x, optax.GradientTransformation),
    )

    # derivative keys with only eq_params updates for the gradient steps over eq_params
    # Again this is a dict for eqch eq_params
    eq_gd_steps_derivative_keys = jax.tree.map(
        lambda l: jax.tree.map(lambda ll: l, loss.derivative_keys),
        eq_params_masks,
    )

    # we checked that it was OK up to there but TODO is to introduce unit tests
    # for the pytree manipulations above

    def _one_alternate_iteration(carry):
        (params, data, loss, nn_opt_state, eq_opt_states) = carry

        # Some gradient descent steps over nn_params
        # Here we resort to the legacy DerivativeKeys to stop the update of
        # eq_params because params and nn_opt_state do include them

        loss = eqx.tree_at(
            lambda pt: pt.derivative_keys, loss, nn_gd_steps_derivative_keys
        )
        params, loss_values, _, data, loss, nn_opt_state, _, _, _ = solve(
            n_iter=nn_n_iter,
            init_params=params,
            data=data,
            loss=loss,
            optimizer=nn_optimizer,
            opt_state=nn_opt_state,
            ahead_of_time=False,
            params_mask=nn_params_mask,
            **kwargs_solve,
        )

        # Some gradient descent steps over the parameters in eq_params
        # We go for a for loop because, it is hard to mathematically
        # conceptualize a tree map over jinns.solve()
        eq_opt_states_ = {}
        for eq_param, eq_optim in eq_optimizers:
            loss = eqx.tree_at(
                lambda pt: pt.derivative_keys,
                loss,
                eq_gd_steps_derivative_keys[eq_param],
            )
            params, loss_values, _, data, loss, eq_opt_state, _, _, _ = solve(
                n_iter=eq_n_iters[eq_param],
                init_params=params,
                data=data,
                loss=loss,
                optimizer=eq_optim,
                opt_state=eq_opt_states[eq_param],
                ahead_of_time=False,
                params_mask=eq_params_masks[eq_param],
                **kwargs_solve,
            )
            eq_opt_states_ = eq_opt_states_ | {eq_param: eq_opt_state}
        eq_opt_states = eq_opt_states_
        carry = (params, data, loss, nn_opt_state, eq_opt_states)
        return carry

    carry = (init_params, data, loss, nn_opt_state, eq_opt_states)

    jax.lax.while_loop(main_break_fun, _one_alternate_iteration, carry)
