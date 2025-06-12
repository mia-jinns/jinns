"""
`jinns.solve_alternate()` to efficiently resolve inverse problems
"""

from typing import TYPE_CHECKING
import jax
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

    nn_optimizer = optimizers.nn_params
    eq_optimizers = optimizers.eq_params

    nn_opt_state = nn_optimizer.init(init_params) # NOTE that this is not
    # init_params.nn_params here because the gradient descent step on nn_params
    # will be totally classical. We afford this because of the hypothesis that
    # eq_params is much smaller than nn_params
    eq_opt_state = jax.tree_map(
        lambda opt, params: opt.init(params), eq_optimizers, init_params.eq_params
    )

    main_break_fun = _get_break_fun(
        n_iter,
        verbose,
        conditions_str=("bool_max_iter", "bool_nan_in_params")
    )

    # Even if provided by the user we get rid of the following keys because we
    # need to control their value for the alternate optimization
    for k in ("opt_state", "ahead_of_time"):
        kwargs_solve.pop(k, None)

    # derivative keys with only nn_params updates for the gradient steps over nn_params
    nn_gd_steps_derivative_keys = jax.tree.map(
        lambda l: Params(
            nn_params=True,
            eq_params=jax.tree.map(lambda ll: False, init_params.eq_params)),
        loss.derivative_keys,
    )

    def _one_alternate_iteration(carry):
        (
            params,
            data,
            loss,
            nn_opt_state,
            eq_opt_state
        ) = carry

        # Some gradient descent steps over nn_params
        # Here we resort to the legacy DerivativeKeys to stop the update of
        # eq_params because params and nn_opt_state do include them

        loss = eqx.tree_at(
            lambda pt: pt.derivative_keys,
            loss,
            nn_gd_steps_derivative_keys
        )
        out = solve(
            n_iter=nn_n_iter,
            init_params=params, # NOTE we do not get rid of eq_params here
            data=data,
            loss=loss,
            optimizer=nn_optimizer,
            opt_state=nn_opt_state,
            ahead_of_time=False,
            **kwargs_solve,
        )

        # Some gradient descent steps over eq_params
        return carry

    carry = (
        init_params,
        data,
        loss,
        nn_opt_state,
        eq_opt_state
    )

    jax.lax.while_loop(main_break_fun, _one_alternate_iteration, carry)
