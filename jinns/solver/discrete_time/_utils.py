import warnings

import jax.numpy as jnp

from jinns.loss import PDEStatio
from jinns.loss import LossPDEStatio
from jinns.loss._loss_weights import LossWeightsPDEStatio
from jinns.parameters._derivative_keys import DerivativeKeysPDEStatio
from jinns.solver._solve import solve


def _fit_ic(
    n_iter_ic,
    optimizer_ic,
    opt_state_ic,
    data_ic,
    param_data_ic,
    loss,
    init_params_ic,
    extra_optax_args_and_kwargs_ic,
    ahead_of_time,
    initial_condition_fun,
    verbose,
):
    # The trick is to interpret the IC as a dynamic loss
    class InitialConditionAsDynamicLoss(PDEStatio):
        def equation(self, x, u, params):
            return u(x, params) - initial_condition_fun(x)

    ic_as_dyn_loss = InitialConditionAsDynamicLoss()
    with warnings.catch_warnings():
        # Here we catch the "Missing BC" warning as it is intended
        # in the neural Galerking approach
        warnings.simplefilter("ignore", UserWarning)

        loss_ic = LossPDEStatio(
            u=loss.u,
            dynamic_loss=ic_as_dyn_loss,
            loss_weights=LossWeightsPDEStatio(dyn_loss=jnp.array(1.0)),
            derivative_keys=DerivativeKeysPDEStatio.from_str(
                dyn_loss="nn_params", params=init_params_ic
            ),
        )
        res = solve(
            n_iter=n_iter_ic,
            init_params=init_params_ic,
            data=data_ic,
            loss=loss_ic,
            optimizer=optimizer_ic,
            print_loss_every=n_iter_ic // 10,
            opt_state=opt_state_ic,
            param_data=param_data_ic,
            extra_optax_args_and_kwargs=extra_optax_args_and_kwargs_ic,
            ahead_of_time=ahead_of_time,
            verbose=verbose,
        )
    return res[0]
