"""
This modules implements the resolution of a PDE with Neural Galerkin approach
"""

from __future__ import (
    annotations,
)  # https://docs.python.org/3/library/typing.html#constant

import time
from typing import TYPE_CHECKING, Callable, Literal, cast
from dataclasses import asdict
import optax
import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, PRNGKeyArray
from jinns.loss._DynamicLossAbstract import PDEStatio
from jinns.loss._loss_weights import LossWeightsPDEStatio
from jinns.parameters._derivative_keys import DerivativeKeysPDEStatio
from jinns.solver._solve import solve
from jinns.parameters._params import Params
from jinns.utils._DictToModuleMeta import DictToModuleMeta
from jinns.data._CubicMeshPDEStatio import CubicMeshPDEStatio
from jinns.loss._LossPDE import LossPDEStatio
from jinns.utils._containers import OptimizationContainer
from jinns.solver.discrete_time._utils import (
    _save_nn_params,
    _print_fn_time_discrete,
)


if TYPE_CHECKING:
    from jinns.data._DataGeneratorParameter import DataGeneratorParameter
    from jinns.solver._utils import GetJinnsVariableName
    from jinns.utils._types import SolveCarry


def _solve_discrete_pinn(
    *,
    params_t0: Params[Array],
    times: Array,
    times_saved: Array,
    data: CubicMeshPDEStatio,
    loss: LossPDEStatio,
    n_iter: int,
    optimizer: optax.GradientTransformation | optax.GradientTransformationExtraArgs,
    print_loss_every: int | None = None,
    param_data: DataGeneratorParameter | None = None,
    verbose: bool = True,
    extra_optax_args_and_kwargs: dict[str, Callable | GetJinnsVariableName]
    | None = None,
    key: PRNGKeyArray | None = None,
    scheme: Literal[
        "rk4", "rk-4", "explicit-euler", "rk1", "rk-1", "rk2", "rk-2"
    ] = "rk4",
):
    """
    Solve the PDE with the discrete PINN approach

    This function solves PDE with time dependency but, in this methodology,
    the dependency on time only appears through the parameters.
    Therefore the loss that is passed (and the PINN) are here defined as StatioPDE (and therefore PINN
    without time as input). The loss must only evaluate the dynamic_loss attribute (attributes inducing other terms must
    be None).
    Same for the DataGenerator, it must then be a CubicMeshPDEStatio.
    On the other hand, the initial condition and the time at which we start the resolution is passed directly as arguments

    The methodology starts with fitting the initial condition and then XXX

    NOTE the initial condition is the function that that network must fit at times[0]

    Note that the boundary conditions must be hardcoded in this approach.

    Observations cannot be incorporated in this approach

    References:
    - Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations, M. Raissi et al., 2019
    - Neural semi-Lagrangian method for high-dimensional advection-diffusion problems, E. Franck et al., 2025
    """

    dt = times[1] - times[0]
    n_time_steps = len(times)
    assert loss.dynamic_loss is not None  # for pyright
    ################################
    # 2) Get the parameter dynamic #
    ################################
    print("\n\n #-- Starting the optimizations at each time step.")
    times_saved = jnp.array(times_saved)
    n_times_saved = len(times_saved)

    # We need to modify internally Params to be able to store the previous parameters
    # for evaluation in the dynamic loss.abs
    # # We then make a little abuse of what Params.eq_params is meant for
    class _EqParams(metaclass=DictToModuleMeta):
        """ """

        pass

    # we precompile the solve functions that will be used iteratively
    # many times
    if scheme in ["rk1", "rk-1", "explicit-euler"]:
        _eq_params = _EqParams(
            asdict(params_t0.eq_params) | {"prev_params": params_t0}, "_EqParams"
        )
        params_t0_ = eqx.tree_at(lambda pt: pt.eq_params, params_t0, _eq_params)
        # we need to recreate _eq_params_heterogeneity as a _EqParams instance
        _eq_params_heterogeneity = _EqParams(
            asdict(loss.dynamic_loss[0].eq_params_heterogeneity)
            | {"prev_params": None},
            "_EqParams",  # type: ignore
        )
        dyn_loss_rk = _get_dyn_loss_rk1or2(loss.dynamic_loss[0], "prev_params", dt)(
            eq_params_heterogeneity=_eq_params_heterogeneity,
        )
        compiled_solve_rk1, carry = _get_compiled_solve_rk(
            dyn_loss_rk,
            loss,
            params_t0_,
            n_iter,
            data,
            optimizer,
            param_data,
            extra_optax_args_and_kwargs,
        )

        def compiled_solve(carry):
            return compiled_solve_rk1(carry)

    elif scheme in ["rk2", "rk-2"]:
        # The second equation of RK2 needs 3 sets of parameters
        _eq_params = _EqParams(
            asdict(params_t0.eq_params)
            | {"prev_params": params_t0, "rk2_params": params_t0},
            "_EqParams",
        )
        params_t0_ = eqx.tree_at(lambda pt: pt.eq_params, params_t0, _eq_params)
        _eq_params_heterogeneity = _EqParams(
            asdict(loss.dynamic_loss[0].eq_params_heterogeneity)
            | {"prev_params": None}
            | {"rk2_params": None},
            "_EqParams",  # type: ignore
        )
        # https://fr.wikipedia.org/wiki/M%C3%A9thodes_de_Runge-Kutta#Sch%C3%A9ma_pr%C3%A9dicteur-correcteur_explicite_(RK2)
        # equations from the previous linked paragraph with f being the stationary differential operator
        dyn_loss_rk1 = _get_dyn_loss_rk1or2(
            loss.dynamic_loss[0], "prev_params", dt / 2
        )(
            eq_params_heterogeneity=_eq_params_heterogeneity,
        )
        compiled_solve_rk1, carry = _get_compiled_solve_rk(
            dyn_loss_rk1,
            loss,
            params_t0_,
            n_iter,
            data,
            optimizer,
            param_data,
            extra_optax_args_and_kwargs,
        )
        dyn_loss_rk2 = _get_dyn_loss_rk1or2(loss.dynamic_loss[0], "rk2_params", dt)(
            eq_params_heterogeneity=_eq_params_heterogeneity, params=params_t0_
        )
        compiled_solve_rk2, carry2 = _get_compiled_solve_rk(
            dyn_loss_rk2,
            loss,
            params_t0_,
            n_iter,
            data,
            optimizer,
            param_data,
            extra_optax_args_and_kwargs,
        )
        dyn_loss1 = carry[1].dynamic_loss[0]  # pyright: ignore
        dyn_loss2 = carry2[1].dynamic_loss[0]  # pyright: ignore

        def compiled_solve(carry):
            carry = eqx.tree_at(  # we need to change the loss of the carry because the dynamic loss is a static
                # arg which is different in each compiled function
                lambda pt: pt[1].dynamic_loss[0],
                carry,
                dyn_loss1,
            )
            carry = compiled_solve_rk1(carry)  # this computes first equation XXX
            carry = eqx.tree_at(  # see comment above
                lambda pt: pt[1].dynamic_loss[0], carry, dyn_loss2
            )
            carry, _ = update_carry_and_xxx_params(
                carry, "rk2_params"
            )  # prev_params stays the same and
            # rk2_params enters the place ie the parameters which parameterize the solution of the first equation
            # ie. the derivative at the middle of the integration step
            carry = compiled_solve_rk2(carry)  # this computes second equation XXX
            return carry
    elif scheme in ["rk4", "rk-4"]:
        _eq_params = _EqParams(
            asdict(params_t0.eq_params)
            | {
                "prev_params": params_t0,
                "rk4_params_1": params_t0,
                "rk4_params_2": params_t0,
                "rk4_params_3": params_t0,
            },
            "_EqParams",
        )
        params_t0_ = eqx.tree_at(lambda pt: pt.eq_params, params_t0, _eq_params)
        _eq_params_heterogeneity = _EqParams(
            asdict(loss.dynamic_loss[0].eq_params_heterogeneity)
            | {"prev_params": None}
            | {"rk4_params_1": None}
            | {"rk4_params_2": None}
            | {"rk4_params_3": None},
            "_EqParams",
        )
        dyn_loss_rk1 = _get_dyn_loss_rk1or2(
            loss.dynamic_loss[0], "prev_params", dt / 2
        )(
            eq_params_heterogeneity=_eq_params_heterogeneity,
        )
        compiled_solve_rk1, carry = _get_compiled_solve_rk(
            dyn_loss_rk1,
            loss,
            params_t0_,
            n_iter,
            data,
            optimizer,
            param_data,
            extra_optax_args_and_kwargs,
        )
        dyn_loss_rk2 = _get_dyn_loss_rk1or2(
            loss.dynamic_loss[0], "rk4_params_1", dt / 2
        )(
            eq_params_heterogeneity=_eq_params_heterogeneity,
        )
        compiled_solve_rk2, carry2 = _get_compiled_solve_rk(
            dyn_loss_rk2,
            loss,
            params_t0_,
            n_iter,
            data,
            optimizer,
            param_data,
            extra_optax_args_and_kwargs,
        )
        dyn_loss_rk2_2 = _get_dyn_loss_rk1or2(loss.dynamic_loss[0], "rk4_params_2", dt)(
            eq_params_heterogeneity=_eq_params_heterogeneity,
        )
        compiled_solve_rk2_2, carry3 = _get_compiled_solve_rk(
            dyn_loss_rk2_2,
            loss,
            params_t0_,
            n_iter,
            data,
            optimizer,
            param_data,
            extra_optax_args_and_kwargs,
        )
        dyn_loss_rk4 = _get_dyn_loss_rk4(loss.dynamic_loss[0], dt / 6)(
            eq_params_heterogeneity=_eq_params_heterogeneity,
        )
        compiled_solve_rk4, carry4 = _get_compiled_solve_rk(
            dyn_loss_rk4,
            loss,
            params_t0_,
            n_iter,
            data,
            optimizer,
            param_data,
            extra_optax_args_and_kwargs,
        )

        dyn_loss1 = carry[1].dynamic_loss[0]  # pyright: ignore
        dyn_loss2 = carry2[1].dynamic_loss[0]  # pyright: ignore
        dyn_loss3 = carry3[1].dynamic_loss[0]  # pyright: ignore
        dyn_loss4 = carry4[1].dynamic_loss[0]  # pyright: ignore

        def compiled_solve(carry):
            carry = eqx.tree_at(  # we need to change the loss of the carry because the dynamic loss is a static
                # arg which is different in each compiled function
                lambda pt: pt[1].dynamic_loss[0],
                carry,
                dyn_loss1,
            )
            carry = compiled_solve_rk1(carry)  # this computes y_n + h / 2 * k_1
            carry = eqx.tree_at(  # see comment above
                lambda pt: pt[1].dynamic_loss[0], carry, dyn_loss2
            )
            carry, _ = update_carry_and_xxx_params(
                carry, "rk4_params_1"
            )  # prev_params stays the same and rk2_params are added
            # rk2_params are set (and fixed) to the values that parametrize y_n + h / 2 * k_1
            carry = compiled_solve_rk2(carry)  # this computes y_n + h / 2 * k_2
            carry = eqx.tree_at(  # see comment above
                lambda pt: pt[1].dynamic_loss[0], carry, dyn_loss3
            )
            carry, _ = update_carry_and_xxx_params(
                carry, "rk4_params_2"
            )  # prev_params stays the same and rk2_params are added
            # rk2_params are set (and fixed) to the values that parametrize y_n + h / 2 * k_2
            carry = compiled_solve_rk2_2(carry)  # this computes y_n + h * k_3
            carry = eqx.tree_at(  # see comment above
                lambda pt: pt[1].dynamic_loss[0], carry, dyn_loss4
            )
            carry, _ = update_carry_and_xxx_params(
                carry, "rk4_params_3"
            )  # prev_params stays the same and rk2_params are added
            carry = compiled_solve_rk4(carry)  # this computes y_n + h * k_3

            return carry
    else:
        raise ValueError("Wrong value for scheme!")

    n_loss_values_by_step = len(carry[6].train_loss_values)
    all_train_loss_values_saved = jnp.zeros(
        ((n_time_steps - 1) * n_loss_values_by_step),
    )

    params_t0_fl, unravel_params = jax.flatten_util.ravel_pytree(params_t0.nn_params)  # pyright: ignore
    # Only JAX arrays can be index with traced value (the result from jnp.argwhere)
    # hence we store it in a flattened way

    nn_params_saved = jnp.stack([params_t0_fl for _ in range(n_times_saved)], axis=0)

    # since we use an AOT compiled function, this cannot be used
    # in another JIT compiled function
    step = 1  # start at one!
    start = time.time()
    while step < n_time_steps:
        t = times[step]
        if verbose:
            _print_fn_time_discrete(
                step,
                carry[6].train_loss_values[-1],
                print_loss_every,
                prefix="[Train Discrete PINN] ",
                t=jnp.round(t, 2),
                tmax=times[-1],
            )

        # TODO Here we could have a RAR step...

        carry = compiled_solve(carry)

        carry, new_params = update_carry_and_xxx_params(carry)

        # save the loss terms
        idx = (step - 1) * n_loss_values_by_step
        all_train_loss_values_saved = jax.lax.dynamic_update_slice(
            all_train_loss_values_saved, carry[6].train_loss_values, (idx,)
        )

        # save the parameters at requested times
        nn_params_saved = _save_nn_params(t, times_saved, new_params, nn_params_saved)

        step += 1

    jax.block_until_ready(carry)
    end = time.time()
    if verbose:
        print("\nTraining took\n", end - start, "\n")

    params_saved = tuple(
        eqx.tree_at(
            lambda pt: pt.nn_params, params_t0, unravel_params(nn_params_saved[i])
        )
        for i in range(nn_params_saved.shape[0])
    )

    loss = cast(LossPDEStatio, carry[1])
    train_data = carry[4]
    params_final = carry[2].params

    return (
        loss,
        train_data.data,
        train_data.param_data,
        params_final,
        params_saved,
        all_train_loss_values_saved,
    )


@jax.jit(static_argnames=["xxx_params"])
def update_carry_and_xxx_params(carry, xxx_params="prev_params"):
    """
    Set the value of xxx_params of eq_params of the carry current parameters (carry[2].params)
    to the value of nn_params of the carry current parameters (carry[2].params)

    Also reset the carry counter to 0
    """
    new_params = carry[2].params
    new_params = eqx.tree_at(
        lambda pt: getattr(
            pt.eq_params, xxx_params
        ).nn_params,  # NOTE this forbids eq_params updates!
        new_params,
        new_params.nn_params,
    )

    # refresh carry for next time step
    carry = eqx.tree_at(
        lambda pt: (pt[0], pt[2]),
        carry,
        (
            0,  # this needs to be reset
            OptimizationContainer(
                params=new_params,
                last_non_nan_params=carry[2].last_non_nan_params,
                opt_state=carry[2].opt_state,
            ),
        ),
    )
    return carry, new_params


def _get_dyn_loss_rk1or2(dyn_loss, space_operator_params_name, dt):
    """
    From the PDE residuals (without the time derivative component) (ie the space operator <=> dyn_loss),
    construct the dyn_loss (ie the residuals) that will be minimized in the RK step

    RK1 <=> space_operator_params_name == "prev_params"
    """

    # we modify the dynamic loss accordingly
    class DynLossRK(PDEStatio):
        def equation(self, x, u, params):
            return (
                u(x, params)
                - jax.lax.stop_gradient(
                    u(x, params.eq_params.prev_params)
                    - dt
                    * dyn_loss.equation(
                        x,
                        u,
                        eqx.tree_at(  # NOTE we need to make sure that params.eq_params are the actual eq_params that are used
                            # because they have undergone the eq_params_heterogeneity function
                            # hence the tree_at to just swap the nn_params
                            lambda pt: pt.nn_params,
                            params,
                            getattr(
                                params.eq_params, space_operator_params_name
                            ).nn_params,
                        ),
                    )
                )
            )

    return DynLossRK


def _get_dyn_loss_rk4(dyn_loss, dt):
    class DynLossRK4(PDEStatio):
        def equation(self, x, u, params):
            return (
                u(x, params)
                - jax.lax.stop_gradient(
                    u(x, params.eq_params.prev_params)
                    - dt
                    * (
                        dyn_loss.equation(
                            x,
                            u,
                            eqx.tree_at(  # NOTE we need to make sure that params.eq_params are the actual eq_params that are used
                                # because they have undergone the eq_params_heterogeneity function
                                # hence the tree_at to just swap the nn_params
                                lambda pt: pt.nn_params,
                                params,
                                params.eq_params.prev_params.nn_params,
                            ),
                        )
                        + 2
                        * dyn_loss.equation(
                            x,
                            u,
                            eqx.tree_at(  # NOTE see comment above
                                lambda pt: pt.nn_params,
                                params,
                                params.eq_params.rk4_params_1.nn_params,
                            ),
                        )
                        + 2
                        * dyn_loss.equation(
                            x,
                            u,
                            eqx.tree_at(
                                lambda pt: pt.nn_params,
                                params,
                                params.eq_params.rk4_params_2.nn_params,
                            ),
                        )
                        + dyn_loss.equation(
                            x,
                            u,
                            eqx.tree_at(
                                lambda pt: pt.nn_params,
                                params,
                                params.eq_params.rk4_params_3.nn_params,
                            ),
                        )
                    )
                )
            )

    return DynLossRK4


def _get_compiled_solve_rk(
    dyn_loss,
    loss,
    params_t0_,
    n_iter,
    data,
    optimizer,
    param_data,
    extra_optax_args_and_kwargs,
):
    # below we update the dynamic loss and the derivative keys accordingly
    # we also set the loss wegihts to a default useless value
    loss_discrete_pinn = eqx.tree_at(
        lambda pt: (pt.dynamic_loss, pt.derivative_keys, pt.loss_weights),
        loss,
        (
            (dyn_loss,),
            DerivativeKeysPDEStatio(params=params_t0_),
            LossWeightsPDEStatio(dyn_loss=jnp.array(1.0)),
        ),
    )
    compiled_solve_rk, carry = solve(  # pyright: ignore
        n_iter=n_iter,
        init_params=params_t0_,
        data=data,
        loss=loss_discrete_pinn,
        optimizer=optimizer,
        param_data=param_data,
        extra_optax_args_and_kwargs=extra_optax_args_and_kwargs,
        ahead_of_time=True,
        verbose=False,
        compile_only=True,
    )
    if TYPE_CHECKING:
        carry = cast(SolveCarry, carry)
        compiled_solve_rk = cast(Callable[[SolveCarry], SolveCarry], compiled_solve_rk)

    return compiled_solve_rk, carry
