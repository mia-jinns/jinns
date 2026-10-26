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
from jinns.data._CubicMeshPDEStatio import CubicMeshPDEStatio
from jinns.loss._LossPDE import LossPDEStatio
from jinns.loss._DynamicLossAbstract import PDEStatio
from jinns.loss._loss_weights import LossWeightsPDEStatio
from jinns.parameters._derivative_keys import DerivativeKeysPDEStatio
from jinns.solver._solve import solve
from jinns.solver._utils import (
    _check_batch_size,
)
from jinns.parameters._params import Params
from jinns.optimizers._natural_gradient import NGDState
from jinns.utils._DictToModuleMeta import DictToModuleMeta

from jinns.utils._containers import OptimizationContainer
from jinns.solver.discrete_time._utils import (
    _fit_ic,
    _save_nn_params,
    _print_fn_time_discrete,
)


if TYPE_CHECKING:
    from jinns.loss._abstract_loss import AbstractLoss
    from jinns.data._DataGeneratorParameter import DataGeneratorParameter
    from jinns.data._AbstractDataGenerator import AbstractDataGenerator
    from jinns.solver._utils import GetJinnsVariableName
    from jinns.utils._types import SolveCarry


def solve_discrete_pinn(
    *,
    times,
    times_saved: Array,
    init_params: Params[Array],
    data: AbstractDataGenerator,
    loss: AbstractLoss,
    n_iter: int,
    optimizer: optax.GradientTransformation | optax.GradientTransformationExtraArgs,
    initial_condition_fun,
    n_iter_ic: int,
    optimizer_ic: optax.GradientTransformation | optax.GradientTransformationExtraArgs,
    data_ic: AbstractDataGenerator | None = None,
    print_loss_every: int | None = None,
    opt_state_ic: optax.OptState | NGDState | None = None,
    param_data: DataGeneratorParameter | None = None,
    param_data_ic: DataGeneratorParameter | None = None,
    init_params_ic: Params[Array] | None = None,
    verbose: bool = True,
    ahead_of_time: bool = True,
    extra_optax_args_and_kwargs: dict[str, Callable | GetJinnsVariableName]
    | None = None,
    extra_optax_args_and_kwargs_ic: dict[str, Callable | GetJinnsVariableName]
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
    if data.rar_parameters is not None and key is None:
        raise ValueError(
            "`key` argument must be passed to jinns.solve() when using RAR procedure"
        )

    assert isinstance(loss, LossPDEStatio)
    assert loss.boundary_condition is None
    assert loss.norm_samples is None
    assert isinstance(data, CubicMeshPDEStatio)

    dt = times[1] - times[0]
    n_time_steps = len(times)
    initialization_time = time.time()

    if print_loss_every is None:
        print_loss_every = max(n_time_steps // 10, 1)

    if data_ic is None:
        data_ic = data

    if param_data_ic is None:
        param_data_ic = param_data

    if init_params_ic is None:
        init_params_ic = init_params

    if param_data is not None:
        if param_data.param_batch_size is not None:
            # We need to check that batch sizes will all be compliant for
            # correct vectorization
            _check_batch_size(param_data, data, "param_batch_size")
        else:
            # If DataGeneratorParameter does not have a batch size we will
            # vectorization using `n`, and the same checks must be done
            _check_batch_size(param_data, data, "n")

    # initialize the PyTree for stored loss weights values
    if loss.update_weight_method is not None:
        raise ValueError(
            " Cannot set a value for `loss.update_weight_method` with solve_discrete_pinn"
        )

    if verbose:
        print("Initialization time:", time.time() - initialization_time)

    ################################################
    # 1) Fit the initial condition for t0=times[0] #
    ################################################

    print("\n\n #-- Fitting the initial condition for t0=", times[0])
    params_t0 = _fit_ic(
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
    )

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
        """
        However, static type checkers cannot know that DGParams inherit from
        eqx.Module and explicit casting to the latter class will be needed
        """

        pass

    # we precompile the solve functions that will be used iteratively
    # many times
    if scheme in ["rk1", "rk-1", "explicit-euler"]:
        _eq_params = _EqParams(
            asdict(params_t0.eq_params) | {"prev_params": params_t0}, "_EqParams"
        )
        params_t0_ = eqx.tree_at(lambda pt: pt.eq_params, params_t0, _eq_params)
        compiled_solve_rk1, carry = _get_compiled_solve_rk1(
            dt,
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
        _eq_params = _EqParams(
            asdict(params_t0.eq_params)
            | {"prev_params": params_t0, "rk2_params": params_t0},
            "_EqParams",
        )
        params_t0_ = eqx.tree_at(lambda pt: pt.eq_params, params_t0, _eq_params)
        # https://fr.wikipedia.org/wiki/M%C3%A9thodes_de_Runge-Kutta#Sch%C3%A9ma_pr%C3%A9dicteur-correcteur_explicite_(RK2)
        compiled_solve_rk1, carry = _get_compiled_solve_rk1(
            dt / 2,
            loss,
            params_t0_,
            n_iter,
            data,
            optimizer,
            param_data,
            extra_optax_args_and_kwargs,
        )
        compiled_solve_rk2, carry2 = _get_compiled_solve_rk2(
            dt,
            loss,
            params_t0_,
            n_iter,
            data,
            optimizer,
            param_data,
            extra_optax_args_and_kwargs,
        )
        dyn_loss1 = carry[1].dynamic_loss[0]
        dyn_loss2 = carry2[1].dynamic_loss[0]

        def compiled_solve(carry):
            carry = eqx.tree_at(  # we need to change the loss of the carry because the dynamic loss is a static
                # arg which is different in each compiled function
                lambda pt: pt[1].dynamic_loss[0],
                carry,
                dyn_loss1,
            )
            carry = compiled_solve_rk1(carry)
            carry = eqx.tree_at(  # see comment above
                lambda pt: pt[1].dynamic_loss[0], carry, dyn_loss2
            )
            carry, _ = update_carry_and_rk2_params(
                carry
            )  # prev_params stays the same but rk2_params enters the place
            carry = compiled_solve_rk2(carry)
            return carry

    else:
        raise ValueError("Wrong value for scheme!")

    n_loss_values_by_step = len(carry[6].train_loss_values)
    all_train_loss_values_saved = jnp.zeros(
        ((n_time_steps - 1) * n_loss_values_by_step),
    )

    params_t0_fl, unravel_params = jax.flatten_util.ravel_pytree(params_t0.nn_params)
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

        carry = compiled_solve(carry)

        carry, new_params = update_carry_and_prev_params(carry)

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
            lambda pt: pt.nn_params, init_params, unravel_params(nn_params_saved[i])
        )
        for i in range(nn_params_saved.shape[0])
    )

    loss = carry[1]
    train_data = carry[4]

    return (
        loss,
        train_data.data,
        train_data.param_data,
        params_saved,
        all_train_loss_values_saved,
    )


@jax.jit
def update_carry_and_prev_params(carry):
    # update prev_params
    new_params = carry[2].params
    new_params = eqx.tree_at(
        lambda pt: pt.eq_params.prev_params.nn_params,  # NOTE this forbids eq_params updates!
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


@jax.jit
def update_carry_and_rk2_params(carry):
    # update prev_params
    new_params = carry[2].params
    new_params = eqx.tree_at(
        lambda pt: pt.eq_params.rk2_params.nn_params,  # NOTE this forbids eq_params updates!
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


def _get_compiled_solve_rk1(
    dt,
    loss,
    params_t0_,
    n_iter,
    data,
    optimizer,
    param_data,
    extra_optax_args_and_kwargs,
):
    # we modify the dynamic loss accordingly
    class DynLossRK1(PDEStatio):
        def equation(self, x, u, params):
            return (
                u(x, params)
                - u(x, params.eq_params.prev_params)
                + dt * loss.dynamic_loss[0].equation(x, u, params.eq_params.prev_params)
            )

    dyn_loss_rk1 = DynLossRK1()
    # below we update the dynamic loss and the derivative keys accordingly
    # we also set the loss wegihts to a default useless value
    loss_discrete_pinn = eqx.tree_at(
        lambda pt: (pt.dynamic_loss, pt.derivative_keys, pt.loss_weights),
        loss,
        (
            (dyn_loss_rk1,),
            DerivativeKeysPDEStatio(params=params_t0_),
            LossWeightsPDEStatio(dyn_loss=jnp.array(1.0)),
        ),
    )
    compiled_solve_rk1, carry = solve(  # pyright: ignore
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
        compiled_solve_rk1 = cast(
            Callable[[SolveCarry], SolveCarry], compiled_solve_rk1
        )

    return compiled_solve_rk1, carry


def _get_compiled_solve_rk2(
    dt,
    loss,
    params_t0_,
    n_iter,
    data,
    optimizer,
    param_data,
    extra_optax_args_and_kwargs,
):
    # we modify the dynamic loss accordingly
    class DynLossRK2(PDEStatio):
        def equation(self, x, u, params):
            return (
                u(x, params)
                - u(x, params.eq_params.prev_params)
                + dt * loss.dynamic_loss[0].equation(x, u, params.eq_params.rk2_params)
            )

    dyn_loss_rk2 = DynLossRK2()
    # below we update the dynamic loss and the derivative keys accordingly
    # we also set the loss wegihts to a default useless value
    loss_discrete_pinn = eqx.tree_at(
        lambda pt: (pt.dynamic_loss, pt.derivative_keys, pt.loss_weights),
        loss,
        (
            (dyn_loss_rk2,),
            DerivativeKeysPDEStatio(params=params_t0_),
            LossWeightsPDEStatio(dyn_loss=jnp.array(1.0)),
        ),
    )
    compiled_solve_rk2, carry = solve(  # pyright: ignore
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
        compiled_solve_rk2 = cast(
            Callable[[SolveCarry], SolveCarry], compiled_solve_rk2
        )

    return compiled_solve_rk2, carry
