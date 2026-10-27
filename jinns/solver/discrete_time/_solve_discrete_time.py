"""
Entry point for the discrete time methods
"""

import time
from typing import Literal, Callable
from jaxtyping import Array, PRNGKeyArray
import optax

from jinns.data._CubicMeshPDEStatio import CubicMeshPDEStatio
from jinns.loss._LossPDE import LossPDEStatio
from jinns.solver.discrete_time._utils import _fit_ic
from jinns.solver.discrete_time._solve_discrete_pinn import _solve_discrete_pinn
from jinns.solver.discrete_time._solve_neural_galerkin import _solve_neural_galerkin
from jinns.solver._utils import (
    _check_batch_size,
)
from jinns.optimizers._natural_gradient import NGDState
from jinns.parameters._params import Params

from jinns.data._DataGeneratorParameter import DataGeneratorParameter
from jinns.solver._utils import GetJinnsVariableName


def solve_discrete_time(
    *,
    times,
    times_saved: Array,
    init_params: Params[Array],
    method: Literal["discrete_pinn", "neural_galerkin"],
    data: CubicMeshPDEStatio,
    loss: LossPDEStatio,
    initial_condition_fun,
    n_iter_ic: int,
    optimizer_ic: optax.GradientTransformation | optax.GradientTransformationExtraArgs,
    n_iter: int | None = None,
    optimizer: optax.GradientTransformation
    | optax.GradientTransformationExtraArgs
    | None = None,
    data_ic: CubicMeshPDEStatio | None = None,
    print_loss_every: int | None = None,
    opt_state_ic: optax.OptState | NGDState | None = None,
    param_data: DataGeneratorParameter | None = None,
    param_data_ic: DataGeneratorParameter | None = None,
    verbose: bool = True,
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

    Parameters
    ----------
    times
        The time discretization for of the discrete time method as a jnp array
    times_saved
        The times at which the parameters will be saved
    init_params
        The initial set of parameters
    method
        The discrete time method to used. Currently, Discrete-PINN and Neural Galerkin are the two approaches available
    data
        A `jinns.data.CubicMeshPDEStatio` object to retrieve batches of collocation points at each
        optimization of each time step where the problem is seen as stationary
    loss
        A `jinns.loss.LossPDEStatio` to minimize at each time step where
        the problem is seen as stationary
    initial_condition_fun
        The initial condition to be fitted
    n_iter_ic
        The maximum number of iterations in the optimization for the initial condition
    optimizer_ic
        The optimizer for the fitting of the initial condition
    n_iter
        Only for `method==discrete_pinn` otherwise unused!
        The maximum number of iterations in the optimization at each time step
    optimizer
        Only for `method==discrete_pinn` otherwise unused!
        The optimizer for the optimization at each time step
    data_ic
        A `jinns.data.CubicMeshPDEStatio` object to retrieve batches of collocation points
        for the fitting of the initial condition. To be used if this should be different
        then `data`. Default is None and `data` is used for the fitting of the initial condition
    print_loss_every
        Default is None (meaning `n_iter // 10`).
        The rate at which we print the loss value in the opitimization.
    param_data
        Default `None`. A `jinns.data.DataGeneratorParameter` object which can be used to
        sample equation parameters.
    param_data_ic
        Default `None`. A `jinns.data.DataGeneratorParameter` object which can be used to
        sample equation parameters while fitting the initial condition
    verbose
        Default `True`. If `False`, no output (loss or cause of
        exiting the optimization loop) will be produced.
    extra_optax_args_and_kwargs
        Only for `method==discrete_pinn` otherwise unused!
        Use this for optax optimizers used at each time steps that require extra arguments.
        See `jinns.solver.solve` documentation for all the details.
        Default is None.
    extra_optax_args_and_kwargs_ic
        Use this for optax optimizers used for fitting initial condition (ie `optimizer_ic`)
        that require extra arguments.
        See `jinns.solver.solve` documentation for all the details.
        Default is None.
    key
        Default `None`. A JAX random key that can be used for diverse purpose in
        the main iteration loop.
    scheme
        Default 'RK4`. The Runge Kutta scheme that is used in the discrete time approach.

        - Euler explicit scheme available with values "explicit-euler", "rk1", "rk-1"
        - Runge Kutta second order scheme with values "rk2" or "rk-2"
        - Runge Kutta fourth order scheme with values "rk4" or "rk-4"

    Returns
    -------
    XXX
    """
    if data.rar_parameters is not None and key is None:
        raise ValueError(
            "`key` argument must be passed to jinns.solve() when using RAR procedure"
        )

    assert isinstance(loss, LossPDEStatio)
    assert loss.boundary_condition is None
    assert loss.norm_samples is None
    assert isinstance(data, CubicMeshPDEStatio)

    n_time_steps = len(times)
    initialization_time = time.time()

    if print_loss_every is None:
        print_loss_every = max(n_time_steps // 10, 1)

    if data_ic is None:
        data_ic = data

    if param_data_ic is None:
        param_data_ic = param_data

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

    # COMMON FOR BOTH METHODS
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
        init_params,
        extra_optax_args_and_kwargs_ic,
        initial_condition_fun,
        verbose,
    )

    if method == "discrete_pinn":
        if n_iter is None:
            raise ValueError("n_iter can not be None if method==discrete_pinn")
        if optimizer is None:
            raise ValueError("optimizer can not be None if method==discrete_pinn")

        return _solve_discrete_pinn(
            params_t0=params_t0,
            times=times,
            times_saved=times_saved,
            data=data,
            loss=loss,
            n_iter=n_iter,
            optimizer=optimizer,
            print_loss_every=print_loss_every,
            param_data=param_data,
            verbose=verbose,
            extra_optax_args_and_kwargs=extra_optax_args_and_kwargs,
            scheme=scheme,
        )
    elif method == "neural_galerkin":
        return _solve_neural_galerkin(
            params_t0=params_t0,
            times=times,
            times_saved=times_saved,
            data=data,
            loss=loss,
            print_loss_every=print_loss_every,
            param_data=param_data,
            verbose=verbose,
            key=key,
            scheme=scheme,
        )
