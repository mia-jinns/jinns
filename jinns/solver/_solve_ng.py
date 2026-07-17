"""
This modules implements the resolution of a PDE with Neural Galerkin approach
"""

from __future__ import (
    annotations,
)  # https://docs.python.org/3/library/typing.html#constant

import time
from typing import TYPE_CHECKING, Callable
import optax
import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array
from jinns.data._CubicMeshPDEStatio import CubicMeshPDEStatio
from jinns.loss._LossPDE import LossPDEStatio
from jinns.loss._DynamicLossAbstract import PDEStatio
from jinns.loss._loss_weights import LossWeightsPDEStatio
from jinns.parameters._derivative_keys import DerivativeKeysPDEStatio
from jinns.solver._solve import solve
from jinns.solver._utils import (
    _check_batch_size,
    _build_get_batch,
    _print_fn,
)
from jinns.nn._hyperpinn import _get_param_nb
from jinns.parameters._params import Params
from jinns.utils._containers import (
    DataGeneratorContainer,
)
from jinns.optimizers._natural_gradient import NGDState

if TYPE_CHECKING:
    from jinns.loss._abstract_loss import AbstractLoss
    from jinns.data._DataGeneratorParameter import DataGeneratorParameter
    from jinns.data._AbstractDataGenerator import AbstractDataGenerator
    from jinns.solver._utils import GetJinnsVariableName


def solve_ng(
    *,
    times,
    times_saved: Array,
    init_params: Params[Array],
    data: AbstractDataGenerator,
    loss: AbstractLoss,
    initial_condition_fun,
    n_iter_ic: int,
    optimizer_ic: optax.GradientTransformation | optax.GradientTransformationExtraArgs,
    print_loss_every: int | None = None,
    store_prev_params: bool = False,
    opt_state_ic: optax.OptState | NGDState | None = None,
    param_data: DataGeneratorParameter | None = None,
    verbose: bool = True,
    ahead_of_time: bool = True,
    extra_optax_args_and_kwargs_ic: dict[str, Callable | GetJinnsVariableName]
    | None = None,
):
    """
    Solve the PDE with the Neural Galerkin approach. In this approach the variation of the parameters
    of the PINN are seen as the solution of an ODE. From those, we can reconstruct the parameter dynamics
    hence the solution at each desired time.

    This function solves PDE with time dependency but, in this methodology,
    the dependency on time only appears through the parameters.
    Therefore the loss that is passed (and therefore the PINN) are here defined as StatioPDE (and therefore PINN
    without time as input). The loss must only evaluate the dynamic_loss attribute (attributes inducing other terms must
    be None).
    Same for the DataGenerator, it must then be a CubicMeshPDEStatio.
    On the other hand, the initial condition and the time at which we start the resolution is passed directly as arguments

    The methodology starts with fitting the initial condition and then an ODE is resolved to get the parameters
    for each of the times that are queried.

    Note that the boundary conditions must be hardcoded in this approach.

    Observations cannot be incorporated in this approach

    References:
    - Evolutional Deep Neural Network, Y. Du et al, 2021
    - Neural semi-Lagrangian method for high-dimensional advection-diffusion problems, E. Franck et al., 2025
    """
    assert isinstance(loss, LossPDEStatio)
    assert loss.boundary_condition is None
    assert loss.norm_samples is None
    assert isinstance(data, CubicMeshPDEStatio)

    dt = times[1] - times[0]
    n_iter = len(times)
    initialization_time = time.time()

    if print_loss_every is None:
        print_loss_every = n_iter // 10

    if param_data is not None:
        if param_data.param_batch_size is not None:
            # We need to check that batch sizes will all be compliant for
            # correct vectorization
            _check_batch_size(param_data, data, "param_batch_size")
        else:
            # If DataGeneratorParameter does not have a batch size we will
            # vectorization using `n`, and the same checks must be done
            _check_batch_size(param_data, data, "n")

    # depending on obs_batch_sharding we will get the simple get_batch or the
    # get_batch with device_put, the latter is not jittable
    get_batch = _build_get_batch(None)

    # initialize the PyTree for stored loss weights values
    if loss.update_weight_method is not None:
        raise ValueError(
            " Cannot set a value for `loss.update_weight_method` with solve_ng"
        )

    train_data = DataGeneratorContainer(data=data, param_data=param_data, obs_data=None)

    if verbose:
        print("Initialization time:", time.time() - initialization_time)

    ################################
    # 1) Fit the initial condition #
    ################################
    def _fit_ic(
        n_iter_ic,
        optimizer_ic,
        opt_state_ic,
        data,
        param_data,
        loss,
        init_params,
        extra_optax_args_and_kwargs_ic,
        ahead_of_time,
        initial_condition_fun,
    ):
        # The trick is to intererpret the IC as a dynamic loss
        class InitialConditionAsDynamicLoss(PDEStatio):
            def equation(self, x, u, params):
                return u(x, params) - initial_condition_fun(x)

        ic_as_dyn_loss = InitialConditionAsDynamicLoss()
        loss_ic = LossPDEStatio(
            u=loss.u,
            dynamic_loss=ic_as_dyn_loss,
            loss_weights=LossWeightsPDEStatio(dyn_loss=1.0),
            derivative_keys=DerivativeKeysPDEStatio.from_str(
                dyn_loss="nn_params", params=init_params
            ),
        )
        res = solve(
            n_iter=n_iter_ic,
            init_params=init_params,
            data=data,
            loss=loss_ic,
            optimizer=optimizer_ic,
            print_loss_every=n_iter_ic // 10,
            opt_state=opt_state_ic,
            param_data=param_data,
            extra_optax_args_and_kwargs=extra_optax_args_and_kwargs_ic,
            ahead_of_time=ahead_of_time,
        )
        return res[0]

    print("\n\n 1 - Fitting the initial condition")
    params_t0 = _fit_ic(
        n_iter_ic,
        optimizer_ic,
        opt_state_ic,
        data,
        param_data,
        loss,
        init_params,
        extra_optax_args_and_kwargs_ic,
        ahead_of_time,
        initial_condition_fun,
    )

    ################################
    # 2) Get the parameter dynamic #
    ################################
    print("\n\n 2 - Resolving the parameter dynamic")

    def _one_time_step(carry, t):
        (loss, params, train_data, params_saved) = carry

        batch, data, param_data, _ = get_batch(
            train_data.data, train_data.param_data, None
        )

        if store_prev_params:
            params = eqx.tree_at(
                lambda pt: pt.eq_params.prev_params, params, params.nn_params
            )

        params = _rk4_step(batch=batch, loss=loss, params=params, dt=dt)

        # Print train loss value during optimization
        if verbose:
            _print_fn(t, None, print_loss_every, prefix="[train Neural Galerkin] ")

        params_saved = jax.lax.cond(
            jnp.isin(t, jnp.array(times_saved)),
            lambda _: eqx.tree_at(lambda pt: pt[0], params_saved, params),
            lambda _: params_saved,
            None,
        )

        return (
            loss,
            params,
            DataGeneratorContainer(data, param_data, None),
            params_saved,
        ), None

    params_saved = tuple(params_t0 for _ in range(len(times_saved)))

    carry = (loss, params_t0, train_data, params_saved)

    def train_fun(carry):
        return jax.lax.scan(_one_time_step, carry, times)

    if ahead_of_time:
        start = time.time()
        compiled_train_fun = jax.jit(train_fun).lower(carry).compile()
        end = time.time()
        if verbose:
            print("\nCompilation took\n", end - start, "\n")

        start = time.time()
        carry, _ = compiled_train_fun(carry)
        jax.block_until_ready(carry)
        end = time.time()
        if verbose:
            print("\nTraining took\n", end - start, "\n")
    else:
        carry, _ = train_fun(carry)

    (loss, params_final, train_data, params_saved) = carry

    return params_final, params_saved


def _rk4_step(batch, loss, params, dt):
    """
    Compte the next value of the parameters following Runge Kutta scheme of
    4th order.

    """
    dnu_dt_k1 = _get_dnu_dt(batch, loss, params)
    dnu_dt_k2 = _get_dnu_dt(
        batch,
        loss,
        eqx.tree_at(
            lambda pt: pt.nn_params,
            params,
            jax.tree.map(
                lambda b, c: b + c * dt / 2, params.nn_params, dnu_dt_k1.nn_params
            ),
        ),
    )
    dnu_dt_k3 = _get_dnu_dt(
        batch,
        loss,
        eqx.tree_at(
            lambda pt: pt.nn_params,
            params,
            jax.tree.map(
                lambda b, c: b + c * dt / 2, params.nn_params, dnu_dt_k2.nn_params
            ),
        ),
    )
    dnu_dt_k4 = _get_dnu_dt(
        batch,
        loss,
        eqx.tree_at(
            lambda pt: pt.nn_params,
            params,
            jax.tree.map(
                lambda b, c: b + c * dt, params.nn_params, dnu_dt_k3.nn_params
            ),
        ),
    )

    return eqx.tree_at(
        lambda pt: pt.nn_params,
        params,
        jax.tree.map(
            lambda a, b, c, d, e: a
            + (1 / 6 * b + 1 / 3 * c + 1 / 3 * d + 1 / 6 * e) * dt,
            params.nn_params,
            dnu_dt_k1.nn_params,
            dnu_dt_k2.nn_params,
            dnu_dt_k3.nn_params,
            dnu_dt_k4.nn_params,
        ),
    )


def _get_dnu_dt(batch, loss, params):
    residuals, du_dnu = loss.values_and_grad_per_sample(params, batch)
    residuals = residuals.dyn_loss
    du_dnu = du_dnu.dyn_loss
    du_dnu = du_dnu[0].nn_params  # only keep gradients wrt to nn_params
    M, M_tmp = _process_du_dnu(du_dnu, batch.domain_batch.shape[0])

    L = _process_residuals(residuals, M_tmp)
    dnu_dt = jnp.linalg.solve(M + 1e-5 * jnp.eye(M.shape[0]), L)
    nn_params = _params_array_to_pytree(dnu_dt, params.nn_params)
    return eqx.tree_at(lambda pt: pt.nn_params, params, nn_params)


def _process_du_dnu(du_dnu, batch_size):
    r"""
    To construct M (as defined in Franck et al. 2025)

    $$
        M(\nu) = \int_Omega\nabla_\nu u_{\nu(t)}(x)\otimes\nabla_\nu u_{\nu(t)}(x)\mathrm{d}x
    $$
    """
    # params on the same last axis
    M_tmp = jax.tree.map(lambda l: l.reshape((batch_size, -1)), du_dnu)
    # array of params from pytre (with batch dim)
    M_tmp = jnp.concatenate(jax.tree.leaves(M_tmp), axis=1)

    # outer product of the param vector with itself for each coloc
    M = jax.vmap(lambda l_: jnp.outer(l_, l_))(M_tmp)

    # avg on coloc points (approximation of the integral)
    M = jnp.mean(M, axis=0)
    return M, M_tmp


def _process_residuals(residuals, M_tmp):
    """
    To construct L
    """

    # Process L
    L = residuals[0] * M_tmp
    # avg on coloc points (approximation of the integral)
    L = jnp.mean(L, axis=0)
    return L


def _params_array_to_pytree(dnu_dt, nn_params):
    """
    nn_params serves only for the structure
    """
    _, params_cumsum = _get_param_nb(nn_params)
    dnu_dt_flat = eqx.tree_at(
        jax.tree.leaves,
        nn_params,
        jnp.split(dnu_dt, params_cumsum[:-1]),
    )

    dnu_dt = jax.tree.map(
        lambda a, b: a.reshape(b.shape),
        dnu_dt_flat,
        nn_params,
        is_leaf=eqx.is_inexact_array,
    )
    return dnu_dt
