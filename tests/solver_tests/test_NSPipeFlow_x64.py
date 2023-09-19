import pytest

import jax
import jax.numpy as jnp
from jax import random
import equinox as eqx
import optax
from jaxopt import OptaxSolver
import jinns


@pytest.fixture
def train_NSPipeFlow_init():
    jax.config.update("jax_enable_x64", True)
    print(jax.config.FLAGS.jax_enable_x64)
    print(jax.devices())
    key = random.PRNGKey(2)

    eqx_list = [
        [eqx.nn.Linear, 2, 50],
        [jax.nn.tanh],
        [eqx.nn.Linear, 50, 50],
        [jax.nn.tanh],
        [eqx.nn.Linear, 50, 50],
        [jax.nn.tanh],
        [eqx.nn.Linear, 50, 2],
    ]
    key, subkey = random.split(key)
    u_init_param_fn, u_raw = jinns.utils.create_PINN(subkey, eqx_list, "statio_PDE", 2)

    eqx_list = [
        [eqx.nn.Linear, 2, 50],
        [jax.nn.tanh],
        [eqx.nn.Linear, 50, 50],
        [jax.nn.tanh],
        [eqx.nn.Linear, 50, 50],
        [jax.nn.tanh],
        [eqx.nn.Linear, 50, 1],
    ]
    key, subkey = random.split(key)
    p_init_param_fn, p_raw = jinns.utils.create_PINN(subkey, eqx_list, "statio_PDE", 2)

    u_init_nn_params = u_init_param_fn()
    p_init_nn_params = p_init_param_fn()

    L = 1
    R = 0.05

    p_out = 0
    p_in = 0.1

    n = 1000
    nb = None
    omega_batch_size = 32
    omega_border_batch_size = None
    dim = 2
    xmin = 0
    xmax = xmin + L
    ymin = -R
    ymax = ymin + 2 * R

    method = "uniform"
    key, subkey = random.split(key)
    train_data = jinns.data.CubicMeshPDEStatio(
        subkey,
        n,
        nb,
        omega_batch_size,
        omega_border_batch_size,
        dim,
        (xmin, ymin),
        (xmax, ymax),
        method,
    )

    u = lambda x, nn_params, eq_params: u_raw(x, nn_params, eq_params) * (
        R**2 - x[1] ** 2
    )  # multiplies the 2 componentse
    p = (
        lambda x, nn_params, eq_params: (x[0] - xmin) / (xmax - xmin) * p_out
        + (xmax - x[0]) / (xmax - xmin) * p_in
        + (xmin - x[0]) * (xmax - x[0]) * p_raw(x, nn_params, eq_params)
    )

    rho = 1.0
    nu = 0.00061

    Delta_p = 0.1
    d = 2 * R

    # initiate parameters dictionary
    init_params = {}
    init_params["nn_params"] = {"u": u_init_nn_params, "p": p_init_nn_params}
    init_params["eq_params"] = {"rho": rho, "nu": nu}

    mc_loss = jinns.loss.MassConservation2DStatio(nn_key="u")
    ns_loss = jinns.loss.NavierStokes2DStatio(u_key="u", p_key="p")

    loss_weights = {"dyn_loss": 1.0}

    loss = jinns.loss.SystemLossPDE(
        u_dict={"u": u, "p": p},
        loss_weights=loss_weights,
        dynamic_loss_dict={"mass_conservation": mc_loss, "navier_stokes": ns_loss},
        nn_type_dict={"u": "nn_statio", "p": "nn_statio"},
    )

    return init_params, loss, train_data


@pytest.fixture
def train_NSPipeFlow_10it(train_NSPipeFlow_init):
    """
    Fixture that requests a fixture
    """
    init_params, loss, train_data = train_NSPipeFlow_init

    # NOTE we need to waste one get_batch() here to stay synchronized with the
    # notebook
    _ = loss.evaluate(init_params, train_data.get_batch())[0]

    params = init_params

    tx = optax.adam(learning_rate=1e-4)
    solver = OptaxSolver(
        opt=tx,
        fun=loss,
        has_aux=True,  # because the objective has aux output
        maxiter=500000,
    )
    n_iter = 10
    pinn_solver = jinns.solver.PinnSolver(optax_solver=solver, loss=loss, n_iter=n_iter)
    params, total_loss_list, loss_by_term_dict, _, _, _, _ = pinn_solver.solve(
        init_params=params, data=train_data
    )
    return total_loss_list[9]


def test_initial_loss_NSPipeFlow(train_NSPipeFlow_init):
    init_params, loss, train_data = train_NSPipeFlow_init

    assert jnp.round(
        loss.evaluate(init_params, train_data.get_batch())[0], 5
    ) == jnp.round(0.005286878980768049, 5)


def test_10it_NSPipeFlow(train_NSPipeFlow_10it):
    total_loss_val = train_NSPipeFlow_10it
    assert jnp.round(total_loss_val, 5) == jnp.round(0.005041058138862686, 5)
