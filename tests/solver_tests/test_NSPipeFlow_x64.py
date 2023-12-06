import pytest

import jax
import jax.numpy as jnp
from jax import random
import equinox as eqx
import optax
import jinns


@pytest.fixture
def train_NSPipeFlow_init():
    jax.config.update("jax_enable_x64", True)

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
    u_output_transform = lambda pinn_in, pinn_out: pinn_out * (R**2 - pinn_in[1] ** 2)
    # This output transform is equivalent to defining afterwards:
    # u = lambda x, nn_params, eq_params: u_raw(x, nn_params, eq_params) * (
    #    R**2 - x[1] ** 2
    # )  # multiplies the 2 components
    u = jinns.utils.create_PINN(
        subkey, eqx_list, "statio_PDE", 2, output_transform=u_output_transform
    )

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
    p_output_transform = lambda pinn_in, pinn_out: (
        (pinn_in[0] - xmin) / (xmax - xmin) * p_out
        + (xmax - pinn_in[0]) / (xmax - xmin) * p_in
        + (xmin - pinn_in[0]) * (xmax - pinn_in[0]) * pinn_out
    )
    # This output transform is equivalent to defining afterwards:
    p = jinns.utils.create_PINN(
        subkey, eqx_list, "statio_PDE", 2, output_transform=p_output_transform
    )

    u_init_nn_params = u.init_params()
    p_init_nn_params = p.init_params()

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
    n_iter = 10
    params, total_loss_list, loss_by_term_dict, _, _, _, _ = jinns.solve(
        init_params=params, data=train_data, optimizer=tx, loss=loss, n_iter=n_iter
    )
    return total_loss_list[9]


def test_initial_loss_NSPipeFlow(train_NSPipeFlow_init):
    init_params, loss, train_data = train_NSPipeFlow_init

    assert jnp.round(
        loss.evaluate(init_params, train_data.get_batch())[0], 5
    ) == jnp.round(0.01055, 5)


def test_10it_NSPipeFlow(train_NSPipeFlow_10it):
    total_loss_val = train_NSPipeFlow_10it
    assert jnp.round(total_loss_val, 5) == jnp.round(0.01008, 5)
