import pytest

import jax
import jax.numpy as jnp
from jax import random
import equinox as eqx
import optax
import jinns


@pytest.fixture
def train_NSPipeFlow_init():
    jax.config.update("jax_enable_x64", False)

    key = random.PRNGKey(2)

    d_ = 2
    r = 32
    m = 2
    eqx_list = [
        [eqx.nn.Linear, 1, 25],
        [jax.nn.tanh],
        [eqx.nn.Linear, 25, 25],
        [jax.nn.tanh],
        [eqx.nn.Linear, 25, 25],
        [jax.nn.tanh],
        [eqx.nn.Linear, 25, r * m],
    ]
    key, subkey = random.split(key)
    u = jinns.utils.create_SPINN(subkey, d_, r, eqx_list, "statio_PDE", m)

    d_ = 2
    r = 32
    eqx_list = [
        [eqx.nn.Linear, 1, 25],
        [jax.nn.tanh],
        [eqx.nn.Linear, 25, 25],
        [jax.nn.tanh],
        [eqx.nn.Linear, 25, 25],
        [jax.nn.tanh],
        [eqx.nn.Linear, 25, r],
    ]
    key, subkey = random.split(key)
    p = jinns.utils.create_SPINN(subkey, d_, r, eqx_list, "statio_PDE")

    u_init_nn_params = u.init_params()
    p_init_nn_params = p.init_params()

    L = 1
    R = 0.05

    p_out = 0
    p_in = 0.1

    n = 500
    nb = 500
    omega_batch_size = 32
    omega_border_batch_size = 32
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

    rho = 1.0
    nu = 0.00061

    Delta_p = 0.1
    d = 2 * R

    # initiate parameters dictionary
    init_params = {}
    init_params["nn_params"] = {"u": u_init_nn_params, "p": p_init_nn_params}
    init_params["eq_params"] = {"rho": rho, "nu": nu}

    p_omega_boundary_fun = {
        "xmin": lambda x: p_in,
        "xmax": lambda x: p_out,
        "ymin": None,
        "ymax": None,
    }
    p_omega_boundary_condition = {
        "xmin": "dirichlet",
        "xmax": "dirichlet",
        "ymin": None,
        "ymax": None,
    }

    u_omega_boundary_fun = {
        "xmin": None,
        "xmax": None,
        "ymin": lambda x: jnp.array([0.0, 0.0]),
        "ymax": lambda x: jnp.array([0.0, 0.0]),
    }
    u_omega_boundary_condition = {
        "xmin": None,
        "xmax": None,
        "ymin": "dirichlet",
        "ymax": "dirichlet",
    }
    mc_loss = jinns.loss.MassConservation2DStatio(nn_key="u")
    ns_loss = jinns.loss.NavierStokes2DStatio(u_key="u", p_key="p")
    loss_weights = {"dyn_loss": 1.0, "boundary_loss": 1.0}

    loss = jinns.loss.SystemLossPDE(
        u_dict={"u": u, "p": p},
        loss_weights=loss_weights,
        dynamic_loss_dict={"mass_conservation": mc_loss, "navier_stokes": ns_loss},
        nn_type_dict={"u": "nn_statio", "p": "nn_statio"},
        omega_boundary_fun_dict={"u": u_omega_boundary_fun, "p": p_omega_boundary_fun},
        omega_boundary_condition_dict={
            "u": u_omega_boundary_condition,
            "p": p_omega_boundary_condition,
        },
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

    tx = optax.adamw(learning_rate=1e-3)
    n_iter = 10
    params, total_loss_list, loss_by_term_dict, _, _, _, _ = jinns.solve(
        init_params=params, data=train_data, optimizer=tx, loss=loss, n_iter=n_iter
    )
    return total_loss_list[9]


def test_initial_loss_NSPipeFlow(train_NSPipeFlow_init):
    init_params, loss, train_data = train_NSPipeFlow_init

    assert jnp.round(
        loss.evaluate(init_params, train_data.get_batch())[0], 5
    ) == jnp.round(0.10145999, 5)


def test_10it_NSPipeFlow(train_NSPipeFlow_10it):
    total_loss_val = train_NSPipeFlow_10it
    assert jnp.round(total_loss_val, 5) == jnp.round(0.01431, 5)
