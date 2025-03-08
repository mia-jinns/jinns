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

    eqx_list = (
        (eqx.nn.Linear, 2, 50),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 50, 50),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 50, 50),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 50, 2),
    )
    key, subkey = random.split(key)
    u_output_transform = lambda pinn_in, pinn_out, params: pinn_out * (
        R**2 - pinn_in[1] ** 2
    )
    # This output transform is equivalent to defining afterwards:
    # u = lambda x, nn_params, eq_params: u_raw(x, nn_params, eq_params) * (
    #    R**2 - x(1) ** 2
    # )  # multiplies the 2 components
    u, u_init_nn_params = jinns.nn.PINN_MLP.create(
        key=subkey,
        eqx_list=eqx_list,
        eq_type="statio_PDE",
        output_transform=u_output_transform,
    )

    eqx_list = (
        (eqx.nn.Linear, 2, 50),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 50, 50),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 50, 50),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 50, 1),
    )
    key, subkey = random.split(key)
    p_output_transform = lambda pinn_in, pinn_out, params: (
        (pinn_in[0] - xmin) / (xmax - xmin) * p_out
        + (xmax - pinn_in[0]) / (xmax - xmin) * p_in
        + (xmin - pinn_in[0]) * (xmax - pinn_in[0]) * pinn_out
    )
    # This output transform is equivalent to defining afterwards:
    p, p_init_nn_params = jinns.nn.PINN_MLP.create(
        key=subkey,
        eqx_list=eqx_list,
        eq_type="statio_PDE",
        output_transform=p_output_transform,
    )

    method = "uniform"
    key, subkey = random.split(key)
    train_data = jinns.data.CubicMeshPDEStatio(
        key=subkey,
        n=n,
        nb=nb,
        omega_batch_size=omega_batch_size,
        omega_border_batch_size=omega_border_batch_size,
        dim=dim,
        min_pts=(xmin, ymin),
        max_pts=(xmax, ymax),
        method=method,
    )

    rho = 1.0
    nu = 0.00061

    Delta_p = 0.1
    d = 2 * R

    # initiate parameters dictionary
    init_params = jinns.parameters.ParamsDict(
        nn_params={"u": u_init_nn_params, "p": p_init_nn_params},
        eq_params={"rho": rho, "nu": nu},
    )

    mc_loss = jinns.loss.MassConservation2DStatio(nn_key="u")
    ns_loss = jinns.loss.NavierStokes2DStatio(u_key="u", p_key="p")

    loss_weights = jinns.loss.LossWeightsPDEDict(dyn_loss=1.0)

    # Catching an expected UserWarning since no border condition is given
    # for this specific PDE (Fokker-Planck).
    with pytest.warns(UserWarning):
        loss = jinns.loss.SystemLossPDE(
            u_dict={"u": u, "p": p},
            loss_weights=loss_weights,
            dynamic_loss_dict={"mass_conservation": mc_loss, "navier_stokes": ns_loss},
            params_dict=init_params,
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

    _, batch = train_data.get_batch()
    _ = loss.evaluate(init_params, batch)[0]

    params = init_params

    tx = optax.adam(learning_rate=1e-4)
    n_iter = 10

    params, total_loss_list, loss_by_term_dict, _, _, _, _, _, _ = jinns.solve(
        init_params=params, data=train_data, optimizer=tx, loss=loss, n_iter=n_iter
    )
    return total_loss_list[9]


def test_initial_loss_NSPipeFlow(train_NSPipeFlow_init):
    init_params, loss, train_data = train_NSPipeFlow_init

    _, batch = train_data.get_batch()
    assert jnp.allclose(loss.evaluate(init_params, batch)[0], 0.01134, atol=1e-1)


def test_10it_NSPipeFlow(train_NSPipeFlow_10it):
    total_loss_val = train_NSPipeFlow_10it
    assert jnp.allclose(total_loss_val, 0.01133, atol=1e-1)
