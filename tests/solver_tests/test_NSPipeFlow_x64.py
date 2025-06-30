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

    key, subkey = random.split(key)

    def u_p_output_transform(pinn_in, pinn_out, params):
        return jnp.concatenate(
            [
                pinn_out[0:2] * (R**2 - pinn_in[1] ** 2),
                (pinn_in[0] - xmin) / (xmax - xmin) * p_out
                + (xmax - pinn_in[0]) / (xmax - xmin) * p_in
                + (xmin - pinn_in[0]) * (xmax - pinn_in[0]) * pinn_out[2:3],
            ],
            axis=-1,
        )

    u_p, u_p_init_nn_params = jinns.nn.PPINN_MLP.create(
        key=subkey,
        eqx_list_list=[
            (
                (eqx.nn.Linear, 2, 50),
                (jax.nn.tanh,),
                (eqx.nn.Linear, 50, 50),
                (jax.nn.tanh,),
                (eqx.nn.Linear, 50, 50),
                (jax.nn.tanh,),
                (eqx.nn.Linear, 50, 2),
            ),
            (
                (eqx.nn.Linear, 2, 50),
                (jax.nn.tanh,),
                (eqx.nn.Linear, 50, 50),
                (jax.nn.tanh,),
                (eqx.nn.Linear, 50, 50),
                (jax.nn.tanh,),
                (eqx.nn.Linear, 50, 1),
            ),
        ],
        eq_type="statio_PDE",
        output_transform=u_p_output_transform,
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
    init_params = jinns.parameters.Params(
        nn_params=u_p_init_nn_params,
        eq_params={"rho": rho, "nu": nu},
    )

    dyn_loss = jinns.loss.NavierStokesMassConservation2DStatio()
    loss_weights = jinns.loss.LossWeightsPDEStatio(dyn_loss=1.0)

    # Catching an expected UserWarning since no border condition is given
    # for this specific PDE (Fokker-Planck).
    with pytest.warns(UserWarning):
        loss = jinns.loss.LossPDEStatio(
            u=u_p,
            loss_weights=loss_weights,
            dynamic_loss=dyn_loss,
            obs_slice=jnp.s_[0:1],  # we only observe the first slice
            params=init_params,
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

    params, total_loss_list, loss_by_term_dict, _, _, _, _, _, _, _ = jinns.solve(
        init_params=params, data=train_data, optimizer=tx, loss=loss, n_iter=n_iter
    )
    return total_loss_list[9]


def test_initial_loss_NSPipeFlow(train_NSPipeFlow_init):
    init_params, loss, train_data = train_NSPipeFlow_init

    assert jnp.allclose(
        loss.evaluate(init_params, train_data.get_batch()[1])[0], 0.01055, atol=1e-1
    )


def test_10it_NSPipeFlow(train_NSPipeFlow_10it):
    total_loss_val = train_NSPipeFlow_10it
    assert jnp.allclose(total_loss_val, 0.01061, atol=1e-1)
