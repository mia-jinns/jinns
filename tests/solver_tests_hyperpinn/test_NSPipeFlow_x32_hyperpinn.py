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

    L = 1
    R = 0.05

    p_out = 0
    p_in = 0.1

    n = 10000
    nb = None
    omega_batch_size = 128
    omega_border_batch_size = None
    dim = 2
    xmin = 0
    xmax = xmin + L
    ymin = -R
    ymax = ymin + 2 * R

    rho = 1.0

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

    method = "grid"
    key, subkey = random.split(key)
    np = 1000
    param_batch_size = 128  # must be equal to batch size of the main DataGenerator
    param_train_data = jinns.data.DataGeneratorParameter(
        subkey,
        np,
        param_batch_size,
        {"nu": (2e-4, 1.9e-3)},
        method,
    )

    eqx_list = (
        (eqx.nn.Linear, 2, 16),
        (jax.nn.swish,),
        (eqx.nn.Linear, 16, 16),
        (jax.nn.swish,),
        (eqx.nn.Linear, 16, 16),
        (jax.nn.swish,),
        (eqx.nn.Linear, 16, 16),
        (jax.nn.swish,),
        (eqx.nn.Linear, 16, 16),
        (jax.nn.swish,),
        (eqx.nn.Linear, 16, 16),
        (jax.nn.swish,),
        (eqx.nn.Linear, 16, 16),
        (jax.nn.swish,),
        (eqx.nn.Linear, 16, 3),
    )

    eqx_list_hyper = (
        (eqx.nn.Linear, 1, 32),  # input is of size 1 for scalar viscosity nu
        (jax.nn.tanh,),
        (eqx.nn.Linear, 32, 32),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 32, 32),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 32, 32),
        (jax.nn.tanh,),
        (
            eqx.nn.Linear,
            32,
            1000,
        ),  # 1000 is a random guess, it will automatically be filled with the correct value
    )

    def u_output_transform(pinn_in, pinn_out, _):
        u = pinn_out[:2] * (R**2 - pinn_in[1] ** 2)
        return u

    def p_output_transform(pinn_in, pinn_out, _):
        p = (
            (pinn_in[0] - xmin) / (xmax - xmin) * p_out
            + (xmax - pinn_in[0]) / (xmax - xmin) * p_in
            + (xmin - pinn_in[0]) * (xmax - pinn_in[0]) * pinn_out[2:3]
        )
        return p

    key, subkey = random.split(key)
    hyperparams = ["nu"]
    hypernet_input_size = 1
    u_hyper, u_init_nn_params = jinns.nn.HyperPINN.create(
        eq_type="statio_PDE",
        hyperparams=hyperparams,
        hypernet_input_size=hypernet_input_size,
        key=subkey,
        eqx_list=eqx_list,
        eqx_list_hyper=eqx_list_hyper,
        output_transform=u_output_transform,
        slice_solution=jnp.s_[:2],
    )
    p_hyper, _ = jinns.nn.HyperPINN.create(
        eq_type="statio_PDE",
        hyperparams=hyperparams,
        hypernet_input_size=hypernet_input_size,
        key=subkey,
        eqx_list=eqx_list,
        eqx_list_hyper=eqx_list_hyper,
        output_transform=p_output_transform,
        slice_solution=jnp.s_[2],
    )
    p_init_nn_params = u_init_nn_params
    param_train_data, param_batch = param_train_data.get_batch()
    init_params_hyper = jinns.parameters.ParamsDict(
        nn_params={"u": u_init_nn_params, "p": p_init_nn_params},
        eq_params={"rho": rho, **param_batch},
    )

    mc_loss = jinns.loss.MassConservation2DStatio(nn_key="u")
    ns_loss = jinns.loss.NavierStokes2DStatio(u_key="u", p_key="p")

    loss_weights = jinns.loss.LossWeightsPDEDict(dyn_loss=1.0)

    # Catching an expected UserWarning since no border condition is given
    # for this specific PDE (Fokker-Planck).
    with pytest.warns(UserWarning):
        loss_hyper = jinns.loss.SystemLossPDE(
            u_dict={"u": u_hyper, "p": p_hyper},
            loss_weights=loss_weights,
            dynamic_loss_dict={"mass_conservation": mc_loss, "navier_stokes": ns_loss},
            params_dict=init_params_hyper,
        )

    return init_params_hyper, loss_hyper, train_data, param_train_data


@pytest.fixture
def train_NSPipeFlow_10it(train_NSPipeFlow_init):
    """
    Fixture that requests a fixture
    """
    init_params, loss, train_data, param_train_data = train_NSPipeFlow_init

    params = init_params

    tx = optax.adamw(learning_rate=1e-4)
    n_iter = 10

    params, total_loss_list, loss_by_term_dict, _, _, _, _, _, _ = jinns.solve(
        init_params=params,
        data=train_data,
        param_data=param_train_data,
        optimizer=tx,
        loss=loss,
        n_iter=n_iter,
    )
    return total_loss_list[9]


def test_10it_NSPipeFlow(train_NSPipeFlow_10it):
    total_loss_val = train_NSPipeFlow_10it
    assert jnp.allclose(total_loss_val, 0.00970628, atol=1e-1)
