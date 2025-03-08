import pytest

import jax
import jax.numpy as jnp
from jax import random
import equinox as eqx
import optax
import jinns


@pytest.fixture
def train_Burgers_init():
    jax.config.update("jax_enable_x64", False)
    key = random.PRNGKey(2)
    eqx_list = (
        (eqx.nn.Linear, 2, 20),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 20, 20),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 20, 20),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 20, 1),
    )
    key, subkey = random.split(key)
    u, init_nn_params = jinns.nn.PINN_MLP.create(
        key=subkey, eqx_list=eqx_list, eq_type="nonstatio_PDE"
    )

    n = 1000
    ni = 1000
    omega_batch_size = 32
    initial_batch_size = 20
    dim = 1
    xmin = -1
    xmax = 1
    tmin = 0
    tmax = 1
    Tmax = 1
    method = "uniform"

    train_data = jinns.data.CubicMeshPDENonStatio(
        key=subkey,
        n=n,
        ni=ni,
        nb=2,
        omega_batch_size=omega_batch_size,
        initial_batch_size=initial_batch_size,
        dim=dim,
        min_pts=(xmin,),
        max_pts=(xmax,),
        tmin=tmin,
        tmax=tmax,
        method=method,
    )

    nu = 1 / (100 * jnp.pi)
    init_params = jinns.parameters.Params(
        nn_params=init_nn_params, eq_params={"nu": nu}
    )

    def u0(x):
        return -jnp.sin(jnp.pi * x)

    be_loss = jinns.loss.BurgersEquation(Tmax=Tmax)

    loss_weights = jinns.loss.LossWeightsPDENonStatio(
        dyn_loss=1, initial_condition=5, boundary_loss=1
    )

    loss = jinns.loss.LossPDENonStatio(
        u=u,
        loss_weights=loss_weights,
        dynamic_loss=be_loss,
        omega_boundary_fun=lambda t_dx: 0,
        omega_boundary_condition="dirichlet",
        initial_condition_fun=u0,
        params=init_params,
    )

    return init_params, loss, train_data


@pytest.fixture
def train_Burgers_10it(train_Burgers_init):
    """
    Fixture that requests a fixture
    """
    init_params, loss, train_data = train_Burgers_init
    params = init_params

    tracked_params = jinns.parameters.Params(nn_params=None, eq_params={"nu": True})

    tx = optax.adam(learning_rate=1e-3)
    n_iter = 10
    params, total_loss_list, loss_by_term_dict, _, _, _, tracked_params, _, _ = (
        jinns.solve(
            init_params=params,
            data=train_data,
            optimizer=tx,
            loss=loss,
            n_iter=n_iter,
            tracked_params=tracked_params,
        )
    )
    return tracked_params, params.eq_params["nu"]


def test_tracked_params_value(train_Burgers_10it):
    tracked_params, nu = train_Burgers_10it
    assert jnp.array_equal(tracked_params.eq_params["nu"], jnp.ones((10,)) * nu)
