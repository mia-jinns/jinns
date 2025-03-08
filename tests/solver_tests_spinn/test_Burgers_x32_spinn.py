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
    d = 2
    r = 256
    eqx_list = (
        (eqx.nn.Linear, 1, 128),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 128, 128),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 128, 128),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 128, r),
    )
    key, subkey = random.split(key)
    key, subkey = random.split(key)
    u, init_nn_params = jinns.nn.SPINN_MLP.create(
        subkey, d, r, eqx_list, "nonstatio_PDE"
    )

    n = 5000
    ni = 200
    nb = 200
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
        nb=nb,
        ni=ni,
        domain_batch_size=32,
        border_batch_size=32,
        initial_batch_size=32,
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
        dyn_loss=1, initial_condition=10, boundary_loss=1
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

    tx = optax.adamw(learning_rate=1e-4)
    n_iter = 10
    params, total_loss_list, loss_by_term_dict, _, _, _, _, _, _ = jinns.solve(
        init_params=params, data=train_data, optimizer=tx, loss=loss, n_iter=n_iter
    )
    return total_loss_list[-1]


def test_initial_loss_Burgers(train_Burgers_init):
    init_params, loss, train_data = train_Burgers_init

    assert jnp.allclose(
        loss.evaluate(init_params, train_data.get_batch()[1])[0], 5.701347, atol=1e-1
    )


def test_10it_Burgers(train_Burgers_10it):
    total_loss_val = train_Burgers_10it
    assert jnp.allclose(total_loss_val, 3.3629522, atol=1e-1)
