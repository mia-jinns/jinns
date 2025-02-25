import pytest

import jax
import jax.numpy as jnp
from jax import random
from jax.scipy.stats import norm
import equinox as eqx
import optax
import jinns


@pytest.fixture
def train_Fisher_init():
    jax.config.update("jax_enable_x64", True)
    key = random.PRNGKey(2)
    eqx_list = (
        (eqx.nn.Linear, 2, 50),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 50, 50),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 50, 50),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 50, 50),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 50, 50),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 50, 1),
        (jnp.exp,),
    )
    key, subkey = random.split(key)
    u, init_nn_params = jinns.nn.PINN_MLP.create(
        key=subkey, eqx_list=eqx_list, eq_type="nonstatio_PDE"
    )

    n = 2500
    nb = 500
    ni = 500
    dim = 1
    xmin = -1
    xmax = 1
    tmin = 0
    tmax = 1
    method = "uniform"

    Tmax = 5
    key, subkey = random.split(key)
    key, subkey = random.split(key)
    train_data = jinns.data.CubicMeshPDENonStatio(
        key=subkey,
        n=n,
        nb=nb,
        ni=ni,
        dim=dim,
        min_pts=(xmin,),
        max_pts=(xmax,),
        tmin=tmin,
        tmax=tmax,
        method=method,
    )

    sigma_init = 0.2 * jnp.ones((1))
    mu_init = 0 * jnp.ones((1))

    def u0(x):
        return norm.pdf(x, loc=mu_init, scale=sigma_init)[0]  # output a scalar

    D = 1.0
    r = 4.0
    g = 3.0

    boundary_condition = "dirichlet"

    omega_boundary_fun = lambda t_dx: 0  # cte func returning 0

    init_params = jinns.parameters.Params(
        nn_params=init_nn_params,
        eq_params={"D": jnp.array([D]), "r": jnp.array([r]), "g": jnp.array([g])},
    )

    fisher_dynamic_loss = jinns.loss.FisherKPP(Tmax=Tmax)

    loss_weights = jinns.loss.LossWeightsPDENonStatio(
        dyn_loss=1, initial_condition=1 * Tmax, boundary_loss=3 / 4 * Tmax
    )

    loss = jinns.loss.LossPDENonStatio(
        u=u,
        loss_weights=loss_weights,
        dynamic_loss=fisher_dynamic_loss,
        omega_boundary_fun=omega_boundary_fun,
        omega_boundary_condition=boundary_condition,
        initial_condition_fun=u0,
        params=init_params,
    )

    return init_params, loss, train_data


@pytest.fixture
def train_Fisher_10it(train_Fisher_init):
    """
    Fixture that requests a fixture
    """
    init_params, loss, train_data = train_Fisher_init

    params = init_params

    tx = optax.adamw(learning_rate=1e-4)
    n_iter = 10
    params, total_loss_list, loss_by_term_dict, _, _, _, _, _, _ = jinns.solve(
        init_params=params, data=train_data, optimizer=tx, loss=loss, n_iter=n_iter
    )
    return total_loss_list[9]


def test_initial_loss_Fisher(train_Fisher_init):
    init_params, loss, train_data = train_Fisher_init
    train_data, batch = train_data.get_batch()
    assert jnp.allclose(loss.evaluate(init_params, batch)[0], 42.47814883, atol=1e-1)


def test_10it_Fisher(train_Fisher_10it):
    total_loss_val = train_Fisher_10it
    assert jnp.allclose(total_loss_val, 38.52351677, atol=1e-1)
