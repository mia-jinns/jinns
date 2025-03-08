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

    n = 2500
    nb = 500
    ni = 500
    domain_batch_size = 32
    initial_batch_size = 32
    border_batch_size = 32
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
        domain_batch_size=domain_batch_size,
        border_batch_size=border_batch_size,
        initial_batch_size=initial_batch_size,
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
        return jnp.squeeze(norm.pdf(x, loc=mu_init, scale=sigma_init))

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
        dyn_loss=1, initial_condition=1, boundary_loss=1
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

    tx = optax.adam(learning_rate=1e-4)
    n_iter = 10
    params, total_loss_list, loss_by_term_dict, _, _, _, _, _, _ = jinns.solve(
        init_params=params, data=train_data, optimizer=tx, loss=loss, n_iter=n_iter
    )
    return total_loss_list[-1]


def test_initial_loss_Fisher(train_Fisher_init):
    init_params, loss, train_data = train_Fisher_init
    assert jnp.allclose(
        loss.evaluate(init_params, train_data.get_batch()[1])[0], 9.163609, atol=1e-1
    )


def test_10it_Fisher(train_Fisher_10it):
    total_loss_val = train_Fisher_10it
    assert jnp.allclose(total_loss_val, 0.93934155, atol=1e-1)
