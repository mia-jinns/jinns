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
    d = 2
    r = 25
    eqx_list = (
        (eqx.nn.Linear, 1, 8),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 8, r),
    )
    key, subkey = random.split(key)
    key, subkey = random.split(key)
    u, init_nn_params = jinns.nn.SPINN_MLP.create(
        subkey, d, r, eqx_list, "PDENonStatio"
    )

    n = 25
    nb = 40
    ni = 40
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
        return norm.pdf(x, loc=mu_init, scale=sigma_init)[..., None]

    D = 1.0
    r = 4.0
    g = 3.0

    boundary_condition = jinns.loss.Dirichlet()

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
        boundary_condition=boundary_condition,
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
    params, total_loss_list, loss_by_term_dict, _, _, _, _, _, _, _, _, _ = jinns.solve(
        init_params=params, data=train_data, optimizer=tx, loss=loss, n_iter=n_iter
    )
    return total_loss_list[-1]


def test_initial_loss_Fisher(train_Fisher_init):
    init_params, loss, train_data = train_Fisher_init
    assert jnp.allclose(
        loss.evaluate(init_params, train_data.get_batch()[1])[0], 16.00568561, atol=1e-5
    )


def test_10it_Fisher(train_Fisher_10it):
    total_loss_val = train_Fisher_10it
    assert jnp.allclose(total_loss_val, 14.24510319, atol=1e-5)
