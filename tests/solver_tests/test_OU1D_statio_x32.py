import pytest

import jax
import jax.numpy as jnp
from jax import random
import equinox as eqx
import optax
from jax.scipy.stats import multivariate_normal
import jinns


@pytest.fixture
def train_OU_init():
    jax.config.update("jax_enable_x64", False)
    key = random.PRNGKey(12345)
    n = 1000
    omega_batch_size = 1000
    xmin = -3
    xmax = 3
    method = "uniform"

    key, subkey = random.split(key)
    train_data = jinns.data.CubicMeshPDEStatio(
        key=subkey,
        n=n,
        nb=None,
        omega_batch_size=omega_batch_size,
        dim=1,
        min_pts=(xmin,),
        max_pts=(xmax,),
        method=method,
    )

    key, subkey = random.split(key)
    eqx_list = (
        (eqx.nn.Linear, 1, 30),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 30, 30),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 30, 30),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 30, 30),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 30, 1),
        (jnp.exp,),  # force positivity of the PINN output
    )
    key, subkey = random.split(key)
    u, init_nn_params = jinns.nn.PINN_MLP.create(
        key=subkey, eqx_list=eqx_list, eq_type="statio_PDE"
    )
    sigma = jnp.array(0.5)
    alpha = jnp.array(6.0)
    mu = jnp.array(2.0)
    init_params = jinns.parameters.Params(
        nn_params=init_nn_params,
        eq_params={
            "mu": mu,
            "gamma": sigma / jnp.sqrt(alpha),  # free parameter for statio distrib
        },
    )

    good_mc_params = {"int_xmin": 1, "int_xmax": 3}
    volume = good_mc_params["int_xmax"] - good_mc_params["int_xmin"]
    good_mc_params["norm_weights"] = volume

    n_mc = 1000
    key, subkey = jax.random.split(key, 2)
    good_mc_samples = jax.random.uniform(
        subkey,
        shape=(n_mc, 1),
        minval=good_mc_params["int_xmin"],
        maxval=good_mc_params["int_xmax"],
    )
    loss_weights = jinns.loss.LossWeightsPDENonStatio(
        dyn_loss=1.0,
        norm_loss=1.0,
    )

    class OUStatio1DLoss(jinns.loss.PDEStatio):

        def equation(self, x, u, params):
            u_ = lambda x: u(x, params)
            return jax.grad(lambda x: ((x - params.eq_params["mu"]) * u_(x)).squeeze())(
                x
            ) + params.eq_params["gamma"] ** 2 / 2 * jax.grad(
                lambda x: jax.grad(lambda x: u_(x).squeeze())(x).squeeze()
            )(
                x
            )

    OU_statio_1D_loss = OUStatio1DLoss()

    # Catching an expected UserWarning since no border condition is given
    # for this specific PDE (Fokker-Planck).
    with pytest.warns(UserWarning):
        loss = jinns.loss.LossPDEStatio(
            u=u,
            loss_weights=loss_weights,
            dynamic_loss=OU_statio_1D_loss,
            norm_weights=good_mc_params["norm_weights"],
            norm_samples=good_mc_samples,
            params=init_params,
        )

    return init_params, loss, train_data


@pytest.fixture
def train_OU_10it(train_OU_init):
    """
    Fixture that requests a fixture
    """
    init_params, loss, train_data = train_OU_init

    params = init_params

    tx = optax.adamw(learning_rate=1e-4)
    n_iter = 10
    params, total_loss_list, loss_by_term_dict, _, _, _, _, _, _ = jinns.solve(
        init_params=params, data=train_data, optimizer=tx, loss=loss, n_iter=n_iter
    )
    return total_loss_list[-1]


def test_initial_loss_OU(train_OU_init):
    init_params, loss, train_data = train_OU_init
    _, batch = train_data.get_batch()
    l_init, _ = loss.evaluate(init_params, batch)
    assert jnp.allclose(l_init, 2.4449294, atol=1e-1)


def test_10it_OU(train_OU_10it):
    total_loss_val = train_OU_10it
    assert jnp.allclose(total_loss_val, 2.2827492, atol=1e-1)
