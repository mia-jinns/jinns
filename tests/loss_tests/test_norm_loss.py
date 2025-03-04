import pytest

import jax
import jax.numpy as jnp
from jax import random
import equinox as eqx
from jax.scipy.stats import multivariate_normal
import jinns
import jinns.data
import jinns.loss


@pytest.fixture
def train_OU_init():
    key = random.PRNGKey(2)
    key, subkey = random.split(key)
    eqx_list = [
        [eqx.nn.Linear, 3, 5],
        [
            jax.nn.tanh,
        ],
        [eqx.nn.Linear, 5, 5],
        [
            jnp.exp,
        ],
    ]
    key, subkey = random.split(key)
    u, init_nn_params = jinns.nn.PINN_MLP.create(
        key=subkey, eqx_list=eqx_list, eq_type="nonstatio_PDE"
    )

    Tmax = 5
    sigma = 0.5 * jnp.ones((2))
    alpha = 0.5 * jnp.ones((2))
    mu = jnp.zeros((2))

    init_params = jinns.parameters.Params(
        nn_params=init_nn_params,
        eq_params={"sigma": sigma, "alpha": alpha, "mu": mu},
    )

    def u0(x):
        return multivariate_normal.pdf(x, mean=jnp.array([1, 1]), cov=0.1 * jnp.eye(2))

    int_xmin, int_xmax = -3, 3
    int_ymin, int_ymax = -3, 3

    n_mc_samples = int(13)
    volume = (int_xmax - int_xmin) * (int_ymax - int_ymin)
    key, subkey1, subkey2 = random.split(key, 3)
    mc_samples = jnp.concatenate(
        [
            random.uniform(
                subkey1, shape=(n_mc_samples, 1), minval=int_xmin, maxval=int_xmax
            ),
            random.uniform(
                subkey2, shape=(n_mc_samples, 1), minval=int_ymin, maxval=int_ymax
            ),
        ],
        axis=-1,
    )

    loss_weights = jinns.loss.LossWeightsPDENonStatio(
        dyn_loss=1.0,
        initial_condition=1 * Tmax,
        norm_loss=1 * Tmax,
    )
    dynamic_loss = jinns.loss.OU_FPENonStatioLoss2D(Tmax=Tmax)

    return u, init_params, loss_weights, dynamic_loss, u0, mc_samples, volume


def test_unidimensionality(train_OU_init):
    u, init_params, loss_weights, dynamic_loss, u0, mc_samples, volume = train_OU_init
    loss = jinns.loss.LossPDENonStatio(
        u=u,
        loss_weights=loss_weights,
        dynamic_loss=dynamic_loss,
        initial_condition_fun=u0,
        norm_weights=volume,
        norm_samples=mc_samples,
        params=init_params,
    )
    nb = 3
    batch = jinns.data.PDENonStatioBatch(
        domain_batch=jnp.zeros((nb, 3)),
        border_batch=jnp.ones((nb, 2, 2)),
        initial_batch=jnp.ones((nb, 2)),
    )
    with pytest.raises(AssertionError) as assErr:
        loss.evaluate(init_params, batch)
    assert str(assErr.value) == "norm loss expects unidimensional *PINN"
