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
    key = random.PRNGKey(2)
    d = 3
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
    u, init_nn_params = jinns.nn.SPINN_MLP.create(
        subkey, d, r, eqx_list, "nonstatio_PDE"
    )

    n = 5000
    ni = 196
    nb = None
    domain_batch_size = 32
    initial_batch_size = 32
    dim = 2
    xmin = -3
    xmax = 3
    ymin = -3
    ymax = 3
    tmin = 0
    tmax = 1
    Tmax = 5
    method = "uniform"

    key, subkey = random.split(key)
    train_data = jinns.data.CubicMeshPDENonStatio(
        key=subkey,
        n=n,
        nb=nb,
        ni=ni,
        domain_batch_size=domain_batch_size,
        initial_batch_size=initial_batch_size,
        dim=2,
        min_pts=(xmin, ymin),
        max_pts=(xmax, ymax),
        tmin=tmin,
        tmax=tmax,
        method=method,
    )

    sigma = 0.5 * jnp.ones((2))
    alpha = 0.5 * jnp.ones((2))
    mu = jnp.zeros((2))

    init_params = jinns.parameters.Params(
        nn_params=init_nn_params,
        eq_params={"sigma": sigma, "alpha": alpha, "mu": mu},
    )

    def u0(x):
        # TODO: find a reshape of init_func to prevent UserWarning from jinns
        return multivariate_normal.pdf(x, mean=jnp.array([1, 1]), cov=0.1 * jnp.eye(2))

    int_xmin, int_xmax = -3, 3
    int_ymin, int_ymax = -3, 3

    n_samples = 32
    volume = (int_xmax - int_xmin) * (int_ymax - int_ymin)
    norm_weights = volume
    key, subkey1, subkey2 = random.split(key, 3)
    mc_samples = jnp.concatenate(
        [
            random.uniform(
                subkey1, shape=(n_samples, 1), minval=int_xmin, maxval=int_xmax
            ),
            random.uniform(
                subkey2, shape=(n_samples, 1), minval=int_ymin, maxval=int_ymax
            ),
        ],
        axis=-1,
    )

    loss_weights = jinns.loss.LossWeightsPDENonStatio(
        dyn_loss=1, initial_condition=1 * Tmax, norm_loss=1 * Tmax
    )
    OU_fpe_non_statio_2D_loss = jinns.loss.OU_FPENonStatioLoss2D(Tmax=Tmax)

    # Catching an expected UserWarning since no border condition is given
    # for this specific PDE (Fokker-Planck).
    with pytest.warns(UserWarning):
        loss = jinns.loss.LossPDENonStatio(
            u=u,
            loss_weights=loss_weights,
            dynamic_loss=OU_fpe_non_statio_2D_loss,
            initial_condition_fun=u0,
            norm_weights=norm_weights,
            norm_samples=mc_samples,
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

    return total_loss_list[9]


def test_initial_loss_OU(train_OU_init):
    init_params, loss, train_data = train_OU_init
    assert jnp.allclose(
        loss.evaluate(init_params, train_data.get_batch()[1])[0], 8.43018, atol=1e-1
    )


def test_10it_OU(train_OU_10it):
    total_loss_val = train_OU_10it
    assert jnp.allclose(total_loss_val, 0.20094508, atol=1e-1)
