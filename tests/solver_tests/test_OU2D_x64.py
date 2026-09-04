import pytest

import jax
import jax.numpy as jnp
from jax import random
import equinox as eqx
import optax
from jax.scipy.stats import multivariate_normal
import jinns

from jinns.data._RARParameters import RARParameters
from jinns.loss._NormalizationSamples import NormalizationSamples


@pytest.fixture
def train_OU_init():
    jax.config.update("jax_enable_x64", True)
    key = random.PRNGKey(2)
    key, subkey = random.split(key)
    eqx_list = (
        (eqx.nn.Linear, 3, 3),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 3, 1),
        (jnp.exp,),
    )
    key, subkey = random.split(key)
    u, init_nn_params = jinns.nn.PINN_MLP.create(
        key=subkey, eqx_list=eqx_list, eq_type="PDENonStatio"
    )

    n = 25
    nb = None
    ni = 20
    xmin = -3
    xmax = 3
    ymin = -3
    ymax = 3
    tmin = 0
    tmax = 1
    method = "uniform"

    key, subkey = random.split(key)
    train_data = jinns.data.CubicMeshPDENonStatio(
        key=subkey,
        n=n,
        nb=nb,
        ni=ni,
        dim=2,
        min_pts=(xmin, ymin),
        max_pts=(xmax, ymax),
        tmin=tmin,
        tmax=tmax,
        method=method,
        rar_parameters=RARParameters(
            start_iter=0,
            update_every=10,
            novelty_proportion=0.05,
            method="D",
            k=2.0,
            c=0.0,
        ),
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

    n_samples = 10
    volume = (int_xmax - int_xmin) * (int_ymax - int_ymin)
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

    norm_samples = NormalizationSamples(
        samples=mc_samples,
        weights=volume,
        min_pts=(int_xmin, int_ymin),
        max_pts=(int_xmax, int_ymax),
        time_slices=100,
    )

    loss_weights = jinns.loss.LossWeightsPDENonStatio(
        dyn_loss=1.0,
        initial_condition=1 * Tmax,
        norm_loss=1 * Tmax,
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
            norm_samples=norm_samples,
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

    tx = optax.adamw(learning_rate=1e-3)
    n_iter = 10
    params, total_loss_list, loss_by_term_dict, _, _, _, _, _, _, _, _, _ = jinns.solve(
        init_params=params,
        data=train_data,
        optimizer=tx,
        loss=loss,
        n_iter=n_iter,
        key=random.PRNGKey(2),
    )
    return total_loss_list[-1]


def test_initial_loss_OU(train_OU_init):
    init_params, loss, train_data = train_OU_init
    _, batch = train_data.get_batch()
    l_init, _ = loss.evaluate(init_params, batch)
    assert jnp.allclose(l_init, 9453.18520921, atol=1e-5)


def test_10it_OU(train_OU_10it):
    total_loss_val = train_OU_10it
    assert jnp.allclose(total_loss_val, 9071.12968974, atol=1e-5)


@pytest.fixture(params=["time_subsample", "time_array"])
def train_OU_init_parametrized(request):
    jax.config.update("jax_enable_x64", True)
    key = random.PRNGKey(2)
    key, subkey = random.split(key)
    eqx_list = (
        (eqx.nn.Linear, 3, 3),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 3, 1),
        (jnp.exp,),
    )
    key, subkey = random.split(key)
    u, init_nn_params = jinns.nn.PINN_MLP.create(
        key=subkey, eqx_list=eqx_list, eq_type="PDENonStatio"
    )

    n = 25
    nb = None
    ni = 20
    xmin = -3
    xmax = 3
    ymin = -3
    ymax = 3
    tmin = 0
    tmax = 1
    method = "uniform"

    key, subkey = random.split(key)
    train_data = jinns.data.CubicMeshPDENonStatio(
        key=subkey,
        n=n,
        nb=nb,
        ni=ni,
        dim=2,
        min_pts=(xmin, ymin),
        max_pts=(xmax, ymax),
        tmin=tmin,
        tmax=tmax,
        method=method,
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

    n_samples = 10
    volume = (int_xmax - int_xmin) * (int_ymax - int_ymin)
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

    if request.param == "time_subsample":
        norm_samples = NormalizationSamples(
            samples=mc_samples,
            weights=volume,
            min_pts=(int_xmin, int_ymin),
            max_pts=(int_xmax, int_ymax),
            time_slices=100,
        )
    if request.param == "time_array":
        norm_samples = NormalizationSamples(
            samples=mc_samples,
            weights=volume,
            min_pts=(int_xmin, int_ymin),
            max_pts=(int_xmax, int_ymax),
            time_slices=train_data.domain[:100, 0:1],
        )
    else:
        raise ValueError

    loss_weights = jinns.loss.LossWeightsPDENonStatio(
        dyn_loss=1.0,
        initial_condition=1 * Tmax,
        norm_loss=1 * Tmax,
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
            norm_samples=norm_samples,
            params=init_params,
        )

    return init_params, loss, train_data


def test_initial_loss_OU_parametrized(train_OU_init_parametrized):
    init_params, loss, train_data = train_OU_init_parametrized
    _, batch = train_data.get_batch()
    l_init, _ = loss.evaluate(init_params, batch)
    assert jnp.allclose(l_init, 9453.18520921, atol=1e-5)
