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
    key, subkey = random.split(key)
    eqx_list = [
        [eqx.nn.Linear, 3, 30],
        [
            jax.nn.tanh,
        ],
        [eqx.nn.Linear, 30, 30],
        [
            jax.nn.tanh,
        ],
        [eqx.nn.Linear, 30, 30],
        [
            jax.nn.tanh,
        ],
        [eqx.nn.Linear, 30, 1],
        [
            jnp.exp,
        ],
    ]
    key, subkey = random.split(key)
    u, init_nn_params = jinns.nn.PINN_MLP.create(
        key=subkey, eqx_list=eqx_list, eq_type="nonstatio_PDE"
    )
    rar_parameters = {
        "start_iter": 1000,  # the gradient step at which RAR algo starts (enables a burn in period)
        "update_every": 500,  # nb of gradient steps between two RAR procedures
        "sample_size": 22,  # the number of new candidates time points
        "selected_sample_size": 5,
    }
    n_start = 500  # the initial number of spatial collocation points at beginning

    n = 2500
    nb = None
    ni = 196
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
        rar_parameters=rar_parameters,
        n_start=n_start,
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

    n_samples = int(1e3)
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

    tx = optax.adamw(learning_rate=1e-3)
    n_iter = 10
    params, total_loss_list, loss_by_term_dict, _, _, _, _, _, _ = jinns.solve(
        init_params=params, data=train_data, optimizer=tx, loss=loss, n_iter=n_iter
    )
    return total_loss_list[-1]


def test_initial_loss_OU(train_OU_init):
    init_params, loss, train_data = train_OU_init
    _, batch = train_data.get_batch()
    l_init, _ = loss.evaluate(init_params, batch)
    assert jnp.allclose(l_init, 9044.623, atol=1e-1)


def test_10it_OU(train_OU_10it):
    total_loss_val = train_OU_10it
    assert jnp.allclose(total_loss_val, 5730.3555, atol=1e-1)
