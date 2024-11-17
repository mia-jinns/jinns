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
    u = jinns.utils.create_PINN(subkey, eqx_list, "nonstatio_PDE", 2)
    rar_parameters = {
        "start_iter": 1000,  # the gradient step at which RAR algo starts (enables a burn in period)
        "update_every": 500,  # nb of gradient steps between two RAR procedures
        "sample_size_times": 22,  # the number of new candidates time points
        "selected_sample_size_times": 5,  # the number of selected times collocation points from the sample, to join the dataset.
        "sample_size_omega": 22,  # the number of new candidates space points
        "selected_sample_size_omega": 5,
    }
    n_start = 500  # the initial number of spatial collocation points at beginning
    nt_start = 500  # the initial number of temporal collocation points at beginning
    init_nn_params = u.init_params()

    n = 1000
    nb = 4  # not used here
    nt = 999  # can be != n
    omega_batch_size = 32
    omega_border_batch_size = None  # not used here
    temporal_batch_size = 32
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
        nt=nt,
        omega_batch_size=omega_batch_size,
        omega_border_batch_size=omega_border_batch_size,
        temporal_batch_size=temporal_batch_size,
        dim=2,
        min_pts=(xmin, ymin),
        max_pts=(xmax, ymax),
        tmin=tmin,
        tmax=tmax,
        method=method,
        rar_parameters=rar_parameters,
        n_start=n_start,
        nt_start=nt_start,
    )

    # the next line is to be able to use the the same test values as the legacy
    # DataGenerators. We need to align the object parameters because their
    # respective init is not the same
    train_data = eqx.tree_at(
        lambda m: (m.curr_omega_idx, m.curr_time_idx, m.omega, m.times),
        train_data,
        (
            0,
            0,
            random.choice(
                jnp.array([1200475246, 4276610528], dtype=jnp.uint32),
                train_data.omega,
                shape=(train_data.omega.shape[0],),
                replace=False,
                p=train_data.p_omega,
            ),
            random.choice(
                jnp.array([1112151499, 1480465624], dtype=jnp.uint32),
                train_data.times,
                shape=(train_data.times.shape[0],),
                replace=False,
                p=train_data.p_times,
            ),
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

    int_xmin, int_xmax = -5, 5
    int_ymin, int_ymax = -5, 5

    n_samples = int(1e3)
    int_length = (int_xmax - int_xmin) * (int_ymax - int_ymin)
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
        norm_loss=0.1 * Tmax,
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
            norm_int_length=int_length,
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

    # NOTE we need to waste one get_batch() here to stay synchronized with the
    # notebook
    train_data, _ = train_data.get_batch()

    params = init_params

    tx = optax.adamw(learning_rate=1e-3)
    n_iter = 10
    params, total_loss_list, loss_by_term_dict, _, _, _, _, _, _ = jinns.solve(
        init_params=params, data=train_data, optimizer=tx, loss=loss, n_iter=n_iter
    )
    return total_loss_list[9]


def test_initial_loss_OU(train_OU_init):
    init_params, loss, train_data = train_OU_init
    _, batch = train_data.get_batch()
    l_init, all = loss.evaluate(init_params, batch)
    assert jnp.allclose(l_init, 3924.7366, atol=1e-1)


def test_10it_OU(train_OU_10it):
    total_loss_val = train_OU_10it
    assert jnp.allclose(total_loss_val, 2564.442, atol=1e-1)
