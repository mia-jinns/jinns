import pytest

import jax
import jax.numpy as jnp
from jax import random
from jax.scipy.stats import norm
import equinox as eqx
import optax
import jinns


@pytest.fixture
def train_ReacDiff_init():
    jax.config.update("jax_enable_x64", True)
    key = random.PRNGKey(2)
    d = 3
    r = 256
    eqx_list = [
        [eqx.nn.Linear, 1, 128],
        [jax.nn.tanh],
        [eqx.nn.Linear, 128, 128],
        [jax.nn.tanh],
        [eqx.nn.Linear, 128, 128],
        [jax.nn.tanh],
        [eqx.nn.Linear, 128, r],
    ]
    key, subkey = random.split(key)
    u_spinn = jinns.utils.create_SPINN(subkey, d, r, eqx_list, "nonstatio_PDE")

    init_nn_params_spinn = u_spinn.init_params()

    n = 600
    nb = 600
    nt = 600
    omega_batch_size = 32
    temporal_batch_size = 32
    omega_border_batch_size = 32
    dim = 2
    xmin = 0
    xmax = 1
    ymin = 0
    ymax = 1
    tmin = 0
    tmax = 1
    method = "uniform"

    Tmax = 2
    key, subkey = random.split(key)
    train_data = jinns.data.CubicMeshPDENonStatio(
        subkey,
        n,
        nb,
        nt,
        omega_batch_size,
        omega_border_batch_size,
        temporal_batch_size,
        dim,
        (xmin, ymin),
        (xmax, ymax),
        tmin,
        tmax,
        method,
    )

    from jax.scipy.stats import norm

    sigma_init = 1 * jnp.ones((2))
    mu_init = jnp.array([0.7, 0.15])

    def u0(x):
        return jnp.exp(-jnp.linalg.norm(x - mu_init, axis=-1))

    D = 0.05
    r1, r2, r3 = 0.0, -4.0, 2.0
    g = 1.0
    l = xmax - xmin

    init_params = {
        "nn_params": init_nn_params_spinn,
        "eq_params": {
            "D": jnp.array([D]),
            "r": jnp.array([r1, r2, r3]),
            "g": jnp.array([g]),
        },
    }

    from jinns.utils._utils import _get_grid

    def r_fun(t, x, eq_params):
        """must be a jittable function"""
        r1, r2, r3 = eq_params["r"]

        # Next is required to be able to call plot_2D and see the map
        if x.shape[0] != x.shape[1]:
            x = _get_grid(x.squeeze())

        # By default put r1 everywhere
        r_map_batch = jnp.full(x.shape[:2], r1)
        # But if the next cond is True, update to r2
        r_map_batch = jnp.where(
            jnp.logical_or(
                jnp.logical_and(x[..., 0] > 6 / 20, x[..., 0] < 8 / 20),
                jnp.logical_and(x[..., 1] > 8 / 20, x[..., 1] < 10 / 20),
            ),
            r2,
            r_map_batch,
        )
        # Again if the next cond is True, update to r3
        r_map_batch = jnp.where(
            jnp.logical_or(
                (x[..., 0] - 0.15) ** 2 + (x[..., 1] - 0.15) ** 2 < 0.015,
                (x[..., 0] - 0.8) ** 2 + (x[..., 1] - 0.80) ** 2 < 0.03,
            ),
            r3,
            r_map_batch,
        )
        return r_map_batch

    fisher_dynamic_loss = jinns.loss.FisherKPP(
        Tmax=Tmax, eq_params_heterogeneity={"D": None, "r": r_fun, "g": None}
    )
    loss_weights = {
        "dyn_loss": 1,
        "initial_condition": 10 * Tmax,
        "boundary_loss": 1 * Tmax,
    }

    loss = jinns.loss.LossPDENonStatio(
        u=u_spinn,
        loss_weights=loss_weights,
        dynamic_loss=fisher_dynamic_loss,
        omega_boundary_fun=lambda t, dx: 0,
        omega_boundary_condition="neumann",
        initial_condition_fun=u0,
        norm_key=None,
        norm_borders=None,
        norm_samples=None,
    )

    return init_params, loss, train_data


@pytest.fixture
def train_ReacDiff_10it(train_ReacDiff_init):
    """
    Fixture that requests a fixture
    """
    init_params, loss, train_data = train_ReacDiff_init

    # NOTE we need to waste one get_batch() here to stay synchronized with the
    # notebook
    _ = loss.evaluate(init_params, train_data.get_batch())[0]

    params = init_params

    tx = optax.adamw(learning_rate=1e-3)
    n_iter = 10
    params, total_loss_list, loss_by_term_dict, _, _, _, _ = jinns.solve(
        init_params=params, data=train_data, optimizer=tx, loss=loss, n_iter=n_iter
    )
    return total_loss_list[9]


def test_initial_loss_ReacDiff(train_ReacDiff_init):
    init_params, loss, train_data = train_ReacDiff_init
    assert jnp.round(
        loss.evaluate(init_params, train_data.get_batch())[0], 5
    ) == jnp.round(8.77175, 5)


def test_10it_ReacDiff(train_ReacDiff_10it):
    total_loss_val = train_ReacDiff_10it
    assert jnp.round(total_loss_val, 5) == jnp.round(1.84888, 5)
