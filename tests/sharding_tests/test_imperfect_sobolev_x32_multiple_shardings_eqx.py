import pytest

import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count=2"

import jax
import jax.numpy as jnp
from jax import random
import equinox as eqx
import optax
import jinns


@pytest.fixture
def train_imperfect_sobolev_init_sharding():
    jax.config.update("jax_enable_x64", False)

    # We have forced CPU computations and 2 fake CPU devices
    # in order to test the sharding
    cpu1, cpu2 = jax.devices("cpu")

    # Default device is cpu1
    jax.config.update("jax_default_device", cpu1)

    # Playing with obs_data on cpu2
    cpu2_sharding = jax.sharding.SingleDeviceSharding(cpu2)

    key = random.PRNGKey(2)

    # create obs
    def u_star(t, x):
        return jnp.exp(t - x) + 0.1 * jnp.cos(2 * jnp.pi * x)

    n = 2000
    key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)
    t_obs = jax.random.uniform(subkey1, (n,), minval=0.001, maxval=0.5)
    x_obs = jax.random.uniform(subkey2, (n,), minval=0.001, maxval=1.0)
    obs = u_star(t_obs, x_obs) + jax.random.normal(subkey3, (n,)) * jnp.sqrt(1e-2)

    eqx_list = [
        [eqx.nn.Linear, 2, 50],
        [jax.nn.tanh],
        [eqx.nn.Linear, 50, 50],
        [jax.nn.tanh],
        [eqx.nn.Linear, 50, 50],
        [jax.nn.tanh],
        [eqx.nn.Linear, 50, 50],
        [jax.nn.tanh],
        [eqx.nn.Linear, 50, 1],
    ]
    key, subkey = random.split(key)
    u = jinns.utils.create_PINN(subkey, eqx_list, "nonstatio_PDE", 1)

    init_nn_params = u.init_params()

    n = 500
    nt = 500
    nb = 2
    omega_batch_size = 32
    temporal_batch_size = 32
    omega_border_batch_size = 1
    dim = 1
    xmin = 0.001
    xmax = 1
    tmin = 0
    tmax = 1
    Tmax = 1
    method = "uniform"

    train_data = jinns.data.CubicMeshPDENonStatio_eqx(
        key=subkey,
        n=n,
        nb=nb,
        nt=nt,
        omega_batch_size=omega_batch_size,
        omega_border_batch_size=omega_border_batch_size,
        temporal_batch_size=temporal_batch_size,
        dim=dim,
        min_pts=(xmin,),
        max_pts=(xmax,),
        tmin=tmin,
        tmax=tmax,
        method=method,
    )

    # the next line is to be able to use the the same test values as the legacy
    # DataGenerators. We need to align the object parameters because their
    # respective init is not the same
    train_data = eqx.tree_at(
        lambda m: (
            m.curr_omega_idx,
            m.curr_omega_border_idx,
            m.curr_time_idx,
            m.omega,
            m.times,
        ),
        train_data,
        (
            0,
            0,
            0,
            random.choice(
                jnp.array([2330110495, 1427500313], dtype=jnp.uint32),
                train_data.omega,
                shape=(train_data.omega.shape[0],),
                replace=False,
                p=train_data.p_omega,
            ),
            random.choice(
                jnp.array([1474180313, 1830033527], dtype=jnp.uint32),
                train_data.times,
                shape=(train_data.times.shape[0],),
                replace=False,
                p=train_data.p_times,
            ),
        ),
    )

    key, subkey = random.split(key)
    obs_data = jinns.data.DataGeneratorObservations_eqx(
        subkey,
        obs_batch_size=omega_batch_size * temporal_batch_size,
        observed_pinn_in=jnp.stack([t_obs, x_obs], axis=1),
        observed_values=obs,
    )
    obs_data = eqx.tree_at(
        lambda m: (
            m.curr_idx,
            m.indices,
        ),
        obs_data,
        (
            0,
            random.choice(
                jnp.array([2823058779, 1116524360], dtype=jnp.uint32),
                obs_data.indices,
                shape=(obs_data.indices.shape[0],),
                replace=False,
                p=None,
            ),
        ),
    )

    init_params = {"nn_params": init_nn_params, "eq_params": {}}

    def u_init(x):
        return jnp.exp(-x)

    def u_boundary(t, dx):
        return jnp.exp(t)

    omega_boundary_fun = {"xmin": u_boundary, "xmax": None}
    omega_boundary_condition = {"xmin": "dirichlet", "xmax": None}

    from jinns.loss import PDENonStatio

    class advection_loss(PDENonStatio):
        def __init__(self, Tmax=1, eq_params_heterogeneity=None):
            super().__init__(Tmax, eq_params_heterogeneity)

        def evaluate(self, t, x, u, params):
            u_ = lambda t, x: u(t, x, params)[0]
            du_dt = jax.grad(u_, 0)
            du_dx = jax.grad(u_, 1)

            return du_dt(t, x) + du_dx(t, x)

    dynamic_loss = advection_loss(Tmax=Tmax)

    lambda_d = jnp.sqrt(10 * n)
    lambda_t = 0.1 / jnp.log(n)
    loss_weights = {
        "dyn_loss": 1,
        "initial_condition": 1,
        "boundary_loss": 1,
        "observations": lambda_d,
        "sobolev": lambda_t,
    }

    loss = jinns.loss.LossPDENonStatio(
        u=u,
        loss_weights=loss_weights,
        dynamic_loss=dynamic_loss,
        omega_boundary_fun=omega_boundary_fun,
        omega_boundary_condition=omega_boundary_condition,
        initial_condition_fun=u_init,
        sobolev_m=1,
    )

    return init_params, loss, train_data, obs_data, cpu2_sharding


@pytest.fixture
def train_imperfect_sobolev_10it_sharding(train_imperfect_sobolev_init_sharding):
    """
    Fixture that requests a fixture
    """
    init_params, loss, train_data, obs_data, cpu2_sharding = (
        train_imperfect_sobolev_init_sharding
    )

    # NOTE we need to waste one get_batch() here to stay synchronized with the
    # notebook
    train_data, _ = train_data.get_batch()
    obs_data, _ = obs_data.get_batch()

    params = init_params

    lambda_ = 1 / jnp.sqrt(10000)

    tx = optax.adamw(learning_rate=1e-3, weight_decay=lambda_)
    n_iter = 10
    params, total_loss_list, loss_by_term_dict, _, _, _, _, _, _ = jinns.solve(
        init_params=params,
        data=train_data,
        optimizer=tx,
        loss=loss,
        n_iter=n_iter,
        obs_data=obs_data,
        obs_batch_sharding=cpu2_sharding,
    )
    return total_loss_list[9]


@pytest.fixture
def train_imperfect_sobolev_init_no_sharding():
    jax.config.update("jax_enable_x64", False)
    key = random.PRNGKey(2)

    # create obs
    def u_star(t, x):
        return jnp.exp(t - x) + 0.1 * jnp.cos(2 * jnp.pi * x)

    n = 2000
    key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)
    t_obs = jax.random.uniform(subkey1, (n,), minval=0.001, maxval=0.5)
    x_obs = jax.random.uniform(subkey2, (n,), minval=0.001, maxval=1.0)
    obs = u_star(t_obs, x_obs) + jax.random.normal(subkey3, (n,)) * jnp.sqrt(1e-2)

    eqx_list = [
        [eqx.nn.Linear, 2, 50],
        [jax.nn.tanh],
        [eqx.nn.Linear, 50, 50],
        [jax.nn.tanh],
        [eqx.nn.Linear, 50, 50],
        [jax.nn.tanh],
        [eqx.nn.Linear, 50, 50],
        [jax.nn.tanh],
        [eqx.nn.Linear, 50, 1],
    ]
    key, subkey = random.split(key)
    u = jinns.utils.create_PINN(subkey, eqx_list, "nonstatio_PDE", 1)

    init_nn_params = u.init_params()

    n = 500
    nt = 500
    nb = 2
    omega_batch_size = 32
    temporal_batch_size = 32
    omega_border_batch_size = 1
    dim = 1
    xmin = 0.001
    xmax = 1
    tmin = 0
    tmax = 1
    Tmax = 1
    method = "uniform"

    train_data = jinns.data.CubicMeshPDENonStatio_eqx(
        key=subkey,
        n=n,
        nb=nb,
        nt=nt,
        omega_batch_size=omega_batch_size,
        omega_border_batch_size=omega_border_batch_size,
        temporal_batch_size=temporal_batch_size,
        dim=dim,
        min_pts=(xmin,),
        max_pts=(xmax,),
        tmin=tmin,
        tmax=tmax,
        method=method,
    )

    # the next line is to be able to use the the same test values as the legacy
    # DataGenerators. We need to align the object parameters because their
    # respective init is not the same
    train_data = eqx.tree_at(
        lambda m: (
            m.curr_omega_idx,
            m.curr_omega_border_idx,
            m.curr_time_idx,
            m.omega,
            m.times,
        ),
        train_data,
        (
            0,
            0,
            0,
            random.choice(
                jnp.array([2330110495, 1427500313], dtype=jnp.uint32),
                train_data.omega,
                shape=(train_data.omega.shape[0],),
                replace=False,
                p=train_data.p_omega,
            ),
            random.choice(
                jnp.array([1474180313, 1830033527], dtype=jnp.uint32),
                train_data.times,
                shape=(train_data.times.shape[0],),
                replace=False,
                p=train_data.p_times,
            ),
        ),
    )

    key, subkey = random.split(key)
    obs_data = jinns.data.DataGeneratorObservations_eqx(
        subkey,
        obs_batch_size=omega_batch_size * temporal_batch_size,
        observed_pinn_in=jnp.stack([t_obs, x_obs], axis=1),
        observed_values=obs,
    )
    obs_data = eqx.tree_at(
        lambda m: (
            m.curr_idx,
            m.indices,
        ),
        obs_data,
        (
            0,
            random.choice(
                jnp.array([2823058779, 1116524360], dtype=jnp.uint32),
                obs_data.indices,
                shape=(obs_data.indices.shape[0],),
                replace=False,
                p=None,
            ),
        ),
    )

    init_params = {"nn_params": init_nn_params, "eq_params": {}}

    def u_init(x):
        return jnp.exp(-x)

    def u_boundary(t, dx):
        return jnp.exp(t)

    omega_boundary_fun = {"xmin": u_boundary, "xmax": None}
    omega_boundary_condition = {"xmin": "dirichlet", "xmax": None}

    from jinns.loss import PDENonStatio

    class advection_loss(PDENonStatio):
        def __init__(self, Tmax=1, eq_params_heterogeneity=None):
            super().__init__(Tmax, eq_params_heterogeneity)

        def evaluate(self, t, x, u, params):
            u_ = lambda t, x: u(t, x, params)[0]
            du_dt = jax.grad(u_, 0)
            du_dx = jax.grad(u_, 1)

            return du_dt(t, x) + du_dx(t, x)

    dynamic_loss = advection_loss(Tmax=Tmax)

    lambda_d = jnp.sqrt(10 * n)
    lambda_t = 0.1 / jnp.log(n)
    loss_weights = {
        "dyn_loss": 1,
        "initial_condition": 1,
        "boundary_loss": 1,
        "observations": lambda_d,
        "sobolev": lambda_t,
    }

    loss = jinns.loss.LossPDENonStatio(
        u=u,
        loss_weights=loss_weights,
        dynamic_loss=dynamic_loss,
        omega_boundary_fun=omega_boundary_fun,
        omega_boundary_condition=omega_boundary_condition,
        initial_condition_fun=u_init,
        sobolev_m=1,
    )

    return init_params, loss, train_data, obs_data


@pytest.fixture
def train_imperfect_sobolev_10it_no_sharding(train_imperfect_sobolev_init_no_sharding):
    """
    Fixture that requests a fixture
    """
    init_params, loss, train_data, obs_data = train_imperfect_sobolev_init_no_sharding

    # NOTE we need to waste one get_batch() here to stay synchronized with the
    # notebook
    train_data, _ = train_data.get_batch()
    obs_data, _ = obs_data.get_batch()

    params = init_params

    lambda_ = 1 / jnp.sqrt(10000)

    tx = optax.adamw(learning_rate=1e-3, weight_decay=lambda_)
    n_iter = 10
    params, total_loss_list, loss_by_term_dict, _, _, _, _, _, _ = jinns.solve(
        init_params=params,
        data=train_data,
        optimizer=tx,
        loss=loss,
        n_iter=n_iter,
        obs_data=obs_data,
    )
    return total_loss_list[9]


def test_10it_imperfect_sobolev_sharding(
    train_imperfect_sobolev_10it_sharding, train_imperfect_sobolev_10it_no_sharding
):
    total_loss_val_for = train_imperfect_sobolev_10it_sharding
    total_loss_val_scan = train_imperfect_sobolev_10it_no_sharding
    assert jnp.allclose(total_loss_val_for, total_loss_val_scan, atol=1e-1)