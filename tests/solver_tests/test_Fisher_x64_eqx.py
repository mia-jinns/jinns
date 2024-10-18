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
    eqx_list = (
        (eqx.nn.Linear, 2, 50),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 50, 50),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 50, 50),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 50, 50),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 50, 50),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 50, 1),
        (jnp.exp,),
    )
    key, subkey = random.split(key)
    u = jinns.utils.create_PINN(subkey, eqx_list, "nonstatio_PDE", 1)

    init_nn_params = u.init_params()

    n = 1000
    nb = 2
    nt = 1000
    omega_batch_size = 32
    temporal_batch_size = 20
    omega_border_batch_size = 2
    dim = 1
    xmin = -1
    xmax = 1
    tmin = 0
    tmax = 1
    method = "uniform"

    Tmax = 5
    key, subkey = random.split(key)
    train_data = jinns.data.CubicMeshPDENonStatio(
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
                jnp.array([3403514854, 2121154009], dtype=jnp.uint32),
                train_data.omega,
                shape=(train_data.omega.shape[0],),
                replace=False,
                p=train_data.p_omega,
            ),
            random.choice(
                jnp.array([1414439136, 3381782969], dtype=jnp.uint32),
                train_data.times,
                shape=(train_data.times.shape[0],),
                replace=False,
                p=train_data.p_times,
            ),
        ),
    )

    sigma_init = 0.2 * jnp.ones((1))
    mu_init = 0 * jnp.ones((1))

    def u0(x):
        return norm.pdf(x, loc=mu_init, scale=sigma_init)[0]  # output a scalar

    D = 1.0
    r = 3.0
    g = 3.0
    l = xmax - xmin

    boundary_condition = "dirichlet"

    if boundary_condition == "dirichlet":
        omega_boundary_fun = lambda t, dx: 0  # cte func returning 0

    elif boundary_condition == "neumann":
        omega_boundary_fun = lambda t, dx: 0  # cte func returning 0

    init_params = jinns.parameters.Params(
        nn_params=init_nn_params,
        eq_params={"D": jnp.array([D]), "r": jnp.array([r]), "g": jnp.array([g])},
    )

    fisher_dynamic_loss = jinns.loss.FisherKPP(Tmax=Tmax)

    loss_weights = jinns.loss.LossWeightsPDENonStatio(
        dyn_loss=1, initial_condition=1 * Tmax, boundary_loss=3 / 4 * Tmax
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

    # NOTE we need to waste one get_batch() here to stay synchronized with the
    # notebook
    train_data, _ = train_data.get_batch()

    params = init_params

    tx = optax.adam(learning_rate=1e-4)
    n_iter = 10
    params, total_loss_list, loss_by_term_dict, _, _, _, _, _, _ = jinns.solve(
        init_params=params, data=train_data, optimizer=tx, loss=loss, n_iter=n_iter
    )
    return total_loss_list[9]


def test_initial_loss_Fisher(train_Fisher_init):
    init_params, loss, train_data = train_Fisher_init
    assert jnp.allclose(
        loss.evaluate(init_params, train_data.get_batch()[1])[0], 12.15913, atol=1e-1
    )


def test_10it_Fisher(train_Fisher_10it):
    total_loss_val = train_Fisher_10it
    assert jnp.allclose(total_loss_val, 10.89091, atol=1e-1)
