import pytest

import jax
import jax.numpy as jnp
from jax import random
from jax.scipy.stats import norm
import equinox as eqx
import optax
from jaxopt import OptaxSolver
import jinns


@pytest.fixture
def train_Fisher_init():
    jax.config.update("jax_enable_x64", True)
    print(jax.config.FLAGS.jax_enable_x64)
    print(jax.devices())
    key = random.PRNGKey(2)
    eqx_list = [
        [eqx.nn.Linear, 2, 50],
        [jax.nn.tanh],
        [eqx.nn.Linear, 50, 50],
        [jax.nn.tanh],
        [eqx.nn.Linear, 50, 50],
        [jax.nn.tanh],
        [eqx.nn.Linear, 50, 50],
        [jax.nn.tanh],
        [eqx.nn.Linear, 50, 50],
        [jax.nn.tanh],
        [eqx.nn.Linear, 50, 1],
        [jnp.exp],
    ]
    key, subkey = random.split(key)
    init_param_fn, u = jinns.utils.create_PINN(subkey, eqx_list, "nonstatio_PDE", 1)

    init_nn_params = init_param_fn()

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
        subkey,
        n,
        nb,
        nt,
        omega_batch_size,
        omega_border_batch_size,
        temporal_batch_size,
        dim,
        (xmin,),
        (xmax,),
        tmin,
        tmax,
        method,
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
        omega_boundary_fun = lambda dx: 0  # cte func returning 0

    elif boundary_condition == "neumann":
        omega_boundary_fun = lambda dx: 0  # cte func returning 0

    init_params = {
        "nn_params": init_nn_params,
        "eq_params": {"D": jnp.array([D]), "r": jnp.array([r]), "g": jnp.array([g])},
    }

    fisher_dynamic_loss = jinns.loss.FisherKPP(Tmax=Tmax)

    loss_weights = {
        "dyn_loss": 1,
        "temporal_loss": 1 * Tmax,
        "boundary_loss": 3 / 4 * Tmax,
    }

    loss = jinns.loss.LossPDENonStatio(
        u=u,
        loss_weights=loss_weights,
        dynamic_loss=fisher_dynamic_loss,
        omega_boundary_fun=omega_boundary_fun,
        omega_boundary_condition=boundary_condition,
        temporal_boundary_fun=u0,
        norm_key=None,
        norm_borders=None,
        norm_samples=None,
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
    _ = loss.evaluate(init_params, train_data.get_batch())[0]

    params = init_params

    tx = optax.adam(learning_rate=1e-4)
    solver = OptaxSolver(
        opt=tx,
        fun=loss,
        has_aux=True,  # because the objective has aux output
        maxiter=500000,
    )
    n_iter = 10
    pinn_solver = jinns.solver.PinnSolver(optax_solver=solver, loss=loss, n_iter=n_iter)
    params, total_loss_list, loss_by_term_dict, _, _ = pinn_solver.solve(
        init_params=params, data=train_data
    )
    return total_loss_list[9]


def test_initial_loss_Fisher(train_Fisher_init):
    init_params, loss, train_data = train_Fisher_init
    assert jnp.round(
        loss.evaluate(init_params, train_data.get_batch())[0], 5
    ) == jnp.round(12.155556509238087, 5)


def test_10it_Fisher(train_Fisher_10it):
    total_loss_val = train_Fisher_10it
    assert jnp.round(total_loss_val, 5) == jnp.round(10.961348468863452, 5)
