import pytest

import jax
import jax.numpy as jnp
from jax import random
import equinox as eqx
import optax
from jaxopt import OptaxSolver
import jinns


@pytest.fixture
def train_GLV_init():
    jax.config.update("jax_enable_x64", True)
    print(jax.config.FLAGS.jax_enable_x64)
    print(jax.devices())
    key = random.PRNGKey(2)
    key, subkey = random.split(key)
    eqx_list = [
        [eqx.nn.Linear, 1, 20],
        [jax.nn.tanh],
        [eqx.nn.Linear, 20, 20],
        [jax.nn.tanh],
        [eqx.nn.Linear, 20, 20],
        [jax.nn.tanh],
        [eqx.nn.Linear, 20, 1],
        [jnp.exp],
    ]
    key, subkey = random.split(key)
    init_param_fn, u = jinns.utils.create_PINN(subkey, eqx_list, "ODE")

    init_nn_params = init_param_fn()

    n = 320
    batch_size = 32
    method = "uniform"
    tmin = 0
    tmax = 1

    Tmax = 30
    key, subkey = random.split(key)
    train_data = jinns.data.DataGeneratorODE(subkey, n, tmin, tmax, batch_size, method)

    init_nn_params_list = []
    for _ in range(3):
        key, subkey = random.split(key)
        init_param_fn, _ = jinns.utils.create_PINN(subkey, eqx_list, "ODE", 0)
        init_nn_params = init_param_fn()
        init_nn_params_list.append(init_nn_params)

    N_0 = jnp.array([10.0, 7.0, 4.0])
    growth_rates = jnp.array([0.1, 0.5, 0.8])
    carrying_capacities = jnp.array([0.04, 0.02, 0.02])
    interactions = -jnp.array([[0, 0.001, 0.001], [0, 0.001, 0.001], [0, 0.001, 0.001]])

    init_params = {}
    init_params["nn_params"] = {str(i): init_nn_params_list[i] for i in range(3)}
    init_params["eq_params"] = {
        str(i): {
            "carrying_capacity": carrying_capacities[i],
            "growth_rate": growth_rates[i],
            "interactions": interactions[i, :],
        }
        for i in range(3)
    }

    N1_dynamic_loss = jinns.loss.GeneralizedLotkaVolterra(
        key_main="0", keys_other=["1", "2"], Tmax=Tmax
    )
    N2_dynamic_loss = jinns.loss.GeneralizedLotkaVolterra(
        key_main="1", keys_other=["0", "2"], Tmax=Tmax
    )
    N3_dynamic_loss = jinns.loss.GeneralizedLotkaVolterra(
        key_main="2", keys_other=["0", "1"], Tmax=Tmax
    )

    loss_weights = {"dyn_loss": 1, "initial_condition": 1 * Tmax}

    loss = jinns.loss.SystemLossODE(
        u_dict={"0": u, "1": u, "2": u},
        loss_weights=loss_weights,
        dynamic_loss_dict={
            "0": N1_dynamic_loss,
            "1": N2_dynamic_loss,
            "2": N3_dynamic_loss,
        },
        initial_condition_dict={
            "0": (float(tmin), N_0[0]),
            "1": (float(tmin), N_0[1]),
            "2": (float(tmin), N_0[2]),
        },
    )

    return init_params, loss, train_data


@pytest.fixture
def train_GLV_10it(train_GLV_init):
    """
    Fixture that requests a fixture
    """
    init_params, loss, train_data = train_GLV_init

    # NOTE we need to waste one get_batch() here to stay synchronized with the
    # notebook
    _ = loss.evaluate(init_params, train_data.get_batch())[0]

    params = init_params

    tx = optax.adam(learning_rate=1e-3)
    solver = OptaxSolver(
        opt=tx,
        fun=loss,
        has_aux=True,  # because the objective has aux output
        maxiter=500000,
    )
    n_iter = 10
    pinn_solver = jinns.solver.PinnSolver(optax_solver=solver, loss=loss, n_iter=n_iter)
    params, total_loss_list, loss_by_term_dict, _, _, _, _ = pinn_solver.solve(
        init_params=params, data=train_data
    )
    return total_loss_list[9]


def test_initial_loss_GLV(train_GLV_init):
    init_params, loss, train_data = train_GLV_init
    assert jnp.round(
        loss.evaluate(init_params, train_data.get_batch())[0], 5
    ) == jnp.round(4233.245370360475, 5)


def test_10it_GLV(train_GLV_10it):
    total_loss_val = train_GLV_10it
    assert jnp.round(total_loss_val, 5) == jnp.round(3867.97425219, 5)
