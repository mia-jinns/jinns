import pytest

import jax
import jax.numpy as jnp
from jax import random
import equinox as eqx
import optax
import jinns


@pytest.fixture
def train_GLV_init():
    jax.config.update("jax_enable_x64", True)
    key = random.PRNGKey(2)
    key, subkey = random.split(key)
    eqx_list = (
        (eqx.nn.Linear, 1, 20),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 20, 20),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 20, 20),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 20, 1),
        (jnp.exp,),
    )
    key, subkey = random.split(key)
    u, init_nn_params = jinns.nn.PINN_MLP.create(
        key=subkey, eqx_list=eqx_list, eq_type="ODE"
    )

    n = 320
    batch_size = 32
    method = "uniform"
    tmin = 0
    tmax = 1

    Tmax = 30
    key, subkey = random.split(key)
    train_data = jinns.data.DataGeneratorODE(
        key=subkey,
        nt=n,
        tmin=tmin,
        tmax=tmax,
        temporal_batch_size=batch_size,
        method=method,
    )

    key, subkey = random.split(key)
    u, init_nn_params = jinns.nn.PINN_MLP.create(
        key=subkey, eqx_list=eqx_list, eq_type="ODE"
    )

    N_0 = jnp.array([10.0, 7.0, 4.0])
    growth_rates = jnp.array([0.1, 0.5, 0.8])
    carrying_capacities = jnp.array([0.04, 0.02, 0.02])
    interactions = (
        -jnp.array([0, 0.001, 0.001]),
        -jnp.array([0.001, 0, 0.001]),
        -jnp.array([0.001, 0.001, 0]),
    )
    init_params = jinns.parameters.Params(
        nn_params=init_nn_params,
        eq_params={
            "carrying_capacities": carrying_capacities,
            "growth_rates": growth_rates,
            "interactions": interactions,
        },
    )
    dynamic_loss = jinns.loss.GeneralizedLotkaVolterra(Tmax=Tmax)

    loss_weights = jinns.loss.LossWeightsODE(dyn_loss=1, initial_condition=1 * Tmax)
    loss = jinns.loss.LossODE(
        u=u,
        loss_weights=loss_weights,
        dynamic_loss=dynamic_loss,
        initial_condition=(float(tmin), jnp.array([N_0[0], N_0[1], N_0[2]])),
        params=init_params,
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
    train_data, batch = train_data.get_batch()
    _ = loss.evaluate(init_params, batch)[0]
    # NOTE the following line is not accurate as it skips one batch
    # but this is to comply with behaviour of 1st gen of DataGenerator
    # (which had this bug see issue #5) so that we can keep old tests values to
    # know we are doing the same
    train_data = eqx.tree_at(
        lambda m: m.curr_time_idx, train_data, train_data.temporal_batch_size
    )

    params = init_params

    tx = optax.adam(learning_rate=1e-3)
    n_iter = 10
    params, total_loss_list, loss_by_term_dict, _, _, _, _, _, _, _ = jinns.solve(
        init_params=params, data=train_data, optimizer=tx, loss=loss, n_iter=n_iter
    )
    return total_loss_list[9]


def test_initial_loss_GLV(train_GLV_init):
    init_params, loss, train_data = train_GLV_init
    assert jnp.allclose(
        loss.evaluate(init_params, train_data.get_batch()[1])[0],
        4770.75105231,
        atol=1e-1,
    )


def test_10it_GLV(train_GLV_10it):
    total_loss_val = train_GLV_10it
    assert jnp.allclose(total_loss_val, 4571.6202204, atol=1e-1)
