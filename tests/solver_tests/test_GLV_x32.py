import pytest

import jax
import jax.numpy as jnp
from jax import random
import equinox as eqx
import optax
import jinns
from jinns.data._DataGenerators import DataGeneratorODE


@pytest.fixture
def train_GLV_init():
    jax.config.update("jax_enable_x64", False)
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
    train_data = DataGeneratorODE(
        key=subkey,
        nt=n,
        tmin=tmin,
        tmax=tmax,
        temporal_batch_size=batch_size,
        method=method,
    )

    init_nn_params_list = []
    for _ in range(3):
        key, subkey = random.split(key)
        nn, init_nn_params = jinns.nn.PINN_MLP.create(
            key=subkey, eqx_list=eqx_list, eq_type="ODE"
        )
        init_nn_params_list.append(init_nn_params)

    N_0 = jnp.array([10.0, 7.0, 4.0])
    growth_rates = jnp.array([0.1, 0.5, 0.8])
    carrying_capacities = jnp.array([0.04, 0.02, 0.02])
    interactions = -jnp.array([[0, 0.001, 0.001], [0, 0.001, 0.001], [0, 0.001, 0.001]])

    init_params = jinns.parameters.ParamsDict(
        nn_params={str(i): init_nn_params_list[i] for i in range(3)},
        eq_params={
            str(i): {
                "carrying_capacity": carrying_capacities[i],
                "growth_rate": growth_rates[i],
                "interactions": interactions[i, :],
            }
            for i in range(3)
        },
    )

    N1_dynamic_loss = jinns.loss.GeneralizedLotkaVolterra(
        key_main="0", keys_other=["1", "2"], Tmax=Tmax
    )
    N2_dynamic_loss = jinns.loss.GeneralizedLotkaVolterra(
        key_main="1", keys_other=["0", "2"], Tmax=Tmax
    )
    N3_dynamic_loss = jinns.loss.GeneralizedLotkaVolterra(
        key_main="2", keys_other=["0", "1"], Tmax=Tmax
    )

    loss_weights = jinns.loss.LossWeightsODEDict(dyn_loss=1, initial_condition=1 * Tmax)

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
        params_dict=init_params,
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
    params, total_loss_list, loss_by_term_dict, data, _, _, _, _, _ = jinns.solve(
        n_iter=n_iter,
        loss=loss,
        optimizer=tx,
        init_params=params,
        data=train_data,
    )
    return total_loss_list[9]


def test_initial_loss_GLV(train_GLV_init):
    init_params, loss, train_data = train_GLV_init
    _, batch = train_data.get_batch()
    assert jnp.allclose(
        loss.evaluate(init_params, batch)[0],
        4750.7188,
        atol=1e-1,
    )


def test_10it_GLV(train_GLV_10it):
    total_loss_val = train_GLV_10it
    assert jnp.allclose(total_loss_val, 4479.3237, atol=1e-1)
