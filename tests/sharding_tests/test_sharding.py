import pytest

import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"

import jinns
import jax
import jax.numpy as jnp
import equinox as eqx
import optax


@pytest.fixture
def solve_without_sharding():
    key = jax.random.PRNGKey(2)
    key, subkey = jax.random.split(key)

    # We have forced CPU computations and 2 fake CPU devices
    # in order to test the sharding
    cpu1, cpu2 = jax.devices("cpu")

    # Default device is cpu1
    jax.config.update("jax_default_device", cpu1)

    obs_data = jinns.data.DataGeneratorObservations(
        key=subkey,
        observed_pinn_in=(jnp.arange(100), jnp.arange(70)),
        observed_values=(jnp.ones((100,)), jnp.ones((70, 2))),
    )

    eqx_list = (
        (eqx.nn.Linear, 1, 20),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 20, 20),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 20, 20),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 20, 3),
    )
    key, subkey = jax.random.split(key)
    u, init_nn_params = jinns.nn.PINN_MLP.create(
        key=subkey, eqx_list=eqx_list, eq_type="ODE"
    )

    n = 1000
    tmin = 0
    tmax = 1

    train_data = jinns.data.DataGeneratorODE(
        key=subkey,
        nt=n,
        tmin=tmin,
        tmax=tmax,
    )

    init_params = jinns.parameters.Params(
        nn_params=init_nn_params, eq_params={"nu": 0.0}
    )

    loss_weights = jinns.loss.LossWeightsODE(observations=(1.0, 1.0))
    loss = jinns.loss.LossODE(
        u=u,
        dynamic_loss=None,
        params=init_params,
        loss_weights=loss_weights,
        obs_slice=(jnp.s_[0:1], jnp.s_[1:3]),
    )

    tx = optax.adam(learning_rate=1e-3)
    n_iter = 10
    _, total_loss_list, _, _, _, _, _, _, _, _, _, _ = jinns.solve(
        init_params=init_params,
        data=train_data,
        optimizer=tx,
        loss=loss,
        n_iter=n_iter,
        obs_data=obs_data,
    )
    return total_loss_list


@pytest.fixture
def solve_with_sharding():
    key = jax.random.PRNGKey(2)
    key, subkey = jax.random.split(key)

    # We have forced CPU computations and 2 fake CPU devices
    # in order to test the sharding
    cpu1, cpu2 = jax.devices("cpu")

    # Default device is cpu1
    jax.config.update("jax_default_device", cpu1)
    cpu1_sharding = jax.sharding.SingleDeviceSharding(cpu1)

    # Playing with obs_data on cpu2
    cpu2_sharding = jax.sharding.SingleDeviceSharding(cpu2)

    obs_data = jinns.data.DataGeneratorObservations(
        key=subkey,
        observed_pinn_in=(jnp.arange(100), jnp.arange(70)),
        observed_values=(jnp.ones((100,)), jnp.ones((70, 2))),
        sharding_device=cpu2_sharding,
    )

    eqx_list = (
        (eqx.nn.Linear, 1, 20),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 20, 20),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 20, 20),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 20, 3),
    )
    key, subkey = jax.random.split(key)
    u, init_nn_params = jinns.nn.PINN_MLP.create(
        key=subkey, eqx_list=eqx_list, eq_type="ODE"
    )

    n = 1000
    tmin = 0
    tmax = 1

    train_data = jinns.data.DataGeneratorODE(
        key=subkey,
        nt=n,
        tmin=tmin,
        tmax=tmax,
    )

    init_params = jinns.parameters.Params(
        nn_params=init_nn_params, eq_params={"nu": 0.0}
    )

    loss_weights = jinns.loss.LossWeightsODE(observations=(1.0, 1.0))
    loss = jinns.loss.LossODE(
        u=u,
        dynamic_loss=None,
        params=init_params,
        loss_weights=loss_weights,
        obs_slice=(jnp.s_[0:1], jnp.s_[1:3]),
    )

    tx = optax.adam(learning_rate=1e-3)
    n_iter = 10
    _, total_loss_list, _, _, _, _, _, _, _, _, _, _ = jinns.solve(
        init_params=init_params,
        data=train_data,
        optimizer=tx,
        loss=loss,
        n_iter=n_iter,
        obs_data=obs_data,
        obs_batch_sharding=cpu1_sharding,
    )
    return total_loss_list


def test_sharding_results(solve_with_sharding, solve_without_sharding):
    total_loss_list_sharding = solve_with_sharding
    total_loss_list_no_sharding = solve_without_sharding

    assert jnp.allclose(total_loss_list_sharding, total_loss_list_no_sharding)
