import pytest
import jax
import jax.numpy as jnp
import jinns


@pytest.fixture
def create_DataGeneratorODE():
    key = jax.random.PRNGKey(2)
    key, subkey = jax.random.split(key)

    n = 10000
    batch_size = 64
    method = "uniform"
    tmin = 0
    tmax = 1

    return jinns.data.DataGeneratorODE(subkey, n, tmin, tmax, batch_size, method)


def test_t_range_DataGeneratorODE(create_DataGeneratorODE):
    data_generator_ode = create_DataGeneratorODE
    assert jnp.all(data_generator_ode.times >= data_generator_ode.tmin) and jnp.all(
        data_generator_ode.times <= data_generator_ode.tmax
    )


def test_get_batch(create_DataGeneratorODE):
    data_generator_ode = create_DataGeneratorODE
    times_batch = data_generator_ode.get_batch().temporal_batch
    assert jnp.all(times_batch[:] >= data_generator_ode.tmin) and jnp.all(
        times_batch[:] <= data_generator_ode.tmax
    )
