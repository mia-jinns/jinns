import pytest
import jax
import jax.numpy as jnp
import jinns

n = 64


@pytest.fixture
def create_DataGeneratorParameter():
    key = jax.random.PRNGKey(2)
    key, subkey = jax.random.split(key)

    param_batch_size = 64
    method = "uniform"
    param_ranges = {"theta": (10.0, 11.0)}
    user_data = {"nu": jnp.arange(n)}

    return jinns.data.DataGeneratorParameter(
        subkey,
        n,
        param_batch_size,
        param_ranges=param_ranges,
        method=method,
        user_data=user_data,
    )


def test_get_batch(create_DataGeneratorParameter):
    data_generator_parameters = create_DataGeneratorParameter
    param_batch = data_generator_parameters.get_batch()
    assert jnp.allclose(jnp.sort(jnp.unique(param_batch["nu"])), jnp.arange(n)) and (
        jnp.all(param_batch["theta"] >= 10.0) and jnp.all(param_batch["theta"] <= 11.0)
    )
