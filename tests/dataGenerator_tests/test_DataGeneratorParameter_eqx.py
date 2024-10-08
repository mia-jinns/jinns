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


@pytest.fixture
def create_DataGeneratorParameter_only_user_data():
    key = jax.random.PRNGKey(2)
    key, subkey = jax.random.split(key)

    param_batch_size = 64
    method = "uniform"
    user_data = {"nu": jnp.arange(n)}

    return jinns.data.DataGeneratorParameter(
        subkey,
        n,
        param_batch_size,
        method=method,
        user_data=user_data,
    )


def test_get_batch(create_DataGeneratorParameter):
    data_generator_parameters = create_DataGeneratorParameter
    _, param_batch = data_generator_parameters.get_batch()
    assert jnp.allclose(jnp.sort(jnp.unique(param_batch["nu"])), jnp.arange(n)) and (
        jnp.all(param_batch["theta"] >= 10.0) and jnp.all(param_batch["theta"] <= 11.0)
    )


def test_get_batch_only_user_data(create_DataGeneratorParameter_only_user_data):
    data_generator_parameters = create_DataGeneratorParameter_only_user_data
    _, param_batch = data_generator_parameters.get_batch()
    assert jnp.allclose(jnp.sort(jnp.unique(param_batch["nu"])), jnp.arange(n))


def test_raise_error_with_wrong_shape_for_user_data():
    key = jax.random.PRNGKey(2)
    key, subkey = jax.random.split(key)

    param_batch_size = 64
    method = "uniform"
    param_ranges = {"theta": (10.0, 11.0)}
    # user_data is not (n,) or (n,1)
    user_data = {"nu": jnp.ones((n, 1, 1))}

    with pytest.raises(ValueError) as e_info:
        # __init__ calls self.generate_data() that we are testing for
        data_generator_parameters = jinns.data.DataGeneratorParameter(
            subkey,
            n,
            param_batch_size,
            param_ranges=param_ranges,
            method=method,
            user_data=user_data,
        )
