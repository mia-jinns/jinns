import pytest
import jax
import jax.numpy as jnp
import jinns


@pytest.fixture
def create_DataGeneratorObservations():
    key = jax.random.PRNGKey(2)
    key, subkey = jax.random.split(key)

    batch_size = (50, None)

    return jinns.data.DataGeneratorObservations(
        key=subkey,
        observed_pinn_in=(jnp.arange(100), jnp.arange(70)),
        observed_values=(jnp.ones((100,)), 2 * jnp.ones((70,))),
        obs_batch_size=batch_size,
    )


def test_batch_sizes(create_DataGeneratorObservations):
    dg = create_DataGeneratorObservations

    dg, obs_batch_tuple = dg.get_batch()

    assert len(obs_batch_tuple) == 2
    assert obs_batch_tuple[0]["pinn_in"].shape == (50, 1)
    assert obs_batch_tuple[0]["val"].shape == (50, 1)
    assert obs_batch_tuple[1]["pinn_in"].shape == (70, 1)
    assert obs_batch_tuple[1]["val"].shape == (70, 1)


# TODO add tests for observed_eq_params and for all the checks made in
# __post_init__
