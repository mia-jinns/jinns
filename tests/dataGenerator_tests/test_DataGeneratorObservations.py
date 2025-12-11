import pytest
import jax
import jax.numpy as jnp
import jinns


def test_batch_sizes():
    key = jax.random.PRNGKey(2)
    key, subkey = jax.random.split(key)

    batch_size = (50, None)

    dg = jinns.data.DataGeneratorObservations(
        key=subkey,
        observed_pinn_in=(jnp.arange(100), jnp.arange(70)),
        observed_values=(jnp.ones((100,)), 2 * jnp.ones((70,))),
        obs_batch_size=batch_size,
    )

    dg, obs_batch_tuple = dg.get_batch()

    assert len(obs_batch_tuple) == 2
    assert obs_batch_tuple[0]["pinn_in"].shape == (50, 1)
    assert obs_batch_tuple[0]["val"].shape == (50, 1)
    assert obs_batch_tuple[1]["pinn_in"].shape == (70, 1)
    assert obs_batch_tuple[1]["val"].shape == (70, 1)

    dg = jinns.data.DataGeneratorObservations(
        key=subkey,
        observed_pinn_in=jnp.arange(100),
        observed_values=(jnp.ones((100, 2)), 2 * jnp.ones((100, 5))),
        obs_batch_size=batch_size,
    )

    dg, obs_batch_tuple = dg.get_batch()

    assert len(obs_batch_tuple) == 2
    assert obs_batch_tuple[0]["pinn_in"].shape == (50, 1)
    assert obs_batch_tuple[0]["val"].shape == (50, 2)
    assert obs_batch_tuple[1]["pinn_in"].shape == (100, 1)
    assert obs_batch_tuple[1]["val"].shape == (100, 5)


def test_dataset_alignements():
    key = jax.random.PRNGKey(2)
    key, subkey = jax.random.split(key)

    with pytest.raises(ValueError, match="Each matching elements"):
        jinns.data.DataGeneratorObservations(
            key=subkey,
            observed_pinn_in=(jnp.arange(100), jnp.arange(40)),
            observed_values=(jnp.ones((100,)), 2 * jnp.ones((70,))),
        )
    with pytest.raises(ValueError, match="Each matching elements"):
        jinns.data.DataGeneratorObservations(
            key=subkey,
            observed_pinn_in=jnp.arange(100),
            observed_values=(jnp.ones((100,)), 2 * jnp.ones((70,))),
        )
    with pytest.raises(ValueError, match="Each matching elements"):
        jinns.data.DataGeneratorObservations(
            key=subkey,
            observed_pinn_in=jnp.arange(10),
            observed_values=jnp.ones((100,)),
        )


def test_dataset_ndim_checks():
    key = jax.random.PRNGKey(2)
    key, subkey = jax.random.split(key)

    with pytest.raises(ValueError, match="have 2 dimensions"):
        jinns.data.DataGeneratorObservations(
            key=subkey,
            observed_pinn_in=jnp.arange(100),
            observed_values=jnp.ones((100, 3, 3)),
        )
