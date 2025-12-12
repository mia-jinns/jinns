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

    # here pinn_in will be used for all values
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

    # here values will be used for all pinn_in
    dg = jinns.data.DataGeneratorObservations(
        key=subkey,
        observed_pinn_in=(jnp.ones(100), jnp.zeros((100,))),
        observed_values=jnp.ones((100, 5)),
        obs_batch_size=batch_size,
    )

    dg, obs_batch_tuple = dg.get_batch()

    assert len(obs_batch_tuple) == 2
    assert obs_batch_tuple[0]["pinn_in"].shape == (50, 1)
    assert obs_batch_tuple[0]["val"].shape == (50, 5)
    assert obs_batch_tuple[1]["pinn_in"].shape == (100, 1)
    assert obs_batch_tuple[1]["val"].shape == (100, 5)
    assert jnp.allclose(jnp.ones((50,)), obs_batch_tuple[0]["pinn_in"])
    assert jnp.allclose(jnp.zeros((100,)), obs_batch_tuple[1]["pinn_in"])

    # here pinn_in and values will be used for all eq_params
    dg = jinns.data.DataGeneratorObservations(
        key=subkey,
        observed_pinn_in=jnp.ones(100),
        observed_values=jnp.ones((100, 5)),
        observed_eq_params=({"a": jnp.ones((100, 3))}, {"a": jnp.ones((100,))}),
        obs_batch_size=batch_size,
    )

    dg, obs_batch_tuple = dg.get_batch()

    assert len(obs_batch_tuple) == 2
    assert obs_batch_tuple[0]["val"].shape == (50, 5)
    assert obs_batch_tuple[1]["val"].shape == (100, 5)
    assert jnp.allclose(jnp.ones((50,)), obs_batch_tuple[0]["pinn_in"])
    assert jnp.allclose(jnp.ones((50, 3)), obs_batch_tuple[0]["eq_params"].a)
    assert jnp.allclose(jnp.ones((100,)), obs_batch_tuple[1]["eq_params"].a)

    dg = jinns.data.DataGeneratorObservations(
        key=subkey,
        observed_pinn_in=(jnp.arange(100), jnp.ones((10,)), jnp.ones((10,))),
        observed_values=(jnp.ones((100, 3)), jnp.ones((10, 3)), jnp.ones((10, 3))),
        obs_batch_size=10,
    )
    dg, obs_batch_tuple = dg.get_batch()
    assert obs_batch_tuple[0]["pinn_in"].shape == (10, 1)
    assert obs_batch_tuple[1]["pinn_in"].shape == (10, 1)
    assert obs_batch_tuple[2]["pinn_in"].shape == (10, 1)
    assert obs_batch_tuple[0]["val"].shape == (10, 3)
    assert obs_batch_tuple[1]["val"].shape == (10, 3)
    assert obs_batch_tuple[2]["val"].shape == (10, 3)


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
            observed_pinn_in=(jnp.ones((100,)), 2 * jnp.ones((70,))),
            observed_values=jnp.ones((100,)),
        )
    with pytest.raises(ValueError, match="Each matching elements"):
        jinns.data.DataGeneratorObservations(
            key=subkey,
            observed_pinn_in=jnp.arange(10),
            observed_values=jnp.ones((100,)),
        )
    with pytest.raises(ValueError, match="Each matching elements"):
        jinns.data.DataGeneratorObservations(
            key=subkey,
            observed_pinn_in=jnp.arange(100),
            observed_values=jnp.ones((100,)),
            observed_eq_params={"a": jnp.ones((50,))},
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
    with pytest.raises(ValueError, match="have 2 dimensions"):
        jinns.data.DataGeneratorObservations(
            key=subkey,
            observed_pinn_in=jnp.arange(100),
            observed_values=jnp.ones((100, 1)),
            observed_eq_params={"a": jnp.ones((100, 3, 3))},
        )


def test_batch_size_checks():
    key = jax.random.PRNGKey(2)
    key, subkey = jax.random.split(key)

    with pytest.raises(ValueError, match="obs_batch_size is a tuple"):
        jinns.data.DataGeneratorObservations(
            key=subkey,
            observed_pinn_in=jnp.arange(100),
            observed_values=(
                jnp.ones((100, 3)),
                jnp.ones((100, 3)),
                jnp.ones((100, 3)),
            ),
            obs_batch_size=(None, None),
        )

    with pytest.raises(ValueError, match="obs_batch_size is a tuple"):
        jinns.data.DataGeneratorObservations(
            key=subkey,
            observed_pinn_in=(jnp.arange(100), jnp.ones((10,)), jnp.ones((10,))),
            observed_values=(jnp.ones((100, 3)), jnp.ones((10, 3)), jnp.ones((10, 3))),
            obs_batch_size=(None, 10),
        )
