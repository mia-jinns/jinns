from itertools import zip_longest
from functools import partial
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


def test_batch_equality():
    """
    A more complex batch equality test
    To avoid problems of jitting a bound method, we decorticated the get_bathc
    function of DataGeneratorObservations here
    """

    def get_obs_batch_fun(dg):
        args = (
            dg.observed_eq_params,
            dg.observed_pinn_in,
            dg.observed_values,
            dg.n,
            dg.obs_batch_size,
            dg.curr_idx,
            dg.key,
            dg.indices,
        )

        tree_map_args = tuple(
            (
                {"observed_eq_params": e},
                {"observed_pinn_in": p},
                {"observed_values": v},
                {"n": n},
                {"obs_batch_size": b},
                {"curr_idx": c},
                {"key": k},
                {"indices": i},
            )
            for e, p, v, n, b, c, k, i in zip_longest(*args)
        )
        fixed_args = ()
        obs_batch_fun = jinns.data.DataGeneratorObservations.obs_batch
        if len(set(map(len, args))) > 1:  # at least 2 lengths differ
            # but since values, pinn_in and equations are the arguments that
            # generates all the others, it suffices to potentially fix the
            # former
            if len(dg.observed_pinn_in) == 1:
                obs_batch_fun = partial(
                    obs_batch_fun, observed_pinn_in=dg.observed_pinn_in[0]
                )
                fixed_args = fixed_args + ("observed_pinn_in",)
            if len(dg.observed_eq_params) == 1:
                obs_batch_fun = partial(
                    obs_batch_fun, observed_eq_params=dg.observed_eq_params[0]
                )
                fixed_args = fixed_args + ("observed_eq_params",)
            if len(dg.observed_values) == 1:
                obs_batch_fun = partial(
                    obs_batch_fun, observed_values=dg.observed_values[0]
                )
                fixed_args = fixed_args + ("observed_values",)

        fun = lambda args: jax.tree.map(
            jinns.data._DataGeneratorObservations._merge_dict_arguments(
                obs_batch_fun, fixed_args
            ),
            args,
            is_leaf=lambda x: (isinstance(x, tuple) and isinstance(x[0], dict)),
        )
        return fun, tree_map_args

    key = jax.random.PRNGKey(2)
    key, subkey = jax.random.split(key)
    subkeys2 = jax.random.split(key, 20)

    dg1 = jinns.data.DataGeneratorObservations(
        key=subkey,
        observed_pinn_in=tuple(
            jax.random.normal(key, shape=(100000, 3)) for _ in range(len(subkeys2))
        ),
        observed_values=tuple(
            jax.random.normal(k, shape=(100000, 15)) for k in subkeys2
        ),
        observed_eq_params=tuple(
            {"a": jax.random.normal(key, shape=(100000,))} for _ in range(len(subkeys2))
        ),
    )
    fun, tree_map_args = get_obs_batch_fun(dg1)
    compiled_1 = jax.jit(fun).lower(tree_map_args).compile()
    # print(compiled_1.cost_analysis()["bytes accessed"])
    obs_batch_duplicated = jax.tree.map(
        lambda l: l[3],
        compiled_1(tree_map_args),
        is_leaf=lambda x: isinstance(x, tuple) and len(x) == 4,
    )

    key = jax.random.PRNGKey(2)
    key, subkey = jax.random.split(key)

    dg2 = jinns.data.DataGeneratorObservations(
        key=subkey,
        observed_pinn_in=jax.random.normal(key, shape=(100000, 3)),
        observed_values=tuple(
            jax.random.normal(k, shape=(100000, 15)) for k in subkeys2
        ),
        observed_eq_params={"a": jax.random.normal(key, shape=(100000,))},
    )

    fun, tree_map_args = get_obs_batch_fun(dg2)
    compiled_2 = jax.jit(fun).lower(tree_map_args).compile()
    # print(compiled_2.cost_analysis()["bytes accessed"])
    obs_batch_nonduplicated = jax.tree.map(
        lambda l: l[3],
        compiled_2(tree_map_args),
        is_leaf=lambda x: isinstance(x, tuple) and len(x) == 4,
    )

    assert all(
        tuple(
            jnp.allclose(
                obs_batch_duplicated[i]["pinn_in"],
                obs_batch_nonduplicated[i]["pinn_in"],
            )
            for i in range(len(obs_batch_duplicated))
        )
    )
    assert all(
        tuple(
            jnp.allclose(
                obs_batch_duplicated[i]["val"], obs_batch_nonduplicated[i]["val"]
            )
            for i in range(len(obs_batch_duplicated))
        )
    )
