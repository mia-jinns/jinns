import pytest

import jax
import jax.numpy as jnp
from jax import random
import equinox as eqx
import jinns
from jinns.nn import save_pinn, load_pinn


@pytest.fixture
def save_reload_with_params(tmpdir):
    jax.config.update("jax_enable_x64", False)
    key = random.PRNGKey(2)
    eqx_list = (
        (eqx.nn.Linear, 2, 128),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 128, 128),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 128, 128),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 128, 1),
    )
    key, subkey = random.split(key)
    kwargs_creation = {
        "key": subkey,
        "eqx_list": eqx_list,
        "eq_type": "nonstatio_PDE",
    }
    u, params = jinns.nn.PINN_MLP.create(**kwargs_creation)

    params = jinns.parameters.Params(nn_params=params, eq_params={})
    # Save
    filename = str(tmpdir.join("test"))
    save_pinn(filename, u, params, kwargs_creation)

    # Reload
    u_reloaded, params_reloaded = load_pinn(filename, type_="pinn_mlp")
    return key, params, u, params_reloaded, u_reloaded


def test_equality_save_reload_with_params(save_reload_with_params):
    """
    Test if we have correctly reloaded the same model
    """
    key, params, u, params_reloaded, u_reloaded = save_reload_with_params
    key, subkey = jax.random.split(key, 2)
    test_points = jax.random.normal(subkey, shape=(10, 2))
    v_u = jax.vmap(u, (0, None))
    v_u_reloaded = jax.vmap(u_reloaded, (0, None))

    assert jnp.allclose(
        v_u(test_points, params),
        v_u_reloaded(test_points, params_reloaded),
        atol=1e-3,
    )


def test_jitting_reloaded_pinn_with_params(save_reload_with_params):
    """
    This test ensures that the reloaded pinn is jit-able.
    Some conversion of onp.array nodes can arise when reloading.
    See this MR : https://gitlab.com/mia_jinns/jinns/-/merge_requests/32
    jinns v0.8.2 uses eqx.field(static=True) to solve the problem.
    This tests is here for testimony.
    """

    key, _, _, params_reloaded, u_reloaded = save_reload_with_params

    key, subkey = jax.random.split(key, 2)
    test_points = jax.random.normal(subkey, shape=(10, 2))
    v_u_reloaded = jax.vmap(u_reloaded, (0, None))
    v_u_reloaded_jitted = jax.jit(v_u_reloaded)

    v_u_reloaded_jitted(test_points, params_reloaded)


@pytest.fixture
def save_reload_with_paramsdict(tmpdir):
    jax.config.update("jax_enable_x64", False)
    key = random.PRNGKey(2)
    key, subkey = random.split(key)
    eqx_list = [
        [eqx.nn.Linear, 1, 20],
        [jax.nn.tanh],
        [eqx.nn.Linear, 20, 20],
        [jax.nn.tanh],
        [eqx.nn.Linear, 20, 20],
        [jax.nn.tanh],
        [eqx.nn.Linear, 20, 1],
        [jnp.exp],
    ]
    key, subkey = random.split(key)
    u, init_nn_params = jinns.nn.PINN_MLP.create(
        key=subkey, eqx_list=eqx_list, eq_type="ODE"
    )

    init_nn_params_list = []
    for _ in range(3):
        key, subkey = random.split(key)
        _, init_nn_params = jinns.nn.PINN_MLP.create(
            key=subkey, eqx_list=eqx_list, eq_type="ODE"
        )
        init_nn_params_list.append(init_nn_params)

    growth_rates = jnp.array([0.1, 0.5, 0.8])
    carrying_capacities = jnp.array([0.04, 0.02, 0.02])
    interactions = -jnp.array([[0, 0.001, 0.001], [0, 0.001, 0.001], [0, 0.001, 0.001]])

    params = jinns.parameters.ParamsDict(
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

    # Save
    filename = str(tmpdir.join("test"))
    kwargs_creation = {
        "key": subkey,
        "eqx_list": eqx_list,
        "eq_type": "ODE",
    }
    save_pinn(filename, u, params, kwargs_creation)

    # Reload
    u_reloaded, params_reloaded = load_pinn(
        filename, type_="pinn_mlp", key_list_for_paramsdict=["0", "1", "2"]
    )

    return key, params, u, params_reloaded, u_reloaded


def test_equality_save_reload_with_paramsdict(save_reload_with_paramsdict):
    """
    Test if we have correctly reloaded the same model
    """
    key, params, u, params_reloaded, u_reloaded = save_reload_with_paramsdict
    key, subkey = jax.random.split(key, 2)
    test_points = jax.random.normal(subkey, shape=(10, 1))
    v_u = jax.vmap(u, (0, None))
    v_u_reloaded = jax.vmap(u_reloaded, (0, None))

    assert jnp.allclose(
        v_u(test_points[:, 0:1], params.nn_params["0"]),
        v_u_reloaded(test_points[:, 0:1], params_reloaded.nn_params["0"]),
        atol=1e-3,
    )


def test_jitting_reloaded_pinn_with_paramsdict(save_reload_with_paramsdict):
    """
    This test ensures that the reloaded pinn is jit-able.
    Some conversion of onp.array nodes can arise when reloading.
    See this MR : https://gitlab.com/mia_jinns/jinns/-/merge_requests/32
    jinns v0.8.2 uses eqx.field(static=True) to solve the problem.
    This tests is here for testimony.
    """

    key, _, _, params_reloaded, u_reloaded = save_reload_with_paramsdict

    key, subkey = jax.random.split(key, 2)
    test_points = jax.random.normal(subkey, shape=(10, 1))
    v_u_reloaded = jax.vmap(u_reloaded, (0, None))
    v_u_reloaded_jitted = jax.jit(v_u_reloaded)

    v_u_reloaded_jitted(test_points[:, 0:1], params_reloaded.nn_params["0"])
