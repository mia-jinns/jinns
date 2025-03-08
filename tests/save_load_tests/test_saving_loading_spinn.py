import pytest

import jax
import jax.numpy as jnp
from jax import random
import equinox as eqx
import jinns
from jinns.nn import save_pinn, load_pinn


@pytest.fixture
def save_reload(tmpdir):
    jax.config.update("jax_enable_x64", False)
    key = random.PRNGKey(2)
    d = 2
    r = 256
    eqx_list = (
        (eqx.nn.Linear, 1, 128),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 128, 128),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 128, 128),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 128, r),
    )
    key, subkey = random.split(key)
    u, params = jinns.nn.SPINN_MLP.create(subkey, d, r, eqx_list, "nonstatio_PDE")

    params = jinns.parameters.Params(nn_params=params, eq_params={})

    # Save
    filename = str(tmpdir.join("test"))
    kwargs_creation = {
        "key": subkey,
        "d": d,
        "r": r,
        "eqx_list": eqx_list,
        "eq_type": "nonstatio_PDE",
    }
    save_pinn(filename, u, params, kwargs_creation)

    # Reload
    u_reloaded, params_reloaded = load_pinn(filename, type_="spinn_mlp")
    return key, params, u, params_reloaded, u_reloaded


def test_equality_save_reload(save_reload):
    """
    Test if we have correctly reloaded the same model
    """
    key, params, u, params_reloaded, u_reloaded = save_reload
    key, subkey = jax.random.split(key, 2)
    test_points = jax.random.normal(subkey, shape=(10, 2))

    assert jnp.allclose(
        u(test_points, params),
        u_reloaded(test_points, params_reloaded),
        atol=1e-3,
    )


def test_jitting_reloaded_spinn(save_reload):
    """
    This test ensures that the reloaded spinn is jit-able.
    Some conversion of onp.array nodes can arise when reloading.
    See this MR : https://gitlab.com/mia_jinns/jinns/-/merge_requests/32
    jinns v0.8.2 uses eqx.field(static=True) to solve the problem.
    This tests is here for testimony.
    """

    key, _, _, params_reloaded, u_reloaded = save_reload

    key, subkey = jax.random.split(key, 2)
    test_points = jax.random.normal(subkey, shape=(10, 2))

    u_reloaded_jitted = jax.jit(u_reloaded.__call__)
    u_reloaded_jitted(test_points, params_reloaded)
