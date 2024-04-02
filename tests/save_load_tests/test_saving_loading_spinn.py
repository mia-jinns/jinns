import pytest

import jax
import jax.numpy as jnp
from jax import random
import equinox as eqx
import jinns
from jinns.utils import save_pinn, load_pinn


@pytest.fixture
def save_reload():
    jax.config.update("jax_enable_x64", False)
    key = random.PRNGKey(2)
    d = 2
    r = 256
    eqx_list = [
        [eqx.nn.Linear, 1, 128],
        [jax.nn.tanh],
        [eqx.nn.Linear, 128, 128],
        [jax.nn.tanh],
        [eqx.nn.Linear, 128, 128],
        [jax.nn.tanh],
        [eqx.nn.Linear, 128, r],
    ]
    key, subkey = random.split(key)
    u = jinns.utils.create_SPINN(subkey, d, r, eqx_list, "nonstatio_PDE")

    params = u.init_params()
    params = {"nn_params": params, "eq_params": {}}

    # Save
    filename = "./test"
    kwargs_creation = {
        "key": subkey,
        "d": d,
        "r": r,
        "eqx_list": eqx_list,
        "eq_type": "nonstatio_PDE",
    }
    save_pinn(filename, u, params, kwargs_creation)

    # Reload
    u_reloaded, params_reloaded = load_pinn(filename, type_="spinn")
    return key, params, u, params_reloaded, u_reloaded


def test_equality_save_reload(save_reload):
    """
    Test if we have correctly reloaded the same model
    """
    key, params, u, params_reloaded, u_reloaded = save_reload
    key, subkey = jax.random.split(key, 2)
    test_points = jax.random.normal(subkey, shape=(10, 2))

    assert jnp.allclose(
        u(test_points[:, 0:1], test_points[:, 1:], params),
        u_reloaded(test_points[:, 0:1], test_points[:, 1:], params_reloaded),
        atol=1e-3,
    )
