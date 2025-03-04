import pytest

import jax
import jax.numpy as jnp
from jax import random
import equinox as eqx
import jinns
from jinns.nn import save_pinn, load_pinn
from jinns.parameters._params import _get_vmap_in_axes_params


@pytest.fixture
def save_reload(tmpdir):
    jax.config.update("jax_enable_x64", False)
    key = random.PRNGKey(2)

    eqx_list = (
        (eqx.nn.Linear, 3, 16),
        (jax.nn.swish,),
        (eqx.nn.Linear, 16, 16),
        (jax.nn.swish,),
        (eqx.nn.Linear, 16, 16),
        (jax.nn.swish,),
        (eqx.nn.Linear, 16, 16),
        (jax.nn.swish,),
        (eqx.nn.Linear, 16, 16),
        (jax.nn.swish,),
        (eqx.nn.Linear, 16, 16),
        (jax.nn.swish,),
        (eqx.nn.Linear, 16, 1),
    )

    eqx_list_hyper = (
        (eqx.nn.Linear, 2, 32),  # input is of size 2 for scalar D and scalar r
        (jax.nn.tanh,),
        (eqx.nn.Linear, 32, 32),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 32, 32),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 32, 32),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 32, 32),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 32, 32),
        (jax.nn.tanh,),
        (
            eqx.nn.Linear,
            32,
            1000,
        ),  # 1000 is a random guess, it will automatically be filled with the correct value
    )

    key, subkey = random.split(key)

    hyperparams = ["D", "r"]
    hypernet_input_size = 2

    kwargs_creation = {
        "key": subkey,
        "eqx_list": eqx_list,
        "eq_type": "nonstatio_PDE",
        "hyperparams": hyperparams,
        "hypernet_input_size": hypernet_input_size,
        "eqx_list_hyper": eqx_list_hyper,
    }

    u, params = jinns.nn.HyperPINN.create(**kwargs_creation)

    params = jinns.parameters.Params(
        nn_params=params,
        eq_params={"D": jnp.empty((10, 1)), "r": jnp.empty((10, 1))},
    )
    # Save
    filename = str(tmpdir.join("test"))
    save_pinn(filename, u, params, kwargs_creation)

    # Reload
    u_reloaded, params_reloaded = load_pinn(filename, type_="hyperpinn")
    return key, params, u, params_reloaded, u_reloaded


def test_equality_save_reload(save_reload):
    """
    Test if we have correctly reloaded the same model
    """
    key, params, u, params_reloaded, u_reloaded = save_reload
    key, subkey = jax.random.split(key, 2)
    test_points = jax.random.normal(subkey, shape=(10, 3))
    vmap_axes_params = _get_vmap_in_axes_params({"D": None, "r": None}, params)
    v_u = jax.vmap(
        u,
        (0,) + vmap_axes_params,
    )
    vmap_axes_params = _get_vmap_in_axes_params({"D": None, "r": None}, params_reloaded)
    v_u_reloaded = jax.vmap(
        u_reloaded,
        (0,) + vmap_axes_params,
    )

    assert jnp.allclose(
        v_u(test_points, params),
        v_u_reloaded(test_points, params_reloaded),
        atol=1e-3,
    )


def test_jitting_reloaded_hyperpinn(save_reload):
    """
    This test ensures that the reloaded hyperpinn is jit-able.
    Some conversion of onp.array nodes can arise when reloading.
    See this MR : https://gitlab.com/mia_jinns/jinns/-/merge_requests/32
    jinns v0.8.2 uses eqx.field(static=True) to solve the problem.
    This tests is here for testimony.
    """

    key, _, _, params_reloaded, u_reloaded = save_reload

    v_u_reloaded = jax.vmap(
        u_reloaded,
        (0, jinns.parameters.Params(nn_params=None, eq_params={"D": 0, "r": 0})),
    )
    v_u_reloaded_jitted = jax.jit(v_u_reloaded)

    key, subkey = jax.random.split(key, 2)
    test_points = jax.random.normal(subkey, shape=(10, 3))
    v_u_reloaded_jitted(test_points, params_reloaded)
