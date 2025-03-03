import pytest

import jax
import jax.numpy as jnp
from jax import random
import equinox as eqx
import jinns


@pytest.fixture
def create_PPINN_MLP():
    jax.config.update("jax_enable_x64", False)
    key = random.PRNGKey(2)
    eqx_list_list = [
        (
            (eqx.nn.Linear, 2, 128),
            (jax.nn.tanh,),
            (eqx.nn.Linear, 128, 128),
            (jax.nn.tanh,),
            (eqx.nn.Linear, 128, 128),
            (jax.nn.tanh,),
            (eqx.nn.Linear, 128, 2),
        ),
        (
            (eqx.nn.Linear, 2, 128),
            (jax.nn.tanh,),
            (eqx.nn.Linear, 128, 128),
            (jax.nn.tanh,),
            (eqx.nn.Linear, 128, 2),
        ),
        (
            (eqx.nn.Linear, 2, 12),
            (jax.nn.relu,),
            (eqx.nn.Linear, 12, 12),
            (jax.nn.sigmoid,),
            (eqx.nn.Linear, 12, 12),
            (jax.nn.squareplus,),
            (eqx.nn.Linear, 12, 12),
        ),
    ]
    key, subkey = random.split(key)
    u, params = jinns.nn.PPINN_MLP.create(
        key=subkey, eqx_list_list=eqx_list_list, eq_type="nonstatio_PDE"
    )

    return u, params


@pytest.fixture
def create_3_MLPs():
    """ """
    jax.config.update("jax_enable_x64", False)

    key = random.PRNGKey(2)
    eqx_list_list = [
        (
            (eqx.nn.Linear, 2, 128),
            (jax.nn.tanh,),
            (eqx.nn.Linear, 128, 128),
            (jax.nn.tanh,),
            (eqx.nn.Linear, 128, 128),
            (jax.nn.tanh,),
            (eqx.nn.Linear, 128, 2),
        ),
        (
            (eqx.nn.Linear, 2, 128),
            (jax.nn.tanh,),
            (eqx.nn.Linear, 128, 128),
            (jax.nn.tanh,),
            (eqx.nn.Linear, 128, 2),
        ),
        (
            (eqx.nn.Linear, 2, 12),
            (jax.nn.relu,),
            (eqx.nn.Linear, 12, 12),
            (jax.nn.sigmoid,),
            (eqx.nn.Linear, 12, 12),
            (jax.nn.squareplus,),
            (eqx.nn.Linear, 12, 12),
        ),
    ]
    eqx_networks = []
    _, key = random.split(key)  # burn one jey for equivalency

    for i in range(3):
        key, subkey = jax.random.split(key, 2)
        eqx_networks.append(jinns.nn.MLP(key=subkey, eqx_list=eqx_list_list[i]))

    return eqx_networks


def test_equality(create_PPINN_MLP, create_3_MLPs):
    """
    Test that a PPIN_MLP is actually 3 parallel PINN MLP
    """
    u1, p1 = create_PPINN_MLP
    eqx_networks = create_3_MLPs
    key = random.PRNGKey(2)
    key, subkey = jax.random.split(key, 2)
    test_points = jax.random.normal(subkey, shape=(10, 2))
    results1 = jax.vmap(u1, (0, None))(test_points, p1)
    results2 = jnp.concatenate([jax.vmap(u)(test_points) for u in eqx_networks], axis=1)
    assert jnp.allclose(
        results1,
        results2,
        atol=1e-3,
    )
