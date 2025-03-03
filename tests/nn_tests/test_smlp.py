import pytest

import jax
import jax.numpy as jnp
from jax import random
import equinox as eqx
import jinns


@pytest.fixture
def create_SMLP():
    jax.config.update("jax_enable_x64", False)
    key = random.PRNGKey(2)
    d = 3
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

    return u, params


@pytest.fixture
def create_3_MLPs():
    jax.config.update("jax_enable_x64", False)

    key = random.PRNGKey(2)
    eqx_list = (
        (eqx.nn.Linear, 1, 128),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 128, 128),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 128, 128),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 128, 256),
    )
    eqx_networks = []
    d = 3
    key, subkey = random.split(key)  # burn one jey for equivalency
    keys = jax.random.split(subkey, d)
    for i in range(d):
        eqx_networks.append(jinns.nn.MLP(key=keys[i], eqx_list=eqx_list))

    return eqx_networks


def test_equality(create_SMLP, create_3_MLPs):
    """
    Test that a SMLP is actually 3 separated MLPs
    """
    u1, p1 = create_SMLP
    eqx_networks = create_3_MLPs
    key = random.PRNGKey(2)
    key, subkey = jax.random.split(key, 2)
    test_points = jax.random.normal(subkey, shape=(10, 3))
    results1 = jax.vmap(eqx.combine(p1, u1.static))(test_points)
    results2 = [
        jax.vmap(u)(test_points[:, idx][:, None]) for idx, u in enumerate(eqx_networks)
    ]
    assert jnp.allclose(
        jnp.array(results1),
        jnp.swapaxes(jnp.array(results2), 0, 1),
        atol=1e-3,
    )
