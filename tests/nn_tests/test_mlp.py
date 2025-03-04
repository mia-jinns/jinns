import pytest

import jax
import jax.numpy as jnp
from jax import random
import equinox as eqx
import jinns


@pytest.fixture
def create_MLP_1():
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
    u, params = jinns.nn.PINN_MLP.create(
        key=subkey, eqx_list=eqx_list, eq_type="nonstatio_PDE"
    )

    return u, params


@pytest.fixture
def create_MLP_2():
    jax.config.update("jax_enable_x64", False)
    key = random.PRNGKey(2)
    key, subkey = random.split(key)
    eqx_network = eqx.nn.MLP(2, 1, 128, 3, jax.nn.tanh, key=subkey)
    u, params = jinns.nn.PINN_MLP.create(
        eqx_network=eqx_network, eq_type="nonstatio_PDE"
    )

    return u, params


@pytest.fixture
def create_MLP_3():
    """
    Illustrates the minimal requirements to inherit from PINN
    """
    jax.config.update("jax_enable_x64", False)

    class MyPINN(jinns.nn.PINN):
        def __call__(self, inputs, params):
            model = eqx.combine(params, self.static)
            return model(inputs)

    key = random.PRNGKey(2)
    key, subkey = random.split(key)
    eqx_list = (
        (eqx.nn.Linear, 2, 128),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 128, 128),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 128, 128),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 128, 1),
    )
    eqx_network = jinns.nn.MLP(key=subkey, eqx_list=eqx_list)

    u = MyPINN(eqx_network=eqx_network, eq_type="nonstatio_PDE")
    params = u.init_params

    return u, params


# No, we do not have equivalency with a eqx.nn.MLP because of
# we have a different PRNG scheme
# def test_equality_1(create_MLP_1, create_MLP_2):
#    u1, p1 = create_MLP_1
#    u2, p2 = create_MLP_2
#    key = random.PRNGKey(2)
#    key, subkey = jax.random.split(key, 2)
#    test_points = jax.random.normal(subkey, shape=(10, 2))
#    v_u1 = jax.vmap(u1, (0, None))
#    v_u2 = jax.vmap(u2, (0, None))
#    assert jnp.allclose(
#        v_u1(test_points, p1),
#        v_u2(test_points, p2),
#        atol=1e-3,
#    )


def test_equality(create_MLP_1, create_MLP_3):
    u1, p1 = create_MLP_1
    u2, p2 = create_MLP_3
    key = random.PRNGKey(2)
    key, subkey = jax.random.split(key, 2)
    test_points = jax.random.normal(subkey, shape=(10, 2))
    v_u1 = jax.vmap(u1, (0, None))
    v_u2 = jax.vmap(u2, (0, None))
    assert jnp.allclose(
        v_u1(test_points, p1),
        v_u2(test_points, p2),
        atol=1e-3,
    )
