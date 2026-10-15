import pytest

import jax
import jax.numpy as jnp
from jax import random
import equinox as eqx
import jinns


@pytest.fixture
def create_MLP_1():
    jax.config.update("jax_enable_x64", True)
    key = random.PRNGKey(2)
    eqx_list = (
        (eqx.nn.Linear, 2, 5),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 5, 1),
    )
    key, subkey = random.split(key)
    u, params = jinns.nn.PINN_MLP.create(
        key=subkey, eqx_list=eqx_list, eq_type="PDENonStatio"
    )
    params = jinns.parameters.Params(nn_params=params, eq_params={})

    return u, params


@pytest.fixture
def create_MLP_2():
    jax.config.update("jax_enable_x64", True)
    key = random.PRNGKey(2)
    key, subkey = random.split(key)
    eqx_network = eqx.nn.MLP(2, 1, 5, 1, jax.nn.tanh, key=subkey)
    u, params = jinns.nn.PINN_MLP.create(
        eqx_network=eqx_network, eq_type="PDENonStatio"
    )
    params = jinns.parameters.Params(nn_params=params, eq_params={})

    return u, params


@pytest.fixture
def create_MLP_3():
    """
    Illustrates the minimal requirements to inherit from PINN
    """
    jax.config.update("jax_enable_x64", True)

    class MyPINN(jinns.nn.PINN):
        def __call__(self, inputs, params):
            model = eqx.combine(params.nn_params, self.static)
            return model(inputs)

    key = random.PRNGKey(2)
    key, subkey = random.split(key)
    eqx_list = (
        (eqx.nn.Linear, 2, 5),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 5, 1),
    )
    eqx_network = jinns.nn.MLP(key=subkey, eqx_list=eqx_list)

    u = MyPINN(eqx_network=eqx_network, eq_type="PDENonStatio")
    params = u.init_params
    params = jinns.parameters.Params(nn_params=params, eq_params={})

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


@pytest.fixture
def create_datagenerators():
    jax.config.update("jax_enable_x64", True)

    key = random.PRNGKey(2)
    key, subkey = random.split(key)
    param_train_data = jinns.data.DataGeneratorParameter(
        key=subkey,
        n=5,
        param_batch_size=5,
        param_ranges={"a": (0.05, 1.0), "b": (0.05, 0.15)},
        method="grid",
    )

    key, subkey = random.split(key)
    train_data = jinns.data.CubicMeshPDENonStatio(
        key=subkey,
        n=5,
        nb=None,
        ni=5,
        dim=2,
        domain_batch_size=5,
        min_pts=(0, 0),
        max_pts=(1, 1),
        tmin=0,
        tmax=1,
        method="uniform",
    )
    return train_data, param_train_data


def test_hyperparams(create_datagenerators):
    train_data, param_train_data = create_datagenerators

    key = random.PRNGKey(2)
    eqx_list = (
        (eqx.nn.Linear, 5, 5),  # Entry is 5D : space + time + "a" + "b"
        (jax.nn.tanh,),
        (eqx.nn.Linear, 5, 1),
    )
    key, subkey = random.split(key)
    u, params = jinns.nn.PINN_MLP.create(
        key=subkey, eqx_list=eqx_list, eq_type="PDENonStatio", hyperparams=["a", "b"]
    )
    params = jinns.parameters.Params(nn_params=params, eq_params={"a": 0.0, "b": 0.0})
    params = jinns.parameters.update_eq_params(params, param_train_data.get_batch()[1])

    # PyTree can be passed in vmap in_axes arguments to specify vmap axes in complex structure
    # Here we want to vmap over leaf "a" and "b" of Params PyTree
    v_u = jax.vmap(
        u, (0, jinns.parameters.Params(nn_params=None, eq_params={"a": 0, "b": 0}))
    )

    batch = train_data.get_batch()[1]
    assert jnp.allclose(v_u(batch.domain_batch, params)[0], jnp.array([-0.35134127]))
