import pytest
import jax
import jax.numpy as jnp
import equinox as eqx
import jinns


@pytest.fixture
def create_u_statio():
    key = jax.random.PRNGKey(2)
    eqx_list = (
        (eqx.nn.Linear, 2, 20),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 20, 20),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 20, 20),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 20, 2),
    )
    key, subkey = jax.random.split(key)
    return jinns.nn.PINN_MLP.create(key=subkey, eqx_list=eqx_list, eq_type="statio_PDE")


def test_divergence_rev_statio(create_u_statio):
    u_statio, params = create_u_statio
    x = jnp.array([0.4, 1.5])
    divergence = (
        jax.grad(lambda x: u_statio(x, params)[0])(x)[0]
        + jax.grad(lambda x: u_statio(x, params)[1])(x)[1]
    )
    assert jnp.allclose(jinns.loss.divergence_rev(x, u_statio, params), divergence)


@pytest.fixture
def create_u_nonstatio():
    key = jax.random.PRNGKey(2)
    eqx_list = (
        (eqx.nn.Linear, 3, 20),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 20, 20),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 20, 20),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 20, 3),
    )
    key, subkey = jax.random.split(key)
    return jinns.nn.PINN_MLP.create(
        key=subkey, eqx_list=eqx_list, eq_type="nonstatio_PDE"
    )


def test_divergence_rev_nonstatio(create_u_nonstatio):
    u_nonstatio, params = create_u_nonstatio
    t_x = jnp.array([0.5, 0.4, 1.5])
    divergence = (
        jax.grad(lambda t_x: u_nonstatio(t_x, params)[1])(t_x)[1]
        + jax.grad(lambda t_x: u_nonstatio(t_x, params)[2])(t_x)[2]
    )
    assert jnp.allclose(jinns.loss.divergence_rev(t_x, u_nonstatio, params), divergence)
