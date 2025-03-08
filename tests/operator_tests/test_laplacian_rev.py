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
        (eqx.nn.Linear, 20, 1),
    )
    key, subkey = jax.random.split(key)
    return jinns.nn.PINN_MLP.create(key=subkey, eqx_list=eqx_list, eq_type="statio_PDE")


def test_laplacian_rev_statio(create_u_statio):
    u_statio, params = create_u_statio
    x = jnp.array([0.4, 1.5])
    assert jnp.allclose(
        jinns.loss.laplacian_rev(x, u_statio, params, method="trace_hessian_x"),
        jinns.loss.laplacian_rev(x, u_statio, params, method="trace_hessian_t_x"),
    )
    assert jnp.allclose(
        jinns.loss.laplacian_rev(x, u_statio, params, method="trace_hessian_x"),
        jinns.loss.laplacian_rev(x, u_statio, params, method="loop"),
    )


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
        (eqx.nn.Linear, 20, 1),
    )
    key, subkey = jax.random.split(key)
    return jinns.nn.PINN_MLP.create(
        key=subkey, eqx_list=eqx_list, eq_type="nonstatio_PDE"
    )


def test_laplacian_rev_nonstatio(create_u_nonstatio):
    u_nonstatio, params = create_u_nonstatio
    t_x = jnp.array([0.5, 0.4, 1.5])
    assert jnp.allclose(
        jinns.loss.laplacian_rev(t_x, u_nonstatio, params, method="trace_hessian_x"),
        jinns.loss.laplacian_rev(t_x, u_nonstatio, params, method="trace_hessian_t_x"),
    )
    assert jnp.allclose(
        jinns.loss.laplacian_rev(t_x, u_nonstatio, params, method="trace_hessian_x"),
        jinns.loss.laplacian_rev(t_x, u_nonstatio, params, method="loop"),
    )
