import pytest
import jax
import jax.numpy as jnp
import equinox as eqx
import jinns


@pytest.fixture
def create_u_statio():
    key = jax.random.PRNGKey(2)
    eqx_list = (
        (eqx.nn.Linear, 1, 20),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 20, 20),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 20, 20),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 20, 10),
    )
    key, subkey = jax.random.split(key)
    return jinns.nn.SPINN_MLP.create(subkey, 2, 10, eqx_list, "statio_PDE", 1)


def test_laplacian_fwd_statio(create_u_statio):
    u_statio, params = create_u_statio
    x = jnp.stack([jnp.array([0.4, 1.5]) for i in range(10)], axis=0)

    assert jnp.allclose(
        jinns.loss.laplacian_fwd(x, u_statio, params, method="loop"),
        jinns.loss.laplacian_fwd(x, u_statio, params, method="trace_hessian_x"),
    )


@pytest.fixture
def create_u_nonstatio():
    key = jax.random.PRNGKey(2)
    eqx_list = (
        (eqx.nn.Linear, 1, 20),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 20, 20),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 20, 20),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 20, 10),
    )
    key, subkey = jax.random.split(key)
    return jinns.nn.SPINN_MLP.create(subkey, 3, 10, eqx_list, "nonstatio_PDE", 1)


def test_laplacian_fwd_nonstatio(create_u_nonstatio):
    u_nonstatio, params = create_u_nonstatio
    t_x = jnp.stack([jnp.array([0.5, 0.4, 1.5]) for i in range(10)], axis=0)
    assert jnp.allclose(
        jinns.loss.laplacian_fwd(t_x, u_nonstatio, params, method="loop"),
        jinns.loss.laplacian_fwd(t_x, u_nonstatio, params, method="trace_hessian_t_x"),
    )
