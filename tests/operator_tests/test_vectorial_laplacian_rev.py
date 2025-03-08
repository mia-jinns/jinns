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
        (eqx.nn.Linear, 20, 4),
    )
    key, subkey = jax.random.split(key)
    return jinns.nn.PINN_MLP.create(key=subkey, eqx_list=eqx_list, eq_type="statio_PDE")


def test_laplacian_rev_statio(create_u_statio):
    u_statio, params = create_u_statio
    x = jnp.array([0.4, 1.5])
    vec_lap = jnp.array(
        [
            jinns.loss.laplacian_rev(
                x,
                lambda x, params: u_statio(x, params)[0],
                params,
                eq_type="statio_PDE",
            ),
            jinns.loss.laplacian_rev(
                x,
                lambda x, params: u_statio(x, params)[1],
                params,
                eq_type="statio_PDE",
            ),
            jinns.loss.laplacian_rev(
                x,
                lambda x, params: u_statio(x, params)[2],
                params,
                eq_type="statio_PDE",
            ),
            jinns.loss.laplacian_rev(
                x,
                lambda x, params: u_statio(x, params)[3],
                params,
                eq_type="statio_PDE",
            ),
        ]
    )
    assert jnp.allclose(
        jinns.loss.vectorial_laplacian_rev(x, u_statio, params, dim_out=4), vec_lap
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
        (eqx.nn.Linear, 20, 4),
    )
    key, subkey = jax.random.split(key)
    return jinns.nn.PINN_MLP.create(
        key=subkey, eqx_list=eqx_list, eq_type="nonstatio_PDE"
    )


def test_laplacian_rev_nonstatio(create_u_nonstatio):
    u_nonstatio, params = create_u_nonstatio
    t_x = jnp.array([0.5, 0.4, 1.5])
    vec_lap = jnp.array(
        [
            jinns.loss.laplacian_rev(
                t_x,
                lambda t_x, params: u_nonstatio(t_x, params)[0],
                params,
                eq_type="nonstatio_PDE",
            ),
            jinns.loss.laplacian_rev(
                t_x,
                lambda t_x, params: u_nonstatio(t_x, params)[1],
                params,
                eq_type="nonstatio_PDE",
            ),
            jinns.loss.laplacian_rev(
                t_x,
                lambda t_x, params: u_nonstatio(t_x, params)[2],
                params,
                eq_type="nonstatio_PDE",
            ),
            jinns.loss.laplacian_rev(
                t_x,
                lambda t_x, params: u_nonstatio(t_x, params)[3],
                params,
                eq_type="nonstatio_PDE",
            ),
        ]
    )
    assert jnp.allclose(
        jinns.loss.vectorial_laplacian_rev(t_x, u_nonstatio, params, dim_out=4), vec_lap
    )
