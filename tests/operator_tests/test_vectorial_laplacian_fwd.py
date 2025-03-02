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
        (eqx.nn.Linear, 20, 10 * 4),
    )
    key, subkey = jax.random.split(key)
    return jinns.nn.SPINN_MLP.create(subkey, 2, 10, eqx_list, "statio_PDE", 4)


def test_laplacian_fwd_statio(create_u_statio):
    u_statio, params = create_u_statio
    x = jnp.stack([jnp.array([0.4, 1.5]) for i in range(10)], axis=0)
    vec_lap = jnp.array(
        [
            jinns.loss.laplacian_fwd(
                x,
                lambda x, params: u_statio(x, params)[..., 0],
                params,
                eq_type="statio_PDE",
            ),
            jinns.loss.laplacian_fwd(
                x,
                lambda x, params: u_statio(x, params)[..., 1],
                params,
                eq_type="statio_PDE",
            ),
            jinns.loss.laplacian_fwd(
                x,
                lambda x, params: u_statio(x, params)[..., 2],
                params,
                eq_type="statio_PDE",
            ),
            jinns.loss.laplacian_fwd(
                x,
                lambda x, params: u_statio(x, params)[..., 3],
                params,
                eq_type="statio_PDE",
            ),
        ]
    )
    vec_lap = jnp.moveaxis(vec_lap, 0, -1)
    assert jnp.allclose(
        jinns.loss.vectorial_laplacian_fwd(x, u_statio, params, dim_out=4), vec_lap
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
        (eqx.nn.Linear, 20, 10 * 3),
    )
    key, subkey = jax.random.split(key)
    return jinns.nn.SPINN_MLP.create(subkey, 3, 10, eqx_list, "nonstatio_PDE", 3)


def test_laplacian_fwd_nonstatio(create_u_nonstatio):
    u_nonstatio, params = create_u_nonstatio
    t_x = jnp.stack([jnp.array([0.5, 0.4, 1.5]) for i in range(10)], axis=0)
    vec_lap = jnp.array(
        [
            jinns.loss.laplacian_fwd(
                t_x,
                lambda t_x, params: u_nonstatio(t_x, params)[..., 0],
                params,
                eq_type="nonstatio_PDE",
            ),
            jinns.loss.laplacian_fwd(
                t_x,
                lambda t_x, params: u_nonstatio(t_x, params)[..., 1],
                params,
                eq_type="nonstatio_PDE",
            ),
            jinns.loss.laplacian_fwd(
                t_x,
                lambda t_x, params: u_nonstatio(t_x, params)[..., 2],
                params,
                eq_type="nonstatio_PDE",
            ),
            jinns.loss.laplacian_fwd(
                t_x,
                lambda t_x, params: u_nonstatio(t_x, params)[..., 3],
                params,
                eq_type="nonstatio_PDE",
            ),
        ]
    )
    vec_lap = jnp.moveaxis(vec_lap, 0, -1)
    assert jnp.allclose(
        jinns.loss.vectorial_laplacian_fwd(t_x, u_nonstatio, params, dim_out=4), vec_lap
    )
