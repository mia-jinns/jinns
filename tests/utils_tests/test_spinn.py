"""
Test script for custom PINN eqx.Module
"""

import pytest
import jax
import jax.random as random
import jax.numpy as jnp
import equinox as eqx

import jinns
from jinns.utils import create_SPINN
import jinns.utils


d = 5
r = 100  # embedding dim
m = 1  # output dim
eqx_list = (
    (eqx.nn.Linear, 1, 128),
    (jax.nn.tanh,),
    (eqx.nn.Linear, 128, 128),
    (jax.nn.tanh,),
    (eqx.nn.Linear, 128, 128),
    (jax.nn.tanh,),
    (eqx.nn.Linear, 128, r * m),
)


def _assert_attr_equal(u):
    assert u.m == m
    assert u.r == r


@pytest.fixture
def create_SPINN_ode():
    key = random.PRNGKey(2)
    key, subkey = random.split(key)
    u_statio = create_SPINN(subkey, 1, r, eqx_list, "ODE", m)

    return u_statio


@pytest.fixture
def create_SPINN_statio():
    key = random.PRNGKey(2)
    key, subkey = random.split(key)
    u_statio = create_SPINN(subkey, d, r, eqx_list, "statio_PDE", m)

    return u_statio


@pytest.fixture
def create_SPINN_nonstatio():
    key = random.PRNGKey(2)
    key, subkey = random.split(key)
    u_nonstatio = create_SPINN(subkey, d, r, eqx_list, "nonstatio_PDE", m)

    return u_nonstatio


def test_ode_pinn_struct(create_SPINN_ode):

    u_ode = create_SPINN_ode
    assert u_ode.eq_type == "ODE"
    assert isinstance(u_ode, jinns.utils._spinn.SPINN)
    assert u_ode.d == 1
    _assert_attr_equal(u_ode)
    _ = u_ode.init_params()


def test_statio_pinn_struct(create_SPINN_statio):

    u_statio = create_SPINN_statio
    assert u_statio.eq_type == "statio_PDE"
    assert isinstance(u_statio, jinns.utils._spinn.SPINN)

    assert u_statio.d == d
    _assert_attr_equal(u_statio)
    _ = u_statio.init_params()


def test_nonstatio_pinn_struct(create_SPINN_nonstatio):

    u_nonstatio = create_SPINN_nonstatio
    assert u_nonstatio.eq_type == "nonstatio_PDE"
    assert isinstance(u_nonstatio, jinns.utils._spinn.SPINN)
    assert u_nonstatio.d == d  # in non-statio SPINN user should include `t` in `d`
    _assert_attr_equal(u_nonstatio)
    _ = u_nonstatio.init_params()


def test_raising_error_init_SPINN():

    # output_dim != r*m
    with pytest.raises(ValueError) as e:
        wrong_eqx_list = [
            [eqx.nn.Linear, 1, 128],
            [jax.nn.tanh],
            [eqx.nn.Linear, 128, r * m + 1],  # output_dim != r*m
        ]
        _ = create_SPINN(random.PRNGKey(1), d, r, wrong_eqx_list, "nonstatio_PDE", m)

    # d > 24
    with pytest.raises(ValueError) as e:
        _ = create_SPINN(random.PRNGKey(1), 24, r, wrong_eqx_list, "nonstatio_PDE", m)

    # d > 24
    with pytest.raises(ValueError) as e:
        _ = create_SPINN(random.PRNGKey(1), 24, r, wrong_eqx_list, "nonstatio_PDE", m)
