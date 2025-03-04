"""
Test script for custom PINN eqx.Module
"""

import pytest
import jax
import jax.random as random
import jax.numpy as jnp
import equinox as eqx

import jinns
from jinns.nn import PINN_MLP


d = 5


@pytest.fixture
def create_pinn_ode():
    key = random.PRNGKey(2)
    eqx_list = (
        (eqx.nn.Linear, 1, 128),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 128, 1),
    )
    key, subkey = random.split(key)
    u_statio = PINN_MLP.create(key=subkey, eqx_list=eqx_list, eq_type="ODE")[0]

    return u_statio


@pytest.fixture
def create_pinn_statio():
    key = random.PRNGKey(2)
    eqx_list = (
        (eqx.nn.Linear, d, 128),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 128, 1),
    )
    key, subkey = random.split(key)
    u_statio = PINN_MLP.create(key=subkey, eqx_list=eqx_list, eq_type="statio_PDE")[0]

    return u_statio


@pytest.fixture
def create_pinn_nonstatio():
    key = random.PRNGKey(2)
    eqx_list = (
        (eqx.nn.Linear, d, 128),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 128, 1),
    )
    key, subkey = random.split(key)
    u_nonstatio = PINN_MLP.create(
        key=subkey, eqx_list=eqx_list, eq_type="nonstatio_PDE"
    )[0]

    return u_nonstatio


def test_ode_pinn_struct(create_pinn_ode):

    u_ode = create_pinn_ode
    assert u_ode.eq_type == "ODE"
    assert isinstance(u_ode, jinns.nn.PINN)
    assert isinstance(u_ode.slice_solution, slice)


def test_statio_pinn_struct(create_pinn_statio):

    u_statio = create_pinn_statio
    assert u_statio.eq_type == "statio_PDE"
    assert isinstance(u_statio, jinns.nn.PINN)
    print(u_statio.slice_solution)
    assert isinstance(u_statio.slice_solution, slice)


def test_nonstatio_pinn_struct(create_pinn_nonstatio):

    u_nonstatio = create_pinn_nonstatio
    assert u_nonstatio.eq_type == "nonstatio_PDE"
    assert isinstance(u_nonstatio, jinns.nn.PINN)
    assert isinstance(u_nonstatio.slice_solution, slice)
