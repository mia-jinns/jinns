"""
Test script for custom PINN eqx.Module
"""

import pytest
import jax
import jax.random as random
import jax.numpy as jnp
import equinox as eqx

import jinns
from jinns.nn import HyperPINN
import jinns.utils

key = random.PRNGKey(2)
key, subkey = random.split(key)

d = 5
n_param = 42
hyperparams = [f"param {i}" for i in range(n_param)]

EQX_LIST = (
    (jax.nn.swish,),
    (eqx.nn.Linear, 16, 16),
    (jax.nn.swish,),
    (eqx.nn.Linear, 16, 16),
)

eqx_list_hyper = (
    (eqx.nn.Linear, n_param, 32),  # input is of size 42
    (jax.nn.tanh,),
    (eqx.nn.Linear, 32, 16),
    (jax.nn.tanh,),
    (
        eqx.nn.Linear,
        16,
        1000,
    ),  # 1000 is a random guess, it will automatically be filled with the correct value
)


@pytest.fixture
def create_pinn_ode():
    eqx_list = ((eqx.nn.Linear, 1, 16),) + EQX_LIST
    u_ode = HyperPINN.create(
        key=subkey,
        eqx_list=eqx_list,
        eq_type="ODE",
        hyperparams=hyperparams,
        hypernet_input_size=n_param,
        eqx_list_hyper=eqx_list_hyper,
    )[0]
    return u_ode


@pytest.fixture
def create_pinn_statio():
    eqx_list = ((eqx.nn.Linear, d, 16),) + EQX_LIST
    u_statio = HyperPINN.create(
        key=subkey,
        eqx_list=eqx_list,
        eq_type="statio_PDE",
        hyperparams=hyperparams,
        hypernet_input_size=n_param,
        eqx_list_hyper=eqx_list_hyper,
    )[0]
    return u_statio


@pytest.fixture
def create_pinn_nonstatio():
    eqx_list = ((eqx.nn.Linear, d + 1, 16),) + EQX_LIST
    u_nonstatio = HyperPINN.create(
        key=subkey,
        eqx_list=eqx_list,
        eq_type="nonstatio_PDE",
        hyperparams=hyperparams,
        hypernet_input_size=n_param,
        eqx_list_hyper=eqx_list_hyper,
    )[0]

    return u_nonstatio


def test_ode_pinn_struct(create_pinn_ode):

    u_ode = create_pinn_ode
    assert isinstance(u_ode, jinns.nn.PINN)
    assert isinstance(u_ode, jinns.nn.HyperPINN)
    assert u_ode.eq_type == "ODE"
    assert isinstance(u_ode.slice_solution, slice)


def test_statio_pinn_struct(create_pinn_statio):

    u_statio = create_pinn_statio
    assert u_statio.eq_type == "statio_PDE"
    assert isinstance(u_statio, jinns.nn.PINN)
    assert isinstance(u_statio, jinns.nn.HyperPINN)
    assert isinstance(u_statio.slice_solution, slice)


def test_nonstatio_pinn_struct(create_pinn_nonstatio):

    u_nonstatio = create_pinn_nonstatio
    assert u_nonstatio.eq_type == "nonstatio_PDE"
    assert isinstance(u_nonstatio, jinns.nn.PINN)
    assert isinstance(u_nonstatio, jinns.nn.HyperPINN)
    assert isinstance(u_nonstatio.slice_solution, slice)
