"""
Test script for custom PINN eqx.Module
"""

import pytest
import jax
import jax.random as random
import jax.numpy as jnp
import equinox as eqx

import jinns
from jinns.utils import create_PINN


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
    u_statio = create_PINN(subkey, eqx_list, "ODE")

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
    u_statio = create_PINN(subkey, eqx_list, "statio_PDE", d)

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
    u_nonstatio = create_PINN(subkey, eqx_list, "nonstatio_PDE", d)

    return u_nonstatio


@pytest.fixture
def create_pinn_nonstatio_shared_output():
    key = random.PRNGKey(2)
    eqx_list = (
        (eqx.nn.Linear, d, 128),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 128, 1),
    )
    key, subkey = random.split(key)
    # specific argument since we want to have u1 and u2 as separate nns
    shared_pinn_output = (
        jnp.s_[:2],
        jnp.s_[2],
    )

    u1, u2 = jinns.utils.create_PINN(
        subkey,
        eqx_list,
        "nonstatio_PDE",
        d,
        shared_pinn_outputs=shared_pinn_output,
    )
    return u1, u2, shared_pinn_output


def test_ode_pinn_struct(create_pinn_ode):

    u_ode = create_pinn_ode
    assert u_ode.eq_type == "ODE"
    assert isinstance(u_ode, jinns.utils._pinn.PINN)
    assert u_ode.output_slice is None
    assert isinstance(u_ode.slice_solution, slice)
    _ = u_ode.init_params()


def test_statio_pinn_struct(create_pinn_statio):

    u_statio = create_pinn_statio
    assert u_statio.eq_type == "statio_PDE"
    assert isinstance(u_statio, jinns.utils._pinn.PINN)

    assert u_statio.output_slice is None
    assert isinstance(u_statio.slice_solution, slice)
    _ = u_statio.init_params()


def test_nonstatio_pinn_struct(create_pinn_nonstatio):

    u_nonstatio = create_pinn_nonstatio
    assert u_nonstatio.eq_type == "nonstatio_PDE"
    assert isinstance(u_nonstatio, jinns.utils._pinn.PINN)
    assert u_nonstatio.output_slice is None
    assert isinstance(u_nonstatio.slice_solution, slice)
    _ = u_nonstatio.init_params()


def test_nonstatio_pinn_shared_output(create_pinn_nonstatio_shared_output):

    u1, u2, shared_pinn_ouput = create_pinn_nonstatio_shared_output
    assert u1.eq_type == "nonstatio_PDE"
    assert u1.output_slice == shared_pinn_ouput[0]
    assert isinstance(u1.slice_solution, slice)

    assert u2.eq_type == "nonstatio_PDE"
    assert u2.output_slice == shared_pinn_ouput[1]
    assert isinstance(u2.slice_solution, slice)

    param1 = u1.init_params()
    param2 = u2.init_params()

    # the init parameters for the 2 nns should be the same PyTree
    assert eqx.tree_equal(param1, param2)
