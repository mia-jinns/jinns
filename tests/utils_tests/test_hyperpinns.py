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
    u_ode = jinns.utils.create_HYPERPINN(
        subkey, eqx_list, "ODE", hyperparams, n_param, 0, eqx_list_hyper
    )
    return u_ode


@pytest.fixture
def create_pinn_statio():
    eqx_list = ((eqx.nn.Linear, d, 16),) + EQX_LIST
    u_statio = jinns.utils.create_HYPERPINN(
        subkey, eqx_list, "statio_PDE", hyperparams, n_param, d, eqx_list_hyper
    )
    return u_statio


@pytest.fixture
def create_pinn_nonstatio():
    eqx_list = ((eqx.nn.Linear, d + 1, 16),) + EQX_LIST
    u_nonstatio = jinns.utils.create_HYPERPINN(
        subkey, eqx_list, "nonstatio_PDE", hyperparams, n_param, d + 1, eqx_list_hyper
    )

    return u_nonstatio


@pytest.fixture
def create_pinn_nonstatio_shared_output():

    # specific argument since we want to have u1 and u2 as separate nns
    shared_pinn_output = (
        jnp.s_[:2],
        jnp.s_[2],
    )

    eqx_list = ((eqx.nn.Linear, d, 16),) + EQX_LIST
    u1, u2 = jinns.utils.create_HYPERPINN(
        subkey,
        eqx_list,
        "nonstatio_PDE",
        hyperparams,
        n_param,
        d,
        eqx_list_hyper,
        shared_pinn_outputs=shared_pinn_output,
    )
    return u1, u2, shared_pinn_output


def test_ode_pinn_struct(create_pinn_ode):

    u_ode = create_pinn_ode
    assert isinstance(u_ode, jinns.utils._hyperpinn.HYPERPINN)
    assert u_ode.eq_type == "ODE"
    assert u_ode.output_slice is None
    assert isinstance(u_ode.slice_solution, slice)
    _ = u_ode.init_params()


def test_statio_pinn_struct(create_pinn_statio):

    u_statio = create_pinn_statio
    assert u_statio.eq_type == "statio_PDE"
    assert isinstance(u_statio, jinns.utils._hyperpinn.HYPERPINN)
    assert u_statio.output_slice is None
    assert isinstance(u_statio.slice_solution, slice)
    _ = u_statio.init_params()


def test_nonstatio_pinn_struct(create_pinn_nonstatio):

    u_nonstatio = create_pinn_nonstatio
    assert u_nonstatio.eq_type == "nonstatio_PDE"
    assert isinstance(u_nonstatio, jinns.utils._hyperpinn.HYPERPINN)
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
