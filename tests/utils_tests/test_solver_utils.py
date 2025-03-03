"""Module to test for jinns.solve._utils.py helper functions

Mainly throws error for inconsistant batch sizes between DataGenerators
"""

import pytest
import jinns
import jax.numpy as jnp
import jax
import equinox as eqx
import jinns.data
from jinns.solver._utils import _check_batch_size

key = jax.random.PRNGKey(1)

tmin = xmin = ymin = -1.2
tmax = xmax = ymax = 2.3
nt = n = ni = 8
nb = 200  # large enough for not raising error cause of number of points/facet
batch_size = 8

n_obs = 11
obs_batch_size = 11
observed_values = jnp.arange(n_obs)
obs_data = jinns.data.DataGeneratorObservations(
    key=key,
    obs_batch_size=obs_batch_size,
    observed_values=observed_values,
    observed_pinn_in=observed_values,
)

n_param = 12
param_batch_size = 12
param_data = jinns.data.DataGeneratorParameter(
    keys=key,
    n=n_param,
    param_ranges={"param1": (-1, 1)},
    param_batch_size=param_batch_size,
)


def test_wrong_obs_batch_size_ode():
    dg_ode_with_batch = jinns.data.DataGeneratorODE(
        key=key, nt=nt, tmin=tmin, tmax=tmax, temporal_batch_size=batch_size
    )
    dg_ode_without_batch = jinns.data.DataGeneratorODE(
        key=key, nt=nt, tmin=tmin, tmax=tmax, temporal_batch_size=None
    )
    with pytest.raises(ValueError):

        # -- observations
        _check_batch_size(obs_data, dg_ode_with_batch, "obs_batch_size")  # 11 != 8
        _check_batch_size(obs_data, dg_ode_without_batch, "n")

        # -- parameters
        _check_batch_size(param_data, dg_ode_with_batch, "param_batch_size")  # 12 != 8
        _check_batch_size(param_data, dg_ode_without_batch, "n")


def test_wrong_batch_size_pde_statio():

    dg_pde_with_batch = jinns.data.CubicMeshPDEStatio(
        key=key,
        n=n,
        nb=nb,
        omega_batch_size=batch_size,
        omega_border_batch_size=batch_size,
        min_pts=((xmin, ymin)),
        max_pts=((xmax, ymax)),
        dim=2,
    )
    dg_pde_without_batch = jinns.data.CubicMeshPDEStatio(
        key=key,
        n=n,
        nb=nb,
        omega_batch_size=None,
        omega_border_batch_size=None,
        min_pts=((xmin, ymin)),
        max_pts=((xmax, ymax)),
        dim=2,
    )
    with pytest.raises(ValueError):

        # -- observations
        _check_batch_size(obs_data, dg_pde_with_batch, "obs_batch_size")  # 11 != 8
        _check_batch_size(obs_data, dg_pde_without_batch, "n")

        # -- parameters
        _check_batch_size(param_data, dg_pde_with_batch, "param_batch_size")  # 12 != 8
        _check_batch_size(param_data, dg_pde_without_batch, "n")

        # TODO: continue writing test with correct batch size for Omega but
        # not for omega_border and vice versa.

        dg_pde_with_batch = jinns.data.CubicMeshPDEStatio(
            key=key,
            n=n,
            nb=nb,
            omega_batch_size=param_batch_size,  # good
            omega_border_batch_size=batch_size,  # wrong
            min_pts=((xmin, ymin)),
            max_pts=((xmax, ymax)),
            dim=2,
        )
        dg_pde_without_batch = jinns.data.CubicMeshPDEStatio(
            key=key,
            n=n_param,  # good
            nb=nb,  # wrong
            omega_batch_size=None,
            omega_border_batch_size=None,
            min_pts=((xmin, ymin)),
            max_pts=((xmax, ymax)),
            dim=2,
        )

        # -- observations
        _check_batch_size(obs_data, dg_pde_with_batch, "obs_batch_size")  # 11 != 8
        _check_batch_size(obs_data, dg_pde_without_batch, "n")

        # -- parameters
        _check_batch_size(param_data, dg_pde_with_batch, "param_batch_size")  # 12 != 8
        _check_batch_size(param_data, dg_pde_without_batch, "n")

    correct_dg = jinns.data.CubicMeshPDEStatio(
        key=key,
        n=n,
        nb=nb,
        omega_batch_size=param_batch_size,
        omega_border_batch_size=param_batch_size,
        min_pts=((xmin, ymin)),
        max_pts=((xmax, ymax)),
        dim=2,
    )

    # this should not throw an error
    _check_batch_size(param_data, correct_dg, "param_batch_size")


def test_wrong_batch_size_pde_nonstatio():

    # TODO

    assert 1 == 1
