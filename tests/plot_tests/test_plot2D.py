"""
Test script for plot functionalities for functions `f(t,x)` when x is 2D.
"""

import pytest
import jax.numpy as jnp
import jinns
import jinns.parameters
import jinns.plot
import matplotlib.pyplot as plt
import matplotlib as mpl

import jinns.utils

u_statio = lambda t_x: 0
u_nonstatio = lambda x: 0
xmin = ymin = tmin = -1
xmax = ymax = tmax = 1
nx = ny = ntime = 3

val_xy_data = [jnp.linspace(xmin, xmax, nx), jnp.linspace(ymin, ymax, ny)]
val_times = jnp.linspace(tmin, tmax, ntime)


def test_plot2d_statio():

    jinns.plot.plot2d(
        fun=u_statio,
        xy_data=val_xy_data,
    )


def test_plot2d_statio_with_spinn():
    import equinox as eqx
    import jax
    from functools import partial

    d = 2
    r = 1
    eqx_list = ((eqx.nn.Linear, 1, r),)  # don't reproduce this architecture at home ;)

    key, subkey = jax.random.split(jax.random.PRNGKey(1))
    u_spinn, init_nn_params_spinn = jinns.nn.SPINN_MLP.create(
        subkey, d, r, eqx_list, "statio_PDE"
    )

    jinns.plot.plot2d(
        fun=lambda x: u_spinn(x, init_nn_params_spinn),
        xy_data=val_xy_data,
        cmap="viridis",
        spinn=True,
    )


def test_plot2d_nonstatio():
    jinns.plot.plot2d(fun=u_nonstatio, xy_data=val_xy_data, times=val_times)


def test_plot2d_nonstatio_raise_error_for_wrong_xy_data():

    with pytest.raises(ValueError):
        # xy_data of len !=2
        jinns.plot.plot2d(fun=u_nonstatio, xy_data=[1, 2, 3], times=val_times)

        # xy_data not a list
        jinns.plot.plot2d(fun=u_nonstatio, xy_data=(1, 2), times=val_times)


def test_plot2d_nonstatio_raise_error_for_wrong_times():

    with pytest.raises(ValueError):
        jinns.plot.plot2d(fun=u_nonstatio, xy_data=val_xy_data, times=(1, 2))
