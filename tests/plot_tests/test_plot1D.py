"""
Test script for plot functionalities for functions `f(t,x)` when x is 1D.
"""

import jax.numpy as jnp
import jinns
import jinns.parameters
import jinns.plot
import matplotlib.pyplot as plt
import matplotlib as mpl

import jinns.utils

u = lambda t_x: 0
xmin = tmin = -1
xmax = tmax = 1
nx = ntime = 20

val_xdata = jnp.linspace(xmin, xmax, nx)
val_times = jnp.linspace(tmin, tmax, ntime)


def test_plot1d_image():

    fig, ax = jinns.plot.plot1d_image(
        fun=u,
        xdata=val_xdata,
        times=val_times,
        cmap="viridis",
        colorbar=True,
        figsize=(5, 5),
        title="u(t, x)",
    )

    if not isinstance(fig, mpl.figure.Figure):
        raise ValueError("plot1d_image should return a mpl Figure object on first arg")

    if not isinstance(ax, mpl.axes.Axes):
        raise ValueError("plot1d_image should return a mpl Axes object on second arg")


def test_plot1d_slice_without_predefined_ax():
    time_slices = [0, 0.2]
    ax = jinns.plot.plot1d_slice(
        u, xdata=val_xdata, time_slices=time_slices, figsize=(5, 5)
    )

    if not isinstance(ax, mpl.axes.Axes):
        raise ValueError("plot1d_slice should return a mpl Axes object only.")


def test_plot1d_slice_with_predefined_ax():
    fig, ax = plt.subplots(1, 1)
    time_slices = [0, 0.2]
    new_ax = jinns.plot.plot1d_slice(
        u, xdata=val_xdata, time_slices=time_slices, figsize=(5, 5), ax=ax
    )

    if ax is not new_ax:
        raise ValueError(
            "If passed an Axes object, plot1d_slice should return the same Axes object."
        )


def test_plot1d_slice_with_None_times():
    jinns.plot.plot1d_slice(u, xdata=val_xdata, time_slices=None)


def test_plot1d_slice_with_spinn():
    import equinox as eqx
    import jax
    from functools import partial

    d = 2
    r = 1
    eqx_list = ((eqx.nn.Linear, 1, r),)  # don't reproduce this architecture at home ;)

    key, subkey = jax.random.split(jax.random.PRNGKey(1))
    u_spinn, init_nn_params_spinn = jinns.nn.SPINN_MLP.create(
        subkey, d, r, eqx_list, "nonstatio_PDE"
    )

    jinns.plot.plot1d_slice(
        partial(u_spinn, params=init_nn_params_spinn),
        xdata=val_xdata,
        time_slices=[0, 1],
        spinn=True,
    )
