"""
Utility functions for plotting
"""

from functools import partial
import warnings
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import vmap
from mpl_toolkits.axes_grid1 import ImageGrid


def plot2d(
    fun,
    xy_data,
    times=None,
    Tmax=1,
    title="",
    figsize=(7, 7),
    cmap="inferno",
    spinn=False,
    vmin_vmax=None,
    ax_for_plot=None,
):
    r"""Generic function for plotting functions over rectangular 2-D domains
    :math:`\Omega`. It treats both the stationary case :math:`u(x)` or the
    non-stationnary case :math:`u(t, x)`.

    When in the non-stationnary case, the `times` argument gives the time
    slices :math:`t_i` at which to plot :math:`u(t_i, x)`.


    Parameters
    ----------
    fun : _type_
        _description_
    xy_data : _type_
        _description_
    times : _type_, optional
        _description_, by default None
    Tmax : float, only in non-stationary cases
        Useful if you used time re-scaling in the differential equation, by
        default 1
    title : str, optional
        plot title, by default ""
    figsize : tuple, optional
        _description_, by default (7, 7)
    cmap : str, optional
        _description_, by default "inferno"
    vmin_vmax : tuple, optional
        The colorbar minimum and maximum value. Defaults None.
    ax_for_plot : Matplotlib axis, optional
        If None, jinns triggers the plotting. Otherwise this argument
        corresponds to the axis which will host the plot. Default is None.
        NOTE: that this argument will have an effect only if times is None.

    Raises
    ------
    ValueError
        if xy_data is not a list of type 2
    """

    # if not isinstance(xy_data, jnp.ndarray) and not xy_data.shape[-1] == 2:
    if not isinstance(xy_data, list) and not len(xy_data) == 2:
        raise ValueError(
            "xy_data must be a list of length 2 containing"
            "jnp.array of shape (nx,) and (ny,)."
        )
    mesh = jnp.meshgrid(xy_data[0], xy_data[1])  # cartesian product

    if times is None:
        # Statio case : expect a function of one argument fun(x)
        if not spinn:
            v_fun = vmap(fun, 0, 0)
            ret = _plot_2D_statio(
                v_fun,
                mesh,
                plot=not ax_for_plot,
                colorbar=True,
                cmap=cmap,
                figsize=figsize,
                vmin_vmax=vmin_vmax,
            )
        elif spinn:
            values_grid = jnp.squeeze(
                fun(jnp.stack([xy_data[0][..., None], xy_data[1][..., None]], axis=1))
            )
            ret = _plot_2D_statio(
                values_grid,
                mesh,
                plot=not ax_for_plot,
                colorbar=True,
                cmap=cmap,
                spinn=True,
                figsize=figsize,
                vmin_vmax=vmin_vmax,
            )
        if not ax_for_plot:
            plt.title(title)
        else:
            if vmin_vmax is not None:
                im = ax_for_plot.pcolormesh(
                    mesh[0],
                    mesh[1],
                    ret[0],
                    cmap=cmap,
                    vmin=vmin_vmax[0],
                    vmax=vmin_vmax[1],
                )
            else:
                im = ax_for_plot.pcolormesh(mesh[0], mesh[1], ret[0], cmap=cmap)
            ax_for_plot.set_title(title)
            ax_for_plot.cax.colorbar(im, format="%0.2f")

    else:
        if ax_for_plot is not None:
            warnings.warn("ax_for_plot is ignored. jinns will plot the figure")
        if not isinstance(times, list):
            try:
                times = times.tolist()
            except:
                raise ValueError("times must be a list or an array")

        fig = plt.figure(figsize=figsize)
        grid = ImageGrid(
            fig,
            111,
            nrows_ncols=(1, len(times)),
            axes_pad=0.2,
            share_all=True,
            cbar_location="bottom",
            cbar_mode="each",
            cbar_size="7%",
            cbar_pad=0.4,
        )

        for idx, (t, ax) in enumerate(zip(times, grid)):
            if not spinn:
                v_fun_at_t = vmap(lambda x: fun(t=jnp.array([t]), x=x), 0, 0)
                t_slice, _ = _plot_2D_statio(
                    v_fun_at_t,
                    mesh,
                    plot=False,
                    colorbar=False,
                    cmap=None,
                    vmin_vmax=vmin_vmax,
                )
            elif spinn:
                values_grid = jnp.squeeze(
                    fun(
                        t * jnp.ones((xy_data[0].shape[0], 1)),
                        jnp.stack(
                            [xy_data[0][..., None], xy_data[1][..., None]], axis=1
                        ),
                    )[0]
                )
                t_slice, _ = _plot_2D_statio(
                    values_grid,
                    mesh,
                    plot=False,
                    colorbar=True,
                    spinn=True,
                    vmin_vmax=vmin_vmax,
                )
            if vmin_vmax is not None:
                im = ax.pcolormesh(
                    mesh[0],
                    mesh[1],
                    t_slice,
                    cmap=cmap,
                    vmin=vmin_vmax[0],
                    vmax=vmin_vmax[1],
                )
            else:
                im = ax.pcolormesh(mesh[0], mesh[1], t_slice, cmap=cmap)
            ax.set_title(f"t = {times[idx] * Tmax:.2f}")
            ax.cax.colorbar(im, format="%0.2f")


def _plot_2D_statio(
    v_fun,
    mesh,
    plot=True,
    colorbar=True,
    cmap="inferno",
    figsize=(7, 7),
    spinn=False,
    vmin_vmax=None,
):
    """Function that plot the function u(x) with 2-D input x using pcolormesh()


    Parameters
    ----------
    v_fun :
        a vmapped function over jnp.array of shape (*, 2)
    mesh :
        a tuple of size 2, containing the x and y meshgrid.
    plot : bool, optional
        either show or return the plot, by default True
    colorbar : bool, optional
        add a colorbar, by default True
    vmin_vmax: tuple, optional
        The colorbar minimum and maximum value. Defaults None.

    Returns
    -------
    Either None or the values of u() over the meshgrid and the current plt axis

    """

    x_grid, y_grid = mesh
    if not spinn:
        values = v_fun(jnp.vstack([x_grid.flatten(), y_grid.flatten()]).T)
        values_grid = values.reshape(x_grid.shape)
    elif spinn:
        # in this case v_fun is directly the values :)
        values_grid = v_fun.T

    if plot:
        fig = plt.figure(figsize=figsize)
        if vmin_vmax is not None:
            im = plt.pcolormesh(
                x_grid,
                y_grid,
                values_grid,
                cmap=cmap,
                vmin=vmin_vmax[0],
                vmax=vmin_vmax[1],
            )
        else:
            im = plt.pcolormesh(x_grid, y_grid, values_grid, cmap=cmap)
        if colorbar:
            fig.colorbar(im, format="%0.2f")
        # don't plt.show() because it is done in plot2d()
    else:
        return values_grid, plt.gca()


def plot1d_slice(
    fun,
    xdata,
    time_slices=jnp.array([0]),
    Tmax=1,
    title="",
    figsize=(10, 10),
    spinn=False,
):
    """Function for plotting time slices of a function :math:`f(t_i, x)` where
    `t` is time (1-D) and x is 1-D

    Parameters
    ----------
    fun : callable with two arguments `t` and `x`
        f(t, x)
    xdata : jnp.array
        the discretization of space
    time_slices : list, optional
        the time slices :math:`t_i` at which to plot, by default [0]
    Tmax : int, optional
        Useful if you used time re-scaling in the differential equation, by
        default 1
    title : str, optional
        title of the plot, by default ""
    figsize : tuple, optional
        size of the figure, by default (10, 10)
    """
    plt.figure(figsize=figsize)
    for t in time_slices:
        if not spinn:
            # fix t with partial : shape is (1,)
            v_u_tfixed = vmap(partial(fun, t=t * jnp.ones((1,))), 0, 0)
            # add an axis to xdata for the concatenate function in the neural net
            values = v_u_tfixed(x=xdata[:, None])
        elif spinn:
            values = jnp.squeeze(
                fun(t * jnp.ones((xdata.shape[0], 1)), xdata[..., None])[0]
            )
        plt.plot(xdata, values, label=f"$t_i={t * Tmax:.2f}$")
    plt.xlabel("x")
    plt.ylabel(r"$u(t_i, x)$")
    plt.legend()
    plt.title(title)


def plot1d_image(
    fun,
    xdata,
    times,
    Tmax=1,
    title="",
    figsize=(10, 10),
    colorbar=True,
    cmap="inferno",
    spinn=False,
    vmin_vmax=None,
):
    """Function for plotting the 2-D image of a function :math:`f(t, x)` where
    `t` is time (1-D) and x is space (1-D).

    The function f is plotted on a meshgrid with plt.pcolormesh().

    Parameters
    ----------
    fun : callable with two arguments t and x
        the function to plot
    xdata : jnp.array
        the discretization of space
    times : jnp.array
        the discretization of time
    Tmax : int, optional
        _description_, by default 1
    title : str, optional
        , by default ""
    figsize : tuple, optional
        , by default (10, 10)
    vmin_vmax: tuple
        The colorbar minimum and maximum value. Defaults None.
    """

    mesh = jnp.meshgrid(times, xdata)  # cartesian product
    if not spinn:
        # the trick is to use _plot2Dstatio
        v_fun = vmap(lambda tx: fun(t=tx[0, None], x=tx[1, None]), 0, 0)
        t_grid, x_grid = mesh
        values_grid = v_fun(jnp.vstack([t_grid.flatten(), x_grid.flatten()]).T).reshape(
            t_grid.shape
        )
    elif spinn:
        values_grid = jnp.squeeze(fun((times[..., None]), xdata[..., None]).T)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    if vmin_vmax is not None:
        im = ax.pcolormesh(
            mesh[0] * Tmax,
            mesh[1],
            values_grid,
            cmap=cmap,
            vmin=vmin_vmax[0],
            vmax=vmin_vmax[1],
        )
    else:
        im = ax.pcolormesh(mesh[0] * Tmax, mesh[1], values_grid, cmap=cmap)
    if colorbar:
        fig.colorbar(im, format="%0.2f")
    ax.set_title(title)
