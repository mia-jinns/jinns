"""
Utility functions for plotting in 1D and 2D, with and without time.
"""

from functools import partial
import warnings
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import vmap
from mpl_toolkits.axes_grid1 import ImageGrid
from typing import Callable, List
from jaxtyping import Array, Float, Bool


def plot2d(
    fun: Callable,
    xy_data: tuple[Float[Array, "nx"], Float[Array, "ny"]],
    times: Float[Array, "nt"] | List[float] | None = None,
    Tmax: float = 1,
    title: str = "",
    figsize: tuple = (7, 7),
    cmap: str = "inferno",
    spinn: bool = False,
    vmin_vmax: tuple[float, float] = [None, None],
):
    r"""Generic function for plotting functions over rectangular 2-D domains
    $\Omega$. It handles both the

     1. the stationary case $u(x)$
     2. the non-stationnary case $u(t, x)$

    In the non-stationnary case, the `times` argument gives the time
    slices $t_i$ at which to plot $u(t_i, x)$.


    Parameters
    ----------
    fun :
        the function $u$ to plot on the meshgrid, and eventually the time
        slices. It's suppose to have signature `u(x)` in the stationnary case,, and `u(t, x)` in the non-stationnary case. Use `partial` or `lambda to freeze / reorder any other arguments.
    xy_data :
        A list of 2 `jnp.Array` providing grid values for meshgrid creation
    times :
        list or Array of time slices where to plot the function. Use Tmax if
        you trained with time-rescaling.
    Tmax :
        Useful if you used time rescaling in the differential equation for training, default to 1 (no rescaling).
    title :
        plot title, by default ""
    figsize :
        By default (7, 7)
    cmap :
        the matplotlib color map used in the ImageGrid.
    vmin_vmax :
        The colorbar minimum and maximum value. Defaults None.
    spinn :
        True if the function is a `SPINN` object.

    Raises
    ------
    ValueError
        if xy_data is not a list of length 2
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
                colorbar=True,
                cmap=cmap,
                figsize=figsize,
                vmin_vmax=vmin_vmax,
            )
        elif spinn:
            values_grid = jnp.squeeze(fun(jnp.stack([xy_data[0], xy_data[1]], axis=1)))
            ret = _plot_2D_statio(
                values_grid,
                mesh,
                colorbar=True,
                cmap=cmap,
                figsize=figsize,
                vmin_vmax=vmin_vmax,
            )
        else:
            fig, ax = plt.subplots(1, 1, figsize=figsize)

            im = ax.pcolormesh(
                mesh[0],
                mesh[1],
                ret[0],
                cmap=cmap,
                vmin=vmin_vmax[0],
                vmax=vmin_vmax[1],
            )

            ax.set_title(title)
            fig.cax.colorbar(im, format="%0.2f")

    else:
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
                x_grid, y_grid = mesh
                v_fun_at_t = vmap(fun)(
                    jnp.concatenate(
                        [
                            t
                            * jnp.ones((xy_data[0].shape[0] * xy_data[1].shape[0], 1)),
                            jnp.vstack([x_grid.flatten(), y_grid.flatten()]).T,
                        ],
                        axis=-1,
                    )
                )
                t_slice, _ = _plot_2D_statio(
                    v_fun_at_t,
                    mesh,
                    plot=False,  # only use to compute t_slice
                    colorbar=False,
                    cmap=None,
                    vmin_vmax=vmin_vmax,
                )
            elif spinn:
                t_x = jnp.concatenate(
                    [
                        t * jnp.ones((xy_data[0].shape[0], 1)),
                        jnp.concatenate(
                            [xy_data[0][..., None], xy_data[1][..., None]], axis=-1
                        ),
                    ],
                    axis=-1,
                )
                values_grid = jnp.squeeze(fun(t_x)[0]).T
                t_slice, _ = _plot_2D_statio(
                    values_grid,
                    mesh,
                    plot=False,  # only use to compute t_slice
                    colorbar=True,
                    vmin_vmax=vmin_vmax,
                )

            im = ax.pcolormesh(
                mesh[0],
                mesh[1],
                t_slice,
                cmap=cmap,
                vmin=vmin_vmax[0],
                vmax=vmin_vmax[1],
            )
            ax.set_title(f"t = {times[idx] * Tmax:.2f}")
            ax.cax.colorbar(im, format="%0.2f")


def _plot_2D_statio(
    v_fun: Callable | Float[Array, "(nx*ny)^2 1"],
    mesh: Float[Array, "nx*ny nx*ny"],
    plot: Bool = True,
    colorbar: Bool = True,
    cmap: str = "inferno",
    figsize: tuple[int, int] = (7, 7),
    vmin_vmax: tuple[float, float] = [None, None],
):
    """Function that plot the function u(x) with 2-D input x using pcolormesh()


    Parameters
    ----------
    v_fun :
        a vmapped function over jnp.array of shape (*, 2) OR a precomputed array of function values with shape compatible with `mesh`.
    mesh :
        a tuple of size 2, containing the x and y meshgrid.
    plot : bool, optional
        either displays the plot, or silently returns the grid of values `v_fun(mesh)`.
    colorbar : bool, optional
        add a colorbar, by default True
    cmap :
        the matplotlib color map used in the ImageGrid.
    figsize :
        By default (7, 7)
    spinn :
        True if a SPINN is to be plotted. False for PINNs and HyperPINNs
    vmin_vmax: list, optional
        The colorbar minimum and maximum value. Defaults None.

    Returns
    -------
    Either None or the values of u() over the meshgrid and the current plt axis

    """

    x_grid, y_grid = mesh
    if callable(v_fun):
        values = v_fun(jnp.vstack([x_grid.flatten(), y_grid.flatten()]).T)
        values_grid = values.reshape(x_grid.shape)
    else:
        values_grid = v_fun.reshape(x_grid.shape)

    if plot:
        fig = plt.figure(figsize=figsize)
        im = plt.pcolormesh(
            x_grid,
            y_grid,
            values_grid,
            cmap=cmap,
            vmin=vmin_vmax[0],
            vmax=vmin_vmax[1],
        )

        if colorbar:
            fig.colorbar(im, format="%0.2f")
        # don't plt.show() because it is done in plot2d()
    else:
        return values_grid, plt.gca()


def plot1d_slice(
    fun: Callable[[float, float], float],
    xdata: Float[Array, "nx"],
    time_slices: Float[Array, "nt"] | None = None,
    Tmax: float = 1.0,
    title: str = "",
    figsize: tuple[int, int] = (10, 10),
    spinn: Bool = False,
    ax=None,
):
    """Function for plotting time slices of a function :math:`f(t_i, x)` where
    `t_i` is time (1-D) and x is 1-D

    Parameters
    ----------
    fun
        f(t, x)
    xdata
        the discretization of space
    time_slices
        the time slices :math:`t_i` at which to plot.
    Tmax
        Useful if you used time re-scaling in the differential equation, by
        default 1
    title
        title of the plot, by default ""
    figsize
        size of the figure, by default (10, 10)
    spinn
        True if a SPINN is to be plotted. False for PINNs and HyperPINNs
    ax
        A pre-defined `matplotlib.Axes` where you want to plot.

    Returns
    -------
    ax
        A `matplotlib.Axes` object
    """
    if time_slices is None:
        time_slices = jnp.array([0])
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    for t in time_slices:
        t_xdata = jnp.concatenate(
            [t * jnp.ones((xdata.shape[0], 1)), xdata[:, None]], axis=1
        )
        if not spinn:
            # fix t with partial : shape is (1,)
            v_u_tfixed = vmap(fun)
            # add an axis to xdata for the concatenate function in the neural net
            values = v_u_tfixed(t_xdata)
        elif spinn:
            values = jnp.squeeze(fun(t_xdata)[0])
        ax.plot(xdata, values, label=f"$t_i={t * Tmax:.2f}$")
    ax.set_xlabel("x")
    ax.set_ylabel(r"$u(t_i, x)$")
    ax.legend()
    ax.set_title(title)
    return ax


def plot1d_image(
    fun: Callable[[float, float], float],
    xdata: Float[Array, "nx"],
    times: Float[Array, "nt"],
    Tmax: float = 1.0,
    title: str = "",
    figsize: tuple[int, int] = (10, 10),
    colorbar: Bool = True,
    cmap: str = "inferno",
    spinn: Bool = False,
    vmin_vmax: tuple[float, float] = [None, None],
):
    """Function for plotting the 2-D image of a function :math:`f(t, x)` where
    `t` is time (1-D) and x is space (1-D).

    The function f is plotted on a meshgrid with plt.pcolormesh().

    Parameters
    ----------
    fun :
        callable with two arguments t and x the function to plot
    xdata :
        the discretization of space
    times :
        the discretization of time
    Tmax :
        by default 1
    title :
        by default ""
    figsize :
        by default (10, 10)
    colorbar :
        Whether to add a colobar
    cmap :
        the matplotlib color map used in the ImageGrid.
    spinn :
        True if a SPINN is to be plotted. False for PINNs and HyperPINNs
    vmin_vmax:
        The colorbar minimum and maximum value. Defaults None.

    Returns
    -------
    fig, ax
        A `matplotlib` `Figure` and `Axes` objects with the figure.
    """

    mesh = jnp.meshgrid(times, xdata)  # cartesian product
    if not spinn:
        # the trick is to use _plot2Dstatio
        v_fun = vmap(fun)  # lambda tx: fun(t=tx[0, None], x=tx[1, None]), 0, 0)
        t_grid, x_grid = mesh
        values_grid = v_fun(jnp.vstack([t_grid.flatten(), x_grid.flatten()]).T).reshape(
            t_grid.shape
        )
    elif spinn:
        values_grid = jnp.squeeze(
            fun(jnp.concatenate([times[..., None], xdata[..., None]], axis=-1))
        ).T

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    im = ax.pcolormesh(
        mesh[0] * Tmax,
        mesh[1],
        values_grid,
        cmap=cmap,
        vmin=vmin_vmax[0],
        vmax=vmin_vmax[1],
    )

    if colorbar:
        fig.colorbar(im, format="%0.2f")
    ax.set_title(title)

    return fig, ax
