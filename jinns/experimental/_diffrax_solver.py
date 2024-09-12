"""
This is originally based on the diffrax tutorial which indicates how to solve PDEs
using the line method
"""

from typing import Callable
import matplotlib.pyplot as plt
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from jax import vmap
from mpl_toolkits.axes_grid1 import ImageGrid


class SpatialDiscretisation(eqx.Module):
    """A class inspired and adpated from the diffrax tutorial on non linear
    heat PDE which indicates how to solve PDE with diffrax.

    It is an equinox module storing the values on a 2-dimensional grid, along
    with the x and y limits of the grid. Several binary operator such as +, x,
    - are defined to allow for algebraic manipulation of the class.

    Source : https://docs.kidger.site/diffrax/examples/nonlinear_heat_pde/
    """

    xmin: float = eqx.field(static=True)
    xmax: float = eqx.field(static=True)
    ymin: float = eqx.field(static=True)
    ymax: float = eqx.field(static=True)
    vals: Float[Array, "nx ny"]

    @classmethod
    def discretise_fn(
        cls,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
        nx: int,
        ny: int,
        fn: Callable,
    ):
        """
        fn : must be a function with separated inputs f(x, y)
        """

        if nx < 2:
            raise ValueError("Must discretise [xmin, xmax] into at least two points")
        elif ny < 2:
            raise ValueError("Must discretise [ymin, ymax] into at least two points")

        xv, yv = jnp.meshgrid(
            jnp.linspace(xmin, xmax, nx), jnp.linspace(ymin, ymax, ny), indexing="ij"
        )
        vals = vmap(fn)(jnp.vstack([xv.flatten(), yv.flatten()]).T).reshape(xv.shape)
        return cls(xmin, xmax, ymin, ymax, vals)

    @property
    def δx(self):
        return (self.xmax - self.xmin) / (self.vals.shape[0] - 1)

    @property
    def δy(self):
        return (self.ymax - self.ymin) / (self.vals.shape[1] - 1)

    def binop(self, other, fn):
        if isinstance(other, SpatialDiscretisation):
            if self.xmin != other.xmin or self.xmax != other.xmax:
                raise ValueError("Mismatched x-axis spatial discretisations")
            if self.ymin != other.ymin or self.ymax != other.ymax:
                raise ValueError("Mismatched y-axis spatial discretisations")
            other = other.vals
        return SpatialDiscretisation(
            self.xmin, self.xmax, self.ymin, self.ymax, fn(self.vals, other)
        )

    def __add__(self, other):
        return self.binop(other, lambda x, y: x + y)

    def __mul__(self, other):
        return self.binop(other, lambda x, y: x * y)

    def __radd__(self, other):
        return self.binop(other, lambda x, y: y + x)

    def __rmul__(self, other):
        return self.binop(other, lambda x, y: y * x)

    def __sub__(self, other):
        return self.binop(other, lambda x, y: x - y)

    def __rsub__(self, other):
        return self.binop(other, lambda x, y: y - x)


def reaction_diffusion_2d_vector_field(
    t, y: SpatialDiscretisation, args
) -> SpatialDiscretisation:
    r"""
    Matrix stencil implementation of the reaction-diffusion equation using
    finite differences. See e.g. Section 2.6.1 of
    <https://hplgit.github.io/fdm-book/doc/pub/book/pdf/fdm-book-4screen.pdf>

    The reaction-diffusion equation is

    $$
        \partial_t u = D \Delta u + r u (1-u).
    $$
    In diffrax, the `vector_field` is the derivative of the function y with
    respect to time `t` in the ODE $y'(t) = F(y(t), t)$. Here it does not
    depend on $t$.
    """
    D, r = args

    nx, ny = y.vals.shape
    dA = jnp.ones(nx - 1)
    A = jnp.diag(dA, -1) - 2 * jnp.eye(nx) + jnp.diag(dA, 1)

    dB = jnp.ones(ny - 1)
    B = jnp.diag(dB, -1) - 2 * jnp.eye(ny) + jnp.diag(dB, 1)

    # Neumann conditions (corresponding to central finite difference)
    A = A.at[0, 0].set(-1)
    A = A.at[-1, -1].set(-1)
    B = B.at[0, 0].set(-1)
    B = B.at[-1, -1].set(-1)

    A = A / y.δx**2
    B = B / y.δy**2

    # transpose because he works with a (ny, nx) mesh
    step = D * (B @ y.vals.T + y.vals.T @ A) + r * y.vals.T * (1 - y.vals.T)
    return SpatialDiscretisation(y.xmin, y.xmax, y.ymin, y.ymax, step.T)


def laplacian(y: SpatialDiscretisation) -> SpatialDiscretisation:
    """
    NOT USED. Alternative implementation of the discrete laplacian
    """

    dx2, dy2 = y.δx**2, y.δy**2

    lap_kernel = jnp.array([[0, dy2, 0], [dx2, -2 * (dx2 + dy2), dx2], [0, dy2, 0]])

    lap_y = jax.scipy.signal.convolve(y.vals, lap_kernel, mode="same") / (dx2 * dy2)
    y_vals = lap_y

    return SpatialDiscretisation(y.xmin, y.xmax, y.ymin, y.ymax, y_vals)


def dirichlet_boundary_condition(y: SpatialDiscretisation) -> SpatialDiscretisation:
    """
    NOT USED. Dirichlet boundary condition
    """
    y_vals = y.vals.at[0, :].set(jnp.zeros_like(y.vals[0, :]))
    y_vals = y_vals.at[-1, :].set(jnp.zeros_like(y_vals[-1, :]))
    y_vals = y_vals.at[:, 0].set(jnp.zeros_like(y_vals[:, 0]))
    y_vals = y_vals.at[:, -1].set(jnp.zeros_like(y_vals[:, -1]))
    return SpatialDiscretisation(y.xmin, y.xmax, y.ymin, y.ymax, y_vals)


def neumann_boundary_condition(y: SpatialDiscretisation) -> SpatialDiscretisation:
    """
    NOT USED. Neumann boundary condition
    """
    y_vals = y.vals.at[0, :].set(y.vals[1, :])
    y_vals = y_vals.at[-1, :].set(y_vals[-2, :])
    y_vals = y_vals.at[:, 0].set(y_vals[:, 1])
    y_vals = y_vals.at[:, -1].set(y_vals[:, -2])
    return SpatialDiscretisation(y.xmin, y.xmax, y.ymin, y.ymax, y_vals)


def plot_diffrax_solution(diffrax_sol, xbounds, ybounds, t_ind=None, nplot=None):
    """
    Plot a 2D diffrax solution at selected times
    """
    if t_ind is None and nplot is None:
        raise ValueError("At least one of t_ind or nplot must not be None")

    if t_ind is None:
        t_ind = jnp.floor(jnp.linspace(0, len(diffrax_sol.ts), nplot)).astype(int)

    nplot = len(t_ind)

    fig = plt.figure(figsize=(20, 20 * nplot))
    grid = ImageGrid(
        fig,
        111,
        nrows_ncols=(1, nplot),
        axes_pad=0.2,
        share_all=True,
        cbar_location="bottom",
        cbar_mode="each",
        cbar_size="7%",
        cbar_pad=0.4,
    )

    (xmin, xmax) = xbounds
    (ymin, ymax) = ybounds
    for i, ax in enumerate(grid):
        ti = t_ind[i]
        im = ax.imshow(
            diffrax_sol.ys.vals[ti, :, :].T,
            origin="lower",
            extent=(xmin, xmax, ymin, ymax),
            aspect=(xmin - xmax) / (ymin - ymax),
            cmap="inferno",
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y", rotation=0)
        ax.set_title(f"Solution with t={diffrax_sol.ts[ti]:.2f}")
        ax.cax.colorbar(im, format="%0.2f")
    plt.show()
