from typing import Callable

import diffrax
import equinox as eqx  # https://github.com/patrick-kidger/equinox
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float  # https://github.com/google/jaxtyping
from jax import vmap
from functools import partial
import numpy as np


class SpatialDiscretisation(eqx.Module):
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

        # WARNING: meshgrid transposes by default, xv and yv are (ny, nx).
        # Set indexing="ij" for (nx, ny) shape
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


def laplacian(y: SpatialDiscretisation) -> SpatialDiscretisation:
    dx2, dy2 = y.δx**2, y.δy**2

    # Write discrete laplacian as a convolution
    lap_kernel = jnp.array([[0, dy2, 0], [dx2, -2 * (dx2 + dy2), dx2], [0, dy2, 0]])
    Δy = jax.scipy.signal.convolve(y.vals, lap_kernel, mode="same") / (dx2 * dy2)

    # Dirichlet boundary condition
    #     Δy = Δy.at[0, :].set(0)
    #     Δy = Δy.at[-1, :].set(0)
    #     Δy = Δy.at[:, 0].set(0)
    #     Δy = Δy.at[:, -1].set(0)

    # Neumann boundary condition
    Δy = Δy.at[0, :].set(Δy[1, :])
    Δy = Δy.at[-1, :].set(Δy[-2, :])
    Δy = Δy.at[:, 0].set(Δy[:, 1])
    Δy = Δy.at[:, -1].set(Δy[:, -2])

    return SpatialDiscretisation(y.xmin, y.xmax, y.ymin, y.ymax, Δy)


def r_func(x, y, rs):
    r1, r2, r3, r4 = rs
    if x > 6 / 20 and x < 8 / 20:
        return r4
    if y > 8 / 20 and y < 10 / 20:
        return r4
    if (x - 0.15) ** 2 + (y - 0.15) ** 2 < 0.015:
        return r1
    if (x - 0.8) ** 2 + (y - 0.80) ** 2 < 0.03:
        return r1
    if y > 0.4:
        return r3
    return r2


def _create_r_mat(rs):
    xv, yv = jnp.meshgrid(
        jnp.linspace(xmin, xmax, nx), jnp.linspace(ymin, ymax, ny), indexing="ij"
    )
    r_mat = np.vectorize(partial(r_func, rs=rs))(xv, yv)
    r_mat = jnp.array(r_mat)


def diffrax_solver(eq_params, pde_control):
    D = eq_params["D"]
    gamma = eq_params["gamma"]
    rs = eq_params["rs"]

    xmin, xmax = pde_control["xboundary"]
    ymin, ymax = pde_control["yboundary"]
    nx = pde_control["nx"]
    ny = pde_control["ny"]

    def _create_r_mat(rs):
        xv, yv = jnp.meshgrid(
            jnp.linspace(xmin, xmax, nx), jnp.linspace(ymin, ymax, ny), indexing="ij"
        )
        r_mat = np.vectorize(partial(r_func, rs=rs))(xv, yv)
        r_mat = jnp.array(r_mat)
        return r_mat

    # compute new r_mat with given rs params
    r_mat = _create_r_mat(rs)

    # define the vector field of the discretized RD PDE with parameters $D$ and $\gamma$
    def vector_field(t, y, args):
        return D * laplacian(y) + y * (r_mat - gamma * y)

    # Prepare ODE object on discretized problem
    term = diffrax.ODETerm(vector_field)

    # Solve the ODE
    u_sol = diffrax.diffeqsolve(term, **pde_control["ode_hyperparams"])

    return u_sol
