"""
This is an adaptation of the diffrax tutorial which indicates how to solve PDEs

Currently implements the resolution of a Fisher KPP problem
"""

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


def laplacian(y: SpatialDiscretisation, von_neumann: bool) -> SpatialDiscretisation:
    dx2, dy2 = y.δx**2, y.δy**2

    lap_kernel = jnp.array([[0, dy2, 0], [dx2, -2 * (dx2 + dy2), dx2], [0, dy2, 0]])

    if von_neumann:
        y_extended = jnp.pad(y.vals, pad_width=1)
        y_extended = y_extended.at[0, :].set(y_extended[2, :])
        y_extended = y_extended.at[-1, :].set(y_extended[-3, :])
        y_extended = y_extended.at[:, 0].set(y_extended[:, 2])
        y_extended = y_extended.at[:, -1].set(y_extended[:, -3])
        Δy = (
            jax.scipy.signal.convolve(y_extended, lap_kernel, mode="same") / (dx2 * dy2)
        )[1:-1, 1:-1]

    else:
        Δy = y.vals
        Δy = jax.scipy.signal.convolve(Δy, lap_kernel, mode="same") / (dx2 * dy2)

    return SpatialDiscretisation(y.xmin, y.xmax, y.ymin, y.ymax, Δy)


def dirichlet_boundary_condition(y: SpatialDiscretisation) -> SpatialDiscretisation:
    y_vals = y.vals.at[0, :].set(jnp.zeros_like(y.vals[0, :]))
    y_vals = y_vals.at[-1, :].set(jnp.zeros_like(y_vals[-1, :]))
    y_vals = y_vals.at[:, 0].set(jnp.zeros_like(y_vals[:, 0]))
    y_vals = y_vals.at[:, -1].set(jnp.zeros_like(y_vals[:, -1]))
    return SpatialDiscretisation(y.xmin, y.xmax, y.ymin, y.ymax, y.vals)


def diffrax_solver(pde_control, vector_field):
    # Prepare ODE object on discretized problem
    term = diffrax.ODETerm(vector_field)

    # Solve the ODE
    u_sol = diffrax.diffeqsolve(term, **pde_control["ode_hyperparams"])

    return u_sol
