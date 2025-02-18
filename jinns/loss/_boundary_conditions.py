"""
Implements the main boundary conditions for all kinds of losses in jinns
"""

from __future__ import (
    annotations,
)  # https://docs.python.org/3/library/typing.html#constant

from typing import TYPE_CHECKING, Callable
import jax
import jax.numpy as jnp
from jax import vmap, grad
import equinox as eqx
from jinns.utils._utils import get_grid, _subtract_with_check
from jinns.data._Batchs import *
from jinns.nn._pinn import PINN
from jinns.nn._spinn import SPINN

if TYPE_CHECKING:
    from jinns.utils._types import *


def _compute_boundary_loss(
    boundary_condition_type: str,
    f: Callable[
        [Float[Array, "dim"] | Float[Array, "dim + 1"]], Float[Array, "dim_solution"]
    ],
    batch: PDEStatioBatch | PDENonStatioBatch,
    u: eqx.Module,
    params: AnyParams,
    facet: int,
    dim_to_apply: slice,
    vmap_in_axes: tuple,
) -> float:
    r"""A generic function that will compute the mini-batch MSE of a
    boundary condition in the stationary case, resp. non-stationary, given by:

    $$
        D[u](\partial x) = f(\partial x), \forall \partial x \in \partial \Omega
    $$
    resp.,

    $$
        D[u](t, \partial x) = f(\partial x), \forall t \in I, \forall \partial
        x \in \partial \Omega
    $$

    Where $D[\cdot]$ is a differential operator, possibly identity.

    **Note**: if using a batch.param_batch_dict, we need to resolve the
    vmapping axes in the boundary functions,  however params["eq_params"]
    has already been fed with the batch in the `evaluate()` of `LossPDEStatio`,
    resp. `LossPDENonStatio`.

    Parameters
    ----------
    boundary_condition_type
        a string defining the differential operator $D[\cdot]$.
        Currently implements one of "Dirichlet" ($D = Id$) and Von
        Neuman ($D[u] = \nabla u \cdot n$) where $n$ is the
        unitary outgoing vector normal to $\partial\Omega$
    f
        the function to be matched in the boundary condition. It should have
        one argument only (for `t`, `x` or `t_x`) (other are ignored).
    batch
        the batch
    u
        a PINN
    params
        Params or ParamsDict
    facet
        An integer which represents the id of the facet which is currently
        considered (in the order provided by the DataGenerator which is fixed)
    dim_to_apply
        A `jnp.s_` object which indicates which dimension(s) of u will be forced
        to match the boundary condition
    vmap_in_axes
        A tuple object which specifies the in_axes of the vmapping

    Returns
    -------
    scalar
        the MSE computed on `batch`
    """
    if boundary_condition_type.lower() in "dirichlet":
        mse = boundary_dirichlet(f, batch, u, params, facet, dim_to_apply, vmap_in_axes)
    elif any(
        boundary_condition_type.lower() in s
        for s in ["von neumann", "vn", "vonneumann"]
    ):
        mse = boundary_neumann(f, batch, u, params, facet, dim_to_apply, vmap_in_axes)
    else:
        raise ValueError("Wrong type of initial condition")
    return mse


def boundary_dirichlet(
    f: Callable[
        [Float[Array, "dim"] | Float[Array, "dim + 1"]], Float[Array, "dim_solution"]
    ],
    batch: PDEStatioBatch | PDENonStatioBatch,
    u: eqx.Module,
    params: Params | ParamsDict,
    facet: int,
    dim_to_apply: slice,
    vmap_in_axes: tuple,
) -> float:
    r"""
    This omega boundary condition enforces a solution that is equal to `f`
    at `times_batch` x `omega_border` (non stationary case) or at `omega_border`
    (stationary case)

    __Note__: if using a batch.param_batch_dict, we need to resolve the
    vmapping axes here however params["eq_params"] has already been fed with
    the batch in the `evaluate()` of `LossPDE*`.

    Parameters
    ----------
    f
        the constraint function
    batch
        The batch
    u
        The PINN or SPINN
    params
        The dictionary of parameters of the model.
        Typically, it is a dictionary of
        dictionaries: `eq_params` and `nn_params`, respectively the
        differential equation parameters and the neural network parameter
    facet:
        An integer which represents the id of the facet which is currently
        considered (in the order provided wy the DataGenerator which is fixed)
    dim_to_apply
        A jnp.s\_ object. The dimension of u on which to apply the boundary condition
    vmap_in_axes
        A tuple object which specifies the in_axes of the vmapping
    """
    batch_array = batch.border_batch
    batch_array = batch_array[..., facet]

    if isinstance(u, PINN):
        v_u_boundary = vmap(
            lambda inputs, params: _subtract_with_check(
                f(inputs),
                u(
                    inputs,
                    params,
                )[dim_to_apply],
                cause="boundary condition fun",
            ),
            vmap_in_axes,
            0,
        )
        res = v_u_boundary(batch_array, params)
        mse_u_boundary = jnp.sum(
            res**2,
            axis=-1,
        )
    elif isinstance(u, SPINN):
        values = u(batch_array, params)[..., dim_to_apply]
        grid = get_grid(batch_array)
        res = _subtract_with_check(f(grid), values, cause="boundary condition fun")
        mse_u_boundary = jnp.sum(res**2, axis=-1)
    else:
        raise ValueError(f"Bad type for u. Got {type(u)}, expected PINN or SPINN")
    return mse_u_boundary


def boundary_neumann(
    f: Callable[
        [Float[Array, "dim"] | Float[Array, "dim + 1"]], Float[Array, "dim_solution"]
    ],
    batch: PDEStatioBatch | PDENonStatioBatch,
    u: eqx.Module,
    params: Params | ParamsDict,
    facet: int,
    dim_to_apply: slice,
    vmap_in_axes: tuple,
) -> float:
    r"""
    This omega boundary condition enforces a solution where $\nabla u\cdot
    n$ is equal to `f` at the cartesian product of `time_batch` x `omega
    borders`. $n$ is the unitary outgoing vector normal at border
    $\partial\Omega$.

    __Note__: if using a batch.param_batch_dict, we need to resolve the
    vmapping axes here however params["eq_params"] has already been fed with
    the batch in the `evaluate()` of `LossPDENonStatio`.

    Parameters
    ----------
    f:
        the constraint function
    batch
        The batch
    u
        The PINN
    params
        The dictionary of parameters of the model.
        Typically, it is a dictionary of
        dictionaries: `eq_params` and `nn_params``, respectively the
        differential equation parameters and the neural network parameter
    facet:
        An integer which represents the id of the facet which is currently
        considered (in the order provided wy the DataGenerator which is fixed)
    dim_to_apply
        A jnp.s\_ object. The dimension of u on which to apply the boundary condition
    vmap_in_axes
        A tuple object which specifies the in_axes of the vmapping
    """
    batch_array = batch.border_batch
    batch_array = batch_array[..., facet]

    # We resort to the shape of the border_batch to determine the dimension as
    # described in the border_batch function
    if jnp.squeeze(batch_array).ndim == 0:  # case 1D borders (just a scalar)
        n = jnp.array([1, -1])  # the unit vectors normal to the two borders

    else:  # case 2D borders (because 3D borders are not supported yet)
        # they are in the order: left, right, bottom, top so we give the normal
        # outgoing vectors accordingly with shape in concordance with
        # border_batch shape (batch_size, ndim, nfacets)
        n = jnp.array([[-1, 1, 0, 0], [0, 0, -1, 1]])

    if isinstance(u, PINN):

        u_ = lambda inputs, params: jnp.squeeze(u(inputs, params)[dim_to_apply])

        if u.eq_type == "statio_PDE":
            v_neumann = vmap(
                lambda inputs, params: _subtract_with_check(
                    f(inputs),
                    jnp.dot(
                        grad(u_, 0)(inputs, params),
                        n[..., facet],
                    ),
                    cause="boundary condition fun",
                ),
                vmap_in_axes,
                0,
            )
        elif u.eq_type == "nonstatio_PDE":
            v_neumann = vmap(
                lambda inputs, params: _subtract_with_check(
                    f(inputs),
                    jnp.dot(
                        grad(u_, 0)(inputs, params)[1:],  # get rid of time dim
                        n[..., facet],
                    ),
                    cause="boundary condition fun",
                ),
                vmap_in_axes,
                0,
            )
        else:
            raise ValueError("Wrong u.eq_type")
        mse_u_boundary = jnp.sum(
            (
                v_neumann(
                    batch_array,
                    params,
                )
            )
            ** 2,
            axis=-1,
        )

    elif isinstance(u, SPINN):
        # the gradient we see in the PINN case can get gradients wrt to x
        # dimensions at once. But it would be very inefficient in SPINN because
        # of the high dim output of u. So we do 2 explicit forward AD, handling all the
        # high dim output at once
        if (batch_array.shape[0] == 1 and isinstance(batch, PDEStatioBatch)) or (
            batch_array.shape[-1] == 2 and isinstance(batch, PDENonStatioBatch)
        ):
            if u.eq_type == "statio_PDE":
                _, du_dx = jax.jvp(
                    lambda inputs: u(inputs, params)[..., dim_to_apply],
                    (batch_array,),
                    (jnp.ones_like(batch_array),),
                )
                values = du_dx * n[facet]
            if u.eq_type == "nonstatio_PDE":
                _, du_dx = jax.jvp(
                    lambda inputs: u(inputs, params)[..., dim_to_apply],
                    (batch_array,),
                    (jnp.ones_like(batch_array),),
                )
                values = du_dx[..., 1] * n[facet]
        elif (batch_array.shape[-1] == 2 and isinstance(batch, PDEStatioBatch)) or (
            batch_array.shape[-1] == 3 and isinstance(batch, PDENonStatioBatch)
        ):
            if u.eq_type == "statio_PDE":
                tangent_vec_0 = jnp.repeat(
                    jnp.array([1.0, 0.0])[None], batch_array.shape[0], axis=0
                )
                tangent_vec_1 = jnp.repeat(
                    jnp.array([0.0, 1.0])[None], batch_array.shape[0], axis=0
                )
                _, du_dx1 = jax.jvp(
                    lambda inputs: u(inputs, params)[..., dim_to_apply],
                    (batch_array,),
                    (tangent_vec_0,),
                )
                _, du_dx2 = jax.jvp(
                    lambda inputs: u(inputs, params)[..., dim_to_apply],
                    (batch_array,),
                    (tangent_vec_1,),
                )
                values = du_dx1 * n[0, facet] + du_dx2 * n[1, facet]  # dot product
            if u.eq_type == "nonstatio_PDE":
                tangent_vec_0 = jnp.repeat(
                    jnp.array([0.0, 1.0, 0.0])[None], batch_array.shape[0], axis=0
                )
                tangent_vec_1 = jnp.repeat(
                    jnp.array([0.0, 0.0, 1.0])[None], batch_array.shape[0], axis=0
                )
                _, du_dx1 = jax.jvp(
                    lambda inputs: u(inputs, params)[..., dim_to_apply],
                    (batch_array,),
                    (tangent_vec_0,),
                )
                _, du_dx2 = jax.jvp(
                    lambda inputs: u(inputs, params)[..., dim_to_apply],
                    (batch_array,),
                    (tangent_vec_1,),
                )
                values = (
                    du_dx1.squeeze() * n[0, facet] + du_dx2.squeeze() * n[1, facet]
                )  # dot product
        else:
            raise ValueError("Not implemented, we'll do that with a loop")

        grid = get_grid(batch_array)
        res = _subtract_with_check(f(grid), values, cause="boundary condition fun")
        mse_u_boundary = jnp.sum(
            res**2,
            axis=-1,
        )
    else:
        raise ValueError(f"Bad type for u. Got {type(u)}, expected PINN or SPINN")
    return mse_u_boundary
