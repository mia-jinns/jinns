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
from jinns.utils._utils import (
    _get_grid,
    _check_user_func_return,
)
from jinns.parameters._params import _get_vmap_in_axes_params
from jinns.data._Batchs import *
from jinns.utils._pinn import PINN
from jinns.utils._spinn import SPINN

if TYPE_CHECKING:
    from jinns.utils._types import *


def _compute_boundary_loss(
    boundary_condition_type: str,
    f: Callable,
    batch: PDEStatioBatch | PDENonStatioBatch,
    u: eqx.Module,
    params: Params | ParamsDict,
    facet: int,
    dim_to_apply: slice,
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
        one or two arguments only (other are ignored).
    batch
        a PDEStatioBatch or PDENonStatioBatch
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

    Returns
    -------
    scalar
        the MSE computed on `batch`
    """
    mse = None
    if isinstance(batch, PDEStatioBatch):
        if boundary_condition_type.lower() in "dirichlet":
            mse = boundary_dirichlet_statio(f, batch, u, params, facet, dim_to_apply)
        elif any(
            boundary_condition_type.lower() in s
            for s in ["von neumann", "vn", "vonneumann"]
        ):
            mse = boundary_neumann_statio(f, batch, u, params, facet, dim_to_apply)
    elif isinstance(batch, PDENonStatioBatch):
        if boundary_condition_type.lower() in "dirichlet":
            mse = boundary_dirichlet_nonstatio(f, batch, u, params, facet, dim_to_apply)
        elif any(
            boundary_condition_type.lower() in s
            for s in ["von neumann", "vn", "vonneumann"]
        ):
            mse = boundary_neumann_nonstatio(f, batch, u, params, facet, dim_to_apply)
    else:
        raise ValueError("Wrong type of batch")
    return mse


def boundary_dirichlet_statio(
    f: Callable,
    batch: PDEStatioBatch,
    u: eqx.Module,
    params: Params | ParamsDict,
    facet: int,
    dim_to_apply: slice,
) -> float:
    r"""
    This omega boundary condition enforces a solution that is equal to f on
    border batch.

    __Note__: if using a batch.param_batch_dict, we need to resolve the
    vmapping axes here however params["eq_params"] has already been fed with
    the batch in the `evaluate()` of `LossPDEStatio`.

    Parameters
    ----------
    f
        the constraint function
    batch
        A PDEStatioBatch object.
    u
        The PINN
    params
        Params or ParamsDict
    dim_to_apply
        A jnp.s\_ object. The dimension of u on which to apply the boundary condition
    """
    _, border_batch = batch.inside_batch, batch.border_batch
    border_batch = border_batch[..., facet]

    if isinstance(u, PINN):
        vmap_in_axes_params = _get_vmap_in_axes_params(batch.param_batch_dict, params)
        vmap_in_axes_x = (0,)

        v_u_boundary = vmap(
            lambda dx, params: u(dx, params)[dim_to_apply] - f(dx),
            vmap_in_axes_x + vmap_in_axes_params,
            0,
        )

        mse_u_boundary = jnp.sum((v_u_boundary(border_batch, params)) ** 2, axis=-1)
    elif isinstance(u, SPINN):
        values = u(border_batch, params)[..., dim_to_apply]
        x_grid = _get_grid(border_batch)
        boundaries = _check_user_func_return(f(x_grid), values.shape)
        res = values - boundaries
        mse_u_boundary = jnp.sum(
            res**2,
            axis=-1,
        )
    else:
        raise ValueError(f"Bad type for u. Got {type(u)}, expected PINN or SPINN")
    return mse_u_boundary


def boundary_neumann_statio(
    f: Callable,
    batch: PDEStatioBatch,
    u: eqx.Module,
    params: Params | ParamsDict,
    facet: int,
    dim_to_apply: slice,
) -> float:
    r"""
    This omega boundary condition enforces a solution where $\nabla u\cdot
    n$ is equal to `f` on omega borders. $n$ is the unitary
    outgoing vector normal at border $\partial\Omega$.

    __Note__: if using a batch.param_batch_dict, we need to resolve the
    vmapping axes here however params["eq_params"] has already been fed with
    the batch in the `evaluate()` of `LossPDEStatio`.

    Parameters
    ----------
    f
        the constraint function
    batch
        A PDEStatioBatch object.
    u
        The PINN
    params
        The dictionary of parameters of the model.
        Typically, it is a dictionary of
        dictionaries: `eq_params` and `nn_params``, respectively the
        differential equation parameters and the neural network parameter
    facet
        An integer which represents the id of the facet which is currently
        considered (in the order provided wy the DataGenerator which is fixed)
    dim_to_apply
        A jnp.s\_ object. The dimension of u on which to apply the boundary condition
    """
    _, border_batch = batch.inside_batch, batch.border_batch
    border_batch = border_batch[..., facet]

    # We resort to the shape of the border_batch to determine the dimension as
    # described in the border_batch function
    if jnp.squeeze(border_batch).ndim == 0:  # case 1D borders (just a scalar)
        n = jnp.array([1, -1])  # the unit vectors normal to the two borders

    else:  # case 2D borders (because 3D borders are not supported yet)
        # they are in the order: left, right, bottom, top so we give the normal
        # outgoing vectors accordingly with shape in concordance with
        # border_batch shape (batch_size, ndim, nfacets)
        n = jnp.array([[-1, 1, 0, 0], [0, 0, -1, 1]])

    if isinstance(u, PINN):
        u_ = lambda x, params: jnp.squeeze(u(x, params)[dim_to_apply])

        vmap_in_axes_params = _get_vmap_in_axes_params(batch.param_batch_dict, params)
        vmap_in_axes_x = (0,)

        v_neumann = vmap(
            lambda dx, params: jnp.dot(
                grad(u_, 0)(dx, params),
                n[..., facet],
            )
            - f(dx),
            vmap_in_axes_x + vmap_in_axes_params,
            0,
        )
        mse_u_boundary = jnp.sum((v_neumann(border_batch, params)) ** 2, axis=-1)
    elif isinstance(u, SPINN):
        # the gradient we see in the PINN case can get gradients wrt to x
        # dimensions at once. But it would be very inefficient in SPINN because
        # of the high dim output of u. So we do 2 explicit forward AD, handling all the
        # high dim output at once
        if border_batch.shape[0] == 1:  # i.e. case 1D
            _, du_dx = jax.jvp(
                lambda x: u(
                    x,
                    params,
                )[..., dim_to_apply],
                (border_batch,),
                (jnp.ones_like(border_batch),),
            )
            values = du_dx * n[facet]
        elif border_batch.shape[-1] == 2:
            tangent_vec_0 = jnp.repeat(
                jnp.array([1.0, 0.0])[None], border_batch.shape[0], axis=0
            )
            tangent_vec_1 = jnp.repeat(
                jnp.array([0.0, 1.0])[None], border_batch.shape[0], axis=0
            )
            _, du_dx1 = jax.jvp(
                lambda x: u(
                    x,
                    params,
                ),
                (border_batch,),
                (tangent_vec_0,),
            )
            _, du_dx2 = jax.jvp(
                lambda x: u(
                    x,
                    params,
                ),
                (border_batch,),
                (tangent_vec_1,),
            )
            values = du_dx1 * n[0, facet] + du_dx2 * n[1, facet]  # dot product
            # explicitly written
        else:
            raise ValueError("Not implemented, we'll do that with a loop")

        x_grid = _get_grid(border_batch)
        boundaries = _check_user_func_return(f(x_grid), values.shape)
        res = values - boundaries
        mse_u_boundary = jnp.sum(res**2, axis=-1)
    else:
        raise ValueError(f"Bad type for u. Got {type(u)}, expected PINN or SPINN")
    return mse_u_boundary


def boundary_dirichlet_nonstatio(
    f: Callable,
    batch: PDENonStatioBatch,
    u: eqx.Module,
    params: Params | ParamsDict,
    facet: int,
    dim_to_apply: slice,
) -> float:
    r"""
    This omega boundary condition enforces a solution that is equal to `f`
    at `times_batch` x `omega borders`

    __Note__: if using a batch.param_batch_dict, we need to resolve the
    vmapping axes here however params["eq_params"] has already been fed with
    the batch in the `evaluate()` of `LossPDENonStatio`.

    Parameters
    ----------
    f
        the constraint function
    batch
        A PDENonStatioBatch object.
    u
        The PINN
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
    """
    times_batch = batch.times_x_border_batch[:, 0:1, facet]
    omega_border_batch = batch.times_x_border_batch[:, 1:, facet]

    if isinstance(u, PINN):
        vmap_in_axes_params = _get_vmap_in_axes_params(batch.param_batch_dict, params)
        vmap_in_axes_x_t = (0, 0)

        v_u_boundary = vmap(
            lambda t, dx, params: u(
                t,
                dx,
                params,
            )[dim_to_apply]
            - f(t, dx),
            vmap_in_axes_x_t + vmap_in_axes_params,
            0,
        )
        res = v_u_boundary(times_batch, omega_border_batch, params)
        mse_u_boundary = jnp.sum(
            res**2,
            axis=-1,
        )
    elif isinstance(u, SPINN):
        values = u(times_batch, omega_border_batch, params)[..., dim_to_apply]
        tx_grid = _get_grid(jnp.concatenate([times_batch, omega_border_batch], axis=-1))
        boundaries = _check_user_func_return(
            f(tx_grid[..., 0:1], tx_grid[..., 1:]), values.shape
        )
        res = values - boundaries
        mse_u_boundary = jnp.sum(res**2, axis=-1)
    else:
        raise ValueError(f"Bad type for u. Got {type(u)}, expected PINN or SPINN")
    return mse_u_boundary


def boundary_neumann_nonstatio(
    f: Callable,
    batch: PDENonStatioBatch,
    u: eqx.Module,
    params: Params | ParamsDict,
    facet: int,
    dim_to_apply: slice,
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
        A PDENonStatioBatch object.
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
    """
    times_batch = batch.times_x_border_batch[:, 0:1, facet]
    omega_border_batch = batch.times_x_border_batch[:, 1:, facet]

    # We resort to the shape of the border_batch to determine the dimension as
    # described in the border_batch function
    if jnp.squeeze(omega_border_batch).ndim == 0:  # case 1D borders (just a scalar)
        n = jnp.array([1, -1])  # the unit vectors normal to the two borders

    else:  # case 2D borders (because 3D borders are not supported yet)
        # they are in the order: left, right, bottom, top so we give the normal
        # outgoing vectors accordingly with shape in concordance with
        # border_batch shape (batch_size, ndim, nfacets)
        n = jnp.array([[-1, 1, 0, 0], [0, 0, -1, 1]])

    if isinstance(u, PINN):
        vmap_in_axes_params = _get_vmap_in_axes_params(batch.param_batch_dict, params)
        vmap_in_axes_x_t = (0, 0)

        u_ = lambda t, x, params: jnp.squeeze(u(t, x, params)[dim_to_apply])
        v_neumann = vmap(
            lambda t, dx, params: jnp.dot(
                grad(u_, 1)(t, dx, params),
                n[..., facet],
            )
            - f(t, dx),
            vmap_in_axes_x_t + vmap_in_axes_params,
            0,
        )
        mse_u_boundary = jnp.sum(
            (
                v_neumann(
                    times_batch,
                    omega_border_batch,
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
        if omega_border_batch.shape[0] == 1:  # i.e. case 1D
            _, du_dx = jax.jvp(
                lambda x: u(times_batch, x, params)[..., dim_to_apply],
                (omega_border_batch,),
                (jnp.ones_like(omega_border_batch),),
            )
            values = du_dx * n[facet]
        elif omega_border_batch.shape[-1] == 2:
            tangent_vec_0 = jnp.repeat(
                jnp.array([1.0, 0.0])[None], omega_border_batch.shape[0], axis=0
            )
            tangent_vec_1 = jnp.repeat(
                jnp.array([0.0, 1.0])[None], omega_border_batch.shape[0], axis=0
            )
            _, du_dx1 = jax.jvp(
                lambda x: u(times_batch, x, params)[..., dim_to_apply],
                (omega_border_batch,),
                (tangent_vec_0,),
            )
            _, du_dx2 = jax.jvp(
                lambda x: u(times_batch, x, params)[..., dim_to_apply],
                (omega_border_batch,),
                (tangent_vec_1,),
            )
            values = du_dx1 * n[0, facet] + du_dx2 * n[1, facet]  # dot product
            # explicitly written
        else:
            raise ValueError("Not implemented, we'll do that with a loop")

        tx_grid = _get_grid(jnp.concatenate([times_batch, omega_border_batch], axis=-1))
        boundaries = _check_user_func_return(
            f(tx_grid[..., 0:1], tx_grid[..., 1:]), values.shape
        )
        res = values - boundaries
        mse_u_boundary = jnp.sum(
            res**2,
            axis=-1,
        )
    else:
        raise ValueError(f"Bad type for u. Got {type(u)}, expected PINN or SPINN")
    return mse_u_boundary
