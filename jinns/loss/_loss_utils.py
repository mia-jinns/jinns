"""
Interface for diverse loss functions to factorize code
"""

from __future__ import (
    annotations,
)  # https://docs.python.org/3/library/typing.html#constant

from typing import TYPE_CHECKING, Callable, TypeGuard
from types import EllipsisType
import jax
import jax.numpy as jnp
from jax import vmap
from jaxtyping import Float, Array

from jinns.loss._boundary_conditions import (
    _compute_boundary_loss,
)
from jinns.utils._utils import _subtract_with_check, get_grid
from jinns.data._utils import make_cartesian_product
from jinns.parameters._params import _get_vmap_in_axes_params
from jinns.nn._pinn import PINN
from jinns.nn._spinn import SPINN
from jinns.nn._hyperpinn import HyperPINN
from jinns.data._Batchs import PDEStatioBatch, PDENonStatioBatch
from jinns.parameters._params import Params

if TYPE_CHECKING:
    from jinns.utils._types import BoundaryConditionFun
    from jinns.nn._abstract_pinn import AbstractPINN


def dynamic_loss_apply(
    dyn_loss: Callable,
    u: AbstractPINN,
    batch: (
        Float[Array, " batch_size 1"]
        | Float[Array, " batch_size dim"]
        | Float[Array, " batch_size 1+dim"]
    ),
    params: Params[Array],
    vmap_axes: tuple[int, Params[int | None] | None],
    u_type: PINN | HyperPINN | None = None,
) -> Float[Array, " "]:
    """
    Sometimes when u is a lambda function a or dict we do not have access to
    its type here, hence the last argument
    """
    if u_type == PINN or u_type == HyperPINN or isinstance(u, (PINN, HyperPINN)):
        v_dyn_loss = vmap(
            lambda batch, params: dyn_loss(
                batch,
                u,
                params,  # we must place the params at the end
            ),
            vmap_axes,
            0,
        )
        residuals = v_dyn_loss(batch, params)
        mse_dyn_loss = jnp.mean(jnp.sum(residuals**2, axis=-1))
    elif u_type == SPINN or isinstance(u, SPINN):
        residuals = dyn_loss(batch, u, params)
        mse_dyn_loss = jnp.mean(jnp.sum(residuals**2, axis=-1))
    else:
        raise ValueError(f"Bad type for u. Got {type(u)}, expected PINN or SPINN")

    return mse_dyn_loss


def normalization_loss_apply(
    u: AbstractPINN,
    batches: (
        tuple[Float[Array, " nb_norm_samples dim"]]
        | tuple[
            Float[Array, " nb_norm_time_slices 1"], Float[Array, " nb_norm_samples dim"]
        ]
    ),
    params: Params[Array],
    vmap_axes_params: tuple[Params[int | None] | None],
    norm_weights: Float[Array, " nb_norm_samples"],
) -> Float[Array, " "]:
    """
    Note the squeezing on each result. We expect unidimensional *PINN since
    they represent probability distributions
    """
    if isinstance(u, (PINN, HyperPINN)):
        if len(batches) == 1:
            v_u = vmap(
                lambda *b: u(*b)[u.slice_solution],
                (0,) + vmap_axes_params,
                0,
            )
            res = v_u(*batches, params)
            assert res.shape[-1] == 1, "norm loss expects unidimensional *PINN"
            # Monte-Carlo integration using importance sampling
            mse_norm_loss = jnp.abs(jnp.mean(res.squeeze() * norm_weights) - 1) ** 2
        else:
            # NOTE this cartesian product is costly
            batch_cart_prod = make_cartesian_product(
                batches[0],
                batches[1],
            ).reshape(batches[0].shape[0], batches[1].shape[0], -1)
            v_u = vmap(
                vmap(
                    lambda t_x, params_: u(t_x, params_),
                    in_axes=(0,) + vmap_axes_params,
                ),
                in_axes=(0,) + vmap_axes_params,
            )
            res = v_u(batch_cart_prod, params)
            assert res.shape[-1] == 1, "norm loss expects unidimensional *PINN"
            # For all times t, we perform an integration. Then we average the
            # losses over times.
            mse_norm_loss = jnp.mean(
                jnp.abs(jnp.mean(res.squeeze() * norm_weights, axis=-1) - 1) ** 2
            )
    elif isinstance(u, SPINN):
        if len(batches) == 1:
            res = u(*batches, params)
            assert res.shape[-1] == 1, "norm loss expects unidimensional *SPINN"
            mse_norm_loss = (
                jnp.abs(
                    jnp.mean(
                        res.squeeze(),
                    )
                    * norm_weights
                    - 1
                )
                ** 2
            )
        else:
            assert batches[1].shape[0] % batches[0].shape[0] == 0
            rep_t = batches[1].shape[0] // batches[0].shape[0]
            res = u(
                jnp.concatenate(
                    [jnp.repeat(batches[0], rep_t, axis=0), batches[1]], axis=-1
                ),
                params,
            )
            assert res.shape[-1] == 1, "norm loss expects unidimensional *SPINN"
            # the outer mean() below is for the times stamps
            mse_norm_loss = jnp.mean(
                jnp.abs(
                    jnp.mean(
                        res.squeeze(),
                        axis=list(d + 1 for d in range(res.ndim - 2)),
                    )
                    * norm_weights
                    - 1
                )
                ** 2
            )
    else:
        raise ValueError(f"Bad type for u. Got {type(u)}, expected PINN or SPINN")

    return mse_norm_loss


def boundary_condition_apply(
    u: AbstractPINN,
    batch: PDEStatioBatch | PDENonStatioBatch,
    params: Params[Array],
    omega_boundary_fun: BoundaryConditionFun | dict[str, BoundaryConditionFun],
    omega_boundary_condition: str | dict[str, str],
    omega_boundary_dim: slice | dict[str, slice],
) -> Float[Array, " "]:
    assert batch.border_batch is not None
    vmap_in_axes = (0,) + _get_vmap_in_axes_params(batch.param_batch_dict, params)

    def _check_tuple_of_dict(
        val,
    ) -> TypeGuard[
        tuple[
            dict[str, BoundaryConditionFun],
            dict[str, BoundaryConditionFun],
            dict[str, BoundaryConditionFun],
        ]
    ]:
        return all(isinstance(x, dict) for x in val)

    omega_boundary_dicts = (
        omega_boundary_condition,
        omega_boundary_fun,
        omega_boundary_dim,
    )
    if _check_tuple_of_dict(omega_boundary_dicts):
        # We must create the facet tree dictionary as we do not have the
        # enumerate from the for loop to pass the id integer
        if batch.border_batch.shape[-1] == 2:
            # 1D
            facet_tree = {"xmin": 0, "xmax": 1}
        elif batch.border_batch.shape[-1] == 4:
            # 2D
            facet_tree = {"xmin": 0, "xmax": 1, "ymin": 2, "ymax": 3}
        else:
            raise ValueError("Other border batches are not implemented")
        b_losses_by_facet = jax.tree_util.tree_map(
            lambda c, f, fa, d: (
                None
                if c is None
                else jnp.mean(
                    _compute_boundary_loss(c, f, batch, u, params, fa, d, vmap_in_axes)
                )
            ),
            omega_boundary_dicts[0],  # omega_boundary_condition,
            omega_boundary_dicts[1],  # omega_boundary_fun,
            facet_tree,
            omega_boundary_dicts[2],  # omega_boundary_dim,
            is_leaf=lambda x: x is None,
        )  # when exploring leaves with None value (no condition) the returned
        # mse is None and we get rid of the None leaves of b_losses_by_facet
        # with the tree_leaves below
        # Note that to keep the behaviour given in the comment above we neede
        # to specify is_leaf according to the note in the release of 0.4.29
    else:
        facet_tuple = tuple(f for f in range(batch.border_batch.shape[-1]))
        b_losses_by_facet = jax.tree_util.tree_map(
            lambda fa: jnp.mean(
                _compute_boundary_loss(
                    omega_boundary_dicts[0],  # type: ignore -> need TypeIs from 3.13
                    omega_boundary_dicts[1],  # type: ignore -> need TypeIs from 3.13
                    batch,
                    u,
                    params,
                    fa,
                    omega_boundary_dicts[2],  # type: ignore -> need TypeIs from 3.13
                    vmap_in_axes,
                )
            ),
            facet_tuple,
        )
    mse_boundary_loss = jax.tree_util.tree_reduce(
        lambda x, y: x + y, jax.tree_util.tree_leaves(b_losses_by_facet)
    )
    return mse_boundary_loss


def observations_loss_apply(
    u: AbstractPINN,
    batch: Float[Array, " obs_batch_size input_dim"],
    params: Params[Array],
    vmap_axes: tuple[int, Params[int | None] | None],
    observed_values: Float[Array, " obs_batch_size observation_dim"],
    obs_slice: EllipsisType | slice | None,
) -> Float[Array, " "]:
    if isinstance(u, (PINN, HyperPINN)):
        v_u = vmap(
            lambda *args: u(*args)[u.slice_solution],
            vmap_axes,
            0,
        )
        val = v_u(batch, params)[:, obs_slice]
        mse_observation_loss = jnp.mean(
            jnp.sum(
                _subtract_with_check(
                    observed_values, val, cause="user defined observed_values"
                )
                ** 2,
                axis=-1,
            )
        )
    elif isinstance(u, SPINN):
        raise RuntimeError("observation loss term not yet implemented for SPINNs")
    else:
        raise ValueError(f"Bad type for u. Got {type(u)}, expected PINN or SPINN")
    return mse_observation_loss


def initial_condition_apply(
    u: AbstractPINN,
    omega_batch: Float[Array, " dimension"],
    params: Params[Array],
    vmap_axes: tuple[int, Params[int | None] | None],
    initial_condition_fun: Callable,
    t0: Float[Array, " 1"],
) -> Float[Array, " "]:
    n = omega_batch.shape[0]
    t0_omega_batch = jnp.concatenate([t0 * jnp.ones((n, 1)), omega_batch], axis=1)
    if isinstance(u, (PINN, HyperPINN)):
        v_u_t0 = vmap(
            lambda t0_x, params: _subtract_with_check(
                initial_condition_fun(t0_x[1:]),
                u(t0_x, params),
                cause="Output of initial_condition_fun",
            ),
            vmap_axes,
            0,
        )
        res = v_u_t0(t0_omega_batch, params)  # NOTE take the tiled
        # omega_batch (ie omega_batch_) to have the same batch
        # dimension as params to be able to vmap.
        # Recall that by convention:
        # param_batch_dict = times_batch_size * omega_batch_size
        mse_initial_condition = jnp.mean(jnp.sum(res**2, axis=-1))
    elif isinstance(u, SPINN):
        values = lambda t_x: u(
            t_x,
            params,
        )[0]
        omega_batch_grid = get_grid(omega_batch)
        v_ini = values(t0_omega_batch)
        res = _subtract_with_check(
            initial_condition_fun(omega_batch_grid),
            v_ini,
            cause="Output of initial_condition_fun",
        )
        mse_initial_condition = jnp.mean(jnp.sum(res**2, axis=-1))
    else:
        raise ValueError(f"Bad type for u. Got {type(u)}, expected PINN or SPINN")
    return mse_initial_condition
