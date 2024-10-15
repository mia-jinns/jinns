"""
Interface for diverse loss functions to factorize code
"""

from __future__ import (
    annotations,
)  # https://docs.python.org/3/library/typing.html#constant

from typing import TYPE_CHECKING, Callable, Dict
import jax
import jax.numpy as jnp
from jax import vmap
import equinox as eqx
from jaxtyping import Float, Array, PyTree

from jinns.loss._boundary_conditions import (
    _compute_boundary_loss,
)
from jinns.utils._utils import _check_user_func_return, _get_grid
from jinns.data._DataGenerators import (
    append_obs_batch,
)
from jinns.utils._pinn import PINN
from jinns.utils._spinn import SPINN
from jinns.utils._hyperpinn import HYPERPINN
from jinns.data._Batchs import *
from jinns.parameters._params import Params, ParamsDict

if TYPE_CHECKING:
    from jinns.utils._types import *


def dynamic_loss_apply(
    dyn_loss: DynamicLoss,
    u: eqx.Module,
    batches: ODEBatch | PDEStatioBatch | PDENonStatioBatch,
    params: Params | ParamsDict,
    vmap_axes: tuple[int | None, ...],
    loss_weight: float | Float[Array, "dyn_loss_dimension"],
    u_type: PINN | HYPERPINN | None = None,
) -> float:
    """
    Sometimes when u is a lambda function a or dict we do not have access to
    its type here, hence the last argument
    """
    if u_type == PINN or u_type == HYPERPINN or isinstance(u, (PINN, HYPERPINN)):
        v_dyn_loss = vmap(
            lambda *args: dyn_loss(
                *args[:-1], u, args[-1]  # we must place the params at the end
            ),
            vmap_axes,
            0,
        )
        residuals = v_dyn_loss(*batches, params)
        mse_dyn_loss = jnp.mean(jnp.sum(loss_weight * residuals**2, axis=-1))
    elif u_type == SPINN or isinstance(u, SPINN):
        residuals = dyn_loss(*batches, u, params)
        mse_dyn_loss = jnp.mean(jnp.sum(loss_weight * residuals**2, axis=-1))
    else:
        raise ValueError(f"Bad type for u. Got {type(u)}, expected PINN or SPINN")

    return mse_dyn_loss


def normalization_loss_apply(
    u: eqx.Module,
    batches: ODEBatch | PDEStatioBatch | PDENonStatioBatch,
    params: Params | ParamsDict,
    vmap_axes: tuple[int | None, ...],
    int_length: int,
    loss_weight: float,
) -> float:
    # TODO merge stationary and non stationary cases
    if isinstance(u, (PINN, HYPERPINN)):
        if len(batches) == 1:
            v_u = vmap(
                lambda *args: u(*args)[u.slice_solution],
                vmap_axes,
                0,
            )
            mse_norm_loss = loss_weight * jnp.mean(
                jnp.abs(jnp.mean(v_u(*batches, params), axis=-1) * int_length - 1) ** 2
            )
        else:
            v_u = vmap(
                vmap(
                    lambda t, x, params_: u(t, x, params_),
                    in_axes=(None, 0) + vmap_axes[2:],
                ),
                in_axes=(0, None) + vmap_axes[2:],
            )
            res = v_u(*batches, params)
            # the outer mean() below is for the times stamps
            mse_norm_loss = loss_weight * jnp.mean(
                jnp.abs(jnp.mean(res, axis=(-2, -1)) * int_length - 1) ** 2
            )
    elif isinstance(u, SPINN):
        if len(batches) == 1:
            res = u(*batches, params)
            mse_norm_loss = (
                loss_weight
                * jnp.abs(
                    jnp.mean(
                        jnp.mean(res, axis=-1),
                        axis=tuple(range(res.ndim - 1)),
                    )
                    * int_length
                    - 1
                )
                ** 2
            )
        else:
            assert batches[1].shape[0] % batches[0].shape[0] == 0
            rep_t = batches[1].shape[0] // batches[0].shape[0]
            res = u(jnp.repeat(batches[0], rep_t, axis=0), batches[1], params)
            # the outer mean() below is for the times stamps
            mse_norm_loss = loss_weight * jnp.mean(
                jnp.abs(
                    jnp.mean(
                        jnp.mean(res, axis=-1),
                        axis=(d + 1 for d in range(res.ndim - 2)),
                    )
                    * int_length
                    - 1
                )
                ** 2
            )
    else:
        raise ValueError(f"Bad type for u. Got {type(u)}, expected PINN or SPINN")

    return mse_norm_loss


def boundary_condition_apply(
    u: eqx.Module,
    batch: PDEStatioBatch | PDENonStatioBatch,
    params: Params | ParamsDict,
    omega_boundary_fun: Callable,
    omega_boundary_condition: str,
    omega_boundary_dim: int,
    loss_weight: float | Float[Array, "boundary_cond_dim"],
) -> float:
    if isinstance(omega_boundary_fun, dict):
        # We must create the facet tree dictionary as we do not have the
        # enumerate from the for loop to pass the id integer
        if (
            isinstance(batch, PDEStatioBatch) and batch.border_batch.shape[-1] == 2
        ) or (
            isinstance(batch, PDENonStatioBatch)
            and batch.times_x_border_batch.shape[-1] == 2
        ):
            # 1D
            facet_tree = {"xmin": 0, "xmax": 1}
        elif (
            isinstance(batch, PDEStatioBatch) and batch.border_batch.shape[-1] == 4
        ) or (
            isinstance(batch, PDENonStatioBatch)
            and batch.times_x_border_batch.shape[-1] == 4
        ):
            # 2D
            facet_tree = {"xmin": 0, "xmax": 1, "ymin": 2, "ymax": 3}
        else:
            raise ValueError("Other border batches are not implemented")
        b_losses_by_facet = jax.tree_util.tree_map(
            lambda c, f, fa, d: (
                None
                if c is None
                else jnp.mean(
                    loss_weight * _compute_boundary_loss(c, f, batch, u, params, fa, d)
                )
            ),
            omega_boundary_condition,
            omega_boundary_fun,
            facet_tree,
            omega_boundary_dim,
            is_leaf=lambda x: x is None,
        )  # when exploring leaves with None value (no condition) the returned
        # mse is None and we get rid of the None leaves of b_losses_by_facet
        # with the tree_leaves below
        # Note that to keep the behaviour given in the comment above we neede
        # to specify is_leaf according to the note in the release of 0.4.29
    else:
        if isinstance(batch, PDEStatioBatch):
            facet_tuple = tuple(f for f in range(batch.border_batch.shape[-1]))
        else:
            facet_tuple = tuple(f for f in range(batch.times_x_border_batch.shape[-1]))
        b_losses_by_facet = jax.tree_util.tree_map(
            lambda fa: jnp.mean(
                loss_weight
                * _compute_boundary_loss(
                    omega_boundary_condition,
                    omega_boundary_fun,
                    batch,
                    u,
                    params,
                    fa,
                    omega_boundary_dim,
                )
            ),
            facet_tuple,
        )
    mse_boundary_loss = jax.tree_util.tree_reduce(
        lambda x, y: x + y, jax.tree_util.tree_leaves(b_losses_by_facet)
    )
    return mse_boundary_loss


def observations_loss_apply(
    u: eqx.Module,
    batches: ODEBatch | PDEStatioBatch | PDENonStatioBatch,
    params: Params | ParamsDict,
    vmap_axes: tuple[int | None, ...],
    observed_values: Float[Array, "batch_size observation_dim"],
    loss_weight: float | Float[Array, "observation_dim"],
    obs_slice: slice,
) -> float:
    # TODO implement for SPINN
    if isinstance(u, (PINN, HYPERPINN)):
        v_u = vmap(
            lambda *args: u(*args)[u.slice_solution],
            vmap_axes,
            0,
        )
        val = v_u(*batches, params)[:, obs_slice]
        mse_observation_loss = jnp.mean(
            jnp.sum(
                loss_weight
                * (val - _check_user_func_return(observed_values, val.shape)) ** 2,
                # the reshape above avoids a potential missing (1,)
                axis=-1,
            )
        )
    elif isinstance(u, SPINN):
        raise RuntimeError("observation loss term not yet implemented for SPINNs")
    else:
        raise ValueError(f"Bad type for u. Got {type(u)}, expected PINN or SPINN")
    return mse_observation_loss


def initial_condition_apply(
    u: eqx.Module,
    omega_batch: Float[Array, "dimension"],
    params: Params | ParamsDict,
    vmap_axes: tuple[int | None, ...],
    initial_condition_fun: Callable,
    n: int,
    loss_weight: float | Float[Array, "initial_condition_dimension"],
) -> float:
    if isinstance(u, (PINN, HYPERPINN)):
        v_u_t0 = vmap(
            lambda x, params: initial_condition_fun(x) - u(jnp.zeros((1,)), x, params),
            vmap_axes,
            0,
        )
        res = v_u_t0(omega_batch, params)  # NOTE take the tiled
        # omega_batch (ie omega_batch_) to have the same batch
        # dimension as params to be able to vmap.
        # Recall that by convention:
        # param_batch_dict = times_batch_size * omega_batch_size
        mse_initial_condition = jnp.mean(jnp.sum(loss_weight * res**2, axis=-1))
    elif isinstance(u, SPINN):
        values = lambda x: u(
            jnp.repeat(jnp.zeros((1, 1)), n, axis=0),
            x,
            params,
        )[0]
        omega_batch_grid = _get_grid(omega_batch)
        v_ini = values(omega_batch)
        ini = _check_user_func_return(
            initial_condition_fun(omega_batch_grid), v_ini.shape
        )
        res = ini - v_ini
        mse_initial_condition = jnp.mean(jnp.sum(loss_weight * res**2, axis=-1))
    else:
        raise ValueError(f"Bad type for u. Got {type(u)}, expected PINN or SPINN")
    return mse_initial_condition


def constraints_system_loss_apply(
    u_constraints_dict: Dict,
    batch: ODEBatch | PDEStatioBatch | PDENonStatioBatch,
    params_dict: ParamsDict,
    loss_weights: Dict[str, float | Array],
    loss_weight_struct: PyTree,
):
    """
    Same function for systemlossODE and systemlossPDE!
    """
    # Transpose so we have each u_dict as outer structure and the
    # associated loss_weight as inner structure
    loss_weights_T = jax.tree_util.tree_transpose(
        jax.tree_util.tree_structure(loss_weight_struct),
        jax.tree_util.tree_structure(loss_weights["initial_condition"]),
        loss_weights,
    )

    if isinstance(params_dict.nn_params, dict):

        def apply_u_constraint(
            u_constraint, nn_params, eq_params, loss_weights_for_u, obs_batch_u
        ):
            res_dict_for_u = u_constraint.evaluate(
                Params(
                    nn_params=nn_params,
                    eq_params=eq_params,
                ),
                append_obs_batch(batch, obs_batch_u),
            )[1]
            res_dict_ponderated = jax.tree_util.tree_map(
                lambda w, l: w * l, res_dict_for_u, loss_weights_for_u
            )
            return res_dict_ponderated

        # Note in the case of multiple PINNs, batch.obs_batch_dict is a dict
        # with keys corresponding to the PINN and value correspondinf to an
        # original obs_batch_dict. Hence the tree mapping also interates over
        # batch.obs_batch_dict
        res_dict = jax.tree_util.tree_map(
            apply_u_constraint,
            u_constraints_dict,
            params_dict.nn_params,
            (
                params_dict.eq_params
                if params_dict.eq_params.keys() == params_dict.nn_params.keys()
                else {k: params_dict.eq_params for k in params_dict.nn_params.keys()}
            ),  # this manipulation is needed since we authorize eq_params not to have the same structure as nn_params in ParamsDict
            loss_weights_T,
            batch.obs_batch_dict,
            is_leaf=lambda x: (
                not isinstance(x, dict)  # to not traverse more than the first
                # outer dict of the pytrees passed to the function. This will
                # work because u_constraints_dict is a dict of Losses, and it
                # thus stops the traversing of other dict too
            ),
        )
    # TODO try to get rid of this condition?
    else:

        def apply_u_constraint(u_constraint, loss_weights_for_u, obs_batch_u):
            res_dict_for_u = u_constraint.evaluate(
                params_dict,
                append_obs_batch(batch, obs_batch_u),
            )[1]
            res_dict_ponderated = jax.tree_util.tree_map(
                lambda w, l: w * l, res_dict_for_u, loss_weights_for_u
            )
            return res_dict_ponderated

        res_dict = jax.tree_util.tree_map(
            apply_u_constraint, u_constraints_dict, loss_weights_T, batch.obs_batch_dict
        )

    # Transpose back so we have mses as outer structures and their values
    # for each u_dict as inner structures. The tree_leaves transforms the
    # inner structure into a list so we can catch is as leaf it the
    # tree_map below
    res_dict = jax.tree_util.tree_transpose(
        jax.tree_util.tree_structure(
            jax.tree_util.tree_leaves(loss_weights["initial_condition"])
        ),
        jax.tree_util.tree_structure(loss_weight_struct),
        res_dict,
    )
    # For each mse, sum their values on each u_dict
    res_dict = jax.tree_util.tree_map(
        lambda mse: jax.tree_util.tree_reduce(
            lambda x, y: x + y, jax.tree_util.tree_leaves(mse)
        ),
        res_dict,
        is_leaf=lambda x: isinstance(x, list),
    )
    # Total loss
    total_loss = jax.tree_util.tree_reduce(
        lambda x, y: x + y, jax.tree_util.tree_leaves(res_dict)
    )
    return total_loss, res_dict
