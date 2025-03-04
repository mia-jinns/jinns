from __future__ import (
    annotations,
)  # https://docs.python.org/3/library/typing.html#constant

from typing import TYPE_CHECKING, Callable
from functools import partial
import jax
from jax import vmap
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Int, Bool

from jinns.data._Batchs import *
from jinns.loss._LossODE import LossODE, SystemLossODE
from jinns.loss._LossPDE import LossPDEStatio, LossPDENonStatio, SystemLossPDE
from jinns.loss._loss_utils import dynamic_loss_apply
from jinns.data._DataGenerators import (
    DataGeneratorODE,
    CubicMeshPDEStatio,
    CubicMeshPDENonStatio,
)
from jinns.nn._hyperpinn import HyperPINN
from jinns.nn._spinn import SPINN


if TYPE_CHECKING:
    from jinns.utils._types import *


def _proceed_to_rar(data: AnyDataGenerator, i: Int) -> Bool:
    """Utilility function with various check to ensure we can proceed with the rar_step.
    Return True if yes, and False otherwise"""

    # Overall checks
    check_list = [
        # check if burn-in period has ended
        data.rar_parameters["start_iter"] <= i,
        # check if enough iterations since last points added
        (data.rar_parameters["update_every"] - 1) == data.rar_iter_from_last_sampling,
    ]

    # Memory allocation checks
    # check if we still have room to append new collocation points in the
    # allocated jnp.array
    check_list.append(
        data.rar_parameters["selected_sample_size"] <= jnp.count_nonzero(data.p == 0),
    )

    proceed = jnp.all(jnp.array(check_list))
    return proceed


@partial(jax.jit, static_argnames=["_rar_step_true", "_rar_step_false"])
def trigger_rar(
    i: Int,
    loss: AnyLoss,
    params: AnyParams,
    data: AnyDataGenerator,
    _rar_step_true: Callable[[rar_operands], AnyDataGenerator],
    _rar_step_false: Callable[[rar_operands], AnyDataGenerator],
) -> tuple[AnyLoss, AnyParams, AnyDataGenerator]:

    if data.rar_parameters is None:
        # do nothing.
        return loss, params, data
    else:
        # update `data` according to rar scheme.
        data = jax.lax.cond(
            _proceed_to_rar(data, i),
            _rar_step_true,
            _rar_step_false,
            (loss, params, data, i),
        )
        return loss, params, data


def init_rar(
    data: AnyDataGenerator,
) -> tuple[
    AnyDataGenerator,
    Callable[[rar_operands], AnyDataGenerator],
    Callable[[rar_operands], AnyDataGenerator],
]:
    """
    Separated from the main rar, because the initialization to get _true and
    _false cannot be jit-ted.
    """
    # NOTE if a user misspell some entry of ``rar_parameters`` the error
    # risks to be a bit obscure but it should be ok.
    if data.rar_parameters is None:
        _rar_step_true, _rar_step_false = None, None
    else:
        _rar_step_true, _rar_step_false = _rar_step_init(
            data.rar_parameters["sample_size"],
            data.rar_parameters["selected_sample_size"],
        )

        data = eqx.tree_at(lambda m: m.rar_iter_from_last_sampling, data, 0)

    return data, _rar_step_true, _rar_step_false


def _rar_step_init(sample_size: Int, selected_sample_size: Int) -> tuple[
    Callable[[rar_operands], AnyDataGenerator],
    Callable[[rar_operands], AnyDataGenerator],
]:
    """
    This is a wrapper because the sampling size and
    selected_sample_size, must be treated as static
    in order to slice. So they must be set before jitting and not with the jitted
    dictionary values rar["test_points_nb"] and rar["added_points_nb"]

    This is a kind of manual declaration of static argnums
    """

    def rar_step_true(operands: rar_operands) -> AnyDataGenerator:
        loss, params, data, i = operands
        if isinstance(loss.u, HyperPINN) or isinstance(loss.u, SPINN):
            raise NotImplementedError("RAR not implemented for hyperPINN and SPINN")

        if isinstance(data, DataGeneratorODE):

            new_key, subkey = jax.random.split(data.key)
            new_samples = data.sample_in_time_domain(subkey, sample_size)
            data = eqx.tree_at(lambda m: m.key, data, new_key)

        elif isinstance(data, CubicMeshPDEStatio) and not isinstance(
            data, CubicMeshPDENonStatio
        ):
            new_key, *subkeys = jax.random.split(data.key, data.dim + 1)
            new_samples = data.sample_in_omega_domain(subkeys, sample_size)
            data = eqx.tree_at(lambda m: m.key, data, new_key)

        elif isinstance(data, CubicMeshPDENonStatio):
            new_key, subkey = jax.random.split(data.key)
            new_samples_times = data.sample_in_time_domain(subkey, sample_size)
            if data.dim == 1:
                new_key, subkeys = jax.random.split(new_key, 2)
            else:
                new_key, *subkeys = jax.random.split(new_key, data.dim + 1)
            new_samples_omega = data.sample_in_omega_domain(subkeys, sample_size)
            new_samples = jnp.concatenate(
                [new_samples_times, new_samples_omega], axis=1
            )

            data = eqx.tree_at(lambda m: m.key, data, new_key)

        # We can have different types of Loss
        if isinstance(loss, (LossODE, LossPDEStatio, LossPDENonStatio)):
            v_dyn_loss = vmap(
                lambda inputs: loss.dynamic_loss.evaluate(inputs, loss.u, params),
            )
            dyn_on_s = v_dyn_loss(new_samples)

            if dyn_on_s.ndim > 1:
                mse_on_s = (jnp.linalg.norm(dyn_on_s, axis=-1) ** 2).flatten()
            else:
                mse_on_s = dyn_on_s**2
        elif isinstance(loss, SystemLossODE, SystemLossPDE):
            mse_on_s = 0

            for i in loss.dynamic_loss_dict.keys():
                v_dyn_loss = vmap(
                    lambda inputs: loss.dynamic_loss_dict[i].evaluate(
                        inputs, loss.u_dict, params
                    ),
                    (0),
                    0,
                )
                dyn_on_s = v_dyn_loss(new_samples)
                if dyn_on_s.ndim > 1:
                    mse_on_s += (jnp.linalg.norm(dyn_on_s, axis=-1) ** 2).flatten()
                else:
                    mse_on_s += dyn_on_s**2

        ## Select the m points with higher dynamic loss
        higher_residual_idx = jax.lax.dynamic_slice(
            jnp.argsort(mse_on_s),
            (mse_on_s.shape[0] - selected_sample_size,),
            (selected_sample_size,),
        )
        higher_residual_points = new_samples[higher_residual_idx]

        # add the new points
        # start indices of update can be dynamic but the the shape (length)
        # of the slice
        if isinstance(data, DataGeneratorODE):
            new_times = jax.lax.dynamic_update_slice(
                data.times,
                higher_residual_points,
                (data.n_start + data.rar_iter_nb * selected_sample_size,),
            )

            data = eqx.tree_at(lambda m: m.times, data, new_times)
        elif isinstance(data, CubicMeshPDEStatio) and not isinstance(
            data, CubicMeshPDENonStatio
        ):
            new_omega = jax.lax.dynamic_update_slice(
                data.omega,
                higher_residual_points,
                (data.n_start + data.rar_iter_nb * selected_sample_size, data.dim),
            )

            data = eqx.tree_at(lambda m: m.omega, data, new_omega)

        elif isinstance(data, CubicMeshPDENonStatio):
            new_domain = jax.lax.dynamic_update_slice(
                data.domain,
                higher_residual_points,
                (data.n_start + data.rar_iter_nb * selected_sample_size, 1 + data.dim),
            )

            data = eqx.tree_at(lambda m: m.domain, data, new_domain)

        ## rearrange probabilities so that the probabilities of the new
        ## points are non-zero
        new_proba = 1 / (data.n_start + data.rar_iter_nb * selected_sample_size)
        # the next work because nt_start is static
        new_p = data.p.at[: data.n_start].set(new_proba)
        data = eqx.tree_at(
            lambda m: m.p,
            data,
            new_p,
        )

        # the next requires a fori_loop because the range is dynamic
        def update_slices(i, p):
            return jax.lax.dynamic_update_slice(
                p,
                1 / new_proba * jnp.ones((selected_sample_size,)),
                ((data.n_start + i * selected_sample_size),),
            )

        new_rar_iter_nb = data.rar_iter_nb + 1
        new_p = jax.lax.fori_loop(0, new_rar_iter_nb, update_slices, data.p)
        data = eqx.tree_at(
            lambda m: (m.rar_iter_nb, m.p),
            data,
            (new_rar_iter_nb, new_p),
        )

        # update RAR parameters for all cases
        data = eqx.tree_at(lambda m: m.rar_iter_from_last_sampling, data, 0)

        # NOTE must return data to be correctly updated because we cannot
        # have side effects in this function that will be jitted
        return data

    def rar_step_false(operands: rar_operands) -> AnyDataGenerator:
        _, _, data, i = operands

        # Add 1 only if we are after the burn in period
        increment = jax.lax.cond(
            i <= data.rar_parameters["start_iter"],
            lambda: 0,
            lambda: 1,
        )

        new_rar_iter_from_last_sampling = data.rar_iter_from_last_sampling + increment
        if isinstance(data, eqx.Module):
            data = eqx.tree_at(
                lambda m: m.rar_iter_from_last_sampling,
                data,
                new_rar_iter_from_last_sampling,
            )
        else:
            data.rar_iter_from_last_sampling = new_rar_iter_from_last_sampling
        return data

    return rar_step_true, rar_step_false
