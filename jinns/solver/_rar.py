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
from jinns.data._DataGenerators import (
    DataGeneratorODE,
    CubicMeshPDEStatio,
    CubicMeshPDENonStatio,
)
from jinns.utils._hyperpinn import HYPERPINN
from jinns.utils._spinn import SPINN


if TYPE_CHECKING:
    from jinns.utils._types import *


def _proceed_to_rar(data: AnyDataGenerator, i: Int) -> Bool:
    """Utilility function with various check to ensure we can proceed with the rar_step.
    Return True if yes, and False otherwise"""

    # Overall checks (universal for any data generator)
    check_list = [
        # check if burn-in period has ended
        data.rar_parameters["start_iter"] <= i,
        # check if enough iterations since last points added
        (data.rar_parameters["update_every"] - 1) == data.rar_iter_from_last_sampling,
    ]

    # Memory allocation checks (depends on the type of DataGenerator)
    # check if we still have room to append new collocation points in the
    # allocated jnp.array (can concern `data.p_times` or `p_omega`)
    if isinstance(data, (DataGeneratorODE, CubicMeshPDENonStatio)):
        check_list.append(
            data.rar_parameters["selected_sample_size_times"]
            <= jnp.count_nonzero(data.p_times == 0),
        )

    if isinstance(data, (CubicMeshPDEStatio, CubicMeshPDENonStatio)):
        # for now the above check are redundants but there may be a time when
        # we drop inheritence
        check_list.append(
            data.rar_parameters["selected_sample_size_omega"]
            <= jnp.count_nonzero(data.p_omega == 0),
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
        if isinstance(data, DataGeneratorODE):
            # In this case we only need rar parameters related to `times`
            _rar_step_true, _rar_step_false = _rar_step_init(
                data.rar_parameters["sample_size_times"],
                data.rar_parameters["selected_sample_size_times"],
            )
        elif isinstance(data, CubicMeshPDENonStatio):
            # In this case we need rar parameters related to both `times`
            # and`omega`
            _rar_step_true, _rar_step_false = _rar_step_init(
                (
                    data.rar_parameters["sample_size_times"],
                    data.rar_parameters["sample_size_omega"],
                ),
                (
                    data.rar_parameters["selected_sample_size_times"],
                    data.rar_parameters["selected_sample_size_omega"],
                ),
            )
        elif isinstance(data, CubicMeshPDEStatio):
            # In this case we only need rar parameters related to `omega`
            _rar_step_true, _rar_step_false = _rar_step_init(
                data.rar_parameters["sample_size_omega"],
                data.rar_parameters["selected_sample_size_omega"],
            )
        else:
            raise ValueError(f"Wrong type for data got {type(data)}")

        if isinstance(data, eqx.Module):
            data = eqx.tree_at(lambda m: m.rar_iter_from_last_sampling, data, 0)
        else:
            data.rar_iter_from_last_sampling = 0

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

        if isinstance(data, DataGeneratorODE):

            if isinstance(data, eqx.Module):
                new_key, subkey = jax.random.split(data.key)
                new_omega_samples = data.sample_in_time_domain(subkey, sample_size)
                data = eqx.tree_at(lambda m: m.key, data, new_key)
            else:
                new_omega_samples = data.sample_in_time_domain(sample_size)

            # We can have different types of Loss
            if isinstance(loss, LossODE):
                v_dyn_loss = vmap(
                    lambda t: loss.dynamic_loss.evaluate(t, loss.u, params),
                    (0),
                    0,
                )
                dyn_on_s = v_dyn_loss(new_omega_samples)
                if dyn_on_s.ndim > 1:
                    mse_on_s = (jnp.linalg.norm(dyn_on_s, axis=-1) ** 2).flatten()
                else:
                    mse_on_s = dyn_on_s**2
            elif isinstance(loss, SystemLossODE):
                mse_on_s = 0

                for i in loss.dynamic_loss_dict.keys():
                    v_dyn_loss = vmap(
                        lambda t: loss.dynamic_loss_dict[i].evaluate(
                            t, loss.u_dict, params
                        ),
                        (0),
                        0,
                    )
                    dyn_on_s = v_dyn_loss(new_omega_samples)
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
            higher_residual_points = new_omega_samples[higher_residual_idx]

            ## add the new points in times
            # start indices of update can be dynamic but the the shape (length)
            # of the slice
            new_times = jax.lax.dynamic_update_slice(
                data.times,
                higher_residual_points,
                (data.nt_start + data.rar_iter_nb * selected_sample_size,),
            )

            if isinstance(data, eqx.Module):
                data = eqx.tree_at(lambda m: m.times, data, new_times)
            else:
                data.times = new_times
            ## rearrange probabilities so that the probabilities of the new
            ## points are non-zero
            new_proba = 1 / (data.nt_start + data.rar_iter_nb * selected_sample_size)
            # the next work because nt_start is static
            new_p_times = data.p_times.at[: data.nt_start].set(new_proba)
            if isinstance(data, eqx.Module):
                data = eqx.tree_at(
                    lambda m: m.p_times,
                    data,
                    new_p_times,
                )
            else:
                data.p_times = new_p_times

            # the next requires a fori_loop because the range is dynamic
            def update_slices(i, p):
                return jax.lax.dynamic_update_slice(
                    p,
                    1 / new_proba * jnp.ones((selected_sample_size,)),
                    ((data.nt_start + i * selected_sample_size),),
                )

            new_rar_iter_nb = data.rar_iter_nb + 1
            new_p_times = jax.lax.fori_loop(
                0, data.rar_iter_nb, update_slices, data.p_times
            )
            if isinstance(data, eqx.Module):
                data = eqx.tree_at(
                    lambda m: (m.rar_iter_nb, m.p_times),
                    data,
                    (new_rar_iter_nb, new_p_times),
                )
            else:
                data.rar_iter_nb = new_rar_iter_nb
                data.p_times = new_p_times

        elif isinstance(data, CubicMeshPDEStatio) and not isinstance(
            data, CubicMeshPDENonStatio
        ):
            if isinstance(data, eqx.Module):
                new_key, *subkeys = jax.random.split(data.key, data.dim + 1)
                new_omega_samples = data.sample_in_omega_domain(subkeys, sample_size)
                data = eqx.tree_at(lambda m: m.key, data, new_key)
            else:
                new_omega_samples = data.sample_in_omega_domain(sample_size)

            # We can have different types of Loss
            if isinstance(loss, LossPDEStatio):
                v_dyn_loss = vmap(
                    lambda x: loss.dynamic_loss.evaluate(
                        x,
                        loss.u,
                        params,
                    ),
                    (0),
                    0,
                )
                dyn_on_s = v_dyn_loss(new_omega_samples)
                if dyn_on_s.ndim > 1:
                    mse_on_s = (jnp.linalg.norm(dyn_on_s, axis=-1) ** 2).flatten()
                else:
                    mse_on_s = dyn_on_s**2
            elif isinstance(loss, SystemLossODE):
                mse_on_s = 0
                for i in loss.dynamic_loss_dict.keys():
                    # only the case LossPDEStatio here
                    v_dyn_loss = vmap(
                        lambda x: loss.dynamic_loss_dict[i].evaluate(
                            x, loss.u_dict, params
                        ),
                        0,
                        0,
                    )
                    dyn_on_s = v_dyn_loss(new_omega_samples)
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
            higher_residual_points = new_omega_samples[higher_residual_idx]

            ## add the new points in omega
            # start indices of update can be dynamic but not the shape (length)
            # of the slice
            new_omega = jax.lax.dynamic_update_slice(
                data.omega,
                higher_residual_points,
                (data.n_start + data.rar_iter_nb * selected_sample_size, data.dim),
            )

            if isinstance(data, eqx.Module):
                data = eqx.tree_at(lambda m: m.omega, data, new_omega)
            else:
                data.omega = new_omega

            ## rearrange probabilities so that the probabilities of the new
            ## points are non-zero
            new_proba = 1 / (data.n_start + data.rar_iter_nb * selected_sample_size)
            # the next work because n_start is static
            new_p_omega = data.p_omega.at[: data.n_start].set(new_proba)
            if isinstance(data, eqx.Module):
                data = eqx.tree_at(
                    lambda m: m.p_omega,
                    data,
                    new_p_omega,
                )
            else:
                data.p_omega = new_p_omega

            # the next requires a fori_loop because the range is dynamic
            def update_slices(i, p):
                return jax.lax.dynamic_update_slice(
                    p,
                    1 / new_proba * jnp.ones((selected_sample_size,)),
                    ((data.n_start + i * selected_sample_size),),
                )

            new_rar_iter_nb = data.rar_iter_nb + 1
            new_p_omega = jax.lax.fori_loop(
                0, data.rar_iter_nb, update_slices, data.p_omega
            )
            if isinstance(data, eqx.Module):
                data = eqx.tree_at(
                    lambda m: (m.rar_iter_nb, m.p_omega),
                    data,
                    (new_rar_iter_nb, new_p_omega),
                )
            else:
                data.rar_iter_nb = new_rar_iter_nb
                data.p_omega = new_p_omega

        elif isinstance(data, CubicMeshPDENonStatio):
            if isinstance(loss.u, HYPERPINN) or isinstance(loss.u, SPINN):
                raise NotImplementedError("RAR not implemented for hyperPINN and SPINN")

            # NOTE in this case sample_size and selected_sample_size
            # are tuples (times, omega) => we unpack them for clarity
            selected_sample_size_times, selected_sample_size_omega = (
                selected_sample_size
            )
            sample_size_times, sample_size_omega = sample_size

            if isinstance(data, eqx.Module):
                new_key, subkey = jax.random.split(data.key)
                new_times_samples = data.sample_in_time_domain(
                    subkey, sample_size_times
                )
                new_key, *subkeys = jax.random.split(new_key, data.dim + 1)
                new_omega_samples = data.sample_in_omega_domain(
                    subkeys, sample_size_omega
                )
                data = eqx.tree_at(lambda m: m.key, data, new_key)
            else:
                new_times_samples = data.sample_in_time_domain(sample_size_times)
                new_omega_samples = data.sample_in_omega_domain(sample_size_omega)

            if not data.cartesian_product:
                times = new_times_samples
                omega = new_omega_samples
            else:
                # do cartesian product on new points
                omega = jnp.tile(
                    new_omega_samples, reps=(sample_size_times, 1)
                )  # it is tiled
                times = jnp.repeat(new_times_samples, sample_size_omega, axis=0)[
                    ..., None
                ]  # it is repeated + add an axis

            if isinstance(loss, LossPDENonStatio):
                v_dyn_loss = vmap(
                    lambda t, x: loss.dynamic_loss.evaluate(t, x, loss.u, params),
                    (0, 0),
                    0,
                )
                dyn_on_s = v_dyn_loss(times, omega).reshape(
                    (sample_size_times, sample_size_omega)
                )
                mse_on_s = dyn_on_s**2
            elif isinstance(loss, SystemLossPDE):
                dyn_on_s = jnp.zeros((sample_size_times, sample_size_omega))
                for i in loss.dynamic_loss_dict.keys():
                    v_dyn_loss = vmap(
                        lambda t, x: loss.dynamic_loss_dict[i].evaluate(
                            t, x, loss.u_dict, params
                        ),
                        (0, 0),
                        0,
                    )
                    dyn_on_s += v_dyn_loss(times, omega).reshape(
                        (sample_size_times, sample_size_omega)
                    )

            mse_on_s = dyn_on_s**2
            # -- Select the m points with highest average residuals on time and
            # -- space (times in rows / omega in columns)
            # mean_times = mse_on_s.mean(axis=1)
            # mean_omega = mse_on_s.mean(axis=0)
            # times_idx = jax.lax.dynamic_slice(
            #     jnp.argsort(mean_times),
            #     (mse_on_s.shape[0] - selected_sample_size_times,),
            #     (selected_sample_size_times,),
            # )
            # omega_idx = jax.lax.dynamic_slice(
            #     jnp.argsort(mean_omega),
            #     (mse_on_s.shape[1] - selected_sample_size_omega,),
            #     (selected_sample_size_omega,),
            # )

            # -- Select the m worst points (t, x) with highest residuals
            n_select = max(selected_sample_size_times, selected_sample_size_omega)
            _, idx = jax.lax.top_k(mse_on_s.flatten(), k=n_select)
            arr_idx = jnp.unravel_index(idx, mse_on_s.shape)
            times_idx = arr_idx[0][:selected_sample_size_times]
            omega_idx = arr_idx[1][:selected_sample_size_omega]

            higher_residual_points_times = new_times_samples[times_idx]
            higher_residual_points_omega = new_omega_samples[omega_idx]

            ## add the new points in times
            # start indices of update can be dynamic but not the shape (length)
            # of the slice
            new_times = jax.lax.dynamic_update_slice(
                data.times,
                higher_residual_points_times,
                (
                    data.n_start
                    + data.rar_iter_nb  # NOTE typo here nt_start ?
                    * selected_sample_size_times,
                ),
            )

            if isinstance(data, eqx.Module):
                data = eqx.tree_at(lambda m: m.times, data, new_times)
            else:
                data.times = new_times

            ## add the new points in omega
            new_omega = jax.lax.dynamic_update_slice(
                data.omega,
                higher_residual_points_omega,
                (
                    data.n_start + data.rar_iter_nb * selected_sample_size_omega,
                    data.dim,
                ),
            )

            if isinstance(data, eqx.Module):
                data = eqx.tree_at(lambda m: m.omega, data, new_omega)
            else:
                data.omega = new_omega

            ## rearrange probabilities so that the probabilities of the new
            ## points are non-zero
            new_p_times = 1 / (
                data.nt_start + data.rar_iter_nb * selected_sample_size_times
            )
            # the next work because nt_start is static
            if isinstance(data, eqx.Module):
                data = eqx.tree_at(
                    lambda m: m.p_times,
                    data,
                    data.p_times.at[: data.nt_start].set(new_p_times),
                )
            else:
                data.p_times = data.p_times.at[: data.nt_start].set(new_p_times)

            # same for p_omega (work because n_start is static)
            new_p_omega = 1 / (
                data.n_start + data.rar_iter_nb * selected_sample_size_omega
            )
            if isinstance(data, eqx.Module):
                data = eqx.tree_at(
                    lambda m: m.p_omega,
                    data,
                    data.p_omega.at[: data.n_start].set(new_p_omega),
                )
            else:
                data.p_omega = data.p_omega.at[: data.n_start].set(new_p_omega)

            # the part of data.p_* after n_start requires a fori_loop because
            # the range is dynamic
            def create_update_slices(new_val, selected_sample_size):
                def update_slices(i, p):
                    new_p = jax.lax.dynamic_update_slice(
                        p,
                        new_val * jnp.ones((selected_sample_size,)),
                        ((data.n_start + i * selected_sample_size),),
                    )
                    return new_p

                return update_slices

            new_rar_iter_nb = data.rar_iter_nb + 1

            ## update rest of p_times
            update_slices_times = create_update_slices(
                new_p_times, selected_sample_size_times
            )
            new_p_times = jax.lax.fori_loop(
                0,
                data.rar_iter_nb,
                update_slices_times,
                data.p_times,
            )
            ## update rest of p_omega
            update_slices_omega = create_update_slices(
                new_p_omega, selected_sample_size_omega
            )
            new_p_omega = jax.lax.fori_loop(
                0,
                data.rar_iter_nb,
                update_slices_omega,
                data.p_omega,
            )
            if isinstance(data, eqx.Module):
                data = eqx.tree_at(
                    lambda m: (m.rar_iter_nb, m.p_omega, m.p_times),
                    data,
                    (new_rar_iter_nb, new_p_omega, new_p_times),
                )
            else:
                data.rar_iter_nb = new_rar_iter_nb
                data.p_times = new_p_times
                data.p_omega = new_p_omega

        # update RAR parameters for all cases
        if isinstance(data, eqx.Module):
            data = eqx.tree_at(lambda m: m.rar_iter_from_last_sampling, data, 0)
        else:
            data.rar_iter_from_last_sampling = 0

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
