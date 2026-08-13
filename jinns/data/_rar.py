from __future__ import (
    annotations,
)  # https://docs.python.org/3/library/typing.html#constant

from typing import TYPE_CHECKING, TypeAlias, Any, cast
from jaxtyping import Array, Bool, PRNGKeyArray
import jax
import jax.numpy as jnp
import equinox as eqx

from jinns.data._Batchs import ODEBatch, PDENonStatioBatch, PDEStatioBatch
from jinns.data._DataGeneratorParameter import DataGeneratorParameter, DGParams
from jinns.data._DataGeneratorODE import DataGeneratorODE
from jinns.data._CubicMeshPDEStatio import CubicMeshPDEStatio
from jinns.data._CubicMeshPDENonStatio import CubicMeshPDENonStatio
from jinns.data._RARParameters import RARParameters
from jinns.nn._hyperpinn import HyperPINN
from jinns.nn._spinn import SPINN
from jinns.data._utils import append_param_batch
from jinns.utils._types import AnyBatch


if TYPE_CHECKING:
    from jinns.data._AbstractDataGenerator import AbstractDataGenerator
    from jinns.utils._types import AnyLoss
    from jinns.parameters._params import Params

    class DataGeneratorWithRAR(AbstractDataGenerator):
        """
        Add the required RAR operands for type checks
        """

        rar_parameters: RARParameters | None
        n_start: int

    RAROperands: TypeAlias = tuple[
        Any,
        Params,
        DataGeneratorWithRAR,
        DataGeneratorParameter | None,
        AnyBatch,
        PRNGKeyArray,
        int,
    ]
    RARReturns: TypeAlias = tuple[
        DataGeneratorWithRAR, DataGeneratorParameter | None, AnyBatch
    ]


def _proceed_to_rar(data: DataGeneratorWithRAR, i: int) -> Bool[Array, " "]:
    """Utilility function with various check to ensure we can proceed with the rar_step.
    Return True if yes, and False otherwise"""
    assert data.rar_parameters is not None
    # Overall checks
    check_list = [
        # check if burn-in period has ended
        jnp.asarray(data.rar_parameters.start_iter <= i),
        # check if enough iterations since last points added
        jnp.asarray(
            (data.rar_parameters.update_every - 1)
            == data.rar_parameters._rar_iter_from_last_sampling
        ),
    ]

    proceed = jnp.all(jnp.array(check_list))
    return proceed


@jax.jit
def _trigger_rar(
    i: int,
    loss: AnyLoss,
    params: Params,
    data: DataGeneratorWithRAR,
    param_data: DataGeneratorParameter | None,
    batch: AnyBatch,
    key: PRNGKeyArray,
) -> tuple[
    AnyLoss, Params, DataGeneratorWithRAR, DataGeneratorParameter | None, AnyBatch
]:
    if data.rar_parameters is None:
        # do nothing.
        return loss, params, data, param_data, batch
    else:
        # update `data` according to rar scheme.
        data, param_data, batch = jax.lax.cond(
            _proceed_to_rar(data, i),
            _rar_step_true,
            _rar_step_false,
            (loss, params, data, param_data, batch, key, i),
        )
        return loss, params, data, param_data, batch


def _rar_step_true(operands: RAROperands) -> RARReturns:
    loss, params, data, param_data, batch, key, it = operands
    assert data.rar_parameters is not None

    if isinstance(loss.u, HyperPINN) or isinstance(loss.u, SPINN):
        raise NotImplementedError("RAR not implemented for hyperPINN and SPINN")

    # the signature we get from tree.reduce is Array | int
    # we are sure this is Array so we use the cast to get rid of int
    res = cast(
        Array,
        jax.tree.reduce(
            jnp.add,
            loss.values_and_grad_per_sample(params, batch)[0].dyn_loss,
            0,
        ),
    )
    res = jnp.atleast_2d(res)
    res = jnp.sum(res**2, axis=-1)
    res_abs = jnp.abs(res)
    batch_size = res_abs.shape[
        0
    ]  # get the batch_size this way so that we are indepedent
    # of which DG subclass we work with

    # Here we create the novelty that be incorporated to the batch and the DGs
    novelty_sample_size = round(batch_size * data.rar_parameters.novelty_proportion)
    if isinstance(data, DataGeneratorODE):
        key, subkey = jax.random.split(key)
        new_samples = data.sample_in_time_domain(subkey, novelty_sample_size)
    elif isinstance(data, CubicMeshPDEStatio) and not isinstance(
        data, CubicMeshPDENonStatio
    ):
        key, *subkeys = jax.random.split(key, data.dim + 1)
        new_samples = CubicMeshPDEStatio.sample_in_omega_domain(
            subkeys,
            novelty_sample_size,
            data.dim,
            data.method,  # type: ignore
            data.min_pts,
            data.max_pts,
        )
    elif isinstance(data, CubicMeshPDENonStatio):
        key, subkey = jax.random.split(key)
        new_samples_times = CubicMeshPDENonStatio.sample_in_time_domain(
            subkey, novelty_sample_size, data.tmin, data.tmax
        )
        if data.dim == 1:
            key, subkeys = jax.random.split(key, 2)
        else:
            key, *subkeys = jax.random.split(key, data.dim + 1)
        new_samples_omega = CubicMeshPDEStatio.sample_in_omega_domain(
            subkeys,
            novelty_sample_size,
            data.dim,
            data.method,  # type: ignore
            data.min_pts,
            data.max_pts,
        )
        new_samples = jnp.concatenate([new_samples_times, new_samples_omega], axis=1)
    else:
        raise ValueError("Wrong DataGenerator type")
    if param_data is not None:
        key, subkey = jax.random.split(key)
        _, _param_n_samples = param_data.generate_data(subkey, batch_size)
        new_param_samples = DGParams(_param_n_samples, "DGParams")

    # RAR-G
    ## Select the m points with higher dynamic loss, they will be conserved
    if data.rar_parameters.method == "G":
        keep_idx = jnp.argsort(res_abs, descending=True)[
            : round(batch_size * (1 - data.rar_parameters.novelty_proportion))
        ]
    # RAR-D
    elif data.rar_parameters.method == "D":
        assert data.rar_parameters.k is not None
        assert data.rar_parameters.c is not None

        res_normalized = res_abs / (jnp.max(res_abs) + 1e-6)
        prop_weights = res_normalized**data.rar_parameters.k + data.rar_parameters.c
        weights = prop_weights / jnp.sum(prop_weights)
        keep_idx = jax.random.choice(
            key,
            a=jnp.arange(res_abs.shape[0]),
            shape=(round(batch_size * (1 - data.rar_parameters.novelty_proportion)),),
            replace=False,
            p=weights.flatten(),
        )
        jax.debug.print("{x}", x=keep_idx)
    else:
        raise ValueError("Unknown RAR method")

    ## Introduce novelty samples with the novelty_sample_size samples that have been sampled
    ## Begin (Update the batch with the novelty)
    ### for each param with a jax.tree.map
    if param_data is not None:
        param_batch = jax.tree.map(
            lambda b, new_b: jnp.concatenate([b[keep_idx], new_b], axis=0),
            batch.param_batch_dict,
            new_param_samples,
        )
    else:
        param_batch = None
    if isinstance(batch, ODEBatch) and isinstance(data, DataGeneratorODE):
        arr = jnp.concatenate([batch.temporal_batch[keep_idx], new_samples], axis=0)
        batch = eqx.tree_at(lambda pt: pt.temporal_batch, batch, arr)

        # Also add the new points ie update the fixed datasets of the DGs
        if data.temporal_batch_size is not None:
            new_times = jax.lax.dynamic_update_slice_in_dim(
                data.times, batch.temporal_batch, data.curr_time_idx, 0
            )
        else:  # NOTE that this branch is currently out of unit tests
            new_times = batch.temporal_batch
        data = eqx.tree_at(lambda m: m.times, data, new_times)
    elif isinstance(batch, PDEStatioBatch) or isinstance(batch, PDENonStatioBatch):
        arr = jnp.concatenate([batch.domain_batch[keep_idx], new_samples], axis=0)
        batch = eqx.tree_at(lambda pt: pt.domain_batch, batch, arr)
        # Also add the new points ie update the fixed datasets of the DGs
        if isinstance(data, CubicMeshPDEStatio) and not isinstance(
            data, CubicMeshPDENonStatio
        ):
            if data.omega_batch_size is not None:
                new_omega = jax.lax.dynamic_update_slice_in_dim(
                    data.omega, batch.domain_batch, data.curr_omega_idx, 0
                )
            else:  # NOTE that this branch is currently out of unit tests
                new_omega = batch.domain_batch
            data = eqx.tree_at(lambda m: m.omega, data, new_omega)
        elif isinstance(data, CubicMeshPDENonStatio):
            if (
                data.domain_batch_size is not None
            ):  # NOTE that this branch is currently out of unit tests
                new_domain = jax.lax.dynamic_update_slice_in_dim(
                    data.domain, batch.domain_batch, data.curr_domain_idx, 0
                )
            else:
                new_domain = batch.domain_batch
            data = eqx.tree_at(lambda m: m.domain, data, new_domain)
    else:
        raise ValueError
    ## Here is the batch we will return
    batch = append_param_batch(batch, param_batch)
    ## End (Update the batch with the novelty)

    # update RAR parameters for all cases
    data = eqx.tree_at(lambda m: m.rar_parameters._rar_iter_from_last_sampling, data, 0)

    # NOTE must return data to be correctly updated because we cannot
    # have side effects in this function that will be jitted
    return data, param_data, batch


def _rar_step_false(operands: RAROperands) -> RARReturns:
    _, _, data, param_data, batch, _, i = operands

    assert data.rar_parameters is not None  # for type checker

    # Add 1 only if we are after the burn in period
    increment = jax.lax.cond(
        i <= data.rar_parameters.start_iter,
        lambda: 0,
        lambda: 1,
    )

    new_rar_iter_from_last_sampling = (
        data.rar_parameters._rar_iter_from_last_sampling + increment
    )
    data = eqx.tree_at(
        lambda m: m.rar_parameters._rar_iter_from_last_sampling,
        data,
        new_rar_iter_from_last_sampling,
    )
    return data, param_data, batch
