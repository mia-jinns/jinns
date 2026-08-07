from __future__ import (
    annotations,
)  # https://docs.python.org/3/library/typing.html#constant

from types import NoneType
from typing import TYPE_CHECKING, Callable, Literal, TypeAlias, Any, TypedDict, cast
from functools import partial
from jaxtyping import Float, Array, Bool, PRNGKeyArray
import jax
from jax import vmap
import jax.numpy as jnp
import equinox as eqx

from jinns.data._Batchs import ODEBatch, PDENonStatioBatch, PDEStatioBatch
from jinns.data._DataGeneratorParameter import DataGeneratorParameter, DGParams
from jinns.data._DataGeneratorODE import DataGeneratorODE
from jinns.data._CubicMeshPDEStatio import CubicMeshPDEStatio
from jinns.data._CubicMeshPDENonStatio import CubicMeshPDENonStatio
from jinns.loss._DynamicLossAbstract import PDENonStatio
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

        rar_parameters: RARParameterDict | None
        n_start: int
        rar_iter_from_last_sampling: int
        rar_iter_nb: int
        p: Float[Array, " n 1"]

    RAROperands: TypeAlias = tuple[Any, Params, DataGeneratorWithRAR, DataGeneratorParameter | None, AnyBatch, PRNGKeyArray, int]
    RARReturns: TypeAlias = tuple[DataGeneratorWithRAR, DataGeneratorParameter | None, AnyBatch]

class RARParameterDict(TypedDict):
    """
    TypedDict to specify the Residual Adaptative Resampling procedure
    Otherwise a dictionary with keys
    - `start_iter`: the iteration at which we start the RAR sampling scheme (we first have a "burn-in" period).
    - `update_every`: the number of gradient steps taken between
    each update of collocation points in the RAR algo.
    - `novelty_proportion`: the proportion of the batchsize which is replaced
    by new samples at each RAR step
    - `RAR_method`
        - either "G" for RAR-G, ie, new points replace the batch points with the lowest dynamic loss
        - either "D" for RAR-D, ie, new points replace the batch points that have not been selected when
        resampling batch_size * (1 - novelty_proportion) points among the batch points with weigths given 
        by the formula (2) in the article below. In this case, RARParameterDict must specify the keys 'k' and 'c'
        with float values as defined in the formula.
    - `k`: the value of k, only for RAR-D
    - `c`: the value of c, only for RAR-D


    RAR methods as inspired from https://arxiv.org/pdf/2207.10289
    However the critical difference is that the dataset size is fixed. So new points replace others
    """

    start_iter: int
    update_every: int
    novelty_proportion: float
    RAR_method: Literal["G", "D"]
    k: float
    c: float


def _proceed_to_rar(data: DataGeneratorWithRAR, i: int) -> Bool[Array, " "]:
    """Utilility function with various check to ensure we can proceed with the rar_step.
    Return True if yes, and False otherwise"""
    assert data.rar_parameters is not None
    # Overall checks
    check_list = [
        # check if burn-in period has ended
        jnp.asarray(data.rar_parameters["start_iter"] <= i),
        # check if enough iterations since last points added
        jnp.asarray(
            (data.rar_parameters["update_every"] - 1)
            == data.rar_iter_from_last_sampling
        ),
    ]

    proceed = jnp.all(jnp.array(check_list))
    return proceed


@partial(jax.jit, static_argnames=["_rar_step_true", "_rar_step_false"])
def trigger_rar(
    i: int,
    loss: AnyLoss,
    params: Params,
    data: DataGeneratorWithRAR,
    param_data: DataGeneratorParameter | None,
    batch: AnyBatch,
    key: PRNGKeyArray,
    _rar_step_true: Callable[[RAROperands], RARReturns],
    _rar_step_false: Callable[[RAROperands], RARReturns],
) -> tuple[AnyLoss, Params, DataGeneratorWithRAR, DataGeneratorParameter | None, AnyBatch]:
    if data.rar_parameters is None:
        # do nothing.
        return loss, params, data, param_data, batch
    else:
        # update `data` according to rar scheme.
        data = jax.lax.cond(
            _proceed_to_rar(data, i),
            _rar_step_true,
            _rar_step_false,
            (loss, params, data, param_data, batch, key, i),
        )
        return loss, params, data, param_data, batch


def init_rar(
    data: DataGeneratorWithRAR,
) -> tuple[
    DataGeneratorWithRAR,
    Callable[[RAROperands], RARReturns] | None,
    Callable[[RAROperands], RARReturns] | None,
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
            data.rar_parameters["novelty_proportion"]
        )

        data = eqx.tree_at(lambda m: m.rar_iter_from_last_sampling, data, 0)

    return data, _rar_step_true, _rar_step_false


def _rar_step_init(
    novelty_proportion
) -> tuple[
    Callable[[RAROperands], RARReturns],
    Callable[[RAROperands], RARReturns],
]:
    """
    This is a wrapper because the sampling size and
    selected_sample_size, must be treated as static
    in order to slice. So they must be set before jitting and not with the jitted
    dictionary values rar["test_points_nb"] and rar["added_points_nb"]

    This is a kind of manual declaration of static argnums
    """

    def rar_step_true(operands: RAROperands) -> RARReturns:
        loss, params, data, param_data, batch, key, _ = operands
        if isinstance(loss.u, HyperPINN) or isinstance(loss.u, SPINN):
            raise NotImplementedError("RAR not implemented for hyperPINN and SPINN")

        # the signature we get from tree.reduce is Array | int
        # we are sure this is Array so we use the cast to get rid of int
        res = cast(
            Array,
            jax.tree.reduce(
                jnp.add,
                # jax.tree.map(
                #     lambda v: (jnp.linalg.norm(v, axis=-1) ** 2).flatten(), dyn_on_s
                # ),
                loss.values_and_grad_per_sample(params, batch)[0].dyn_loss,
                0,
            ),
        )
        res_abs = jnp.abs(res)
        batch_size = res_abs.shape[0] # get the batch_size this way so that we are indepedent
        # of which DG subclass we work with

        # Here we create the novelty that be incorporated to the batch and the DGs
        novelty_sample_size = int(batch_size * (1 - data.p))
        if isinstance(data, DataGeneratorODE):
            key, subkey = jax.random.split(key)
            new_samples = data.sample_in_time_domain(subkey, novelty_sample_size)
        elif isinstance(data, CubicMeshPDEStatio) and not isinstance(
            data, CubicMeshPDENonStatio
        ):
            key, *subkeys = jax.random.split(key, data.dim + 1)
            new_samples = data.sample_in_omega_domain(subkeys, novelty_sample_size)
        elif isinstance(data, CubicMeshPDENonStatio):
            key, subkey = jax.random.split(key)
            new_samples_times = data.sample_in_time_domain(subkey, novelty_sample_size)
            if data.dim == 1:
                key, subkeys = jax.random.split(key, 2)
            else:
                key, *subkeys = jax.random.split(key, data.dim + 1)
            new_samples_omega = data.sample_in_omega_domain(subkeys, novelty_sample_size)
            new_samples = jnp.concatenate(
                [new_samples_times, new_samples_omega], axis=1
            )
        else:
            raise ValueError("Wrong DataGenerator type")
        if param_data is not None:
            key, subkey = jax.random.split(key)
            _, _param_n_samples = param_data.generate_data(subkey, batch_size)
            new_param_samples = DGParams(_param_n_samples, "DGParams")

        # RAR-G
        ## Select the m points with higher dynamic loss, they will be conserved
        highest_residual_idx = jnp.argsort(res_abs, descending=True)[:int(batch_size * novelty_proportion)]
        # RAR-D
        ## Introduce novelty samples with the novelty_sample_size samples that haev been sampled
        ## Update the batch with the novelty
        ### for each param with a jax.tree.map
        param_batch = jax.tree.map(
            lambda b, new_b: jnp.concatenate([b[highest_residual_idx], new_b], axis=0),
            batch.param_batch_dict, new_param_samples
        )
        if isinstance(batch, ODEBatch):
            arr = jnp.concatenate(
                [batch.temporal_batch[highest_residual_idx], new_samples],
                axis=0
            )
            batch = eqx.tree_at(lambda pt:pt.temporal_batch, batch, arr)
        elif isinstance(batch, PDEStatioBatch) or isinstance(batch, PDENonStatioBatch):
            arr = jnp.concatenate(
                [batch.domain_batch[highest_residual_idx], new_samples],
                axis=0
            )
            batch = eqx.tree_at(lambda pt:pt.domain_batch, batch, arr)
        else:
            raise ValueError

        ## Here is the batch we will return
        batch = append_param_batch(batch, param_batch)

        # add the new points ie update the fixed datasets of the DGs
        if isinstance(data, DataGeneratorODE):
            new_times = data.times.at[data.curr_time_idx:data.curr_time_idx * data.temporal_batch_size].set( # type: ignore
                new_samples
            )
            data = eqx.tree_at(lambda m: m.times, data, new_times)
        elif isinstance(data, CubicMeshPDEStatio) and not isinstance(
            data, CubicMeshPDENonStatio
        ):
            new_omega = data.omega.at[data.curr_omega_idx:data.curr_omega_idx * data.omega_batch_size].set( # type: ignore
                new_samples
            )
            data = eqx.tree_at(lambda m: m.omega, data, new_omega)
        elif isinstance(data, CubicMeshPDENonStatio):
            new_domain = data.domain.at[data.curr_domain_idx:data.curr_domain_idx * data.domain_batch_size].set( # type: ignore
                new_samples
            )
            data = eqx.tree_at(lambda m: m.domain, data, new_domain)

        # update RAR parameters for all cases
        data = eqx.tree_at(lambda m: m.rar_iter_from_last_sampling, data, 0)

        # NOTE must return data to be correctly updated because we cannot
        # have side effects in this function that will be jitted
        return data, param_data, batch

    def rar_step_false(operands: RAROperands) -> RARReturns:
        _, _, data, param_data, batch, _, i = operands

        assert data.rar_parameters is not None # for type checker

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
        return data, param_data, batch

    return rar_step_true, rar_step_false
