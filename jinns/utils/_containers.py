"""
NamedTuples definition
"""

from typing import Union, NamedTuple
from jaxtyping import PyTree
from jax.typing import ArrayLike
import optax
import jax.numpy as jnp
from jinns.loss._LossODE_eqx import LossODE_eqx, SystemLossODE_eqx
from jinns.loss._LossPDE_eqx import (
    LossPDEStatio_eqx,
    LossPDENonStatio_eqx,
    SystemLossPDE_eqx,
)
from jinns.data._DataGenerators_eqx import (
    DataGeneratorODE_eqx,
    CubicMeshPDEStatio_eqx,
    CubicMeshPDENonStatio_eqx,
    DataGeneratorParameter_eqx,
    DataGeneratorObservations_eqx,
    DataGeneratorObservationsMultiPINNs_eqx,
)
from jinns.parameters._params import Params


class DataGeneratorContainer(NamedTuple):
    data: Union[DataGeneratorODE_eqx, CubicMeshPDEStatio_eqx, CubicMeshPDENonStatio_eqx]
    param_data: Union[DataGeneratorParameter_eqx, None] = None
    obs_data: Union[
        DataGeneratorObservations_eqx, DataGeneratorObservationsMultiPINNs_eqx, None
    ] = None


class ValidationContainer(NamedTuple):
    loss: Union[
        LossODE_eqx,
        SystemLossODE_eqx,
        LossPDEStatio_eqx,
        LossPDENonStatio_eqx,
        SystemLossPDE_eqx,
        None,
    ]
    data: DataGeneratorContainer
    hyperparams: PyTree = None
    loss_values: Union[ArrayLike, None] = None


class OptimizationContainer(NamedTuple):
    params: Params
    last_non_nan_params: dict
    opt_state: optax.OptState


class OptimizationExtraContainer(NamedTuple):
    curr_seq: int
    best_val_params: Params
    early_stopping: bool = False


class LossContainer(NamedTuple):
    stored_loss_terms: dict
    train_loss_values: ArrayLike


class StoredObjectContainer(NamedTuple):
    stored_params: Union[list, None]
