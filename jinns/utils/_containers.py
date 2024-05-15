"""
NamedTuples definition
"""

from typing import Union, NamedTuple
from jaxtyping import PyTree
from jax.typing import ArrayLike
import optax
import jax.numpy as jnp
from jinns.loss._LossODE import LossODE, SystemLossODE
from jinns.loss._LossPDE import LossPDEStatio, LossPDENonStatio, SystemLossPDE
from jinns.data._DataGenerators import (
    DataGeneratorODE,
    CubicMeshPDEStatio,
    CubicMeshPDENonStatio,
    DataGeneratorParameter,
    DataGeneratorObservations,
    DataGeneratorObservationsMultiPINNs,
)


class DataGeneratorContainer(NamedTuple):
    data: Union[DataGeneratorODE, CubicMeshPDEStatio, CubicMeshPDENonStatio]
    param_data: Union[DataGeneratorParameter, None] = None
    obs_data: Union[
        DataGeneratorObservations, DataGeneratorObservationsMultiPINNs, None
    ] = None


class ValidationContainer(NamedTuple):
    loss: Union[
        LossODE, SystemLossODE, LossPDEStatio, LossPDENonStatio, SystemLossPDE, None
    ]
    data: DataGeneratorContainer
    hyperparams: PyTree = None
    loss_values: Union[ArrayLike, None] = None


class OptimizationContainer(NamedTuple):
    params: dict
    last_non_nan_params: dict
    opt_state: optax.OptState


class OptimizationExtraContainer(NamedTuple):
    curr_seq: int
    seq2seq: Union[dict, None]
    early_stopping: bool = False


class LossContainer(NamedTuple):
    stored_loss_terms: dict
    train_loss_values: ArrayLike


class StoredObjectContainer(NamedTuple):
    stored_params: Union[list, None]
