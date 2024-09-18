"""
NamedTuples definition
"""

from typing import NamedTuple
from jaxtyping import PyTree, Array, Float, Int, Bool
import optax
from optax import OptState
import jax.numpy as jnp
from jinns.loss._LossODE import LossODE, SystemLossODE
from jinns.loss._LossPDE import (
    LossPDEStatio,
    LossPDENonStatio,
    SystemLossPDE,
)
from jinns.data._DataGenerators import (
    DataGeneratorODE,
    CubicMeshPDEStatio,
    CubicMeshPDENonStatio,
    DataGeneratorParameter,
    DataGeneratorObservations,
    DataGeneratorObservationsMultiPINNs,
)
from jinns.parameters._params import Params


class DataGeneratorContainer(NamedTuple):
    data: DataGeneratorODE | CubicMeshPDEStatio | CubicMeshPDENonStatio
    param_data: DataGeneratorParameter | None = None
    obs_data: DataGeneratorObservations | DataGeneratorObservationsMultiPINNs | None = (
        None
    )


class ValidationContainer(NamedTuple):
    loss: (
        LossODE
        | SystemLossODE
        | LossPDEStatio
        | LossPDENonStatio
        | SystemLossPDE
        | None
    )
    data: DataGeneratorContainer
    hyperparams: PyTree = None
    loss_values: Float[Array, "n_iter"] | None = None


class OptimizationContainer(NamedTuple):
    params: Params
    last_non_nan_params: dict
    opt_state: OptState


class OptimizationExtraContainer(NamedTuple):
    curr_seq: Int
    best_val_params: Params
    early_stopping: Bool = False


class LossContainer(NamedTuple):
    stored_loss_terms: dict
    train_loss_values: Float[Array, "n_iter"]


class StoredObjectContainer(NamedTuple):
    stored_params: list | None
