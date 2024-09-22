"""
equinox Modules used as containers
"""

from typing import Dict
from jaxtyping import PyTree, Array, Float, Bool
from optax import OptState
import equinox as eqx
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


class DataGeneratorContainer(eqx.Module):
    data: DataGeneratorODE | CubicMeshPDEStatio | CubicMeshPDENonStatio
    param_data: DataGeneratorParameter | None = None
    obs_data: DataGeneratorObservations | DataGeneratorObservationsMultiPINNs | None = (
        None
    )


class ValidationContainer(eqx.Module):
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


class OptimizationContainer(eqx.Module):
    params: Params
    last_non_nan_params: Params
    opt_state: OptState


class OptimizationExtraContainer(eqx.Module):
    curr_seq: int
    best_val_params: Params
    early_stopping: Bool = False


class LossContainer(eqx.Module):
    stored_loss_terms: Dict[str, Float[Array, "n_iter"]]
    train_loss_values: Float[Array, "n_iter"]


class StoredObjectContainer(eqx.Module):
    stored_params: list | None
