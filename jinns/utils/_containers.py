"""
equinox Modules used as containers
"""

from __future__ import (
    annotations,
)  # https://docs.python.org/3/library/typing.html#constant

from typing import TYPE_CHECKING, Dict
from jaxtyping import PyTree, Array, Float, Bool
from optax import OptState
import equinox as eqx

if TYPE_CHECKING:
    from jinns.utils._types import *


class DataGeneratorContainer(eqx.Module):
    data: AnyDataGenerator
    param_data: DataGeneratorParameter | None = None
    obs_data: DataGeneratorObservations | DataGeneratorObservationsMultiPINNs | None = (
        None
    )


class ValidationContainer(eqx.Module):
    loss: AnyLoss | None
    data: DataGeneratorContainer
    hyperparams: PyTree = None
    loss_values: Float[Array, "n_iter"] | None = None


class OptimizationContainer(eqx.Module):
    params: Params
    last_non_nan_params: Params
    opt_state: OptState


class OptimizationExtraContainer(eqx.Module):
    curr_seq: int
    best_iter_id: int  # the best iteration number (that which achieves best_val_params and best_val_params)
    best_val_criterion: float  # the best validation criterion at early stopping
    best_val_params: Params  # the best parameter values at early stopping
    early_stopping: Bool = False


class LossContainer(eqx.Module):
    stored_loss_terms: Dict[str, Float[Array, "n_iter"]]
    train_loss_values: Float[Array, "n_iter"]


class StoredObjectContainer(eqx.Module):
    stored_params: list | None
