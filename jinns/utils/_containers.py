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

from jinns.parameters._params import Params

if TYPE_CHECKING:
    from jinns.data._AbstractDataGenerator import AbstractDataGenerator
    from jinns.data._DataGeneratorParameter import DataGeneratorParameter
    from jinns.data._DataGeneratorObservations import DataGeneratorObservations
    from jinns.utils._types import AnyLoss


class DataGeneratorContainer(eqx.Module):
    data: AbstractDataGenerator
    param_data: DataGeneratorParameter | None = None
    obs_data: DataGeneratorObservations | None = None


class ValidationContainer(eqx.Module):
    loss: AnyLoss | None
    data: DataGeneratorContainer
    hyperparams: PyTree = None
    loss_values: Float[Array, " n_iter"] | None = None


class OptimizationContainer(eqx.Module):
    params: Params
    last_non_nan_params: Params
    opt_state: OptState
    params_mask: Params = eqx.field(static=True)  # to make params_mask
    # hashable JAX type. See _gradient_step docstring


class OptimizationExtraContainer(eqx.Module):
    curr_seq: int
    best_iter_id: int  # the best iteration number (that which achieves best_val_params and best_val_params)
    best_val_criterion: float  # the best validation criterion at early stopping
    best_val_params: Params  # the best parameter values at early stopping
    early_stopping: Bool = False


class LossContainer(eqx.Module):
    stored_loss_terms: Dict[str, Float[Array, " n_iter"]]
    train_loss_values: Float[Array, " n_iter"]


class StoredObjectContainer(eqx.Module):
    stored_params: Params[Array | None]
