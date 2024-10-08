"""
Implements some validation functions and their associated hyperparameter
"""

from __future__ import (
    annotations,
)  # https://docs.python.org/3/library/typing.html#constant

import abc
from typing import TYPE_CHECKING, Union
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from jinns.data._DataGenerators import (
    append_obs_batch,
    append_param_batch,
)

if TYPE_CHECKING:
    from jinns.utils._types import *

# Using eqx Module for the DataClass + Pytree inheritance
# Abstract class and abstract/final pattern is used
# see : https://docs.kidger.site/equinox/pattern/


class AbstractValidationModule(eqx.Module):
    """Abstract class representing interface for any validation module. It must
    1. have a ``call_every`` attribute.
    2. implement a ``__call__`` returning ``(AbstractValidationModule, Bool, Array)``
    """

    call_every: eqx.AbstractVar[int] = eqx.field(
        kw_only=True
    )  # Mandatory for all validation step,
    # it tells that the validation step is performed every call_every
    # iterations.

    @abc.abstractmethod
    def __call__(
        self, params: Params | ParamsDict
    ) -> tuple["AbstractValidationModule", bool, Array, bool]:
        raise NotImplementedError


class ValidationLoss(AbstractValidationModule):
    """
    Implementation of a vanilla validation module returning the PINN loss
    on a validation set of collocation points. This can be used as a baseline
    for more complicated validation strategy.
    """

    loss: AnyLoss = eqx.field(kw_only=True)  # NOTE that
    # there used to be a deepcopy here which has been suppressed. 1) No need
    # because loss are now eqx.Module (immutable) so no risk of in-place
    # modification. 2) deepcopy is buggy with equinox, InitVar etc. (see issue
    # #857 on equinox github)
    validation_data: Union[AnyDataGenerator] = eqx.field(kw_only=True)
    validation_param_data: Union[DataGeneratorParameter, None] = eqx.field(
        kw_only=True, default=None
    )
    validation_obs_data: Union[
        DataGeneratorObservations, DataGeneratorObservationsMultiPINNs, None
    ] = eqx.field(kw_only=True, default=None)
    call_every: int = eqx.field(kw_only=True, default=250)  # concrete typing
    early_stopping: bool = eqx.field(
        kw_only=True, default=True
    )  # globally control if early stopping happens

    patience: Union[int] = eqx.field(kw_only=True, default=10)
    best_val_loss: Array = eqx.field(
        converter=jnp.asarray, default_factory=lambda: jnp.array(jnp.inf), kw_only=True
    )

    counter: Array = eqx.field(
        converter=jnp.asarray, default_factory=lambda: jnp.array(0.0), kw_only=True
    )

    def __call__(
        self, params: AnyParams
    ) -> tuple["ValidationLoss", bool, float, AnyParams]:
        # do in-place mutation

        validation_data, val_batch = self.validation_data.get_batch()
        if self.validation_param_data is not None:
            validation_param_data, param_batch = self.validation_param_data.get_batch()
            val_batch = append_param_batch(val_batch, param_batch)
        if self.validation_obs_data is not None:
            validation_obs_data, obs_batch = self.validation_obs_data.get_batch()
            val_batch = append_obs_batch(val_batch, obs_batch)

        validation_loss_value, _ = self.loss(params, val_batch)
        (counter, best_val_loss, update_best_params) = jax.lax.cond(
            validation_loss_value < self.best_val_loss,
            lambda _: (jnp.array(0.0), validation_loss_value, True),  # reset
            lambda operands: (operands[0] + 1, operands[1], False),  # increment
            (self.counter, self.best_val_loss),
        )

        new = eqx.tree_at(lambda t: t.validation_data, self, validation_data)
        if self.validation_param_data is not None:
            new = eqx.tree_at(
                lambda t: t.validation_param_data, new, validation_param_data
            )
        if self.validation_obs_data is not None:
            new = eqx.tree_at(lambda t: t.validation_obs_data, new, validation_obs_data)
        new = eqx.tree_at(lambda t: t.counter, new, counter)
        new = eqx.tree_at(lambda t: t.best_val_loss, new, best_val_loss)

        bool_early_stopping = jax.lax.cond(
            jnp.logical_and(
                jnp.array(self.counter == self.patience),
                jnp.array(self.early_stopping),
            ),
            lambda _: True,
            lambda _: False,
            None,
        )
        # return `new` cause no in-place modification of the eqx.Module
        return (new, bool_early_stopping, validation_loss_value, update_best_params)
