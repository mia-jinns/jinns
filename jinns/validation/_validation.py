"""
Implements some validation functions and their associated hyperparameter
"""

import copy
import abc
from typing import Union
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, PyTree, Int
import jinns
import jinns.data
from jinns.loss import LossODE, LossPDENonStatio, LossPDEStatio
from jinns.data._DataGenerators import (
    DataGeneratorODE,
    CubicMeshPDEStatio,
    CubicMeshPDENonStatio,
    DataGeneratorParameter,
    DataGeneratorObservations,
    DataGeneratorObservationsMultiPINNs,
    append_obs_batch,
    append_param_batch,
)
import jinns.loss

# Using eqx Module for the DataClass + Pytree inheritance
# Abstract class and abstract/final pattern is used
# see : https://docs.kidger.site/equinox/pattern/


class AbstractValidationModule(eqx.Module):
    """Abstract class representing interface for any validation module. It must
    1. have a ``call_every`` attribute.
    2. implement a ``__call__`` returning ``(AbstractValidationModule, Bool, Array)``
    """

    call_every: eqx.AbstractVar[Int]  # Mandatory for all validation step,
    # it tells that the validation step is performed every call_every
    # iterations.

    @abc.abstractmethod
    def __call__(
        self, params: PyTree
    ) -> tuple["AbstractValidationModule", Bool, Array]:
        raise NotImplementedError


class ValidationLoss(AbstractValidationModule):
    """
    Implementation of a vanilla validation module returning the PINN loss
    on a validation set of collocation points. This can be used as a baseline
    for more complicated validation strategy.
    """

    loss: Union[callable, LossODE, LossPDEStatio, LossPDENonStatio] = eqx.field(
        converter=copy.deepcopy
    )
    validation_data: Union[DataGeneratorODE, CubicMeshPDEStatio, CubicMeshPDENonStatio]
    validation_param_data: Union[DataGeneratorParameter, None] = None
    validation_obs_data: Union[
        DataGeneratorObservations, DataGeneratorObservationsMultiPINNs, None
    ] = None
    call_every: Int = 250  # concrete typing
    early_stopping: Bool = True  # globally control if early stopping happens

    patience: Union[Int] = 10
    best_val_loss: Array = eqx.field(
        converter=jnp.asarray, default_factory=lambda: jnp.array(jnp.inf)
    )

    counter: Array = eqx.field(
        converter=jnp.asarray, default_factory=lambda: jnp.array(0.0)
    )

    def __call__(self, params) -> tuple["ValidationLoss", Bool, Array]:
        # do in-place mutation
        val_batch = self.validation_data.get_batch()
        if self.validation_param_data is not None:
            val_batch = append_param_batch(
                val_batch, self.validation_param_data.get_batch()
            )
        if self.validation_obs_data is not None:
            val_batch = append_obs_batch(
                val_batch, self.validation_obs_data.get_batch()
            )

        validation_loss_value, _ = self.loss(params, val_batch)
        (counter, best_val_loss) = jax.lax.cond(
            validation_loss_value < self.best_val_loss,
            lambda _: (jnp.array(0.0), validation_loss_value),  # reset
            lambda operands: (operands[0] + 1, operands[1]),  # increment
            (self.counter, self.best_val_loss),
        )

        # use eqx.tree_at to update attributes
        # (https://github.com/patrick-kidger/equinox/issues/396)
        new = eqx.tree_at(lambda t: t.counter, self, counter)
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
        return (new, bool_early_stopping, validation_loss_value)


if __name__ == "__main__":
    import jax
    import jax.numpy as jnp
    import jax.random as random
    from jinns.loss import BurgerEquation

    key = random.PRNGKey(1)
    key, subkey = random.split(key)

    n = 50
    nb = 2 * 2 * 10
    nt = 10
    omega_batch_size = 10
    omega_border_batch_size = 10
    temporal_batch_size = 4
    dim = 1
    xmin = 0
    xmax = 1
    tmin, tmax = 0, 1
    method = "uniform"

    val_data = jinns.data.CubicMeshPDENonStatio(
        subkey,
        n,
        nb,
        nt,
        omega_batch_size,
        omega_border_batch_size,
        temporal_batch_size,
        dim,
        (xmin,),
        (xmax,),
        tmin,
        tmax,
        method,
    )

    eqx_list = [
        [eqx.nn.Linear, 2, 50],
        [jax.nn.tanh],
        [eqx.nn.Linear, 50, 50],
        [jax.nn.tanh],
        [eqx.nn.Linear, 50, 50],
        [jax.nn.tanh],
        [eqx.nn.Linear, 50, 50],
        [jax.nn.tanh],
        [eqx.nn.Linear, 50, 50],
        [jax.nn.tanh],
        [eqx.nn.Linear, 50, 2],
    ]

    key, subkey = random.split(key)
    u = jinns.utils.create_PINN(
        subkey, eqx_list, "nonstatio_PDE", 2, slice_solution=jnp.s_[:1]
    )
    init_nn_params = u.init_params()

    dyn_loss = BurgerEquation()
    loss_weights = {"dyn_loss": 1, "boundary_loss": 10, "observations": 10}

    key, subkey = random.split(key)
    loss = jinns.loss.LossPDENonStatio(
        u=u,
        loss_weights=loss_weights,
        dynamic_loss=dyn_loss,
        norm_key=subkey,
        norm_borders=(-1, 1),
    )
    print(id(loss))
    validation = ValidationLoss(
        call_every=250,
        early_stopping=True,
        patience=1000,
        loss=loss,
        validation_data=val_data,
        validation_param_data=None,
    )
    print(id(validation.loss) is not id(loss))  # should be True (deepcopy)

    init_params = {"nn_params": init_nn_params, "eq_params": {"nu": 1.0}}

    print(validation.loss is loss)
    loss.evaluate(init_params, val_data.get_batch())
    print(loss.norm_key)
    print("Call validation once")
    validation, _, _ = validation(init_params)
    print(validation.loss is loss)
    print(validation.loss.norm_key == loss.norm_key)
    print("Crate new pytree from validation and call it once")
    new_val = eqx.tree_at(lambda t: t.counter, validation, jnp.array(3.0))
    print(validation.loss is new_val.loss)  # FALSE
    # test if attribute have been modified
    new_val, _, _ = new_val(init_params)
    print(f"{new_val.loss is loss=}")
    print(f"{loss.norm_key=}")
    print(f"{validation.loss.norm_key=}")
    print(f"{new_val.loss.norm_key=}")
    print(f"{new_val.loss.norm_key == loss.norm_key=}")
    print(f"{new_val.loss.norm_key == validation.loss.norm_key=}")
    print(new_val.counter)
    print(validation.counter)
