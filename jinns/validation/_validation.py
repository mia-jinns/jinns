"""
Implements some validation functions and their associated hyperparameter
"""

import copy
from typing import Dict, Union, NamedTuple
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, PyTree, Bool, Int, Float
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


class BaseValidationModule(eqx.Module):
    # Using eqx Module for the DataClass + Pytree inheritance

    call_every: int  # Mandatory for all validation step, tells that the
    # validation step is performed every call_every iterations
    # might be better to expose and use a DataContainer here
    validation_data: Union[DataGeneratorODE, CubicMeshPDEStatio, CubicMeshPDENonStatio]
    validation_param_data: Union[DataGeneratorParameter, None]
    validation_obs_data: Union[
        DataGeneratorObservations, DataGeneratorObservationsMultiPINNs, None
    ]
    early_stopping: Bool  # globally control if early stopping happen


class VanillaValidation(BaseValidationModule):

    loss: Union[callable, LossODE, LossPDEStatio, LossPDENonStatio]
    patience: Union[int, None]
    best_val_loss: jax.Array
    counter: jax.Array  # counts the number of times we did not improve validation loss

    def __init__(
        self,
        loss,
        validation_data,
        validation_param_data,
        validation_obs_data,
        call_every=250,
        early_stopping=True,
        patience=1,
        counter=None,
        best_val_loss=None,
    ):
        super().__init__(
            call_every,
            validation_data,
            validation_param_data,
            validation_obs_data,
            early_stopping,
        )
        self.loss = loss
        self.call_every = call_every
        self.patience = patience
        if best_val_loss is None:
            self.best_val_loss = jnp.array(-jnp.inf)
        else:
            self.best_val_loss = best_val_loss
        if counter is None:
            self.counter = jnp.zeros(1)
        else:
            self.counter = counter

    def __call__(self, params):
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
        print("In-place modif of self.loss.norm_key !")
        # self.loss.norm_key, _ = jax.random.split(self.loss.norm_key)
        (counter, best_val_loss) = jax.lax.cond(
            validation_loss_value < self.best_val_loss,
            lambda _: (jnp.zeros(1), validation_loss_value),  # reset
            lambda operands: (operands[0] + 1, operands[1]),  # increment
            (self.counter, self.best_val_loss),
        )
        print(counter)
        # use eqx.tree_at to update self
        self = eqx.tree_at(lambda t: t.counter, self, counter)
        self = eqx.tree_at(lambda t: t.best_val_loss, self, best_val_loss)
        bool_early_stopping = jax.lax.cond(
            jnp.logical_and(
                jnp.array(self.counter == self.patience)[0],
                jnp.array(self.early_stopping),
            ),
            lambda _: True,
            lambda _: False,
            None,
        )

        return (self, bool_early_stopping, validation_loss_value)


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

    validation = VanillaValidation(
        call_every=250,
        early_stopping=True,
        patience=1000,
        loss=loss,
        validation_data=val_data,
        validation_param_data=None,
        validation_obs_data=None,
        counter=jnp.zeros(1),
    )
    init_params = {"nn_params": init_nn_params, "eq_params": {"nu": 1.0}}

    print(validation.loss is loss)
    loss.evaluate(init_params, val_data.get_batch())
    print(loss.norm_key)
    print("Call validation once")
    validation, _, _ = validation(init_params)
    print(validation.loss is loss)
    print(validation.loss.norm_key == loss.norm_key)
    print("Crate new pytree from validation and call it once")
    new_val = eqx.tree_at(lambda t: t.counter, validation, jnp.array([3.0]))
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
