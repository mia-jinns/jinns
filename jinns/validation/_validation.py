"""
Implements some validation functions and their associated hyperparameter
"""

import abc
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
    best_val_loss: Float = jnp.inf
    counter: Int = 0  # counts the number of times we did not improve validation loss

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

        (counter, best_val_loss) = jax.lax.cond(
            validation_loss_value < self.best_val_loss,
            lambda operands: (0, validation_loss_value),  # reset
            lambda operands: (operands[0] + 1, operands[1]),  # increment
            (self.counter, self.best_val_loss),
        )
        # use eqx.tree_at to update self
        self = eqx.tree_at(lambda t: t.counter, self, counter)
        self = eqx.tree_at(lambda t: t.best_val_loss, self, best_val_loss)

        bool_early_stopping = jax.lax.cond(
            jnp.logical_and(
                jnp.array(self.counter == self.patience), jnp.array(self.early_stopping)
            ),
            lambda _: True,
            lambda _: False,
            None,
        )

        return (bool_early_stopping, validation_loss_value)


if __name__ == "__main__":
    import jax
    import jax.numpy as jnp
    import jax.random as random
    from jinns.loss import BurgerEquation

    key = random.PRNGKey(1)
    key, subkey = random.split(key)

    n = 50
    nb = 2 * 2 * 10
    omega_batch_size = 10
    omega_border_batch_size = 10
    dim = 2
    xmin = 0
    xmax = 1
    method = "uniform"

    val_data = jinns.data.CubicMeshPDEStatio(
        subkey,
        n,
        nb,
        omega_batch_size,
        omega_border_batch_size,
        dim,
        (xmin, xmin),
        (xmax, xmax),
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
        subkey, eqx_list, "statio_PDE", 2, slice_solution=jnp.s_[:1]
    )
    init_nn_params = u.init_params()

    dyn_loss = BurgerEquation()
    loss_weights = {"dyn_loss": 1, "boundary_loss": 10, "observations": 10}

    key, subkey = random.split(key)
    loss = jinns.loss.LossPDEStatio(
        u=u,
        loss_weights=loss_weights,
        dynamic_loss=dyn_loss,
        norm_key=subkey,
        norm_borders=(-1, 1),
    )

    validation = VanillaValidation(
        counter=jnp.array(10.2),
        call_every=250,
        early_stopping=True,
        patience=10,
        loss=loss,
        validation_data=val_data,
        validation_param_data=None,
        validation_obs_data=None,
    )

    # print(eqx.tree_at(lambda t: t.n, val_data, 10))

    # eqx.tree_at(lambda t: t.loss, validation, 2)
    # eqx.tree_at(lambda t: t.counter, validation, 1.2)

    class MyMod(eqx.Module):
        attr: int
        linear0: eqx.Module

    class MyChildMod(MyMod):
        loss: LossPDEStatio
        counter: int = 0

        def __call__(self, a):
            return a + self.count

    mod = MyChildMod(
        attr=None,
        linear0=None,
        counter=10,
        loss=jinns.loss._LossPDE.LossPDEAbstract(u=u, loss_weights={}),
    )
    print(mod)
    eqx.tree_at(lambda t: t.counter, mod, 11)  # WORKS
