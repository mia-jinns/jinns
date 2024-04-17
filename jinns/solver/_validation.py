"""
Implements some validation functions and their associated hyperparameter
"""

from typing import NamedTuple, Union
import jax
import jax.numpy as jnp


class ValidationLossEarlyStoppingHyperparams(NamedTuple):
    patience: Union[int, None]
    best_val_loss: float = jnp.inf
    counter: int = 0  # counts the number of times we did not improve validation loss


def eval_validation_loss_and_early_stopping(
    i,
    hyperparams,
    loss_fn,
    params,
    validation_data,
    validation_param_data,
    validation_obs_data,
):
    """
    The simplest validation loss to implement early stopping
    we ignore validation_param_data and validation_obs_data

    hyperparams is of type ValidationLossEarlyStoppingHyperparams
    """
    if hyperparams.patience <= 0:
        raise ValueError("Early stopping patience parameter must be > 0")

    validation_loss_value = loss_fn(params, validation_data.get_batch())

    (hyperparams.counter, hyperparams.best_val_loss) = jax.lax.cond(
        jnp.logical_and(
            jnp.array(i > 0),
            jnp.array(validation_loss_value < hyperparams.best_val_loss),
        ),
        lambda operands: (0, validation_loss_value),
        lambda operands: operands,
        (hyperparams.counter, hyperparams.best_val_loss),
    )

    bool_early_stopping = jax.lax.cond(
        hyperparams.counter == hyperparams.patience,
        lambda _: True,
        lambda _: False,
        None,
    )

    return (
        bool_early_stopping,
        validation_loss_value,
        validation_data,
        validation_param_data,
        validation_obs_data,
        hyperparams,
    )
