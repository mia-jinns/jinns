"""
Test the lr annealing algorithm
"""

import jax.numpy as jnp
import jinns
from jinns.loss._loss_components import PDENonStatioComponents


def test_lr_annealing_update():
    """
    Note that the process should automatically exclude observation field to be
    correct with the formula
    """
    loss_weights = jinns.loss.LossWeightsPDENonStatio(
        dyn_loss=jnp.array(1.0),
        norm_loss=jnp.array(0.6),
        boundary_loss=jnp.array(1.2),
        initial_condition=jnp.array(0.5),
        update_method="lr_annealing",
    )

    loss_terms = PDENonStatioComponents(
        dyn_loss=jnp.array(0.3),
        norm_loss=jnp.array(0.6),
        boundary_loss=jnp.array(1.2),
        observations=None,
        initial_condition=jnp.array(0.5),
    )
    stored_loss_terms = PDENonStatioComponents(
        dyn_loss=jnp.ones((10,)) * 0.3,
        norm_loss=jnp.ones((10,)) * 0.6,
        boundary_loss=jnp.ones((10,)) * 1.2,
        observations=None,
        initial_condition=jnp.array(0.5),
    )

    grad_terms = PDENonStatioComponents(
        dyn_loss=jnp.array(0.3),
        norm_loss=jnp.array(0.6),
        boundary_loss=jnp.array(1.2),
        observations=None,
        initial_condition=jnp.array(0.5),
    )

    if loss_weights.update is not None:
        loss_weights_new = loss_weights.update(
            loss_terms, stored_loss_terms, grad_terms
        )

    assert jnp.allclose(loss_weights_new.dyn_loss, 1.0, atol=1e-3)
    assert jnp.allclose(
        loss_weights_new.norm_loss, 0.1 * (0.6) + 0.9 * (0.3 / 0.6), atol=1e-3
    )
    assert jnp.allclose(
        loss_weights_new.boundary_loss, 0.1 * (1.2) + 0.9 * (0.3 / 1.2), atol=1e-3
    )
    assert jnp.allclose(
        loss_weights_new.initial_condition, 0.1 * (0.5) + 0.9 * (0.3 / 0.5), atol=1e-3
    )
    assert loss_weights_new.observations is None
