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

    loss = jinns.loss.LossPDENonStatio(
        u=None,
        dynamic_loss=None,
        loss_weights=loss_weights,
        update_weight_method="lr_annealing",
        params=jinns.parameters.Params(eq_params={"a": jnp.array([0])}),
    )
    if loss.update_weight_method is not None:
        loss_new = loss.update_weights(
            1, loss_terms, stored_loss_terms, grad_terms, None
        )

    assert jnp.allclose(loss_new.loss_weights.dyn_loss, 1.0, atol=1e-3)
    assert jnp.allclose(
        loss_new.loss_weights.norm_loss, 0.1 * (0.6) + 0.9 * (0.3 / 0.6), atol=1e-3
    )
    assert jnp.allclose(
        loss_new.loss_weights.boundary_loss, 0.1 * (1.2) + 0.9 * (0.3 / 1.2), atol=1e-3
    )
    assert jnp.allclose(
        loss_new.loss_weights.initial_condition,
        0.1 * (0.5) + 0.9 * (0.3 / 0.5),
        atol=1e-3,
    )
    assert loss_new.loss_weights.observations is None
