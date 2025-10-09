"""
Test the prior_loss implementation
"""

import jax.numpy as jnp
import jinns
from jinns.loss._loss_components import PDEStatioComponents


def test_weight_update_value():
    """
    Note that the process should automatically exclude observation field to be
    correct with the formula
    """
    loss_weights = jinns.loss.LossWeightsPDEStatio(
        dyn_loss=jnp.array(0.3),
        norm_loss=jnp.array(0.6),
        boundary_loss=jnp.array(1.2),
    )

    loss_terms = PDEStatioComponents(
        dyn_loss=jnp.array(0.3),
        norm_loss=jnp.array(0.6),
        boundary_loss=jnp.array(1.2),
        observations=None,
    )
    stored_loss_terms = PDEStatioComponents(
        dyn_loss=jnp.ones((10,)) * 0.3,
        norm_loss=jnp.ones((10,)) * 0.6,
        boundary_loss=jnp.ones((10,)) * 1.2,
        observations=None,
    )

    grad_terms = PDEStatioComponents(
        dyn_loss=jnp.array(0.3),
        norm_loss=jnp.array(0.6),
        boundary_loss=jnp.array(1.2),
        observations=None,
    )

    loss = jinns.loss.LossPDEStatio(
        u=None,
        dynamic_loss=None,
        loss_weights=loss_weights,
        update_weight_method="prior_loss",
        params=jinns.parameters.Params(eq_params={"a": jnp.array(0)}),
        keep_initial_loss_weight_scales=False,
    )
    if loss.update_weight_method is not None:
        loss_new = loss.update_weights(
            1, loss_terms, stored_loss_terms, grad_terms, None
        )

    assert jnp.allclose(loss_new.loss_weights.dyn_loss, 1 / 0.3, atol=1e-3)
    assert jnp.allclose(loss_new.loss_weights.norm_loss, 1 / 0.6, atol=1e-3)
    assert jnp.allclose(loss_new.loss_weights.boundary_loss, 1 / 1.2, atol=1e-3)
    assert loss_new.loss_weights.observations is None
