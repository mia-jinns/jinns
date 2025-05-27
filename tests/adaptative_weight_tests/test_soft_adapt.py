"""
Test the soft adapt implementation
https://docs.nvidia.com/deeplearning/physicsnemo/physicsnemo-sym/user_guide/theory/advanced_schemes.html#softadapt
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
        update_method="soft_adapt",
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

    if loss_weights.update is not None:
        loss_weights_new = loss_weights.update(
            loss_terms, stored_loss_terms, grad_terms
        )

    assert jnp.allclose(loss_weights_new.dyn_loss, 0.333, atol=1e-3)
    assert jnp.allclose(loss_weights_new.norm_loss, 0.333, atol=1e-3)
    assert jnp.allclose(loss_weights_new.boundary_loss, 0.333, atol=1e-3)
    assert loss_weights_new.observations is None
