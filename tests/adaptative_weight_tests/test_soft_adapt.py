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
        update_weight_method="soft_adapt",
        params=jinns.parameters.Params(eq_params={"a": jnp.array(0)}),
        keep_initial_loss_weight_scales=False,
    )
    if loss.update_weight_method is not None:
        loss_new = loss.update_weights(
            1, loss_terms, stored_loss_terms, grad_terms, None
        )
    else:
        loss_new = None

    assert loss_new is not None
    assert jnp.allclose(loss_new.loss_weights.dyn_loss, 0.333, atol=1e-3)
    assert jnp.allclose(loss_new.loss_weights.norm_loss, 0.333, atol=1e-3)
    assert jnp.allclose(loss_new.loss_weights.boundary_loss, 0.333, atol=1e-3)
    assert loss_new.loss_weights.observations is None


def test_weight_update_value_dyn_and_obs_loss_are_tuples():
    """
    Note that the process should automatically include observation field to be
    correct with the formula
    """
    loss_weights = jinns.loss.LossWeightsPDEStatio(
        dyn_loss=(jnp.array(0.1), jnp.array(0.1)),
        norm_loss=jnp.array(0.1),
        boundary_loss=jnp.array(0.1),
        observations=(jnp.array(0.1), jnp.array(0.1)),
    )

    loss_terms = PDEStatioComponents(
        dyn_loss=(jnp.array(0.3), jnp.array(0.3)),
        norm_loss=jnp.array(0.6),
        boundary_loss=jnp.array(1.2),
        observations=(jnp.array(0.5), jnp.array(0.5)),
    )
    stored_loss_terms = PDEStatioComponents(
        dyn_loss=(jnp.ones((10,)) * 0.3, jnp.ones((10,)) * 0.3),
        norm_loss=jnp.ones((10,)) * 0.6,
        boundary_loss=jnp.ones((10,)) * 1.2,
        observations=(jnp.ones((10,)) * 0.5, jnp.ones((10,)) * 0.5),
    )

    grad_terms = PDEStatioComponents(
        dyn_loss=(jnp.array(0.3), jnp.array(0.3)),
        norm_loss=jnp.array(0.6),
        boundary_loss=jnp.array(1.2),
        observations=(jnp.array(0.5), jnp.array(0.5)),
    )

    loss = jinns.loss.LossPDEStatio(
        u=None,
        dynamic_loss=None,
        loss_weights=loss_weights,
        update_weight_method="soft_adapt",
        params=jinns.parameters.Params(eq_params={"a": jnp.array(0)}),
        keep_initial_loss_weight_scales=False,
    )
    if loss.update_weight_method is not None:
        loss_new = loss.update_weights(
            1, loss_terms, stored_loss_terms, grad_terms, None
        )
    assert all(
        jnp.allclose(v, 1 / 6, atol=1e-3) for v in loss_new.loss_weights.dyn_loss
    )
    assert all(
        jnp.allclose(v, 1 / 6, atol=1e-3) for v in loss_new.loss_weights.observations
    )
    assert jnp.allclose(loss_new.loss_weights.norm_loss, 1 / 6, atol=1e-3)
    assert jnp.allclose(loss_new.loss_weights.boundary_loss, 1 / 6, atol=1e-3)


def test_weight_update_value_dyn_and_obs_loss_are_tuples_init_reweighting():
    """
    Note that the process should automatically include observation field to be
    correct with the formula
    """
    loss_weights = jinns.loss.LossWeightsPDEStatio(
        dyn_loss=(jnp.array(0.1), jnp.array(0.1)),
        norm_loss=jnp.array(0.1),
        boundary_loss=jnp.array(0.1),
        observations=(jnp.array(0.1), jnp.array(0.1)),
    )

    loss_terms = PDEStatioComponents(
        dyn_loss=(jnp.array(0.3), jnp.array(0.3)),
        norm_loss=jnp.array(0.6),
        boundary_loss=jnp.array(1.2),
        observations=(jnp.array(0.5), jnp.array(0.5)),
    )
    stored_loss_terms = PDEStatioComponents(
        dyn_loss=(jnp.ones((10,)) * 0.3, jnp.ones((10,)) * 0.3),
        norm_loss=jnp.ones((10,)) * 0.6,
        boundary_loss=jnp.ones((10,)) * 1.2,
        observations=(jnp.ones((10,)) * 0.5, jnp.ones((10,)) * 0.5),
    )

    grad_terms = PDEStatioComponents(
        dyn_loss=(jnp.array(0.3), jnp.array(0.3)),
        norm_loss=jnp.array(0.6),
        boundary_loss=jnp.array(1.2),
        observations=(jnp.array(0.5), jnp.array(0.5)),
    )

    loss = jinns.loss.LossPDEStatio(
        u=None,
        dynamic_loss=None,
        loss_weights=loss_weights,
        update_weight_method="soft_adapt",
        params=jinns.parameters.Params(eq_params={"a": jnp.array(0)}),
        keep_initial_loss_weight_scales=True,
    )
    if loss.update_weight_method is not None:
        loss_new = loss.update_weights(
            1, loss_terms, stored_loss_terms, grad_terms, None
        )
    assert all(
        jnp.allclose(v, 1 / 60, atol=1e-3) for v in loss_new.loss_weights.dyn_loss
    )
    assert all(
        jnp.allclose(v, 1 / 60, atol=1e-3) for v in loss_new.loss_weights.observations
    )
    assert jnp.allclose(loss_new.loss_weights.norm_loss, 1 / 60, atol=1e-3)
    assert jnp.allclose(loss_new.loss_weights.boundary_loss, 1 / 60, atol=1e-3)
