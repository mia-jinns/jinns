import pytest

import jax
import jax.numpy as jnp
import equinox as eqx
import jinns
import jinns.data
from jinns.data import ODEBatch


def test_initial_condition_checks():
    key = jax.random.PRNGKey(2)
    eqx_network = eqx.nn.MLP(2, 1, 128, 3, jax.nn.tanh, key=key)
    u, params = jinns.nn.PINN_MLP.create(eqx_network=eqx_network, eq_type="ODE")
    params = jinns.parameters.Params(nn_params=params)

    loss = jinns.loss.LossODE(
        u=u,
        dynamic_loss=None,
        initial_condition=(1.0, 1.0),
        params=params,
    )
    # check that reshaping was done well in __post_init__ checks
    assert loss.initial_condition[0].shape == (1, 1)
    assert loss.initial_condition[1].shape == (1, 1)

    loss = jinns.loss.LossODE(
        u=u,
        dynamic_loss=None,
        initial_condition=(jnp.array(1.0), jnp.array(1.0)),
        params=params,
    )
    # check that reshaping was done well in __post_init__ checks
    assert loss.initial_condition[0].shape == (1, 1)
    assert loss.initial_condition[1].shape == (1, 1)

    # Catching an expected Error because t0 is badly shaped for specifying more
    # than one condition (expected 2D in this case)
    """
    Added in jinns v1.5.1
    """
    with pytest.raises(ValueError):
        loss = jinns.loss.LossODE(
            u=u,
            dynamic_loss=None,
            initial_condition=(
                jnp.array([2.0, 2.0]),
                (jnp.array([[2.0, 2.0], [3.0, 3.0]])),
            ),
            params=params,
        )

    # Catching an expected Error because t0 and u0 don't match in terms of
    # sequence length
    """
    Added in jinns v1.5.1
    """
    with pytest.raises(ValueError):
        _ = jinns.loss.LossODE(
            u=u,
            dynamic_loss=None,
            initial_condition=(jnp.array([1.0, 2.0]), jnp.array([[1.0], [2.0], [3.0]])),
            params=params,
        )

    # Catching an expected Error because t0 and u0 must be arrays if multiple
    # conditions are given
    """
    Added in jinns v1.5.1
    """
    with pytest.raises(ValueError):
        _ = jinns.loss.LossODE(
            u=u,
            dynamic_loss=None,
            initial_condition=((1.0, 2.0), (1.0, 2.0, 3.0)),
            params=params,
        )


def test_new_initial_condition():
    """
    Added in jinns v1.5.1
    """
    key = jax.random.PRNGKey(2)
    eqx_network = eqx.nn.MLP(1, 2, 128, 3, jax.nn.tanh, key=key)
    u, params = jinns.nn.PINN_MLP.create(eqx_network=eqx_network, eq_type="ODE")
    params = jinns.parameters.Params(nn_params=params)

    # we use a dummy u to be able to easily compute expected values but we
    # still need a fake well formed Param object to construct the loss as well
    # as a fake well form ODEBatch
    loss = jinns.loss.LossODE(
        u=lambda t, p: t,
        dynamic_loss=None,
        initial_condition=(1.0, jnp.array([1.0])),
        params=params,
    )
    mses, _ = loss.evaluate_by_terms(params, ODEBatch(None))
    assert jnp.allclose(mses.initial_condition, jnp.array(0.0))

    loss = jinns.loss.LossODE(
        u=lambda t, p: t,
        dynamic_loss=None,
        initial_condition=(jnp.array([[2.0], [1.0]]), jnp.array([[2.0], [1.0]])),
        params=params,
    )
    mses, _ = loss.evaluate_by_terms(params, ODEBatch(None))
    assert jnp.allclose(mses.initial_condition, jnp.array(0.0))

    loss = jinns.loss.LossODE(
        u=lambda t, p: jnp.concatenate([t, t]),
        dynamic_loss=None,
        initial_condition=(
            jnp.array([[2.0], [1.0]]),
            jnp.array([[2.0, 2.0], [1.0, 1.0]]),
        ),
        params=params,
    )
    mses, _ = loss.evaluate_by_terms(params, ODEBatch(None))
    assert jnp.allclose(mses.initial_condition, jnp.array(0.0))
