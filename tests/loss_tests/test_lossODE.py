import pytest

import jax
import jax.numpy as jnp
import equinox as eqx
import jinns
import jinns.data


def test_initial_condition_checks():
    key = jax.random.PRNGKey(2)
    eqx_network = eqx.nn.MLP(2, 1, 128, 3, jax.nn.tanh, key=key)
    u, params = jinns.nn.PINN_MLP.create(eqx_network=eqx_network, eq_type="ODE")
    params = jinns.parameters.Params(nn_params=params)

    # Catching an expected Error because t0 has a bad shape
    with pytest.raises(ValueError):
        _ = jinns.loss.LossODE(
            u=u,
            dynamic_loss=None,
            initial_condition=(jnp.array([[1.0]]), jnp.array([1.0])),
            params=params,
        )

    loss = jinns.loss.LossODE(
        u=u,
        dynamic_loss=None,
        initial_condition=(1.0, jnp.array([1.0])),
        params=params,
    )
    # check that reshaping was done well in __post_init__ checks
    assert loss.initial_condition[0].shape == (1,)

    loss = jinns.loss.LossODE(
        u=u,
        dynamic_loss=None,
        initial_condition=(jnp.array(1.0), jnp.array([1.0])),
        params=params,
    )
    # check that reshaping was done well in __post_init__ checks
    assert loss.initial_condition[0].shape == (1,)
