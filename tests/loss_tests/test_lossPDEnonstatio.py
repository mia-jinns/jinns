import pytest

import jax
import jax.numpy as jnp
import equinox as eqx
import jinns
import jinns.data


def test_initial_condition_time_checks():
    key = jax.random.PRNGKey(2)
    eqx_network = eqx.nn.MLP(2, 1, 128, 3, jax.nn.tanh, key=key)
    u, params = jinns.nn.PINN_MLP.create(
        eqx_network=eqx_network, eq_type="nonstatio_PDE"
    )
    params = jinns.parameters.Params(nn_params=params)

    # Catching an expected Error because t0 has a bad shape
    with pytest.raises(ValueError):
        _ = jinns.loss.LossPDENonStatio(
            u=u,
            dynamic_loss=None,
            initial_condition_time=jnp.array([[1.0]]),
            params=params,
        )

    loss = jinns.loss.LossPDENonStatio(
        u=u,
        dynamic_loss=None,
        initial_condition_time=1.0,
        params=params,
    )
    # check that reshaping was done well in __post_init__ checks
    assert loss.initial_condition_time.shape == (1,)

    loss = jinns.loss.LossPDENonStatio(
        u=u,
        dynamic_loss=None,
        initial_condition_time=jnp.array(1.0),
        params=params,
    )
    # check that reshaping was done well in __post_init__ checks
    assert loss.initial_condition_time.shape == (1,)

    loss = jinns.loss.LossPDENonStatio(
        u=u,
        dynamic_loss=None,
        initial_condition_time=None,
        params=params,
    )
    # check that reshaping was done well in __post_init__ checks
    assert jnp.array_equal(loss.initial_condition_time, jnp.array([0.0]))
