import pytest

import jax
import jax.numpy as jnp
import equinox as eqx
import jinns
import jinns.data


def test_t0_checks():
    key = jax.random.PRNGKey(2)
    eqx_network = eqx.nn.MLP(2, 1, 128, 3, jax.nn.tanh, key=key)
    u, params = jinns.nn.PINN_MLP.create(
        eqx_network=eqx_network, eq_type="nonstatio_PDE"
    )
    params = jinns.parameters.Params(nn_params=params)

    # Catching an expected ValueError because t0 has a bad shape
    with pytest.warns(UserWarning), pytest.raises(ValueError):
        _ = jinns.loss.LossPDENonStatio(
            u=u,
            dynamic_loss=None,
            t0=jnp.array([[1.0]]),
            params=params,
        )

    # Catching an expected warning because no BC
    with pytest.warns(UserWarning):
        loss = jinns.loss.LossPDENonStatio(
            u=u,
            dynamic_loss=None,
            t0=1.0,
            params=params,
        )
    # check that reshaping was done well in __post_init__ checks
    assert loss.t0.shape == (1,)

    # Catching an expected warning because no BC
    with pytest.warns(UserWarning):
        loss = jinns.loss.LossPDENonStatio(
            u=u,
            dynamic_loss=None,
            t0=jnp.array(1.0),
            params=params,
        )
    # check that reshaping was done well in __post_init__ checks
    assert loss.t0.shape == (1,)

    # Catching an expected warning because no BC
    with pytest.warns(UserWarning):
        loss = jinns.loss.LossPDENonStatio(
            u=u,
            dynamic_loss=None,
            t0=None,
            params=params,
        )
    # check that reshaping was done well in __post_init__ checks
    assert jnp.array_equal(loss.t0, jnp.array([0.0]))
