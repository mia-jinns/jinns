"""
Check that we correctly beark from the main while loop as soon as a parameter is Nan or Inf
"""

import pytest
import jax
import jax.numpy as jnp
import optax
import equinox as eqx

import jinns
from jinns.parameters._params import Params


@pytest.fixture
def init_Params_objects():
    n_iter_by_solver = Params(nn_params=1, eq_params={"theta": 2, "kappa": 3})
    eqx_list = ((eqx.nn.Linear, 1, 1),)
    u, init_nn_params = jinns.nn.PINN_MLP.create(
        key=key, eqx_list=eqx_list, eq_type="ODE"
    )
    init_params = Params(
        nn_params=init_nn_params,
        eq_params={"theta": jnp.array([1.0]), "kappa": jnp.array([2.0])},
    )

    class DummyODE(jinns.loss.ODE):
        Tmax: int = 1

        def equation(self, t, u, params):
            return (
                jnp.array([2.0]) * u(t, params).squeeze() ** 2
                + params.eq_params.theta**2
                + params.eq_params.kappa**2
            )

    dynamic_loss = DummyODE()
    loss_kwargs = {
        "u": u,
        "loss_weights": jinns.loss.LossWeightsODE(dyn_loss=jnp.array(1.0)),
        "dynamic_loss": dynamic_loss,
        "derivative_keys": jinns.parameters.DerivativeKeysODE(
            dyn_loss=jinns.parameters.Params(
                nn_params=True, eq_params={"kappa": True, "theta": True}
            ),
            params=init_params,
        ),
        "params": init_params,
    }
    with pytest.warns(UserWarning):
        loss = jinns.loss.LossODE(**loss_kwargs)

    optimizers = Params(
        nn_params=optax.scale(1e8),
        eq_params={
            "theta": optax.scale(1e8),
            "kappa": optax.scale(1e8),
        },
    )

    return n_iter_by_solver, optimizers, init_params, loss


n_iter = 10


key = jax.random.PRNGKey(0)
train_data = jinns.data.DataGeneratorODE(
    key=key,
    nt=10,
    tmin=0,
    tmax=1,
    temporal_batch_size=None,
)


def test_last_non_nan_params_alternate(init_Params_objects):
    n_iter_by_solver, optimizers, init_params, loss = init_Params_objects
    out = jinns.solve_alternate(
        n_iter=n_iter,
        n_iter_by_solver=n_iter_by_solver,
        init_params=init_params,
        optimizers=optimizers,
        data=train_data,
        loss=loss,
        verbose=True,
    )
    # We see that theta is the first updated causing Nan and causing the break
    assert jnp.allclose(out[0].nn_params.layers[0].weight, jnp.array([[-0.03746729]]))
    assert jnp.allclose(out[0].eq_params.kappa, jnp.array([[1.14649349e256]]))
    assert jnp.allclose(out[0].eq_params.theta, jnp.array([[6.18500811e36]]))
