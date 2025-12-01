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
    eqx_list = (
        (eqx.nn.Linear, 1, 20),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 20, 1),
        (jnp.exp,),
    )
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
            return jnp.array([2.0])

    dynamic_loss = DummyODE()
    loss_kwargs = {
        "u": u,
        "loss_weights": jinns.loss.LossWeightsODE(dyn_loss=1.0),
        "dynamic_loss": dynamic_loss,
        "derivative_keys": None,  # defaults to nn_params everywhere
        "params": init_params,
    }
    loss = jinns.loss.LossODE(**loss_kwargs)

    optimizers = Params(
        nn_params=optax.adam(learning_rate=1e-3),
        eq_params={
            "theta": optax.adam(learning_rate=1e-3),
            "kappa": optax.adam(learning_rate=1e-3),
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


def test_tracked_parameters(init_Params_objects):
    n_iter_by_solver, optimizers, init_params, loss = init_Params_objects
    tracked_params = jinns.parameters.Params(
        nn_params=None, eq_params={"theta": True, "kappa": True}
    )
    out = jinns.solve_alternate(
        n_iter=n_iter,
        n_iter_by_solver=n_iter_by_solver,
        init_params=init_params,
        optimizers=optimizers,
        data=train_data,
        loss=loss,
        tracked_params=tracked_params,
        verbose=False,
    )
    assert jnp.allclose(out[6].eq_params.theta, 1.0 * jnp.ones((6 * n_iter)))
    assert jnp.allclose(out[6].eq_params.kappa, 2.0 * jnp.ones((6 * n_iter)))


def test_tracked_loss_values(init_Params_objects):
    n_iter_by_solver, optimizers, init_params, loss = init_Params_objects
    tracked_params = jinns.parameters.Params(
        nn_params=None, eq_params={"theta": True, "kappa": True}
    )
    out = jinns.solve_alternate(
        n_iter=n_iter,
        n_iter_by_solver=n_iter_by_solver,
        init_params=init_params,
        optimizers=optimizers,
        data=train_data,
        loss=loss,
        tracked_params=tracked_params,
        verbose=False,
    )
    assert jnp.allclose(out[1], 4.0 * jnp.ones((6 * n_iter)))


def test_schedulers(init_Params_objects):
    """
    We check that the count is OK for each schedule of each param that is
    optimized
    """
    _, _, init_params, loss = init_Params_objects

    n_iter_by_solver = Params(nn_params=4, eq_params={"theta": 2, "kappa": 3})
    nn_scheduler = optax.linear_schedule(
        init_value=1.0,
        end_value=0.01,
        transition_steps=n_iter_by_solver.nn_params * n_iter,
    )
    theta_scheduler = optax.linear_schedule(
        init_value=1.0,
        end_value=0.01,
        transition_steps=n_iter_by_solver.eq_params.theta * n_iter,
    )
    kappa_scheduler = optax.linear_schedule(
        init_value=1.0,
        end_value=0.01,
        transition_steps=n_iter_by_solver.eq_params.kappa * n_iter,
    )
    optimizers = Params(
        nn_params=optax.chain(
            optax.scale_by_adam(),
            optax.scale_by_schedule(nn_scheduler),
            optax.scale(-1.0),
        ),
        eq_params={
            "theta": optax.chain(
                optax.scale_by_adam(),
                optax.scale_by_schedule(theta_scheduler),
                optax.scale(-1.0),
            ),
            "kappa": optax.chain(
                optax.scale_by_adam(),
                optax.scale_by_schedule(kappa_scheduler),
                optax.scale(-1.0),
            ),
        },
    )

    tracked_params = jinns.parameters.Params(
        nn_params=None, eq_params={"theta": True, "kappa": True}
    )
    out = jinns.solve_alternate(
        n_iter=n_iter,
        n_iter_by_solver=n_iter_by_solver,
        init_params=init_params,
        optimizers=optimizers,
        data=train_data,
        loss=loss,
        tracked_params=tracked_params,
        verbose=False,
    )
    opt_state = out[5]
    assert opt_state[0][1].count == 40
    assert opt_state[1].theta[1].count == 20
    assert opt_state[1].kappa[1].count == 30
