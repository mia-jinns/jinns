"""
Here we test the vectorial loss weight update in the GLV case. What we call
vectorial is the fact that we require jinns to update one weight for each of
the components of the dynamic loss, the latter is then passed as a tuple of 1D
dynamic loss following the new feature
"""

import pytest

import jax
import jax.numpy as jnp
from jax import random
import equinox as eqx
import optax

import jinns
from jinns.data._DataGeneratorODE import DataGeneratorODE
from jinns.loss import ODE


@pytest.fixture
def train_GLV_init():
    jax.config.update("jax_enable_x64", False)
    key = random.PRNGKey(2)
    key, subkey = random.split(key)
    eqx_list = (
        (eqx.nn.Linear, 1, 20),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 20, 20),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 20, 20),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 20, 1),
        (jnp.exp,),
    )
    key, subkey = random.split(key)
    u, init_nn_params = jinns.nn.PINN_MLP.create(
        key=subkey, eqx_list=eqx_list, eq_type="ODE"
    )

    n = 320
    batch_size = 32
    method = "uniform"
    tmin = 0
    tmax = 1

    Tmax = 30
    key, subkey = random.split(key)
    train_data = DataGeneratorODE(
        key=subkey,
        nt=n,
        tmin=tmin,
        tmax=tmax,
        temporal_batch_size=batch_size,
        method=method,
    )

    key, subkey = random.split(key)
    u, init_nn_params = jinns.nn.PINN_MLP.create(
        key=subkey, eqx_list=eqx_list, eq_type="ODE"
    )

    N_0 = jnp.array([10.0, 7.0, 4.0])
    growth_rates = jnp.array([0.1, 0.5, 0.8])
    carrying_capacities = jnp.array([0.04, 0.02, 0.02])
    interactions = (
        -jnp.array([0, 0.001, 0.001]),
        -jnp.array([0.001, 0, 0.001]),
        -jnp.array([0.001, 0.001, 0]),
    )
    init_params = jinns.parameters.Params(
        nn_params=init_nn_params,
        eq_params={
            "carrying_capacities": carrying_capacities,
            "growth_rates": growth_rates,
            "interactions": interactions,
        },
    )

    class GLV_eq1(ODE):
        """Tmax is inherited via ODE"""

        def equation(self, t, u, params):
            u1 = lambda t: jnp.log(u(t, params))[0]
            du1_dt = jax.grad(u1)(t).squeeze()
            carrying_term = params.eq_params.carrying_capacities[0] * jnp.sum(
                u(t, params)
            )
            interaction_term = jnp.sum(
                params.eq_params.interactions[0] * u(t, params).squeeze()
            )
            return (
                du1_dt.squeeze()
                + self.Tmax
                * (-params.eq_params.growth_rates[0] + interaction_term + carrying_term)
            )[None]

    class GLV_eq2(ODE):
        """Tmax is inherited via ODE"""

        def equation(self, t, u, params):
            u2 = lambda t: jnp.log(u(t, params))[1]
            du2_dt = jax.grad(u2)(t).squeeze()
            carrying_term = params.eq_params.carrying_capacities[1] * jnp.sum(
                u(t, params)
            )
            interaction_term = jnp.sum(
                params.eq_params.interactions[1] * u(t, params).squeeze()
            )
            return (
                du2_dt.squeeze()
                + self.Tmax
                * (-params.eq_params.growth_rates[1] + interaction_term + carrying_term)
            )[None]

    class GLV_eq3(ODE):
        """Tmax is inherited via ODE"""

        def equation(self, t, u, params):
            u3 = lambda t: jnp.log(u(t, params))[2]
            du3_dt = jax.grad(u3)(t).squeeze()
            carrying_term = params.eq_params.carrying_capacities[2] * jnp.sum(
                u(t, params)
            )
            interaction_term = jnp.sum(
                params.eq_params.interactions[2] * u(t, params).squeeze()
            )
            return (
                du3_dt.squeeze()
                + self.Tmax
                * (-params.eq_params.growth_rates[2] + interaction_term + carrying_term)
            )[None]

    dyn_loss_eq1 = GLV_eq1(Tmax=Tmax)
    dyn_loss_eq2 = GLV_eq2(Tmax=Tmax)
    dyn_loss_eq3 = GLV_eq3(Tmax=Tmax)

    loss_weights = jinns.loss.LossWeightsODE(
        dyn_loss=(1, 1, 1), initial_condition=1 * Tmax
    )

    loss = jinns.loss.LossODE(
        u=u,
        loss_weights=loss_weights,
        dynamic_loss=(dyn_loss_eq1, dyn_loss_eq2, dyn_loss_eq3),
        initial_condition=(float(tmin), jnp.array([N_0[0], N_0[1], N_0[2]])),
        params=init_params,
        update_weight_method="soft_adapt",
    )

    return init_params, loss, train_data


def test_initial_loss_GLV(train_GLV_init):
    init_params, loss, train_data = train_GLV_init
    _, batch = train_data.get_batch()
    assert jnp.allclose(
        loss.evaluate(init_params, batch)[0],
        4928.112,  # NOTE that here we compare to the value attained in the
        # classical case, since at init there is no loss weight update yet
        atol=1e-1,
    )


@pytest.fixture
def train_GLV_10it(train_GLV_init):
    """
    Fixture that requests a fixture
    """
    init_params, loss, train_data = train_GLV_init

    # NOTE we need to waste one get_batch() here to stay synchronized with the
    # notebook
    train_data, batch = train_data.get_batch()
    _ = loss.evaluate(init_params, batch)[0]
    # NOTE the following line is not accurate as it skips one batch
    # but this is to comply with behaviour of 1st gen of DataGenerator
    # (which had this bug see issue #5) so that we can keep old tests values to
    # know we are doing the same
    train_data = eqx.tree_at(
        lambda m: m.curr_time_idx, train_data, train_data.temporal_batch_size
    )

    params = init_params

    key = jax.random.PRNGKey(0)
    tx = optax.adam(learning_rate=1e-3)
    n_iter = 10
    params, total_loss_list, loss_by_term_dict, data, _, _, _, _, _, _ = jinns.solve(
        n_iter=n_iter,
        loss=loss,
        optimizer=tx,
        init_params=params,
        data=train_data,
        key=key,
    )
    return total_loss_list[9]


def test_10it_GLV(train_GLV_10it):
    total_loss_val = train_GLV_10it
    assert jnp.allclose(total_loss_val, 1184.3315, atol=1e-1)
