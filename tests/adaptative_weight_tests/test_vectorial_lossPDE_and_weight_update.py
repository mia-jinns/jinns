"""
Here we test a few solve iterations with dummy dyn loss, dummy updates and
dummy PINN, to see if vectorial dyn loss and vectorial DGObs work well with
LossPDEStatio and LossPDENonStatio. We check that looking for the correct loss
weights update (taking easily predictable values).

Note: in the similar tests for LossODE we have a non dummy example.
"""

import pytest

import jax
import jax.numpy as jnp
from jax import random
import equinox as eqx
import optax

import jinns
from jinns.data import CubicMeshPDEStatio, CubicMeshPDENonStatio
from jinns.loss import PDEStatio, PDENonStatio


@pytest.fixture
def train_DummyPDEStatio_init():
    key = random.PRNGKey(2)
    key, subkey = random.split(key)
    eqx_list = ((eqx.nn.Linear, 1, 1),)
    key, subkey = random.split(key)
    u, init_nn_params = jinns.nn.PINN_MLP.create(
        key=subkey, eqx_list=eqx_list, eq_type="PDEStatio"
    )
    init_nn_params = eqx.tree_at(
        lambda pt: (pt.layers[0].weight, pt.layers[0].bias),
        init_nn_params,
        (jnp.zeros((1, 1)), jnp.ones((1,))),
    )

    n = 320
    method = "uniform"

    key, subkey = random.split(key)
    train_data = CubicMeshPDEStatio(
        key=subkey,
        n=n,
        min_pts=(1,),
        max_pts=(2,),
        method=method,
        omega_batch_size=None,
        omega_border_batch_size=None,
        dim=1,
    )

    key, subkey = random.split(key)
    obs_data = jinns.data.DataGeneratorObservations(
        key=subkey,
        observed_pinn_in=(
            jnp.ones((1)),
            jnp.ones((1)),
            jnp.ones((1)),
        ),
        observed_values=(
            2 * jnp.ones((1)),
            2 * jnp.ones((1)),
            2 * jnp.ones((1)),
        ),
        obs_batch_size=(None, None, None),
    )

    class DummyLoss1(PDEStatio):
        def equation(self, x, u, params):
            return jnp.array([1.0])

    class DummyLoss2(PDEStatio):
        def equation(self, x, u, params):
            return jnp.array([1.0])

    class DummyLoss3(PDEStatio):
        def equation(self, x, u, params):
            return jnp.array([1.0])

    dyn_loss_eq1 = DummyLoss1()
    dyn_loss_eq2 = DummyLoss2()
    dyn_loss_eq3 = DummyLoss3()

    loss_weights = jinns.loss.LossWeightsPDEStatio(
        dyn_loss=(1, 1, 1), observations=(1, 1, 1)
    )

    init_params = jinns.parameters.Params(nn_params=init_nn_params, eq_params=None)
    loss = jinns.loss.LossPDEStatio(
        u=u,
        loss_weights=loss_weights,
        dynamic_loss=(dyn_loss_eq1, dyn_loss_eq2, dyn_loss_eq3),
        params=init_params,
        update_weight_method="soft_adapt",
        obs_slice=(jnp.s_[...], jnp.s_[...], jnp.s_[...]),
    )

    return init_params, loss, train_data, obs_data


def test_loss_weights_DummyPDEStatio_10it(train_DummyPDEStatio_init):
    init_params, loss, train_data, obs_data = train_DummyPDEStatio_init

    params = init_params

    key = jax.random.PRNGKey(0)
    tx = optax.adam(learning_rate=0)  # No updates on the weights to have a
    # easily predictable series of loss weight updates
    n_iter = 10
    _, _, _, _, loss_new, _, _, stored_weight_terms, _, _, _, _ = jinns.solve(
        n_iter=n_iter,
        loss=loss,
        optimizer=tx,
        init_params=params,
        data=train_data,
        obs_data=obs_data,
        key=key,
    )
    assert jnp.allclose(
        jnp.asarray(stored_weight_terms.dyn_loss)[0, 1:],
        1
        / 6
        * jnp.ones(
            9,
        ),
        atol=1e-3,
    )  # 1st element will be one
    assert jnp.allclose(
        jnp.asarray(stored_weight_terms.observations)[0, 1:],
        1
        / 6
        * jnp.ones(
            9,
        ),
        atol=1e-3,
    )  # 1st element will be one
    assert jnp.allclose(jnp.asarray(loss_new.loss_weights.dyn_loss), 1 / 6, atol=1e-3)


@pytest.fixture
def train_DummyPDENonStatio_init():
    key = random.PRNGKey(2)
    key, subkey = random.split(key)
    eqx_list = ((eqx.nn.Linear, 2, 1),)
    key, subkey = random.split(key)
    u, init_nn_params = jinns.nn.PINN_MLP.create(
        key=subkey, eqx_list=eqx_list, eq_type="PDENonStatio"
    )
    init_nn_params = eqx.tree_at(
        lambda pt: (pt.layers[0].weight, pt.layers[0].bias),
        init_nn_params,
        (jnp.zeros((1, 2)), jnp.ones((1,))),
    )

    n = 320
    method = "uniform"

    key, subkey = random.split(key)
    train_data = CubicMeshPDENonStatio(
        key=subkey,
        n=n,
        ni=n,
        min_pts=(1,),
        max_pts=(2,),
        tmin=0,
        tmax=1,
        method=method,
        domain_batch_size=None,
        border_batch_size=None,
        initial_batch_size=None,
        dim=1,
    )

    key, subkey = random.split(key)
    obs_data = jinns.data.DataGeneratorObservations(
        key=subkey,
        observed_pinn_in=(
            jnp.ones((1, 2)),
            jnp.ones((1, 2)),
            jnp.ones((1, 2)),
        ),
        observed_values=(
            2 * jnp.ones((1)),
            2 * jnp.ones((1)),
            2 * jnp.ones((1)),
        ),
        obs_batch_size=(None, None, None),
    )

    class DummyLoss1(PDENonStatio):
        def equation(self, t_x, u, params):
            return jnp.array([1.0])

    class DummyLoss2(PDENonStatio):
        def equation(self, t_x, u, params):
            return jnp.array([1.0])

    class DummyLoss3(PDENonStatio):
        def equation(self, t_x, u, params):
            return jnp.array([1.0])

    dyn_loss_eq1 = DummyLoss1()
    dyn_loss_eq2 = DummyLoss2()
    dyn_loss_eq3 = DummyLoss3()

    loss_weights = jinns.loss.LossWeightsPDENonStatio(
        dyn_loss=(1, 1, 1), observations=(1, 1, 1)
    )

    init_params = jinns.parameters.Params(nn_params=init_nn_params, eq_params=None)
    loss = jinns.loss.LossPDENonStatio(
        u=u,
        loss_weights=loss_weights,
        dynamic_loss=(dyn_loss_eq1, dyn_loss_eq2, dyn_loss_eq3),
        params=init_params,
        update_weight_method="soft_adapt",
        obs_slice=(jnp.s_[...], jnp.s_[...], jnp.s_[...]),
    )

    return init_params, loss, train_data, obs_data


def test_loss_weights_DummyPDENonStatio_10it(train_DummyPDENonStatio_init):
    init_params, loss, train_data, obs_data = train_DummyPDENonStatio_init

    params = init_params

    key = jax.random.PRNGKey(0)
    tx = optax.adam(learning_rate=0)  # No updates on the weights to have a
    # easily predictable series of loss weight updates
    n_iter = 10
    _, _, _, _, loss_new, _, _, stored_weight_terms, _, _, _, _ = jinns.solve(
        n_iter=n_iter,
        loss=loss,
        optimizer=tx,
        init_params=params,
        data=train_data,
        obs_data=obs_data,
        key=key,
    )
    assert jnp.allclose(
        jnp.asarray(stored_weight_terms.dyn_loss)[0, 1:],
        1
        / 6
        * jnp.ones(
            9,
        ),
        atol=1e-3,
    )  # 1st element will be one
    assert jnp.allclose(
        jnp.asarray(stored_weight_terms.observations)[0, 1:],
        1
        / 6
        * jnp.ones(
            9,
        ),
        atol=1e-3,
    )  # 1st element will be one
    assert jnp.allclose(jnp.asarray(loss_new.loss_weights.dyn_loss), 1 / 6, atol=1e-3)
