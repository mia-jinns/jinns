"""
Test script for custom PINN eqx.Module
"""

import pytest
import jax
import jax.random as random
import jax.numpy as jnp
import equinox as eqx
import optax

import jinns
from jinns.loss import ODE
from jinns.parameters import Params, DerivativeKeysODE

key = random.PRNGKey(2)


@pytest.fixture
def create_pinn_ode():
    eqx_list = (
        (eqx.nn.Linear, 1, 20),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 20, 20),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 20, 20),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 20, 1),
    )
    _, subkey = random.split(key)
    return jinns.nn.PINN_MLP.create(eq_type="ODE", key=subkey, eqx_list=eqx_list)


@pytest.fixture
def create_datagenerator():
    n = 320
    batch_size = 32
    method = "uniform"
    tmin = 0.3
    tmax = 2

    Tmax = 1
    _, subkey = random.split(key)
    train_data = jinns.data.DataGeneratorODE(
        key=subkey,
        nt=n,
        tmin=tmin,
        tmax=tmax,
        temporal_batch_size=batch_size,
        method=method,
    )
    return train_data, tmin, Tmax


@pytest.fixture
def initialize_parameters(create_pinn_ode):
    a = 1.0
    _, init_nn_params = create_pinn_ode
    init_params = jinns.parameters.Params(
        nn_params=init_nn_params,
        eq_params={"a": a},
    )
    return init_params


def create_loss(tmin, Tmax, u, derivative_keys, init_params):
    u0 = 1.848

    class LinearFODE(ODE):

        def equation(self, t, u, params):
            # in log-space
            u_ = lambda t, p: u(t, p)[0]
            du_dt = jax.grad(u_, 0)(t, params)
            return du_dt - params.eq_params["a"]

    fo_loss = LinearFODE(Tmax=Tmax)
    loss_weights = jinns.loss.LossWeightsODE(dyn_loss=2.0, initial_condition=1.0)

    loss = jinns.loss.LossODE(
        u=u,
        loss_weights=loss_weights,
        dynamic_loss=fo_loss,
        initial_condition=(float(tmin), jnp.log(u0)),
        derivative_keys=derivative_keys,
        params=init_params,
    )
    return loss


def train(train_data, params, loss):
    tx = optax.adam(learning_rate=1e0)
    n_iter = 2
    end_params, _, _, _, _, _, _, _, _ = jinns.solve(
        init_params=params,
        data=train_data,
        optimizer=tx,
        loss=loss,
        n_iter=n_iter,
    )
    return end_params


def test_derivative_keys_via_Params_ValueError(initialize_parameters):
    init_params = initialize_parameters
    with pytest.raises(ValueError):
        # This should fail as `initial_condition=None` and params is not given
        _ = DerivativeKeysODE(dyn_loss=Params(nn_params=True, eq_params={"a": True}))


def test_derivative_keys_via_Params_values_updates1(
    create_pinn_ode, initialize_parameters, create_datagenerator
):
    u, _ = create_pinn_ode
    params = initialize_parameters
    train_data, tmin, Tmax = create_datagenerator

    derivative_keys = DerivativeKeysODE(
        dyn_loss=Params(nn_params=True, eq_params={"a": True}), params=params
    )

    # train
    loss = create_loss(tmin, Tmax, u, derivative_keys, params)
    end_params = train(train_data, params, loss)

    assert not jnp.allclose(
        params.nn_params.layers[0].weight, end_params.nn_params.layers[0].weight
    )
    assert not jnp.allclose(params.eq_params["a"], end_params.eq_params["a"])


def test_derivative_keys_via_Params_values_updates2(
    create_pinn_ode, initialize_parameters, create_datagenerator
):
    u, _ = create_pinn_ode
    params = initialize_parameters
    train_data, tmin, Tmax = create_datagenerator

    derivative_keys = DerivativeKeysODE(
        dyn_loss=Params(nn_params=True, eq_params={"a": False}), params=params
    )

    # train
    loss = create_loss(tmin, Tmax, u, derivative_keys, params)
    end_params = train(train_data, params, loss)

    assert not jnp.allclose(
        params.nn_params.layers[0].weight, end_params.nn_params.layers[0].weight
    )
    assert jnp.allclose(params.eq_params["a"], end_params.eq_params["a"])


def test_derivative_keys_via_Params_values_updates3(
    create_pinn_ode, initialize_parameters, create_datagenerator
):
    u, _ = create_pinn_ode
    params = initialize_parameters
    train_data, tmin, Tmax = create_datagenerator

    derivative_keys = DerivativeKeysODE(
        dyn_loss=Params(nn_params=False, eq_params={"a": True}), params=params
    )

    # train
    loss = create_loss(tmin, Tmax, u, derivative_keys, params)
    end_params = train(train_data, params, loss)

    # nn_params are updated because default value for initial_condition is
    # "nn_params"
    assert not jnp.allclose(
        params.nn_params.layers[0].weight, end_params.nn_params.layers[0].weight
    )
    assert not jnp.allclose(params.eq_params["a"], end_params.eq_params["a"])


def test_derivative_keys_via_Params_values_updates4(
    create_pinn_ode, initialize_parameters, create_datagenerator
):
    u, _ = create_pinn_ode
    params = initialize_parameters
    train_data, tmin, Tmax = create_datagenerator

    derivative_keys = DerivativeKeysODE(
        dyn_loss=Params(nn_params=False, eq_params={"a": False}), params=params
    )

    # train
    loss = create_loss(tmin, Tmax, u, derivative_keys, params)
    end_params = train(train_data, params, loss)

    # nn_params are updated because default value for initial_condition is
    # "nn_params"
    assert not jnp.allclose(
        params.nn_params.layers[0].weight, end_params.nn_params.layers[0].weight
    )
    assert jnp.allclose(params.eq_params["a"], end_params.eq_params["a"])


def test_derivative_keys_via_Params_values_updates5(
    create_pinn_ode, initialize_parameters, create_datagenerator
):
    u, _ = create_pinn_ode
    params = initialize_parameters
    train_data, tmin, Tmax = create_datagenerator

    derivative_keys = DerivativeKeysODE(
        dyn_loss=Params(nn_params=False, eq_params={"a": False}),
        initial_condition=Params(nn_params=True, eq_params={"a": True}),
        params=params,
    )

    # train
    loss = create_loss(tmin, Tmax, u, derivative_keys, params)
    end_params = train(train_data, params, loss)

    # W and B expected to move via the diff of initial_condition
    assert not jnp.allclose(
        params.nn_params.layers[0].weight, end_params.nn_params.layers[0].weight
    )
    # a not expected to move since this parameter has no role in
    # initial_condition
    assert jnp.allclose(params.eq_params["a"], end_params.eq_params["a"])


def test_derivative_keys_via_Str_values_updates1(
    create_pinn_ode, initialize_parameters, create_datagenerator
):
    u, _ = create_pinn_ode
    params = initialize_parameters
    train_data, tmin, Tmax = create_datagenerator

    derivative_keys = DerivativeKeysODE.from_str(dyn_loss="nn_params", params=params)

    # train
    loss = create_loss(tmin, Tmax, u, derivative_keys, params)
    end_params = train(train_data, params, loss)

    assert not jnp.allclose(
        params.nn_params.layers[0].weight, end_params.nn_params.layers[0].weight
    )
    assert jnp.allclose(params.eq_params["a"], end_params.eq_params["a"])


def test_derivative_keys_via_Str_values_updates2(
    create_pinn_ode, initialize_parameters, create_datagenerator
):
    u, _ = create_pinn_ode
    params = initialize_parameters
    train_data, tmin, Tmax = create_datagenerator

    derivative_keys = DerivativeKeysODE.from_str(dyn_loss="eq_params", params=params)

    # train
    loss = create_loss(tmin, Tmax, u, derivative_keys, params)
    end_params = train(train_data, params, loss)

    # both are expected to update since by default above we will have
    # initial_condition="nn_params"!
    assert not jnp.allclose(
        params.nn_params.layers[0].weight, end_params.nn_params.layers[0].weight
    )
    assert not jnp.allclose(params.eq_params["a"], end_params.eq_params["a"])


def test_derivative_keys_via_Str_values_updates3(
    create_pinn_ode, initialize_parameters, create_datagenerator
):
    u, _ = create_pinn_ode
    params = initialize_parameters
    train_data, tmin, Tmax = create_datagenerator

    derivative_keys = DerivativeKeysODE.from_str(
        params=params,
        dyn_loss="eq_params",
        initial_condition="eq_params",
    )

    # train
    loss = create_loss(tmin, Tmax, u, derivative_keys, params)
    end_params = train(train_data, params, loss)

    # Default value set observation="nn_params" above. However, nn_params are
    # not exepected to move since there is no observation loss.
    assert jnp.allclose(
        params.nn_params.layers[0].weight, end_params.nn_params.layers[0].weight
    )
    assert not jnp.allclose(params.eq_params["a"], end_params.eq_params["a"])
