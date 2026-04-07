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
    jax.config.update("jax_enable_x64", True)
    key = random.PRNGKey(2)
    eqx_list = (
        (eqx.nn.Linear, 1, 8),
        (jax.nn.swish,),
        (eqx.nn.Linear, 8, 1),
    )

    eqx_list_hyper = (
        (eqx.nn.Linear, 1, 15),
        (jax.nn.tanh,),
        (
            eqx.nn.Linear,
            15,
            1000,
        ),  # 1000 is a random guess, it will automatically be filled with the correct value
    )
    key, subkey = random.split(key)
    hyperparams = ["a"]
    hypernet_input_size = 1
    u_hyper, u_init_nn_params_hyper = jinns.nn.HyperPINN.create(
        key=subkey,
        eqx_list=eqx_list,
        eq_type="ODE",
        hyperparams=hyperparams,
        hypernet_input_size=hypernet_input_size,
        eqx_list_hyper=eqx_list_hyper,
    )
    n = 32
    batch_size = 32
    method = "uniform"
    tmin = 0
    tmax = 1

    Tmax = 3
    key, subkey = random.split(key)
    train_data = DataGeneratorODE(
        key=subkey,
        nt=n,
        tmin=tmin,
        tmax=tmax,
        temporal_batch_size=batch_size,
        method=method,
    )
    method = "grid"
    key, subkey = random.split(key)
    np = 100
    param_batch_size = 32  # must be equal to batch size of the main DataGenerator
    param_train_data = jinns.data.DataGeneratorParameter(
        key=subkey,
        n=np,
        param_batch_size=param_batch_size,
        param_ranges={"a": (0.1, 2)},
        method=method,
    )
    param_train_data, param_batch = param_train_data.get_batch()

    init_params_hyper = jinns.parameters.Params(
        nn_params=u_init_nn_params_hyper,
        eq_params={"a": None},
    )
    init_params_hyper = jinns.parameters.update_eq_params(
        init_params_hyper, param_batch
    )
    loss_weights = jinns.loss.LossWeightsODE(dyn_loss=1.0, initial_condition=1.0 * Tmax)

    class LinearFODE(ODE):
        def equation(self, t, u, params):
            # in log-space
            u_ = lambda t, p: u(t, p)[0]
            du_dt = jax.grad(u_, 0)(t, params)
            return du_dt - self.Tmax * params.eq_params.a

    fo_loss = LinearFODE(Tmax=Tmax)
    loss_hyper = jinns.loss.LossODE(
        u=u_hyper,
        loss_weights=loss_weights,
        dynamic_loss=fo_loss,
        initial_condition=(float(tmin), jnp.log(1.0)),
        params=init_params_hyper,
    )
    return init_params_hyper, loss_hyper, train_data, param_train_data


@pytest.fixture
def train_GLV_10it(train_GLV_init):
    """
    Fixture that requests a fixture
    """
    init_params_hyper, loss_hyper, train_data, param_train_data = train_GLV_init

    tx = optax.adam(learning_rate=1e-3)
    n_iter = 10
    (
        params_hyper,
        total_loss_list_hyper,
        loss_by_term_dict_hyper,
        train_data,
        loss_hyper,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
    ) = jinns.solve(
        init_params=init_params_hyper,
        data=train_data,
        param_data=param_train_data,
        optimizer=tx,
        loss=loss_hyper,
        n_iter=n_iter,
    )
    return total_loss_list_hyper[9]


def test_initial_loss_linear_fo_hyperpinn(train_GLV_init):
    init_params_hyper, loss_hyper, train_data, param_train_data = train_GLV_init
    _, batch = train_data.get_batch()
    _, param_batch = param_train_data.get_batch()
    batch = jinns.data.append_param_batch(batch, param_batch)
    assert jnp.allclose(
        loss_hyper.evaluate(init_params_hyper, batch)[0],
        12.27783293,
        atol=1e-5,
    )


def test_10it_linear_fo_hyperpinn(train_GLV_10it):
    total_loss_val = train_GLV_10it
    assert jnp.allclose(total_loss_val, 13.51906343, atol=1e-5)
