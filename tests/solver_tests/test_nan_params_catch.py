"""
We put a huge learning rate to introduce NaN in parameters

We check that the solve() function correctly stops for the NaN parameter reason
We check that the returned parameters do not contain NaN values but the last
value before the NaN values
"""

import pytest

import jax
import jax.numpy as jnp
from jax import random
import equinox as eqx
import optax
import jinns

from jinns.loss import ODE
from jinns.utils._utils import _check_nan_in_pytree


@pytest.fixture
def train_init():
    jax.config.update("jax_enable_x64", False)
    key = random.PRNGKey(2)
    eqx_list = (
        (eqx.nn.Linear, 1, 20),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 20, 20),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 20, 20),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 20, 1),
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

    Tmax = 1
    key, subkey = random.split(key)
    train_data = jinns.data.DataGeneratorODE(
        key=subkey,
        nt=n,
        tmin=tmin,
        tmax=tmax,
        temporal_batch_size=batch_size,
        method=method,
    )
    # initial conditions and growth
    u0 = 1.0
    a = 1.0
    init_params = jinns.parameters.Params(
        nn_params=init_nn_params,
        eq_params={"a": a},
    )

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
        params=init_params,
    )

    return init_params, loss, train_data


@pytest.fixture
def train_1it(train_init, capsys):
    """
    Fixture that requests a fixture

    We put a huge learning rate to introduce NaN in parameters
    """
    init_params, loss, train_data = train_init

    params = init_params

    tx = optax.adam(learning_rate=1e90)
    n_iter = 10
    params, total_loss_list, loss_by_term_dict, _, loss, _, _, _, _ = jinns.solve(
        init_params=params, data=train_data, optimizer=tx, loss=loss, n_iter=n_iter
    )
    captured = capsys.readouterr()
    return init_params, params, captured.out


def test_no_nan_params_check(train_1it):
    _, params, _ = train_1it
    assert not _check_nan_in_pytree(params)


def test_init_params_equals_params(train_1it):
    init_params, params, _ = train_1it
    assert jax.tree_util.tree_all(
        jax.tree_util.tree_map(jnp.allclose, params, init_params)
    )


def test_break_reason(train_1it):
    _, _, captured_out = train_1it
    assert "NaN values" in captured_out
