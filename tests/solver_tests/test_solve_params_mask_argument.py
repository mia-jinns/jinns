import pytest

import jax
import jax.numpy as jnp
from jax import random
import equinox as eqx
import optax
import jinns


@pytest.fixture
def train_Burgers_init():
    jax.config.update("jax_enable_x64", False)
    key = random.PRNGKey(2)
    eqx_list = (
        (eqx.nn.Linear, 2, 32),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 32, 32),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 32, 32),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 32, 1),
    )
    key, subkey = random.split(key)
    u, init_nn_params = jinns.nn.PINN_MLP.create(
        key=subkey, eqx_list=eqx_list, eq_type="nonstatio_PDE"
    )

    n = 2500
    ni = 200
    nb = 200
    dim = 1
    xmin = -1
    xmax = 1
    tmin = 0
    tmax = 1
    Tmax = 1
    method = "uniform"

    key, subkey = random.split(key)
    train_data = jinns.data.CubicMeshPDENonStatio(
        key=subkey,
        n=n,
        nb=nb,
        ni=ni,
        dim=dim,
        min_pts=(xmin,),
        max_pts=(xmax,),
        tmin=tmin,
        tmax=tmax,
        method=method,
    )

    nu = 1 / (100 * jnp.pi)
    init_params = jinns.parameters.Params(
        nn_params=init_nn_params, eq_params={"nu": nu}
    )

    def u0(x):
        return -jnp.sin(jnp.pi * x)

    be_loss = jinns.loss.BurgersEquation(Tmax=Tmax)

    loss_weights = jinns.loss.LossWeightsPDENonStatio(
        dyn_loss=1, initial_condition=100, boundary_loss=1
    )

    loss_kwargs = {
        "u": u,
        "loss_weights": loss_weights,
        "dynamic_loss": be_loss,
        "omega_boundary_fun": lambda t_dx: 0,
        "omega_boundary_condition": "dirichlet",
        "initial_condition_fun": u0,
        "params": init_params,
    }

    return init_params, loss_kwargs, train_data


def test_Burgers_10it(train_Burgers_init):
    """
    Fixture that requests a fixture
    """
    init_params, loss_kwargs, train_data = train_Burgers_init

    loss_nn_params = jinns.loss.LossPDENonStatio(
        **loss_kwargs
    )  # here by default the  gradients wrt to nu will be blocked with
    # derivative keys set to "nn_params" for all the losses

    params = init_params
    tx = optax.adamw(learning_rate=1e-3)
    n_iter = 10
    params_nn_params, total_loss_list_nn_params, _, _, _, _, _, _, _, _, _, _ = (
        jinns.solve(
            init_params=params,
            data=train_data,
            optimizer=tx,
            loss=loss_nn_params,
            n_iter=n_iter,
        )
    )

    params_mask = jinns.parameters.Params(
        nn_params=True, eq_params=jax.tree.map(lambda ll: False, init_params.eq_params)
    )
    derivative_keys = jinns.parameters.DerivativeKeysPDENonStatio.from_str(
        params=init_params,
        dyn_loss="both",
        boundary_loss="both",
        initial_condition="both",
    )

    loss_params_mask = jinns.loss.LossPDENonStatio(
        **(loss_kwargs | {"derivative_keys": derivative_keys})
    )  # here the derivative keys are explicitly taken to update nu! But we hope
    # that optimization will not update nu thanks to params mask that is passed
    # to jinns.solve()

    params = init_params
    tx = optax.adamw(learning_rate=1e-3)
    n_iter = 10
    params_params_mask, total_loss_list_params_mask, _, _, _, _, _, _, _, _, _, _ = (
        jinns.solve(
            init_params=params,
            data=train_data,
            optimizer=tx,
            loss=loss_params_mask,
            n_iter=n_iter,
            params_mask=params_mask,
        )
    )

    loss_both_no_mask = jinns.loss.LossPDENonStatio(
        **(loss_kwargs | {"derivative_keys": derivative_keys})
    )  # here the derivative keys are explicitly taken to update nu! But we hope
    # that optimization will not update nu thanks to params mask that is passed
    # to jinns.solve()

    params = init_params
    tx = optax.adamw(learning_rate=1e-3)
    n_iter = 10
    params_both_no_mask, _, _, _, _, _, _, _, _, _, _, _ = jinns.solve(
        init_params=params,
        data=train_data,
        optimizer=tx,
        loss=loss_both_no_mask,
        n_iter=n_iter,
    )  # Here nu should update

    assert jnp.allclose(total_loss_list_nn_params, total_loss_list_params_mask)
    assert jnp.allclose(params_nn_params.eq_params.nu, params_params_mask.eq_params.nu)
    assert params_both_no_mask.eq_params.nu != params_nn_params.eq_params.nu
