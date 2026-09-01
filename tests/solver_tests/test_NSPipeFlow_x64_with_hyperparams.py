import pytest

import jax
import jax.numpy as jnp
from jax import random
import equinox as eqx
import optax
import jinns

from jinns.parameters._params import update_eq_params


# This fixture will be called with both True and False value
# https://pytest-with-eric.com/fixtures/pytest-fixture-with-arguments/#Parameterized-Fixture
@pytest.fixture(params=[True, False])
def train_NSPipeFlow_init_param_update(request):
    jax.config.update("jax_enable_x64", True)

    key = random.PRNGKey(2)

    L = 1
    R = 0.05

    p_out = 0
    p_in = 0.1

    n = 10
    nb = None
    dim = 2
    xmin = 0
    xmax = xmin + L
    ymin = -R
    ymax = ymin + 2 * R

    rho = 1.0

    method = "uniform"
    key, subkey = random.split(key)
    train_data = jinns.data.CubicMeshPDEStatio(
        key=subkey,
        n=n,
        nb=nb,
        dim=dim,
        min_pts=(xmin, ymin),
        max_pts=(xmax, ymax),
        method=method,
        rar_parameters=jinns.data.RARParameters(
            start_iter=1,
            update_every=2,
            novelty_proportion=0.2,
            method="D",
            k=2.0,
            c=0.0,
            update_params=request.param,
        ),
    )

    method = "grid"
    key, subkey = random.split(key)
    np = 10
    param_train_data = jinns.data.DataGeneratorParameter(
        key=subkey,
        n=np,
        param_batch_size=np,
        param_ranges={"nu": (2e-4, 1.9e-3)},
        method=method,
    )

    def u_p_output_transform(pinn_in, pinn_out, params):
        return jnp.concatenate(
            [
                pinn_out[0:2] * (R**2 - pinn_in[1] ** 2),
                (pinn_in[0] - xmin) / (xmax - xmin) * p_out
                + (xmax - pinn_in[0]) / (xmax - xmin) * p_in
                + (xmin - pinn_in[0]) * (xmax - pinn_in[0]) * pinn_out[2:3],
            ],
            axis=-1,
        )

    eqx_list = (
        (eqx.nn.Linear, 3, 8),
        (jax.nn.swish,),
        (eqx.nn.Linear, 8, 3),
    )

    key, subkey = random.split(key)
    hyperparams = ["nu"]
    u_p_hyper, u_p_init_nn_params = jinns.nn.PINN_MLP.create(
        key=subkey,
        eqx_list=eqx_list,
        eq_type="PDEStatio",
        hyperparams=hyperparams,
        output_transform=u_p_output_transform,
    )
    param_train_data, param_batch = param_train_data.get_batch()
    init_params_hyper = jinns.parameters.Params(
        nn_params=u_p_init_nn_params,
        eq_params={"rho": rho, "nu": None},
    )
    init_params_hyper = update_eq_params(init_params_hyper, param_batch)

    dyn_loss = jinns.loss.NavierStokesMassConservation2DStatio()

    loss_weights = jinns.loss.LossWeightsPDEStatio(dyn_loss=1.0)

    # Catching an expected UserWarning since no border condition is given
    # for this specific PDE resolution
    with pytest.warns(UserWarning):
        loss_hyper = jinns.loss.LossPDEStatio(
            u=u_p_hyper,
            loss_weights=loss_weights,
            dynamic_loss=dyn_loss,
            params=init_params_hyper,
        )

    return init_params_hyper, loss_hyper, train_data, param_train_data, key


def test_update_params_with_RAR(train_NSPipeFlow_init_param_update):
    """
    Check that the DataGeneratorParameters is also updated
    """
    init_params, loss, train_data, param_train_data, key = (
        train_NSPipeFlow_init_param_update
    )

    params = init_params

    tx = optax.adamw(learning_rate=1e-4)
    n_iter = 10

    _, _, _, _, _, _, _, _, _, param_train_data_after, _, _ = jinns.solve(
        init_params=params,
        data=train_data,
        param_data=param_train_data,
        optimizer=tx,
        loss=loss,
        n_iter=n_iter,
        key=key,
    )
    if train_data.rar_parameters.update_params:
        assert not jnp.any(
            jnp.array(
                jax.tree.leaves(
                    jax.tree.map(
                        lambda a, b: jnp.allclose(a, b, atol=1e-5),
                        param_train_data.param_n_samples,
                        param_train_data_after.param_n_samples,
                    )
                )
            )
        )
    else:
        assert jnp.all(
            jnp.array(
                jax.tree.leaves(
                    jax.tree.map(
                        lambda a, b: jnp.allclose(a, b, atol=1e-5),
                        param_train_data.param_n_samples,
                        param_train_data_after.param_n_samples,
                    )
                )
            )
        )
