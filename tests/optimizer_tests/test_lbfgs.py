import pytest

import jinns


import jax
from jax import random
import jax.numpy as jnp
import equinox as eqx
import optax

from jinns.loss import PDENonStatio


@pytest.fixture
def train_heat_init():
    jax.config.update("jax_enable_x64", True)
    key = random.PRNGKey(2)
    key, subkey = random.split(key)

    n = 512
    ni = 512
    nb = 512
    dim = 2
    xmin = -1
    xmax = 1
    tmin = 0
    tmax = 1
    method = "uniform"

    train_data = jinns.data.CubicMeshPDENonStatio(
        key=subkey,
        n=n,
        nb=nb,
        ni=ni,
        dim=dim,
        min_pts=(xmin, xmin),
        max_pts=(xmax, xmax),
        tmin=tmin,
        tmax=tmax,
        method=method,
    )

    eqx_list = (
        (eqx.nn.Linear, 3, 25),  # 3 = t + x (2D)
        (jax.nn.tanh,),
        (eqx.nn.Linear, 25, 25),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 25, 1),
    )

    key, subkey = random.split(key)
    u, init_sol_nn_params = jinns.nn.PINN_MLP.create(
        key=subkey, eqx_list=eqx_list, eq_type="PDENonStatio"
    )

    D = jnp.array(0.2)
    init_params = jinns.parameters.Params(
        nn_params=init_sol_nn_params,
        eq_params={"D": D},
    )

    class HeatEquation(PDENonStatio):
        def equation(self, t_x, u, params):
            u_t_x = lambda t_x: u(t_x, params).squeeze()
            u_dt = jax.grad(u_t_x)(t_x)[0:1]
            lap = jinns.loss.laplacian_rev(t_x, u, params, eq_type="PDENonStatio")
            return u_dt - params.eq_params.D * lap

    dyn_loss_heat = HeatEquation()

    boundary_condition = jinns.loss.Dirichlet()

    loss_weights = jinns.loss.LossWeightsPDENonStatio(
        dyn_loss=jnp.array(1.0),
        initial_condition=jnp.array(1.0),
        boundary_loss=None if boundary_condition is None else jnp.array(1.0),
    )

    def u0(x):
        return jnp.sin(jnp.pi * x[0]) * jnp.sin(jnp.pi * x[1])

    loss = jinns.loss.LossPDENonStatio(
        u=u,
        loss_weights=loss_weights,
        dynamic_loss=dyn_loss_heat,
        initial_condition_fun=u0,
        boundary_condition=boundary_condition,
        params=init_params,
    )

    return init_params, loss, train_data, key


def test_heat_ngd_10it(train_heat_init):
    init_params, loss, train_data, key = train_heat_init
    n_iter = 10
    tx = optax.lbfgs()
    params = init_params

    # Below are the stuff passed by the user when they see they have an optax
    # transform with extra args. The example below enables using optax.lbfgs
    # whose update extra args contain in order: value, grad, value_fn,
    # extra_kwargs (for value_fn)
    # the user should then look for the corresponding variable in jinns
    # loss_evaluate_and_standard_gradient and pass the variable name as a
    # string
    extra_optax_args_and_kwargs = {
        "value": "train_loss_value",
        "grad": "params",
        "value_fn": "lambda params, batch: loss.evaluate(params, batch)[0]",
        "batch": "batch",
    }

    key, subkey = random.split(key, 2)
    (
        params,
        total_loss_list,
        loss_by_term_dict,
        train_data,
        loss,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
    ) = jinns.solve(
        init_params=params,
        data=train_data,
        optimizer=tx,
        loss=loss,
        n_iter=n_iter,
        print_loss_every=n_iter // 10,
        extra_optax_args_and_kwargs=extra_optax_args_and_kwargs,
    )

    assert jnp.allclose(total_loss_list[-1], 0.22496643, atol=1e-4)
