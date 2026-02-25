import pytest

import jinns


import jax
from jax import random
import jax.numpy as jnp
import equinox as eqx

import optax
from jinns.optimizers import vanilla_ngd


@pytest.fixture
def train_heat_init():
    key = random.PRNGKey(2)
    key, subkey = random.split(key)

    n = 10000
    ni = n
    nb = n
    domain_batch_size = 400
    initial_batch_size = domain_batch_size
    border_batch_size = domain_batch_size // 4
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
        domain_batch_size=domain_batch_size,
        border_batch_size=border_batch_size,
        initial_batch_size=initial_batch_size,
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

    from jinns.loss import PDENonStatio

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
    ngd_optim = optax.chain(
        optax.sgd(learning_rate=1.0),
        optax.scale_by_backtracking_linesearch(max_backtracking_steps=15, verbose=True),
    )
    tx = vanilla_ngd(ngd_optim)  # use jinns custom wrapper to tell `solve` to use ngd
    ngd_params = init_params

    key, subkey = random.split(key, 2)
    (
        ngd_params,
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
        init_params=ngd_params,
        data=train_data,
        optimizer=tx,
        loss=loss,
        n_iter=n_iter,
        print_loss_every=n_iter // 10,
    )

    assert jnp.allclose(total_loss_list[-1], 0.05048845, atol=1e-1)
    assert ngd_params.eq_params.D == init_params.eq_params.D  # should not move
