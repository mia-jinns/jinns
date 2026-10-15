"""
Tests for Neumann boundary condition
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import jinns

jax.config.update("jax_enable_x64", True)

n = 5
nb = 16
ni = 5
xmin = ymin = 0
xmax = ymax = 1
tmin = 0
tmax = 1

key = jax.random.PRNGKey(0)


def test_1D_statio():
    eqx_list = (
        (eqx.nn.Linear, 1, 2),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 2, 1),
    )
    u, init_nn_params = jinns.nn.PINN_MLP.create(
        key=key, eqx_list=eqx_list, eq_type="PDEStatio"
    )
    params = jinns.parameters.Params(nn_params=init_nn_params, eq_params={})
    train_data = jinns.data.CubicMeshPDEStatio(
        key=key, n=n, nb=nb, dim=1, min_pts=(xmin,), max_pts=(xmax,)
    )
    loss = jinns.loss.LossPDEStatio(
        u=u,
        dynamic_loss=None,
        loss_weights=jinns.loss.LossWeightsPDEStatio(boundary_loss=1.0),
        boundary_condition=jinns.loss.Neumann(),
        params=params,
    )
    assert jnp.allclose(loss.evaluate(params, train_data.get_batch()[1])[0], 0.31723032)


def test_1D_nonstatio():
    eqx_list = (
        (eqx.nn.Linear, 2, 2),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 2, 1),
    )
    u, init_nn_params = jinns.nn.PINN_MLP.create(
        key=key, eqx_list=eqx_list, eq_type="PDENonStatio"
    )
    params = jinns.parameters.Params(nn_params=init_nn_params, eq_params={})
    train_data = jinns.data.CubicMeshPDENonStatio(
        key=key,
        n=n,
        nb=nb,
        ni=ni,
        dim=1,
        tmin=tmin,
        tmax=tmax,
        min_pts=(xmin,),
        max_pts=(xmax,),
    )
    loss = jinns.loss.LossPDENonStatio(
        u=u,
        dynamic_loss=None,
        loss_weights=jinns.loss.LossWeightsPDENonStatio(boundary_loss=1.0),
        boundary_condition=jinns.loss.Neumann(),
        params=params,
    )
    assert jnp.allclose(loss.evaluate(params, train_data.get_batch()[1])[0], 0.00524858)


def test_2D_statio():
    eqx_list = (
        (eqx.nn.Linear, 2, 2),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 2, 1),
    )
    u, init_nn_params = jinns.nn.PINN_MLP.create(
        key=key, eqx_list=eqx_list, eq_type="PDEStatio"
    )
    params = jinns.parameters.Params(nn_params=init_nn_params, eq_params={})
    train_data = jinns.data.CubicMeshPDEStatio(
        key=key, n=n, nb=nb, dim=2, min_pts=(xmin, ymin), max_pts=(xmax, ymax)
    )
    loss = jinns.loss.LossPDEStatio(
        u=u,
        dynamic_loss=None,
        loss_weights=jinns.loss.LossWeightsPDEStatio(boundary_loss=1.0),
        boundary_condition=jinns.loss.Neumann(),
        params=params,
    )
    assert jnp.allclose(loss.evaluate(params, train_data.get_batch()[1])[0], 0.11476325)


def test_2D_nonstatio():
    eqx_list = (
        (eqx.nn.Linear, 3, 2),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 2, 1),
    )
    u, init_nn_params = jinns.nn.PINN_MLP.create(
        key=key, eqx_list=eqx_list, eq_type="PDENonStatio"
    )
    params = jinns.parameters.Params(nn_params=init_nn_params, eq_params={})
    train_data = jinns.data.CubicMeshPDENonStatio(
        key=key,
        n=n,
        nb=nb,
        ni=ni,
        dim=2,
        tmin=tmin,
        tmax=tmax,
        min_pts=(xmin, ymin),
        max_pts=(xmax, ymax),
    )
    loss = jinns.loss.LossPDENonStatio(
        u=u,
        dynamic_loss=None,
        loss_weights=jinns.loss.LossWeightsPDENonStatio(boundary_loss=1.0),
        boundary_condition=jinns.loss.Neumann(),
        params=params,
    )
    assert jnp.allclose(loss.evaluate(params, train_data.get_batch()[1])[0], 0.08888188)
