"""
Simple test for inverse problem workflow inspired from the notebook of the same
name in jinns documentation
"""

import pytest
import jax
import jax.numpy as jnp
import equinox as eqx
import jinns


@pytest.fixture
def create_problem():
    key = jax.random.PRNGKey(42)
    # True is : theta = 1 / (100 * jnp.pi)
    key, subkey = jax.random.split(key)
    theta_ori = jnp.array([1 / (100 * jnp.pi)])
    theta = theta_ori + jax.random.normal(subkey, shape=(1,)) * 0.1
    observations = jnp.load("Notebooks/Tutorials/burgers_solution_grid.npy")
    key, subkey = jax.random.split(key)
    size_subsample = 50
    time_subsample = jax.random.choice(
        subkey,
        jnp.arange(0, observations.shape[0], 1),
        shape=(size_subsample,),
        replace=True,  # we do not have enough observations
    )
    key, subkey = jax.random.split(key)
    size_subsample = 50
    omega_subsample = jax.random.choice(
        subkey,
        jnp.arange(0, observations.shape[1], 1),
        shape=(size_subsample,),
        replace=True,  # we do not have enough observations
    )
    obs_batch = observations[time_subsample, omega_subsample]

    n = 50
    ni = 50
    nb = 2
    domain_batch_size = 32
    initial_batch_size = 32
    dim = 1
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
        initial_batch_size=initial_batch_size,
        dim=dim,
        min_pts=(xmin,),
        max_pts=(xmax,),
        tmin=tmin,
        tmax=tmax,
        method=method,
    )
    key, subkey = jax.random.split(key)
    obs_data = jinns.data.DataGeneratorObservations(
        key=subkey,
        obs_batch_size=32,
        observed_pinn_in=obs_batch[:, :2],
        observed_values=obs_batch[:, 2:3],
    )
    eqx_list = (
        (eqx.nn.Linear, 2, 10),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 10, 1),
    )
    key, subkey = jax.random.split(key)
    u, init_nn_params = jinns.nn.PINN_MLP.create(
        key=subkey, eqx_list=eqx_list, eq_type="nonstatio_PDE"
    )
    init_params = jinns.parameters.Params(
        nn_params=init_nn_params,
        eq_params={
            "nu": theta,
        },
    )
    loss_weights = jinns.loss.LossWeightsPDENonStatio(
        dyn_loss=1.0,
        initial_condition=1.0,
        boundary_loss=1.0,
        observations=1.0,
    )

    def u0(x):
        return -jnp.sin(jnp.pi * x)

    be_loss = jinns.loss.BurgersEquation(Tmax=1)
    derivative_keys_nu_and_theta = jinns.parameters.DerivativeKeysPDENonStatio.from_str(
        dyn_loss=jinns.parameters.Params(
            nn_params=True,
            eq_params={
                "nu": True,
            },
        ),
        boundary_loss="nn_params",
        initial_condition="nn_params",
        observations="nn_params",
        params=init_params,
    )
    loss_nu_and_theta = jinns.loss.LossPDENonStatio(
        u=u,
        loss_weights=loss_weights,
        dynamic_loss=be_loss,
        derivative_keys=derivative_keys_nu_and_theta,
        omega_boundary_fun=lambda t_dx: 0,
        omega_boundary_condition="dirichlet",
        initial_condition_fun=u0,
    )
    return loss_nu_and_theta, init_params, train_data, obs_data


def test_loss_eval(create_problem):
    loss, params, train_data, obs_data = create_problem

    _, train_batch = train_data.get_batch()
    _, obs_batch = obs_data.get_batch()

    batch = jinns.data.append_obs_batch(train_batch, obs_batch)

    assert jnp.allclose(loss.evaluate(params, batch)[0], 1.7419804, atol=1e-1)
