import pytest


import jax
import jinns
import optax

from jax import random
import jax.numpy as jnp
import equinox as eqx


@pytest.fixture
def test_init():
    jax.config.update("jax_platforms", "cpu")
    jax.config.update("jax_enable_x64", True)

    key = random.PRNGKey(42)
    key, subkey = jax.random.split(key)
    nu_ori = jnp.array([1 / (100 * jnp.pi)])
    observations = jnp.load("Notebooks/Tutorials/burgers_solution_grid.npy")
    key, subkey = jax.random.split(key)
    size_subsample = 10
    time_subsample = jax.random.choice(
        subkey,
        jnp.arange(0, observations.shape[0], 1),
        shape=(size_subsample,),
        replace=True,  # we do not have enough observations
    )
    key, subkey = jax.random.split(key)
    omega_subsample = jax.random.choice(
        subkey,
        jnp.arange(0, observations.shape[1], 1),
        shape=(size_subsample,),
        replace=True,  # we do not have enough observations
    )
    obs_batch = observations[time_subsample, omega_subsample]
    n = 10
    ni = 10
    nb = 2
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
        obs_batch_size=10,
        observed_pinn_in=obs_batch[:, :2],
        observed_values=obs_batch[:, 2:3],
    )
    eqx_list = (
        (eqx.nn.Linear, 2, 2),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 2, 1),
    )
    key, subkey = jax.random.split(key)
    u, init_nn_params = jinns.nn.PINN_MLP.create(
        key=subkey, eqx_list=eqx_list, eq_type="PDENonStatio"
    )
    init_params = jinns.parameters.Params(
        nn_params=init_nn_params,
        eq_params={
            "nu": nu_ori,
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

    be_loss = jinns.loss.BurgersEquation()
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
        boundary_condition=jinns.loss.Dirichlet(),
        initial_condition_fun=u0,
    )
    return init_params, loss_nu_and_theta, train_data, obs_data


@pytest.fixture
def train_solve_alternate_ngd_adam(test_init):
    init_params, loss, train_data, obs_data = test_init
    # number of alternate iterations
    n_iter = 7  # number of total iterations for the outer problem
    # number of iterations for each solver
    n_iter_by_solver = jinns.parameters.Params(
        nn_params=1,
        eq_params={"nu": 1},
    )

    alternate_txs = jinns.parameters.Params(
        nn_params=jinns.optimizers.vanilla_ngd(
            sgd_learning_rate=1.0,
            gram_reg=1e-5,
            linesearch=optax.scale_by_backtracking_linesearch(
                max_backtracking_steps=15, verbose=True
            ),
        ),
        eq_params={
            "nu": optax.adam(1e-3),
        },
    )
    (
        _,
        alternate_total_loss_values,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
    ) = jinns.solve_alternate(
        n_iter=n_iter,
        n_iter_by_solver=n_iter_by_solver,
        init_params=init_params,
        data=train_data,
        loss=loss,
        optimizers=alternate_txs,
        verbose=True,
        obs_data=obs_data,
        key=None,
        print_loss_every=n_iter // 10,
    )

    return alternate_total_loss_values


def test_ngd_adam(train_solve_alternate_ngd_adam):
    total_loss_values = train_solve_alternate_ngd_adam

    assert jnp.allclose(total_loss_values[-1], 0.59156366, atol=1e-5)


@pytest.fixture
def train_solve_alternate_ssbroyden_adam(test_init):
    init_params, loss, train_data, obs_data = test_init
    # number of alternate iterations
    n_iter = 7  # number of total iterations for the outer problem
    # number of iterations for each solver
    n_iter_by_solver = jinns.parameters.Params(
        nn_params=1,
        eq_params={"nu": 1},
    )
    alternate_txs = jinns.parameters.Params(
        nn_params=jinns.optimizers.self_scaled_bfgs_or_broyden(broyden=True),
        eq_params={
            "nu": optax.adam(1e-4),
        },
    )

    # for callback fn definition, body code can incorporate global vairaible,
    # other variables must be arguments. Note that the equinox module is global
    # and will be accessible in subsequent calls
    # Note that the callback functions must accept asme arguments (you can use
    # **kwargs to simplify here)
    def callback_value_fn(opt_params, batch, loss, non_opt_params, params_mask_):
        full_params = eqx.combine(opt_params, non_opt_params)
        return loss.evaluate(full_params, batch)[0]

    def callback_grad_fn(opt_params, batch, loss, non_opt_params, params_mask_):
        full_params = eqx.combine(opt_params, non_opt_params)
        grads = loss.values_and_grads(full_params, batch)[1]
        grads = loss.ponderate_and_sum_gradient(grads)
        return eqx.partition(grads, params_mask_)[0]

    def get_grad_pt(params, params_mask):
        return params.partition(params_mask)[0]

    def get_non_opt_params(params, params_mask):
        return eqx.partition(params, params_mask)[1]

    extra_optax_args_for_solvers = jinns.parameters.Params(
        nn_params={
            "value": jinns.solver.GetJinnsVariableName("train_loss_value"),
            "grad_pt": get_grad_pt,
            "value_fn": callback_value_fn,
            "grad_fn": callback_grad_fn,
            "batch": jinns.solver.GetJinnsVariableName("batch"),
            "loss": jinns.solver.GetJinnsVariableName("loss"),
            "non_opt_params": get_non_opt_params,
            "params_mask_": jinns.solver.GetJinnsVariableName("params_mask"),
        },
        eq_params={"nu": {}},
    )
    (
        _,
        alternate_total_loss_values,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
    ) = jinns.solve_alternate(
        n_iter=n_iter,
        n_iter_by_solver=n_iter_by_solver,
        init_params=init_params,
        data=train_data,
        loss=loss,
        optimizers=alternate_txs,
        verbose=True,
        obs_data=obs_data,
        key=None,
        print_loss_every=n_iter // 10,
        extra_optax_args_and_kwargs_for_solvers=extra_optax_args_for_solvers,
    )

    return alternate_total_loss_values


def test_ssbroyden_and_adam(train_solve_alternate_ssbroyden_adam):
    total_loss_values = train_solve_alternate_ssbroyden_adam

    assert jnp.allclose(total_loss_values[-1], 0.7067825, atol=1e-5)
