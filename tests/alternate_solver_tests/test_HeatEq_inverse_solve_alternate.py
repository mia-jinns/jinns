import pytest

from typing import NamedTuple, Any

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

    D_true = jnp.array(0.2)

    def u0(x):
        return jnp.sin(jnp.pi * x[0]) * jnp.sin(jnp.pi * x[1])

    def u_true(t_x):
        t, x = t_x[0], t_x[1:]
        return jnp.exp(-2 * D_true * t * jnp.pi**2) * u0(x)

    # Generate noisy observation
    nobs = 5
    Tmax = 1
    xmin = ymin = -1
    xmax = ymax = 1
    tmin = 0
    tmax = 1

    key, subkey_t, subkey_x, subkey_y = jax.random.split(key, 4)
    noise_lvl = 1e-1
    t_sample = jax.random.uniform(subkey_t, shape=(nobs, 1), minval=0, maxval=Tmax)
    x_sample = jax.random.uniform(subkey_x, shape=(nobs, 1), minval=xmin, maxval=xmax)
    y_sample = jax.random.uniform(subkey_y, shape=(nobs, 1), minval=ymin, maxval=ymax)

    t_x_obs = jnp.concatenate([t_sample, x_sample, y_sample], axis=-1)

    key, subkey = jax.random.split(key, 2)
    u_obs = jax.vmap(u_true)(t_x_obs) + noise_lvl * jax.random.normal(
        subkey, shape=(nobs,)
    )

    # Create a DataGenerator object for the collocation points
    key, subkey = random.split(key)

    n = 20
    ni = n
    nb = n
    dim = 2
    method = "uniform"

    train_data = jinns.data.CubicMeshPDENonStatio(
        key=subkey,
        n=n,
        nb=nb,
        ni=ni,
        dim=dim,
        min_pts=(xmin, xmin),
        max_pts=(ymax, ymax),
        tmin=tmin,
        tmax=tmax,
        method=method,
    )

    # Create a DataGenerator object for the noisy observations

    key, subkey = jax.random.split(key)
    obs_data = jinns.data.DataGeneratorObservations(
        key=subkey,
        obs_batch_size=None,
        observed_pinn_in=t_x_obs,
        observed_values=u_obs,
    )

    eqx_list = (
        (eqx.nn.Linear, 3, 6),  # 3 = t + x (2D)
        (jax.nn.tanh,),
        (eqx.nn.Linear, 6, 1),
    )

    key, subkey = random.split(key)
    u, init_sol_nn_params = jinns.nn.PINN_MLP.create(
        key=subkey, eqx_list=eqx_list, eq_type="PDENonStatio"
    )
    # Initiate parameters : the NN weights + diffusion coef `D`
    delta = -0.15
    D_init = D_true + delta
    init_params = jinns.parameters.Params(
        nn_params=init_sol_nn_params,
        eq_params={"D": D_init},
    )
    # Loss weights
    loss_weights = jinns.loss.LossWeightsODE(
        dyn_loss=1.0, initial_condition=1.0, observations=1.0
    )

    class HeatEquation(jinns.loss.PDENonStatio):
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
        boundary_loss=jnp.array(1.0),
        observations=jnp.array(1.0),
    )

    derivative_keys = jinns.parameters.DerivativeKeysPDENonStatio.from_str(
        dyn_loss=jinns.parameters.Params(
            nn_params=True,
            eq_params={
                "D": True,
            },
        ),
        initial_condition="nn_params",
        boundary_loss="nn_params",
        observations="nn_params",
        params=init_params,
    )

    loss_inverse_pb = jinns.loss.LossPDENonStatio(
        u=u,
        loss_weights=loss_weights,
        dynamic_loss=dyn_loss_heat,
        initial_condition_fun=u0,
        boundary_condition=boundary_condition,
        derivative_keys=derivative_keys,
        params=init_params,
    )

    return init_params, loss_inverse_pb, train_data, obs_data


@pytest.fixture
def train_solve_alternate(test_init):
    init_params, loss, train_data, obs_data = test_init

    # number of alternate iterations
    n_iter = 10  # n_iter_joint
    # number of iterations for each solver
    n_iter_by_solver = jinns.parameters.Params(
        nn_params=1,
        eq_params={"D": 1},
    )

    alternate_txs = jinns.parameters.Params(
        nn_params=optax.adam(learning_rate=1e-3),
        eq_params={
            "D": optax.adam(learning_rate=1e-3),
        },
    )

    key = random.PRNGKey(42)
    key, subkey = random.split(key, 2)

    (
        _,
        total_loss_values,
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
        loss=loss,  # take the complete inverse pb loss!
        optimizers=alternate_txs,
        obs_data=obs_data,
        key=subkey,
    )

    return total_loss_values


@pytest.fixture
def train_alternate_optax(test_init):
    init_params, loss, train_data, obs_data = test_init

    # Code from https://github.com/google-deepmind/optax/issues/993
    class AlternateTxState(NamedTuple):
        step: jnp.ndarray
        tx1_state: Any
        tx2_state: Any

    def alternate_tx(tx1, tx2, every1, every2):
        def init_fn(params):
            return AlternateTxState(
                step=jnp.zeros([], dtype=jnp.int32),
                tx1_state=tx1.init(params),
                tx2_state=tx2.init(params),
            )

        def _update_tx1(updates, state, params=None, **kwargs):
            new_updates, new_state = tx1.update(
                updates, state.tx1_state, params, **kwargs
            )
            return new_updates, state._replace(step=state.step + 1, tx1_state=new_state)

        def _update_tx2(updates, state, params=None, **kwargs):
            new_updates, new_state = tx2.update(
                updates, state.tx2_state, params, **kwargs
            )
            return new_updates, state._replace(step=state.step + 1, tx2_state=new_state)

        def update_fn(updates, state, params=None, **kwargs):
            args = {**{"updates": updates, "state": state, "params": params}, **kwargs}
            return jax.lax.cond(
                state.step % (every1 + every2) >= every1,
                lambda args: _update_tx2(**args),
                lambda args: _update_tx1(**args),
                args,
            )

        return optax.GradientTransformationExtraArgs(init_fn, update_fn)

    n_iter_alternate_optax = 20

    def optimizer_alternate(tx1, tx2, evry1, evry2):
        return alternate_tx(
            optax.partition(
                {"tx": tx1, "zero": optax.set_to_zero()},
                jinns.parameters.Params(nn_params="zero", eq_params={"D": "tx"}),
            ),
            optax.partition(
                {"tx": tx2, "zero": optax.set_to_zero()},
                jinns.parameters.Params(nn_params="tx", eq_params={"D": "zero"}),
            ),
            evry1,
            evry2,
        )

    alternate_txs = optimizer_alternate(
        optax.adam(learning_rate=1e-3), optax.adam(learning_rate=1e-3), 1, 1
    )

    (
        _,
        total_loss_values,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
    ) = jinns.solve(
        init_params=init_params,
        data=train_data,
        optimizer=alternate_txs,
        loss=loss,
        n_iter=n_iter_alternate_optax,
        obs_data=obs_data,
    )

    return total_loss_values


def test_equal_optimization(train_solve_alternate, train_alternate_optax):
    total_loss_values_1 = train_solve_alternate
    total_loss_values_2 = train_alternate_optax

    print(total_loss_values_1, total_loss_values_2)
    assert jnp.allclose(total_loss_values_1, total_loss_values_2, atol=1e-5)
