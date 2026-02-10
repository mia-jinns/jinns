import pytest

import jax
import jax.numpy as jnp
from jax import random
import equinox as eqx
import optax
import jinns


@pytest.fixture
def train_ReacDiff_init():
    key = random.PRNGKey(2)
    eqx_list = (
        (eqx.nn.Linear, 3, 20),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 20, 20),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 20, 20),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 20, 20),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 20, 1),
    )
    key, subkey = random.split(key)
    u, init_nn_params = jinns.nn.PINN_MLP.create(
        key=subkey, eqx_list=eqx_list, eq_type="PDENonStatio"
    )
    n = 2048
    nb = 500
    ni = 500
    domain_batch_size = 32
    initial_batch_size = 32
    border_batch_size = 32
    dim = 2
    xmin = 0
    xmax = 1
    ymin = 0
    ymax = 1
    tmin = 0
    tmax = 1
    method = "uniform"

    Tmax = 2
    key, subkey = random.split(key)
    train_data = jinns.data.CubicMeshPDENonStatio(
        key=subkey,
        n=n,
        nb=nb,
        ni=ni,
        domain_batch_size=domain_batch_size,
        border_batch_size=border_batch_size,
        initial_batch_size=initial_batch_size,
        dim=dim,
        min_pts=(xmin, ymin),
        max_pts=(xmax, ymax),
        tmin=tmin,
        tmax=tmax,
        method=method,
    )

    mu_init = jnp.array([0.7, 0.15])

    def u0(x):
        return jnp.exp(-jnp.linalg.norm(x - mu_init, axis=-1))[..., None]

    D = 0.05
    r1, r2, r3 = 0.0, -4.0, 2.0
    g = 1.0

    init_params = jinns.parameters.Params(
        nn_params=init_nn_params,
        eq_params={
            "D": jnp.array([D]),
            "r": jnp.array([r1, r2, r3]),
            "g": jnp.array([g]),
        },
    )

    def r_fun(t_x, _, params):
        """must be a jittable function"""
        eq_params = params.eq_params
        r1, r2, r3 = eq_params.r
        x = t_x[1:]
        return jax.lax.switch(
            jnp.amax(
                jnp.nonzero(
                    jnp.array(
                        [
                            True,
                            jnp.logical_or(
                                jnp.logical_and(x[0] > 6 / 20, x[0] < 8 / 20),
                                jnp.logical_and(x[1] > 8 / 20, x[1] < 10 / 20),
                            ),
                            jnp.logical_or(
                                (x[0] - 0.15) ** 2 + (x[1] - 0.15) ** 2 < 0.015,
                                (x[0] - 0.8) ** 2 + (x[1] - 0.80) ** 2 < 0.03,
                            ),
                        ]
                    ),
                    size=3,
                )[0]
            ),
            [lambda _: r1, lambda _: r2, lambda _: r3],
            (),
        )

    fisher_dynamic_loss = jinns.loss.FisherKPP(
        Tmax=Tmax,
        eq_params_heterogeneity={"D": None, "r": r_fun, "g": None},
        dim_x=2,
        params=init_params,
    )
    loss_weights = jinns.loss.LossWeightsPDENonStatio(
        dyn_loss=1,
        initial_condition=1 * Tmax,
        boundary_loss=1 * Tmax,
    )

    boundary_condition = jinns.loss.Neumann()

    loss = jinns.loss.LossPDENonStatio(
        u=u,
        loss_weights=loss_weights,
        dynamic_loss=fisher_dynamic_loss,
        boundary_condition=boundary_condition,
        initial_condition_fun=u0,
        params=init_params,
    )

    return init_params, loss, train_data


@pytest.fixture
def train_ReacDiff_10it(train_ReacDiff_init):
    """
    Fixture that requests a fixture
    """
    init_params, loss, train_data = train_ReacDiff_init

    params = init_params

    tx = optax.adamw(learning_rate=1e-3)
    n_iter = 10
    params, total_loss_list, loss_by_term_dict, _, _, _, _, _, _, _, _, _ = jinns.solve(
        init_params=params, data=train_data, optimizer=tx, loss=loss, n_iter=n_iter
    )
    return total_loss_list[9]


def test_initial_loss_ReacDiff(train_ReacDiff_init):
    init_params, loss, train_data = train_ReacDiff_init
    assert jnp.allclose(
        loss.evaluate(init_params, train_data.get_batch()[1])[0], 0.8385, atol=1e-1
    )


def test_10it_ReacDiff(train_ReacDiff_10it):
    total_loss_val = train_ReacDiff_10it
    assert jnp.allclose(total_loss_val, 0.6822, atol=1e-1)
