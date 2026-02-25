import pytest

import jax
import jax.numpy as jnp
from jax import random
import equinox as eqx
import optax
import jinns


@pytest.fixture
def train_NSPipeFlow_init():
    jax.config.update("jax_enable_x64", False)

    key = random.PRNGKey(2)
    key, subkey = random.split(key)
    d_ = 2
    r = 128
    m = 2 + 1
    eqx_list = (
        (eqx.nn.Linear, 1, 50),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 50, 50),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 50, 50),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 50, 50),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 50, 50),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 50, r * m),
    )
    key, subkey = random.split(key)
    u_p, u_p_init_nn_params = jinns.nn.SPINN_MLP.create(
        subkey, d_, r, eqx_list, "PDEStatio", m
    )
    L = 1
    R = 0.05

    p_out = 0
    p_in = 0.1

    n = 500
    nb = 500
    omega_batch_size = 32
    omega_border_batch_size = 32
    dim = 2
    xmin = 0
    xmax = xmin + L
    ymin = -R
    ymax = ymin + 2 * R

    method = "uniform"
    key, subkey = random.split(key)
    train_data = jinns.data.CubicMeshPDEStatio(
        key=subkey,
        n=n,
        nb=nb,
        omega_batch_size=omega_batch_size,
        omega_border_batch_size=omega_border_batch_size,
        dim=dim,
        min_pts=(xmin, ymin),
        max_pts=(xmax, ymax),
        method=method,
    )

    rho = 1.0
    nu = 0.00061

    # initiate parameters dictionary
    init_params = jinns.parameters.Params(
        nn_params=u_p_init_nn_params,
        eq_params={"rho": rho, "nu": nu},
    )

    class DirichletFacets(jinns.loss.BoundaryConditionAbstract):
        """
        For the facet matching pattern, the last axis of `inputs` corresponds to
        0 -> "xmin"
        1 -> "xmax"
        2 -> "ymin"
        3 -> "ymax"
        """

        def equation_u(self, inputs, u, params):
            """
            Note that we must return a tuple for the facets
            Note that u is a SPINN so we get the dimension to apply the
            boundary condition with a ellipsis followed by a slice (eg [...,
            2:3])
            """
            u1 = u_p(inputs[..., 0], params)[..., 2:3]
            u2 = u_p(inputs[..., 1], params)[..., 2:3]
            u3 = u_p(inputs[..., 2], params)[..., 0:2]
            u4 = u_p(inputs[..., 3], params)[..., 0:2]
            return (u1, u2, u3, u4)

        def equation_f(self, _, __, gridify=False):
            f1 = jnp.array(p_in)
            f2 = jnp.array(p_out)
            f3 = jnp.array([0.0, 0.0])
            f4 = jnp.array([0.0, 0.0])
            return (f1, f2, f3, f4)

    boundary_condition = DirichletFacets()

    dyn_loss = jinns.loss.NavierStokesMassConservation2DStatio()
    loss_weights = jinns.loss.LossWeightsPDEStatio(dyn_loss=1.0, boundary_loss=1.0)
    loss = jinns.loss.LossPDEStatio(
        u=u_p,
        loss_weights=loss_weights,
        dynamic_loss=dyn_loss,
        boundary_condition=boundary_condition,
        params=init_params,
    )
    return init_params, loss, train_data


@pytest.fixture
def train_NSPipeFlow_10it(train_NSPipeFlow_init):
    """
    Fixture that requests a fixture
    """
    init_params, loss, train_data = train_NSPipeFlow_init

    # NOTE we need to waste one get_batch() here to stay synchronized with the
    # notebook
    train_data, _ = train_data.get_batch()

    params = init_params

    tx = optax.adamw(learning_rate=1e-3)
    n_iter = 10
    params, total_loss_list, loss_by_term_dict, _, _, _, _, _, _, _, _, _ = jinns.solve(
        init_params=params, data=train_data, optimizer=tx, loss=loss, n_iter=n_iter
    )
    return total_loss_list[9]


def test_initial_loss_NSPipeFlow(train_NSPipeFlow_init):
    init_params, loss, train_data = train_NSPipeFlow_init

    assert jnp.allclose(
        loss.evaluate(init_params, train_data.get_batch()[1])[0], 0.10145999, atol=1e-1
    )


def test_10it_NSPipeFlow(train_NSPipeFlow_10it):
    total_loss_val = train_NSPipeFlow_10it
    assert jnp.allclose(total_loss_val, 0.01241, atol=1e-1)
