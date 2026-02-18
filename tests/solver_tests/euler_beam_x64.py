# Implements the Euler-Beam in 1d
# Not many tests aside custom BC
# It is mainly a test for user-defined BC
import pytest
import jax.numpy as jnp
import jinns
import jax.random as random
import jax

import equinox as eqx

from jaxtyping import Array, Float
from jinns.loss import BoundaryConditionAbstract
from jinns.parameters import Params

from jinns.loss import PDEStatio
from functools import partial

jax.config.update("jax_enable_x64", True)


@pytest.fixture
def create_ebeam_loss():
    key = random.PRNGKey(2)
    key, subkey = random.split(key)

    # Create the neural network architecture for the PINN with `equinox`.
    dim = 1
    eqx_list = (
        (eqx.nn.Linear, dim, 20),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 20, 20),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 20, 20),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 20, 1),
    )
    key, subkey = random.split(key)
    u, init_nn_params = jinns.nn.PINN_MLP.create(
        key=subkey,
        eqx_list=eqx_list,
        eq_type="PDEStatio",
    )

    n = 10
    nb = 2
    omega_batch_size = None  # Use full batch
    omega_border_batch_size = None  # no mini-batches in dimension 1
    dim = 1
    min_pts = (0.0,)
    max_pts = (1.0,)
    method = "grid"

    key, subkey = random.split(key)
    train_data = jinns.data.CubicMeshPDEStatio(
        key=subkey,
        n=n,
        nb=nb,
        omega_batch_size=omega_batch_size,
        omega_border_batch_size=omega_border_batch_size,
        dim=dim,
        min_pts=min_pts,
        max_pts=max_pts,
        method=method,
    )

    class EulerBeamLoss(PDEStatio):
        def equation(self, x, u, params):
            x = x[0]
            u_x = lambda x: partial(u, params=params)(
                x
            ).squeeze()  # squeeze to return a 1d Array -> compatibility with grad()

            dx4 = jax.grad(jax.grad(jax.grad(jax.grad(u_x))))(x)
            return jnp.array([dx4]) + 1

    init_params = jinns.parameters.Params(eq_params={}, nn_params=init_nn_params)

    euler_beam_loss = EulerBeamLoss()
    euler_beam_loss.equation(jnp.zeros((1,)), u, init_params)

    loss_weights = jinns.loss.LossWeightsPDEStatio(
        dyn_loss=jnp.array([1.0]), boundary_loss=jnp.array([1.0])
    )

    return u, init_params, euler_beam_loss, train_data, key, loss_weights


def test_user_defined_bc_euler_beam(create_ebeam_loss):
    u, params, euler_beam_loss, train_data, key, loss_weights = create_ebeam_loss

    # -- Define the boundary of the Euler-Beam problem
    class EulerBeamBC(BoundaryConditionAbstract):
        r"""
        Implements 1d Euler-Beam boundary condition

        This omega boundary condition enforces Euler-Beam boundary conditions.
        This is a stationary PDE in 1D with two high-order boundary condition at
        t=1
        $$
            u''(1) = u'''(1) = 0,
        $$
        and Dirichlet + Neumann at t=0
        $$
            u(0) = u'(0) = 0
        $$

        __Note__: if using a batch.param_batch_dict, we need to resolve the
        vmapping axes here however params["eq_params"] has already been fed with
        the batch in the `evaluate()` of `LossPDE*`.
        """

        def equation_u(
            self,
            inputs: Float[Array, " dim"],
            u: jinns.nn.AbstractPINN,
            params: Params[Array],
        ) -> tuple[Float[Array, " eq_dim"]]:
            inputs = inputs.squeeze()

            # Freeze params + use version of `u` with scalar output
            u_x = lambda x: u(x, params).squeeze()
            d1 = jax.grad(u_x)
            d2 = jax.grad(d1)
            d3 = jax.grad(d2)

            # left boundary (x=0)
            lx = u_x(inputs[..., 0])
            ldx = d1(inputs[..., 0])

            # right boundary(x=1)
            rdx2 = d2(inputs[..., 1])
            rdx3 = d3(inputs[..., 1])

            return tuple(jnp.array([cond]) for cond in [lx, ldx, rdx2, rdx3])

        def equation_f(
            self,
            inputs: Float[Array, " dim"],
            params: Params[Array],
            gridify: bool = False,
        ) -> Float[Array, " eq_dim"]:
            return tuple(jnp.zeros((1,)) for _ in range(4))

    border_batch = train_data.get_batch()[1].border_batch

    ebeam_bc = EulerBeamBC()
    res_bc = ebeam_bc.evaluate(border_batch, u, params)

    assert len(res_bc) == 4
    for cond in res_bc:
        assert isinstance(cond, jnp.ndarray)
        assert cond.shape == (1,)

    loss = jinns.loss.LossPDEStatio(
        u=u,
        loss_weights=loss_weights,
        dynamic_loss=euler_beam_loss,
        boundary_condition=ebeam_bc,
        params=params,
    )

    train_data, batch = train_data.get_batch()
    _ = jax.eval_shape(loss, params, batch=batch)

    # /!\ High-compilation time when jit compiling
    # Deactivate solve() test for now

    # tx = optax.adam(learning_rate=1e-4)
    # n_iter = int(10)  # as in dde, but converges before

    # params, total_loss_list, loss_by_term_dict, data, loss, _, _, _, _, _, _, _ = (
    #     jinns.solve(
    #         init_params=params,
    #         data=train_data,
    #         optimizer=tx,
    #         loss=loss,
    #         n_iter=n_iter,
    #         print_loss_every=1000,
    #     )
    # )
