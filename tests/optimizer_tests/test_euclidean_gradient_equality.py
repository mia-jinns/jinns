"""
jinns now offer two types of gradient descents. Standard GD and natural GD.
Standard gradients (also called Euclidean gradients) appear in the Natural GD
from different computations that in Standard GD. Therefore, in this test, we
ensure the two quantities are always identical
"""

from typing import Any

import jinns


import jax
from jax import random
import jax.numpy as jnp
import equinox as eqx

from jinns.loss._loss_components import PDENonStatioComponents
from jinns.solver._utils import _post_process_pytree_of_grad


class LossPDENonStatio_(jinns.loss.LossPDENonStatio):
    """
    To check the equality we need to compute the standard gradients without the
    mean reduction operation (but with a sum), ie a sum_sum_reduction
    As reduction_functions is declared as a ClassVar it cannot be overriden
    We workaround this by creating a subclass only for the test
    """

    reduction_functions: Any = eqx.field(
        static=True,
        default=PDENonStatioComponents(
            dyn_loss=lambda r: jax.tree.map(
                lambda r_: jnp.sum(jnp.sum(r_**2, axis=-1)), r
            ),
            initial_condition=lambda r: jnp.sum(jnp.sum(r**2, axis=-1)),
            boundary_loss=lambda f: jax.tree.reduce(
                jnp.add, jax.tree.map(lambda r_: jnp.sum(jnp.sum(r_**2, axis=-1)), f), 0
            ),
            norm_loss=None,
            observations=None,
        ),
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def test_euclidean_gradients_equality():
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
        boundary_loss=None if boundary_condition is None else jnp.array(1.0),
    )

    def u0(x):
        return jnp.sin(jnp.pi * x[0]) * jnp.sin(jnp.pi * x[1])

    loss = LossPDENonStatio_(
        u=u,
        loss_weights=loss_weights,
        dynamic_loss=dyn_loss_heat,
        initial_condition_fun=u0,
        boundary_condition=boundary_condition,
        params=init_params,
    )

    _, batch = train_data.get_batch()

    ############################################
    ## STANDARD WAY TO GET EUCLIDEAN GRADIENT ##
    ############################################

    # thanks to the custom loss, we retrieve gradients that have not been
    # averaged (over samples) in each loss terms (also summed in the reduction operation)
    # we will make the average (over samples) globally later on
    _, _, euclidean_grad_std = loss.evaluate_with_standard_gradient(
        init_params,
        batch,
    )
    # sum gradients for each loss term
    euclidean_grad_std = jax.tree.map(
        lambda a, b, c: a + b + c,
        euclidean_grad_std.dyn_loss[0].nn_params,
        euclidean_grad_std.boundary_loss.nn_params,
        euclidean_grad_std.initial_condition.nn_params,
        is_leaf=eqx.is_inexact_array,
    )
    # make the gradients a flat array for the final comparison
    euclidean_grad_std = jnp.concatenate(
        [
            l.flatten()
            for l in jax.tree.leaves(euclidean_grad_std, is_leaf=eqx.is_inexact_array)
        ],
        axis=0,
    )
    # divide by the total number of samples to simulate a global average
    # (similar to what's done in NGD)
    euclidean_grad_std /= 1200

    ############################################
    ## EUCLIDEAN GRADIENT VIA NGD COMPUTATIONS##
    ############################################

    r, g = loss.evaluate_with_natural_gradient(
        init_params,
        batch,
    )

    # Flatten the pytree of params as a big (n, p) matrix
    M = _post_process_pytree_of_grad(g)
    R = jnp.concatenate(jax.tree.leaves(r), axis=0)

    euclidean_grad_ngd = jnp.mean(R * M, axis=0)

    # multiply by 2 which is a factor that appears in a complete derivation of
    # NGD (from the mean square error)
    euclidean_grad_ngd *= 2

    ##############
    # COMPARISON #
    ##############
    assert jnp.allclose(euclidean_grad_ngd, euclidean_grad_std)
