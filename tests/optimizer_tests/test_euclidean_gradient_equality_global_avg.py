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

from jinns.optimizers._utils_ngd import (
    _loss_weights_to_components,
    weighted_jac_and_residuals,
    _get_sqrt_weights_per_sample,
)


class LossPDENonStatio_(jinns.loss.LossPDENonStatio):
    """
    To check the equality we need to compute the standard gradients without the
    mean reduction operation (but with a sum), ie a sum_sum_reduction
    As reduction_functions is declared as a ClassVar it cannot be overriden
    We workaround this by creating a subclass only for the test
    """

    _reduction_functions: Any = eqx.field(
        static=True,
        default=PDENonStatioComponents(
            dyn_loss=lambda r: jax.tree.map(
                # we could do jnp.sum(r_**2) directly below but we code it this way
                # so that the analogy with mean_sum_reduction() is clearer
                lambda r_: jnp.sum(jnp.sum(r_**2, axis=-1)),
                r,
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
    jax.config.update("jax_enable_x64", True)
    key = random.PRNGKey(2)
    key, subkey = random.split(key)

    n = 24
    ni = n
    nb = n
    domain_batch_size = n
    initial_batch_size = domain_batch_size - 1
    border_batch_size = domain_batch_size // 4 - 4
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
        (eqx.nn.Linear, 3, 2),  # 3 = t + x (2D)
        (jax.nn.tanh,),
        (eqx.nn.Linear, 2, 1),
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
        dyn_loss=jnp.array(1.32),
        initial_condition=jnp.array(2.4),
        boundary_loss=None if boundary_condition is None else jnp.array(0.678),
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
    train_loss_terms, euclidean_grad_std = loss.values_and_grads(
        init_params,
        batch,
    )

    # sum gradients for each loss term
    euclidean_grad_std = loss.ponderate_and_sum_gradient(euclidean_grad_std)

    # Get the nn_params gradient (the one of interest for this test
    # and make the gradients a flat array for the final comparison
    euclidean_grad_nn_std, _ = jax.flatten_util.ravel_pytree(
        euclidean_grad_std.nn_params
    )

    # divide by the total number of samples to simulate a global average
    # (similar to what's done in NGD)
    n_sample_std = domain_batch_size + (border_batch_size * 2**dim) + initial_batch_size
    euclidean_grad_nn_std /= n_sample_std
    train_loss_value_std = loss.ponderate_and_sum_loss(train_loss_terms) / n_sample_std

    ############################################
    ## EUCLIDEAN GRADIENT VIA NGD COMPUTATIONS##
    ############################################

    residuals, g = loss.values_and_grad_per_sample(
        init_params,
        batch,
    )

    # Preprocess loss weights
    # NOTE: Here we use batch_norm=False since this test is for the global average computation
    # (no local avging per term)
    lw_ = _loss_weights_to_components(loss.loss_weights, residuals)
    sqrt_weights_per_sample = _get_sqrt_weights_per_sample(
        lw=lw_, r=residuals, batch_norm=False
    )

    # -- Compute the weighted version of r, g and aggregte terms
    # We get a flattenned version of the pytree of jacobians as a big (n * n_equations, p) array
    jacobians_nn, weighted_residuals, _ = weighted_jac_and_residuals(
        residuals, g, sqrt_weights_per_sample, with_eq_params_update=None
    )

    # Get global number of samples
    n_samples_ngd = jax.tree.reduce(
        jnp.add, jax.tree.leaves(jax.tree.map(lambda x: x.shape[0], weighted_residuals))
    )

    assert n_samples_ngd == n_sample_std, (
        "total n_sample should be the same for two method"
    )

    euclidean_grad_pytree = jax.tree.map(
        lambda J, R: jnp.einsum("ij,ij->j", J, R), jacobians_nn, weighted_residuals
    )
    # multiply by 2 which is a factor that appears in a complete derivation of
    # NGD (from the mean square error)

    euclidean_grad_ngd = (
        jax.tree.reduce(jnp.add, euclidean_grad_pytree) * 2 / (n_samples_ngd)
    )

    ##############
    # COMPARISON #
    ##############

    # Loss value equality
    train_loss_value_ngd = jnp.sum(
        jax.flatten_util.ravel_pytree(weighted_residuals)[0] ** 2
    )
    train_loss_value_ngd /= n_samples_ngd  # do global average manually
    assert jnp.allclose(train_loss_value_std, train_loss_value_ngd), (
        "Train loss values do not agree btween the two method of computation"
    )
    # gradient equality
    assert jnp.allclose(euclidean_grad_ngd, euclidean_grad_nn_std), (
        "Euclidean gradients do not agree between the two method of computation"
    )
