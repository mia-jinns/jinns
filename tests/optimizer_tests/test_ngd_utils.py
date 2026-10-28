"""
Test equality of euclidean gradients when the loss is $\sum_{term} w_{term}/n_{term} L_{term}$
"""

import pytest

import jinns


import jax
from jax import random
import jax.numpy as jnp
import equinox as eqx

from jinns.loss import LossPDENonStatio
from jinns.optimizers._utils_ngd import (
    get_euclidean_grads,
    unravel_array_to_pytree,
    _loss_weights_to_components,
    weighted_jac_and_residuals,
    _get_sqrt_weights_per_sample,
)

from jinns.optimizers._natural_gradient import (
    vanilla_ngd,
    compute_ng_full_gramian,
    compute_ng_matrix_free,
)


@pytest.fixture
def create_loss_and_params():
    jax.config.update("jax_enable_x64", True)
    key = random.PRNGKey(2)
    key, subkey = random.split(key)

    n = 24
    ni = n
    nb = n * 4
    domain_batch_size = n
    initial_batch_size = domain_batch_size - 1
    border_batch_size = domain_batch_size - 4
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

    loss = LossPDENonStatio(
        u=u,
        loss_weights=loss_weights,
        dynamic_loss=dyn_loss_heat,
        initial_condition_fun=u0,
        boundary_condition=boundary_condition,
        params=init_params,
    )

    _, batch = train_data.get_batch()

    return loss, init_params, batch


def test_euclidean_gradients_equality(create_loss_and_params):
    loss, init_params, batch = create_loss_and_params

    ############################################
    ## STANDARD WAY TO GET EUCLIDEAN GRADIENT ##
    ############################################

    # Get gradient for each term as in jinns.solve
    train_loss_terms, euclidean_grad_std_terms = loss.values_and_grads(
        init_params,
        batch,
    )

    # Get the full gradient as in jinns
    euclidean_grad_pytree_std = loss.ponderate_and_sum_gradient(
        euclidean_grad_std_terms
    )
    train_loss_value_std = loss.ponderate_and_sum_loss(train_loss_terms)

    ############################################
    ## EUCLIDEAN GRADIENT VIA NGD COMPUTATIONS##
    ############################################

    residuals, g = loss.values_and_grad_per_sample(
        init_params,
        batch,
    )

    # Preprocess loss weights
    lw_ = _loss_weights_to_components(loss.loss_weights, residuals)
    sqrt_weights_per_sample = _get_sqrt_weights_per_sample(
        lw=lw_, r=residuals, batch_norm=True
    )

    # -- Compute the weighted version of r, g and aggregte terms
    jacobians_nn, weighted_residuals, _ = weighted_jac_and_residuals(
        residuals, g, sqrt_weights_per_sample, with_eq_params_update=None
    )

    # Form the euclidean gradient with NGD helper func
    euclidean_grad_array_ngd, _ = get_euclidean_grads(
        jacobians_nn,
        weighted_residuals,
        jacobians_eq_params=None,
        with_eq_params_update=False,
    )

    # Go from array to PyTree of Params for comparison with the other method
    euclidean_grad_pytree_ngd = unravel_array_to_pytree(
        euclidean_grad_array_ngd, init_params, eq_params_array=None
    )

    ##############
    # COMPARISON #
    ##############

    # Loss value equality
    train_loss_value_ngd = jnp.sum(
        jnp.concatenate(
            jax.tree.leaves(
                jax.tree.map(
                    lambda arr: jnp.sum(arr**2, axis=-1),
                    weighted_residuals,
                ),
            ),
            axis=0,
        )
    )
    assert jnp.allclose(train_loss_value_std, train_loss_value_ngd), (
        "Train loss values do not agree btween the two method of computation"
    )

    # Gradient equality at each leaf of Params pytree
    assert jax.tree.all(
        jax.tree.map(
            lambda a, b: jnp.allclose(a, b),
            euclidean_grad_pytree_ngd,
            euclidean_grad_pytree_std,
        )
    ), (
        "Some leaves in both gradient (grad(jinns.evaluate) vs 2 * sum(J * R)) PyTree are not the same"
    )

    # Equality between flattened version (nn structure only) of the Pytree ? -> not guaranteed
    euclidean_grad_std_flat_nn, _ = jax.flatten_util.ravel_pytree(
        euclidean_grad_pytree_std.nn_params
    )

    assert jnp.allclose(euclidean_grad_std_flat_nn, euclidean_grad_array_ngd), (
        "Flattened version of the gradient (std vs NGD) do not coincide. If the test before passes, this means our custom jinns unravel/ravel functions do not flatten in a similar fashion as jax.flatten_util."
    )


def test_ng_all_methods(create_loss_and_params):
    loss, init_params, batch = create_loss_and_params

    tx = vanilla_ngd(method="full")
    ngd_state = tx.init(init_params)

    # Full gram
    (
        natural_grad_array_nn_full_gram,
        euclidean_grad_array_nn_params_full_gram,
        euclidean_grad_array_eq_params_full_gram,
        train_loss_value_full_gram,
        _,
        r_full_gram,
    ) = compute_ng_full_gramian(loss, init_params, batch, ngd_state)

    # Matrix-free via conjugate gradient
    MAXIT = 1000
    RTOL = 1e-6
    (
        natural_grad_array_nn_matrix_free,
        euclidean_grad_array_nn_params_matrix_free,
        euclidean_grad_array_eq_params_matrix_free,
        train_loss_value_matrix_free,
        _,
        r_matrix_free,
    ) = compute_ng_matrix_free(
        loss, init_params, batch, ngd_state, maxit=MAXIT, rtol=RTOL
    )

    assert jnp.allclose(
        euclidean_grad_array_nn_params_full_gram,
        euclidean_grad_array_nn_params_matrix_free,
    ), (
        "[NN params] Euclidan grad do not agress for full gramian vs matrix-free ConjGrad."
    )

    assert jnp.allclose(
        natural_grad_array_nn_full_gram,
        natural_grad_array_nn_matrix_free,
    ), (
        "[NN params] Natural grad are not close up (full gramian vs matrix-free ConjGrad). Try raising maxit. Maybe the allclose tolerance should be higher."
    )

    assert jnp.allclose(
        train_loss_value_full_gram,
        train_loss_value_matrix_free,
    ), "Different loss value between Full gram and matrix-free method"
