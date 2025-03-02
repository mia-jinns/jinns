import jinns


from jax import random
import jax.numpy as jnp
import equinox as eqx
import jinns.parameters
from jinns.validation import ValidationLoss
import optax


def test_validation_module():
    NUM_POINTS = 36

    key = random.PRNGKey(2)
    d = 2
    r = 2
    eqx_list = ((eqx.nn.Linear, 1, r),)
    key, subkey = random.split(key)
    u_spinn, init_nn_params_spinn = jinns.nn.SPINN_MLP.create(
        subkey, d, r, eqx_list, "nonstatio_PDE"
    )

    n = NUM_POINTS
    nb = NUM_POINTS
    ni = NUM_POINTS
    dim = 1
    xmin = -1
    xmax = 1
    tmin = 0
    tmax = 1
    method = "uniform"

    Tmax = 5

    key, subkey = random.split(key)
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
    )

    from jax.scipy.stats import norm

    # true solution N(0,1)
    sigma_init = 0.2 * jnp.ones((1))
    mu_init = 0 * jnp.ones((1))

    def u0(x):
        return norm.pdf(x, loc=mu_init, scale=sigma_init)

    # Example III.29 is persistent with D = 1, r = 4,  g = 3
    D = 1.0
    r = 4.0
    g = 3.0
    l = xmax - xmin
    boundary_condition = "dirichlet"
    omega_boundary_fun = lambda t_dx: 0  # cte func returning 0

    init_params_spinn = jinns.parameters.Params(
        nn_params=init_nn_params_spinn,
        eq_params={"D": jnp.array([D]), "r": jnp.array([r]), "g": jnp.array([g])},
    )

    fisher_dynamic_loss = jinns.loss.FisherKPP(Tmax=Tmax)

    loss_weights = jinns.loss.LossWeightsPDENonStatio(
        dyn_loss=1, initial_condition=1 * Tmax, boundary_loss=1 * Tmax
    )

    loss_spinn = jinns.loss.LossPDENonStatio(
        u=u_spinn,
        loss_weights=loss_weights,
        dynamic_loss=fisher_dynamic_loss,
        omega_boundary_fun=omega_boundary_fun,
        omega_boundary_condition=boundary_condition,
        initial_condition_fun=u0,
        params=init_params_spinn,
    )

    n = NUM_POINTS
    nb = NUM_POINTS
    ni = NUM_POINTS
    dim = 1
    xmin = -1
    xmax = 1
    tmin = 0
    tmax = 1
    method = "uniform"

    Tmax = 5

    key, subkey = random.split(key)
    validation_data = jinns.data.CubicMeshPDENonStatio(
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

    validation = ValidationLoss(
        loss=loss_spinn,
        validation_data=validation_data,
        validation_param_data=None,
        validation_obs_data=None,
        call_every=3,
        early_stopping=True,
        patience=0,
    )

    params_spinn = init_params_spinn

    tx = optax.adamw(learning_rate=1e-4)
    n_iter = 10
    (
        params_spinn,
        train_loss_values,
        _,
        _,
        _,
        _,
        _,
        validation_loss_values,
        best_params_spinn,
    ) = jinns.solve(
        init_params=params_spinn,
        data=train_data,
        optimizer=tx,
        loss=loss_spinn,
        n_iter=n_iter,
        validation=validation,
    )

    assert isinstance(best_params_spinn, jinns.parameters.Params)
