import pytest

import jax
import jinns
from jax import random
import jax.numpy as jnp
import equinox as eqx
from jax.scipy.stats import norm
import optax

xmin = -1
xmax = 1
tmin = 0
tmax = 1


D = 1.0
r = 3.0
g = 3.0

# Init condition N(0,1)
sigma_init = 0.2 * jnp.ones((1))
mu_init = 0 * jnp.ones((1))


def u0(x):
    return jnp.squeeze(norm.pdf(x, loc=mu_init, scale=sigma_init))


# - Implementation of hard constraint for Dirichlet BC is much more straightforward
# - For Neuman we could implement **fourier frequency embedding** (https://arxiv.org/pdf/2504.01093v1) in order to hard constraint Neumann  BC since hard constraint is needed for Neural Galerkin.


def output_transform(x, pinn_out, params):
    """
    this is for Dirichlet
    """
    return (x[0] - xmin) * (x[0] - xmax) * pinn_out


@pytest.fixture
def neural_galerkin_fkpp1d_init():
    jax.config.update("jax_enable_x64", True)
    key = random.PRNGKey(2)

    # Note that input dim is 1 since the network depends on time
    # only through its parameters' ODE
    eqx_list = (
        (eqx.nn.Linear, 1, 3),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 3, 3),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 3, 1),
    )

    n = 30
    dim = 1
    method = "uniform"

    key, subkey = random.split(key)
    u_pinn_ode, init_nn_params_pinn_ode = jinns.nn.PINN_MLP.create(
        key=subkey,
        eqx_list=eqx_list,
        eq_type="PDEStatio",
        output_transform=output_transform,
    )
    init_params_pinn_ode = jinns.parameters.Params(
        nn_params=init_nn_params_pinn_ode,
        eq_params={
            "D": jnp.array([D]),
            "r": jnp.array([r]),
            "g": jnp.array([g]),
        },
    )

    n = 2048
    batch_size = 256
    dim = 1
    xmin = -1
    xmax = 1
    method = "uniform"

    key, subkey = random.split(key)
    train_data_ode = jinns.data.CubicMeshPDEStatio(
        key=subkey,
        n=n,
        omega_batch_size=batch_size,
        dim=dim,
        min_pts=(xmin,),
        max_pts=(xmax,),
        method=method,
    )

    class FisherKPP_statio(jinns.loss.PDEStatio):
        """
        This implements the RHS of the FKPP equation
        """

        def equation(self, x, u, params):
            r"""
            Evaluate the dynamic loss at $x$.

            Parameters
            ---------
            x
                A jnp array a point in $\Omega$
            u
                The PINN
            params
                The dictionary of parameters of the model.
                Typically, it is a dictionary of
                dictionaries: `eq_params` and `nn_params`, respectively the
                differential equation parameters and the neural network parameter
            """
            lap = jinns.loss.laplacian_rev(x, u, params)[..., None]

            return -(
                params.eq_params.D * lap
                + u(x, params)
                * (params.eq_params.r - params.eq_params.g * u(x, params))
            )

    fisher_dynamic_loss_ode = FisherKPP_statio()

    loss_pinn_ode = jinns.loss.LossPDEStatio(
        u=u_pinn_ode,
        loss_weights=jinns.loss.LossWeightsPDEStatio(dyn_loss=1),
        dynamic_loss=fisher_dynamic_loss_ode,
        params=init_params_pinn_ode,
    )

    return init_params_pinn_ode, loss_pinn_ode, train_data_ode


@pytest.fixture
def solve_neural_galerkin_FKPP1D(neural_galerkin_fkpp1d_init):
    """
    Do the ODE solve in NG
    """
    init_params_pinn_ode, loss_pinn_ode, train_data_ode = neural_galerkin_fkpp1d_init
    Tmax = 1
    dt = 0.1
    times = jnp.arange(tmin, Tmax + dt, dt)

    times_saved = [
        0.0,
        0.1,
        1,
    ]
    loss, data, param_data, params_final, params_saved = jinns.solve_neural_galerkin(
        times=times,
        times_saved=times_saved,
        init_params=init_params_pinn_ode,
        data=train_data_ode,
        loss=loss_pinn_ode,
        initial_condition_fun=u0,
        n_iter_ic=100,
        optimizer_ic=optax.adam(1e-3),
        print_loss_every=len(times) // 10,
    )

    return times_saved, loss, data, param_data, params_final, params_saved


def test_ouput_neural_galerkin_fkpp1d(solve_neural_galerkin_FKPP1D):
    """
    Test various aspect of the solve_neural_galerkin output
    """

    times_saved, loss, data, param_data, params_final, params_saved = (
        solve_neural_galerkin_FKPP1D
    )

    assert len(times_saved) == len(params_saved)
    assert all([isinstance(p, jinns.parameters.Params) for p in params_saved])
    assert isinstance(params_final, jinns.parameters.Params)

    # TODO: assert pinn or loss value ?
