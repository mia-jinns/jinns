# test _rar.py
# control shape of p_times and p_omega arguments for various hyper-params

import pytest

import jax.random as random
import jax.numpy as jnp
import jax
from jax import vmap
import jinns
import equinox as eqx
from jax.scipy.stats import multivariate_normal

key = random.PRNGKey(1)
n = 117
nb = None
ni = 4
xmin = -3
xmax = 3
ymin = -3
ymax = 3
tmin = 0
tmax = 1
method = "uniform"


int_xmin, int_xmax = -5, 5
int_ymin, int_ymax = -5, 5

n_samples = int(1e1)
volume = (int_xmax - int_xmin) * (int_ymax - int_ymin)
norm_weights = volume
key, subkey1, subkey2 = random.split(key, 3)
mc_samples = jnp.concatenate(
    [
        random.uniform(subkey1, shape=(100, 1), minval=int_xmin, maxval=int_xmax),
        random.uniform(subkey2, shape=(100, 1), minval=int_ymin, maxval=int_ymax),
    ],
    axis=-1,
)


eqx_list = (
    (eqx.nn.Linear, 3, 30),
    (jax.nn.tanh,),
    (eqx.nn.Linear, 30, 30),
    (jax.nn.tanh,),
    (eqx.nn.Linear, 30, 30),
    (jax.nn.tanh,),
    (eqx.nn.Linear, 30, 1),
    (jnp.exp,),
)
key, subkey = random.split(key)
u, init_nn_params = jinns.nn.PINN_MLP.create(
    key=subkey, eqx_list=eqx_list, eq_type="nonstatio_PDE"
)

# true solution N(0,1)
sigma = 0.5 * jnp.ones((2))
alpha = 0.5 * jnp.ones((2))
mu = jnp.zeros((2))

Tmax = 10


init_params = jinns.parameters.Params(
    nn_params=init_nn_params,
    eq_params={"sigma": sigma, "alpha": alpha, "mu": mu},
)


def u0(x):
    return multivariate_normal.pdf(x, mean=jnp.array([1, 1]), cov=0.1 * jnp.eye(2))


vectorized_u0 = vmap(u0, (0), 0)

OU_fpe_non_statio_2D_loss = jinns.loss.OU_FPENonStatioLoss2D(Tmax=Tmax)

loss_weights = jinns.loss.LossWeightsPDENonStatio(
    dyn_loss=1.0,
    initial_condition=1 * Tmax,
    norm_loss=0.1 * Tmax,
)

with pytest.warns(UserWarning):
    loss = jinns.loss.LossPDENonStatio(
        u=u,
        loss_weights=loss_weights,
        dynamic_loss=OU_fpe_non_statio_2D_loss,
        initial_condition_fun=u0,
        norm_weights=norm_weights,
        norm_samples=mc_samples,
        params=init_params,
    )


# Optimizer
import optax

tx = optax.adamw(learning_rate=1e-3)

n_iter = int(10)
params = init_params


def get_datagenerator_rar(start_iter, update_every):

    rar_parameters = {
        "start_iter": start_iter,  # the gradient step at which RAR algo starts (enables a burn in period)
        "update_every": update_every,  # nb of gradient steps between two RAR procedures
        "sample_size": 20,  # the number of new candidates space points
        "selected_sample_size": 7,
    }
    n_start = 10  # the initial number of spatial collocation points at beginning

    key = random.PRNGKey(12345)

    key, subkey = random.split(key)
    train_data = jinns.data.CubicMeshPDENonStatio(
        key=subkey,
        n=n,
        nb=nb,
        ni=ni,
        dim=2,
        min_pts=(xmin, ymin),
        max_pts=(xmax, ymax),
        tmin=tmin,
        tmax=tmax,
        method=method,
        rar_parameters=rar_parameters,
        n_start=n_start,
    )

    return train_data, rar_parameters


@pytest.fixture
def all_tests(pytestconfig):
    return pytestconfig.getoption("all_tests")


def test_data_proba_shape_before_solve():
    train_data, rar_parameters = get_datagenerator_rar(1, 1)
    assert (train_data.p != 0).sum() == train_data.n_start


def control_shape_after_solve_with_rar(start_iter, update_every):
    train_data, rar_parameters = get_datagenerator_rar(start_iter, update_every)
    _, _, _, train_data, _, _, _, _, _ = jinns.solve(
        init_params=params, data=train_data, optimizer=tx, loss=loss, n_iter=n_iter
    )
    assert (train_data.p != 0).sum() == train_data.n_start + jnp.round(
        (n_iter - rar_parameters["start_iter"]) / rar_parameters["update_every"]
    ) * rar_parameters["selected_sample_size"]


def test_rar_with_various_combination_of_start_and_update_values(all_tests):
    # long test run only if --all_tests
    if all_tests:
        start_iter_list = [0, 3]
        update_every_list = [1, 3]
        for start_iter in start_iter_list:
            for update_every in update_every_list:
                control_shape_after_solve_with_rar(start_iter, update_every)
    else:
        print(
            "\ntest_rar_with_various_combination_of_start_and_update_values "
            "has been skipped due to missing --all_tests option\n"
        )


def test_rar_error_with_SPINN(all_tests):
    # long test run only if --all_tests
    if all_tests:
        train_data, rar_parameters = get_datagenerator_rar(0, 1)
        # ensure same batch size in time & space for SPINN
        d = 3
        r = 256
        eqx_list = [
            [eqx.nn.Linear, 1, 128],
            [jax.nn.tanh],
            [eqx.nn.Linear, 128, 128],
            [jax.nn.tanh],
            [eqx.nn.Linear, 128, 128],
            [jax.nn.tanh],
            [eqx.nn.Linear, 128, r],
        ]
        key = jax.random.PRNGKey(12345)
        key, subkey = random.split(key)
        u, init_nn_params = jinns.nn.create_SPINN(
            subkey, d, r, eqx_list, "nonstatio_PDE"
        )

        init_params = jinns.parameters.Params(
            nn_params=init_nn_params,
            eq_params={"sigma": sigma, "alpha": alpha, "mu": mu},
        )
        loss = jinns.loss.LossPDENonStatio(
            u=u,
            loss_weights=loss_weights,
            dynamic_loss=OU_fpe_non_statio_2D_loss,
            initial_condition_fun=u0,
            norm_norm_weights=norm_weights,
            norm_samples=mc_samples,
            params=init_params,
        )
        # expect error
        with pytest.raises(NotImplementedError):
            tx = optax.adamw(learning_rate=1e-3)
            jinns.solve(
                init_params=init_params,
                data=train_data,
                optimizer=tx,
                loss=loss,
                n_iter=2,
            )
    else:
        print(
            "\ntest_rar_error_with_SPINN has been skipped due not missing "
            "--all_tests option\n"
        )
