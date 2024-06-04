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
nb = 1  # not used here
nt = 59  # is != n
omega_batch_size = 2
omega_border_batch_size = None  # not used here
temporal_batch_size = 3
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
int_length = (int_xmax - int_xmin) * (int_ymax - int_ymin)
key, subkey1, subkey2 = random.split(key, 3)
mc_samples = jnp.concatenate(
    [
        random.uniform(subkey1, shape=(1000, 1), minval=int_xmin, maxval=int_xmax),
        random.uniform(subkey2, shape=(1000, 1), minval=int_ymin, maxval=int_ymax),
    ],
    axis=-1,
)


eqx_list = [
    [eqx.nn.Linear, 3, 30],
    [jax.nn.tanh],
    [eqx.nn.Linear, 30, 30],
    [jax.nn.tanh],
    [eqx.nn.Linear, 30, 30],
    [jax.nn.tanh],
    [eqx.nn.Linear, 30, 1],
    [jnp.exp],
]
key, subkey = random.split(key)
u = jinns.utils.create_PINN(subkey, eqx_list, "nonstatio_PDE", 2)

init_nn_params = u.init_params()

# true solution N(0,1)
sigma = 0.5 * jnp.ones((2))
alpha = 0.5 * jnp.ones((2))
mu = jnp.zeros((2))

Tmax = 10


init_params = {
    "nn_params": init_nn_params,
    "eq_params": {"sigma": sigma, "alpha": alpha, "mu": mu},
}


def u0(x):
    return multivariate_normal.pdf(x, mean=jnp.array([1, 1]), cov=0.1 * jnp.eye(2))


vectorized_u0 = vmap(u0, (0), 0)

OU_fpe_non_statio_2D_loss = jinns.loss.OU_FPENonStatioLoss2D(Tmax=Tmax)

loss_weights = {"dyn_loss": 1, "initial_condition": 1 * Tmax, "norm_loss": 0.1 * Tmax}

with pytest.warns(UserWarning):
    loss = jinns.loss.LossPDENonStatio(
        u=u,
        loss_weights=loss_weights,
        dynamic_loss=OU_fpe_non_statio_2D_loss,
        initial_condition_fun=u0,
        norm_borders=((int_xmin, int_xmax), (int_ymin, int_ymax)),
        norm_samples=mc_samples,
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
        "sample_size_times": 10,  # the number of new candidates time points
        "selected_sample_size_times": 4,  # the number of selected times collocation points from the sample, to join the dataset.
        "sample_size_omega": 20,  # the number of new candidates space points
        "selected_sample_size_omega": 7,
    }
    n_start = 10  # the initial number of spatial collocation points at beginning
    nt_start = 5  # the initial number of temporal collocation points at beginning

    key = random.PRNGKey(12345)

    key, subkey = random.split(key)
    train_data = jinns.data.CubicMeshPDENonStatio(
        subkey,
        n,
        nb,
        nt,
        omega_batch_size,
        omega_border_batch_size,
        temporal_batch_size,
        2,
        (xmin, ymin),
        (xmax, ymax),
        tmin,
        tmax,
        method=method,
        rar_parameters=rar_parameters,
        n_start=n_start,
        nt_start=nt_start,
    )

    return train_data, rar_parameters


@pytest.fixture
def all_tests(pytestconfig):
    return pytestconfig.getoption("all_tests")


def test_data_proba_shape_before_solve():
    train_data, rar_parameters = get_datagenerator_rar(1, 1)
    assert (train_data.p_times != 0).sum() == train_data.nt_start
    assert (train_data.p_omega != 0).sum() == train_data.n_start


def control_shape_after_solve_with_rar(start_iter, update_every):
    train_data, rar_parameters = get_datagenerator_rar(start_iter, update_every)
    _, _, _, train_data, _, _, _ = jinns.solve(
        init_params=params, data=train_data, optimizer=tx, loss=loss, n_iter=n_iter
    )
    assert (train_data.p_times != 0).sum() == train_data.nt_start + jnp.ceil(
        (n_iter - rar_parameters["start_iter"]) / rar_parameters["update_every"]
    ) * rar_parameters["selected_sample_size_times"]

    assert (train_data.p_omega != 0).sum() == train_data.n_start + jnp.ceil(
        (n_iter - rar_parameters["start_iter"]) / rar_parameters["update_every"]
    ) * rar_parameters["selected_sample_size_omega"]


def test_rar_with_various_combination_of_start_and_update_values(all_tests):
    # long test run only if --all_tests
    if all_tests:
        start_iter_list = [0, 3]
        update_every_list = [1, 3]
        for start_iter in start_iter_list:
            for update_every in update_every_list:
                with pytest.warns(UserWarning):
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
        train_data.temporal_batch_size = train_data.omega_batch_size
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
        u = jinns.utils.create_SPINN(subkey, d, r, eqx_list, "nonstatio_PDE")
        init_nn_params = u.init_params()

        # update loss and params
        init_params["nn_params"] = init_nn_params
        loss.u = u
        train_data.temporal_batch_size = train_data.omega_batch_size
        # expect error
        with pytest.raises(NotImplementedError) as e_info, pytest.warns(UserWarning):
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
