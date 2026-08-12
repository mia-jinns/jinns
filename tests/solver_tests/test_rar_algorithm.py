# test _rar.py
# control shape of p_times and p_omega arguments for various hyper-params

import pytest

import jax.random as random
import jax.numpy as jnp
import jax
from jax import vmap
import optax
import jinns
import equinox as eqx
from jax.scipy.stats import multivariate_normal

jax.config.update("jax_enable_x64", True)

# because we did not write a fixture
jinns.parameters.EqParams.clear()
jinns.data.DGParams.clear()

key = random.PRNGKey(1)
n = 99
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
        random.uniform(subkey1, shape=(10, 1), minval=int_xmin, maxval=int_xmax),
        random.uniform(subkey2, shape=(10, 1), minval=int_ymin, maxval=int_ymax),
    ],
    axis=-1,
)

norm_samples = jinns.loss.NormalizationSamples(
    samples=mc_samples,
    weights=volume,
    min_pts=(int_xmin, int_ymin),
    max_pts=(int_xmax, int_ymax),
)


eqx_list = (
    (eqx.nn.Linear, 3, 3),
    (jax.nn.tanh,),
    (eqx.nn.Linear, 3, 1),
    (jnp.exp,),
)
key, subkey = random.split(key)
u, init_nn_params = jinns.nn.PINN_MLP.create(
    key=subkey, eqx_list=eqx_list, eq_type="PDENonStatio"
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
        norm_samples=norm_samples,
        params=init_params,
    )


tx = optax.adamw(learning_rate=1e-3)

n_iter = int(10)
params = init_params


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
    rar_parameters=jinns.data.RARParameters(
        start_iter=0, update_every=1, novelty_proportion=0.05, method="D", k=2.0, c=0.0
    ),
)


# @pytest.fixture
# def all_tests(pytestconfig):
#     return pytestconfig.getoption("all_tests")


def test_rar_proc_OU():
    _, loss_values, _, _, _, _, _, _, _, _, _, _ = jinns.solve(
        init_params=params,
        data=train_data,
        optimizer=tx,
        loss=loss,
        n_iter=n_iter,
        key=key,
    )
    assert jnp.allclose(loss_values[-1], 4978.02179599, atol=1e-5)
