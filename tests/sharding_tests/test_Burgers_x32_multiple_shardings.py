import pytest

import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count=2"

import jax
import jax.numpy as jnp
from jax import random
import equinox as eqx
import optax
import jinns


@pytest.fixture
def train_Burgers_init_sharding():
    jax.config.update("jax_enable_x64", False)
    # We have forced CPU computations and 2 fake CPU devices
    # in order to test the sharding
    cpu1, cpu2 = jax.devices("cpu")

    # Default device is cpu1
    jax.config.update("jax_default_device", cpu1)

    # Playing with obs_data on cpu2
    cpu2_sharding = jax.sharding.SingleDeviceSharding(cpu2)

    key = random.PRNGKey(2)
    eqx_list = (
        (eqx.nn.Linear, 2, 20),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 20, 20),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 20, 20),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 20, 1),
    )
    key, subkey = random.split(key)
    u, init_nn_params = jinns.nn.PINN_MLP.create(
        key=subkey, eqx_list=eqx_list, eq_type="nonstatio_PDE"
    )

    n = 1000
    ni = 1000
    nb = 2
    domain_batch_size = 32
    initial_batch_size = 20
    border_batch_size = 1
    dim = 1
    xmin = -1
    xmax = 1
    tmin = 0
    tmax = 1
    Tmax = 1
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
        min_pts=(xmin,),
        max_pts=(xmax,),
        tmin=tmin,
        tmax=tmax,
        method=method,
    )

    nu = 1 / (100 * jnp.pi)
    init_params = jinns.parameters.Params(
        nn_params=init_nn_params, eq_params={"nu": nu}
    )

    def u0(x):
        return -jnp.sin(jnp.pi * x)

    be_loss = jinns.loss.BurgersEquation(Tmax=Tmax)

    loss_weights = jinns.loss.LossWeightsPDENonStatio(
        dyn_loss=1, initial_condition=5, boundary_loss=1
    )

    loss = jinns.loss.LossPDENonStatio(
        u=u,
        loss_weights=loss_weights,
        dynamic_loss=be_loss,
        omega_boundary_fun=lambda t_dx: 0,
        omega_boundary_condition="dirichlet",
        initial_condition_fun=u0,
        params=init_params,
    )

    return init_params, loss, train_data, cpu2_sharding


@pytest.fixture
def train_Burgers_10it_sharding(train_Burgers_init_sharding):
    """
    Fixture that requests a fixture
    """
    init_params, loss, train_data, cpu2_sharding = train_Burgers_init_sharding

    # NOTE we need to waste one get_batch() here to stay synchronized with the
    # notebook
    train_data, _ = train_data.get_batch()

    params = init_params

    tx = optax.adam(learning_rate=1e-3)
    n_iter = 10
    params, total_loss_list, loss_by_term_dict, _, _, _, _, _, _ = jinns.solve(
        init_params=params,
        data=train_data,
        optimizer=tx,
        loss=loss,
        n_iter=n_iter,
        obs_batch_sharding=cpu2_sharding,
    )
    return total_loss_list[9]


@pytest.fixture
def train_Burgers_init_no_sharding():
    jax.config.update("jax_enable_x64", False)

    key = random.PRNGKey(2)
    eqx_list = [
        [eqx.nn.Linear, 2, 20],
        [jax.nn.tanh],
        [eqx.nn.Linear, 20, 20],
        [jax.nn.tanh],
        [eqx.nn.Linear, 20, 20],
        [jax.nn.tanh],
        [eqx.nn.Linear, 20, 1],
    ]
    key, subkey = random.split(key)
    u, init_nn_params = jinns.nn.PINN_MLP.create(
        key=subkey, eqx_list=eqx_list, eq_type="nonstatio_PDE"
    )

    n = 1000
    ni = 1000
    nb = 2
    domain_batch_size = 32
    initial_batch_size = 20
    border_batch_size = 1
    dim = 1
    xmin = -1
    xmax = 1
    tmin = 0
    tmax = 1
    Tmax = 1
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
        min_pts=(xmin,),
        max_pts=(xmax,),
        tmin=tmin,
        tmax=tmax,
        method=method,
    )

    nu = 1 / (100 * jnp.pi)
    init_params = jinns.parameters.Params(
        nn_params=init_nn_params, eq_params={"nu": nu}
    )

    def u0(x):
        return -jnp.sin(jnp.pi * x)

    be_loss = jinns.loss.BurgersEquation(Tmax=Tmax)

    loss_weights = jinns.loss.LossWeightsPDENonStatio(
        dyn_loss=1, initial_condition=5, boundary_loss=1
    )

    loss = jinns.loss.LossPDENonStatio(
        u=u,
        loss_weights=loss_weights,
        dynamic_loss=be_loss,
        omega_boundary_fun=lambda t_dx: 0,
        omega_boundary_condition="dirichlet",
        initial_condition_fun=u0,
        params=init_params,
    )

    return init_params, loss, train_data


@pytest.fixture
def train_Burgers_10it_no_sharding(train_Burgers_init_no_sharding):
    """
    Fixture that requests a fixture
    """
    init_params, loss, train_data = train_Burgers_init_no_sharding

    # NOTE we need to waste one get_batch() here to stay synchronized with the
    # notebook
    train_data, _ = train_data.get_batch()

    params = init_params

    tx = optax.adam(learning_rate=1e-3)
    n_iter = 10
    params, total_loss_list, loss_by_term_dict, _, _, _, _, _, _ = jinns.solve(
        init_params=params, data=train_data, optimizer=tx, loss=loss, n_iter=n_iter
    )
    return total_loss_list[9]


def test_10it_Burgers_sharding(
    train_Burgers_10it_sharding, train_Burgers_10it_no_sharding
):
    # Test the equivalency sharding / no sharding across two CPUs devices
    # The no sharding example is taken from the test_Burgers_x32 file
    total_loss_val_for = train_Burgers_10it_sharding
    total_loss_val_scan = train_Burgers_10it_no_sharding
    assert jnp.allclose(total_loss_val_for, total_loss_val_scan, atol=1e-1)
