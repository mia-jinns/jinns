import pytest
import jax
import jax.numpy as jnp
import jinns


@pytest.fixture
def create_1DCubicMeshPDEStatio():
    key = jax.random.PRNGKey(2)
    key, subkey = jax.random.split(key)
    n = 1000
    nb = 2
    omega_batch_size = 32
    omega_border_batch_size = 2
    dim = 1
    xmin = -3
    xmax = 3

    return jinns.data.CubicMeshPDEStatio(
        subkey,
        n,
        nb,
        omega_batch_size,
        omega_border_batch_size,
        dim,
        (xmin,),
        (xmax,),
    )


def test_omega_range_1D(create_1DCubicMeshPDEStatio):
    OneD_obj = create_1DCubicMeshPDEStatio
    assert jnp.all(OneD_obj.omega >= OneD_obj.min_pts[0]) and jnp.all(
        OneD_obj.omega <= OneD_obj.max_pts[0]
    )


def test_omega_border_range_1D(create_1DCubicMeshPDEStatio):
    OneD_obj = create_1DCubicMeshPDEStatio
    assert jnp.all(OneD_obj.omega_border >= OneD_obj.min_pts[0]) and jnp.all(
        OneD_obj.omega_border <= OneD_obj.max_pts[0]
    )


def test_get_batch_1D(create_1DCubicMeshPDEStatio):
    OneD_obj = create_1DCubicMeshPDEStatio
    batch = OneD_obj.get_batch()
    inside_batch, border_batch = batch.inside_batch, batch.border_batch
    assert (
        jnp.all(inside_batch[:] >= OneD_obj.min_pts[0])
        and jnp.all(inside_batch[:] <= OneD_obj.max_pts[0])
        and jnp.all(border_batch[:] >= OneD_obj.min_pts[0])
        and jnp.all(border_batch[:] <= OneD_obj.max_pts[0])
    )


@pytest.fixture
def create_2DCubicMeshPDEStatio():
    key = jax.random.PRNGKey(2)
    key, subkey = jax.random.split(key)
    n = 1024
    nb = 8
    omega_batch_size = 32
    omega_border_batch_size = 2
    dim = 2
    xmin = -3
    xmax = 3
    ymin = -5
    ymax = 5

    return jinns.data.CubicMeshPDEStatio(
        subkey,
        n,
        nb,
        omega_batch_size,
        omega_border_batch_size,
        dim,
        (xmin, ymin),
        (xmax, ymax),
    )


def test_omega_ranges_2D(create_2DCubicMeshPDEStatio):
    TwoD_obj = create_2DCubicMeshPDEStatio
    assert all(
        [
            (
                jnp.all(TwoD_obj.omega[:, i] >= TwoD_obj.min_pts[i])
                and jnp.all(TwoD_obj.omega[:, i] <= TwoD_obj.max_pts[i])
            )
            for i in range(TwoD_obj.dim)
        ]
    )


def test_omega_border_ranges_2D(create_2DCubicMeshPDEStatio):
    TwoD_obj = create_2DCubicMeshPDEStatio
    assert all(
        [
            (
                jnp.all(TwoD_obj.omega_border[:, i] >= TwoD_obj.min_pts[i])
                and jnp.all(TwoD_obj.omega_border[:, i] <= TwoD_obj.max_pts[i])
            )
            for i in range(TwoD_obj.dim)
        ]
    )


def test_get_batch_2D(create_2DCubicMeshPDEStatio):
    TwoD_obj = create_2DCubicMeshPDEStatio
    batch = TwoD_obj.get_batch()
    inside_batch, border_batch = batch.inside_batch, batch.border_batch
    assert all(
        [
            (
                jnp.all(inside_batch[:, i] >= TwoD_obj.min_pts[i])
                and jnp.all(inside_batch[:, i] <= TwoD_obj.max_pts[i])
            )
            for i in range(TwoD_obj.dim)
        ]
    ) and all(
        [
            (
                jnp.all(border_batch[:, i] >= TwoD_obj.min_pts[i])
                and jnp.all(border_batch[:, i] <= TwoD_obj.max_pts[i])
            )
            for i in range(TwoD_obj.dim)
        ]
    )
