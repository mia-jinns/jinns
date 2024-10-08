import pytest
import jax
import jax.numpy as jnp
import jinns


@pytest.fixture
def create_1DCubicMeshPDENonStatio():
    key = jax.random.PRNGKey(2)
    key, subkey = jax.random.split(key)
    n = 1000
    nb = 2
    nt = 1000
    omega_batch_size = 32
    temporal_batch_size = 20
    omega_border_batch_size = 2
    dim = 1
    xmin = -3
    xmax = 3
    tmin = 0
    tmax = 1

    return jinns.data.CubicMeshPDENonStatio(
        key=subkey,
        n=n,
        nb=nb,
        nt=nt,
        omega_batch_size=omega_batch_size,
        omega_border_batch_size=omega_border_batch_size,
        temporal_batch_size=temporal_batch_size,
        dim=dim,
        min_pts=(xmin,),
        max_pts=(xmax,),
        tmin=tmin,
        tmax=tmax,
    )


def test_t_range_1D(create_1DCubicMeshPDENonStatio):
    OneD_obj = create_1DCubicMeshPDENonStatio
    assert jnp.all(OneD_obj.times >= OneD_obj.tmin) and jnp.all(
        OneD_obj.times <= OneD_obj.tmax
    )


def test_omega_range_1D(create_1DCubicMeshPDENonStatio):
    OneD_obj = create_1DCubicMeshPDENonStatio
    assert jnp.all(OneD_obj.omega >= OneD_obj.min_pts[0]) and jnp.all(
        OneD_obj.omega <= OneD_obj.max_pts[0]
    )


def test_omega_border_range_1D(create_1DCubicMeshPDENonStatio):
    OneD_obj = create_1DCubicMeshPDENonStatio
    assert jnp.all(OneD_obj.omega_border >= OneD_obj.min_pts[0]) and jnp.all(
        OneD_obj.omega_border <= OneD_obj.max_pts[0]
    )


def test_get_batch_1D(create_1DCubicMeshPDENonStatio):
    OneD_obj = create_1DCubicMeshPDENonStatio
    _, batch = OneD_obj.get_batch()
    t_x, t_dx = batch.times_x_inside_batch, batch.times_x_border_batch
    times_batch = t_x[:, 0]
    inside_batch = t_x[:, 1:]
    border_batch = t_dx[:, 1:]
    assert (
        jnp.all(inside_batch[:] >= OneD_obj.min_pts[0])
        and jnp.all(inside_batch[:] <= OneD_obj.max_pts[0])
        and jnp.all(border_batch[:] >= OneD_obj.min_pts[0])
        and jnp.all(border_batch[:] <= OneD_obj.max_pts[0])
        and jnp.all(times_batch[:] >= OneD_obj.tmin)
        and jnp.all(times_batch[:] <= OneD_obj.tmax)
    )


@pytest.fixture
def create_2DCubicMeshPDENonStatio():
    key = jax.random.PRNGKey(2)
    key, subkey = jax.random.split(key)
    n = 1024
    nb = 8
    nt = 1000
    omega_batch_size = 32
    temporal_batch_size = 20
    omega_border_batch_size = 2
    dim = 2
    xmin = -3
    xmax = 3
    ymin = -5
    ymax = 5
    tmin = 0
    tmax = 1

    return jinns.data.CubicMeshPDENonStatio(
        key=subkey,
        n=n,
        nb=nb,
        nt=nt,
        omega_batch_size=omega_batch_size,
        omega_border_batch_size=omega_border_batch_size,
        temporal_batch_size=temporal_batch_size,
        dim=dim,
        min_pts=(xmin, ymin),
        max_pts=(xmax, ymax),
        tmin=tmin,
        tmax=tmax,
    )


def test_t_range_2D(create_2DCubicMeshPDENonStatio):
    TwoD_obj = create_2DCubicMeshPDENonStatio
    assert jnp.all(TwoD_obj.times >= TwoD_obj.tmin) and jnp.all(
        TwoD_obj.times <= TwoD_obj.tmax
    )


def test_omega_ranges_2D(create_2DCubicMeshPDENonStatio):
    TwoD_obj = create_2DCubicMeshPDENonStatio
    assert all(
        [
            (
                jnp.all(TwoD_obj.omega[:, i] >= TwoD_obj.min_pts[i])
                and jnp.all(TwoD_obj.omega[:, i] <= TwoD_obj.max_pts[i])
            )
            for i in range(TwoD_obj.dim)
        ]
    )


def test_omega_border_ranges_2D(create_2DCubicMeshPDENonStatio):
    TwoD_obj = create_2DCubicMeshPDENonStatio
    assert all(
        [
            (
                jnp.all(TwoD_obj.omega_border[:, i] >= TwoD_obj.min_pts[i])
                and jnp.all(TwoD_obj.omega_border[:, i] <= TwoD_obj.max_pts[i])
            )
            for i in range(TwoD_obj.dim)
        ]
    )


def test_get_batch_2D(create_2DCubicMeshPDENonStatio):
    TwoD_obj = create_2DCubicMeshPDENonStatio
    _, batch = TwoD_obj.get_batch()
    t_x, t_dx = batch.times_x_inside_batch, batch.times_x_border_batch
    times_batch = t_x[:, 0]
    inside_batch = t_x[:, 1:]
    border_batch = t_dx[:, 1:]
    assert (
        all(
            [
                (
                    jnp.all(inside_batch[:, i] >= TwoD_obj.min_pts[i])
                    and jnp.all(inside_batch[:, i] <= TwoD_obj.max_pts[i])
                )
                for i in range(TwoD_obj.dim)
            ]
        )
        and all(
            [
                (
                    jnp.all(border_batch[:, i] >= TwoD_obj.min_pts[i])
                    and jnp.all(border_batch[:, i] <= TwoD_obj.max_pts[i])
                )
                for i in range(TwoD_obj.dim)
            ]
        )
        and all(
            [
                (
                    jnp.all(times_batch[:] >= TwoD_obj.tmin)
                    and jnp.all(times_batch[:] <= TwoD_obj.tmax)
                )
                for i in range(TwoD_obj.dim)
            ]
        )
    )
