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
    ni = 1000
    domain_batch_size = 32
    initial_batch_size = 20
    border_batch_size = None  # NOTE that a border_batch_size for nb=2 in dim=1
    # is totally useless
    dim = 1
    xmin = -3
    xmax = 3
    tmin = 0
    tmax = 1

    return jinns.data.CubicMeshPDENonStatio(
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
    )


def test_t_range_1D(create_1DCubicMeshPDENonStatio):
    OneD_obj = create_1DCubicMeshPDENonStatio
    assert jnp.all(OneD_obj.domain[:, 0] >= OneD_obj.tmin) and jnp.all(
        OneD_obj.domain[:, 0] <= OneD_obj.tmax
    )


def test_omega_range_1D(create_1DCubicMeshPDENonStatio):
    OneD_obj = create_1DCubicMeshPDENonStatio
    assert jnp.all(OneD_obj.domain[:, 1] >= OneD_obj.min_pts[0]) and jnp.all(
        OneD_obj.domain[:, 1] <= OneD_obj.max_pts[0]
    )


def test_omega_border_range_1D(create_1DCubicMeshPDENonStatio):
    OneD_obj = create_1DCubicMeshPDENonStatio
    assert jnp.all(OneD_obj.border[:, 1] >= OneD_obj.min_pts[0]) and jnp.all(
        OneD_obj.border[:, 1] <= OneD_obj.max_pts[0]
    )


def test_get_batch_1D(create_1DCubicMeshPDENonStatio):
    OneD_obj = create_1DCubicMeshPDENonStatio
    _, batch = OneD_obj.get_batch()
    t_x, t_dx = batch.domain_batch, batch.border_batch
    times_batch = t_x[:, 0]
    domain_batch = t_x[:, 1:]
    border_batch = t_dx[:, 1:]
    assert (
        jnp.all(domain_batch[:] >= OneD_obj.min_pts[0])
        and jnp.all(domain_batch[:] <= OneD_obj.max_pts[0])
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
    ni = 1000
    omega_batch_size = 32
    initial_batch_size = 20
    border_batch_size = 2
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
        ni=ni,
        omega_batch_size=omega_batch_size,
        border_batch_size=border_batch_size,
        initial_batch_size=initial_batch_size,
        dim=dim,
        min_pts=(xmin, ymin),
        max_pts=(xmax, ymax),
        tmin=tmin,
        tmax=tmax,
    )


def test_t_range_2D(create_2DCubicMeshPDENonStatio):
    TwoD_obj = create_2DCubicMeshPDENonStatio
    assert jnp.all(TwoD_obj.domain[:, 0] >= TwoD_obj.tmin) and jnp.all(
        TwoD_obj.domain[:, 0] <= TwoD_obj.tmax
    )


def test_omega_ranges_2D(create_2DCubicMeshPDENonStatio):
    TwoD_obj = create_2DCubicMeshPDENonStatio
    assert all(
        [
            (
                jnp.all(TwoD_obj.domain[:, i + 1] >= TwoD_obj.min_pts[i])
                and jnp.all(TwoD_obj.domain[:, i + 1] <= TwoD_obj.max_pts[i])
            )
            for i in range(TwoD_obj.dim)
        ]
    )


def test_omega_border_ranges_2D(create_2DCubicMeshPDENonStatio):
    TwoD_obj = create_2DCubicMeshPDENonStatio
    assert all(
        [
            (
                jnp.all(TwoD_obj.border[:, i + 1] >= TwoD_obj.min_pts[i])
                and jnp.all(TwoD_obj.border[:, i + 1] <= TwoD_obj.max_pts[i])
            )
            for i in range(TwoD_obj.dim)
        ]
    )


def test_get_batch_2D(create_2DCubicMeshPDENonStatio):
    TwoD_obj = create_2DCubicMeshPDENonStatio
    _, batch = TwoD_obj.get_batch()
    t_x, t_dx = batch.domain_batch, batch.border_batch
    times_batch = t_x[:, 0]
    domain_batch = t_x[:, 1:]
    border_batch = t_dx[:, 1:]
    assert (
        all(
            [
                (
                    jnp.all(domain_batch[:, i] >= TwoD_obj.min_pts[i])
                    and jnp.all(domain_batch[:, i] <= TwoD_obj.max_pts[i])
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


def test_n_samples_in_grid_sampling():
    key = jax.random.PRNGKey(2)
    key, subkey = jax.random.split(key)
    n = 99
    ni = 8
    domain_batch_size = 32
    dim = 2
    xmin = -3
    xmax = 3
    method = "grid"

    with pytest.warns(UserWarning):
        datagenerator = jinns.data.CubicMeshPDENonStatio(
            key=subkey,
            n=n,
            ni=ni,
            domain_batch_size=domain_batch_size,
            dim=dim,
            min_pts=(xmin, xmin),
            max_pts=(xmax, xmax),
            tmin=0,
            tmax=1,
            method=method,
        )
    assert datagenerator.n == int(jnp.round(jnp.sqrt(datagenerator.n)) ** 2)
