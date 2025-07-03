import jax
import jax.numpy as jnp
from jinns.data import CubicMeshPDENonStatio

# Set random seed for reproducibility
key = jax.random.PRNGKey(42)

data_gen_args_1D = {
    "key": key,
    "n": 1024,
    "nb": 1024 * 4,
    "ni": 1024,
    "dim": 1,
    "tmin": 0,
    "tmax": 1,
    "domain_batch_size": 512,
    "initial_batch_size": 512,
    "border_batch_size": 512,
    "min_pts": (0.0,),
    "max_pts": (50.0,),
}

data_gen_args_2D = {
    "key": key,
    "n": 1024,
    "nb": 1024 * 4,
    "ni": 1024,
    "dim": 2,
    "tmin": 0,
    "tmax": 1,
    "domain_batch_size": 512,
    "initial_batch_size": 512,
    "border_batch_size": 512,
    "min_pts": (0.0, 0.0),
    "max_pts": (50.0, 50.0),
}


def test_1D_qmc_sampling():
    """
    Test Sobol and Halton sampling in omega domain for 1D case.
    """
    dg = CubicMeshPDENonStatio(**data_gen_args_1D, method="sobol")
    batch_object = dg.get_batch()[1]

    domain_batch = batch_object.domain_batch
    border_batch = batch_object.border_batch

    # Check shape and bounds
    assert domain_batch.shape == (512, 2)
    assert border_batch.shape == (512, 2, 2)
    assert jnp.all(domain_batch[:, 0] >= 0.0)
    assert jnp.all(domain_batch[:, 0] <= 1.0)
    assert jnp.all(domain_batch[:, 1:] >= 0.0)
    assert jnp.all(domain_batch[:, 1:] <= 50.0)

    # Check points are on the boundary
    for facet in range(2):
        facet_points = border_batch[..., facet][:, 1:]
        # Points should be on one of the boundaries (x=0, x=50)
        assert jnp.any(
            jnp.isclose(facet_points[:, 0], 0.0) | jnp.isclose(facet_points[:, 0], 50.0)
        )


def test_2D_qmc_sampling():
    """
    Test Sobol and Halton sampling in omega domain for 1D and 2D cases.
    """

    dg = CubicMeshPDENonStatio(**data_gen_args_2D, method="sobol")
    batch_object = dg.get_batch()[1]

    domain_batch = batch_object.domain_batch
    border_batch = batch_object.border_batch

    # Check shape and bounds
    assert domain_batch.shape == (512, 3)
    assert border_batch.shape == (512, 3, 4)
    assert jnp.all(domain_batch[:, 0] >= 0.0)
    assert jnp.all(domain_batch[:, 0] <= 1.0)
    assert jnp.all(domain_batch[:, 1:] >= 0.0)
    assert jnp.all(domain_batch[:, 1:] <= 50.0)

    # Check points are on the boundary
    for facet in range(4):
        facet_points = border_batch[..., facet][:, 1:]
        # Points should be on one of the boundaries (x=0, x=50., y=0, or y=50.)
        assert (
            jnp.any(jnp.isclose(facet_points[:, 0], 0.0))
            or jnp.any(jnp.isclose(facet_points[:, 0], 50.0))
            or jnp.any(jnp.isclose(facet_points[:, 1], 0.0))
            or jnp.any(jnp.isclose(facet_points[:, 1], 50.0))
        )
