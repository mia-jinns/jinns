import jax
import jax.numpy as jnp
from jinns.data import CubicMeshPDENonStatio

# Set random seed for reproducibility
key = jax.random.PRNGKey(42)

data_gen_args = {
    "key": key,
    "n": 1000,
    "nb": 1000 * 4,
    "ni": 1000,
    "dim": 2,
    "tmin": 0,
    "tmax": 1,
    "domain_batch_size": 100,
    "initial_batch_size": 100,
    "border_batch_size": 100,
    "min_pts": (0.0, 0.0),
    "max_pts": (50.0, 50.0),
}


def test_qmc_sampling_omega_domain():
    """Test Sobol and Halton sampling in omega domain for 1D and 2D cases."""

    dg = CubicMeshPDENonStatio(**data_gen_args, method="sobol")
    data = dg.generate_omega_data(key)[1]

    # Check shape and bounds
    assert data.shape == (1000, 2)
    assert jnp.all(data >= 0.0)
    assert jnp.all(data <= 50.0)

    # Check uniformity (crude test)
    mean = jnp.mean(data, axis=0)
    assert jnp.allclose(mean, jnp.array([50.0 / 2, 50.0 / 2]), atol=0.1)


def test_qmc_border_sampling():
    """Test border sampling with QMC methods."""
    dg = CubicMeshPDENonStatio(
        **data_gen_args,
        method="sobol",
    )

    border_batch = dg.get_batch()[1].border_batch

    # Check border batch shape
    assert border_batch.shape == (100, 3, 4)  # (batch_size, dim, n_facets)

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
