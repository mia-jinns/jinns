from jax import random

## The jinns library developped at MIA Paris Saclay by Nicolas Jouvin and Hugo Gangloff
import jinns

key = random.PRNGKey(2)
data_gen_args = {
    "key": key,
    "n": 1000,
    "nb": 1000 * 4,
    "ni": 1000,
    "dim": 2,
    "tmin": 0,
    "tmax": 1,
    "domain_batch_size": 1000,
    "initial_batch_size": 1000,
    "border_batch_size": 1000,
    "min_pts": (0.0, 0.0),
    "max_pts": (50.0, 50.0),
}


def test_sampling_methods():
    train_data_generator = jinns.data.CubicMeshPDENonStatio(
        **data_gen_args,
        method="uniform",
    )

    full_data = train_data_generator.generate_omega_data(key)[1]
    full_batch_data = train_data_generator.get_batch()[1].domain_batch

    assert full_data.shape == full_batch_data.shape
