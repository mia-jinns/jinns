import equinox as eqx
from jaxtyping import Float, Array


class ODEBatch(eqx.Module):
    temporal_batch: Float[Array, "batch_size"]
    param_batch_dict: dict = eqx.field(default=None)
    obs_batch_dict: dict = eqx.field(default=None)


class PDENonStatioBatch(eqx.Module):
    times_x_inside_batch: (
        Float[Array, "batch_size dimension"] | Float[Array, "(batch_size**2) dimension"]
    )
    times_x_border_batch: (
        Float[Array, "border_batch_size dimension n_facets"]
        | Float[Array, "(border_batch_size**2) dimension n_facets"]
    )
    param_batch_dict: dict = eqx.field(default=None)
    obs_batch_dict: dict = eqx.field(default=None)


class PDEStatioBatch(eqx.Module):
    inside_batch: Float[Array, "batch_size dimension"]
    border_batch: Float[Array, "batch_size dimension n_facets"]
    param_batch_dict: dict = eqx.field(default=None)
    obs_batch_dict: dict = eqx.field(default=None)
