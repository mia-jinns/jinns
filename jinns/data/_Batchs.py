import equinox as eqx
from jaxtyping import Float, Array


class ODEBatch(eqx.Module):
    temporal_batch: Float[Array, "batch_size"]
    param_batch_dict: dict = eqx.field(default=None)
    obs_batch_dict: dict = eqx.field(default=None)


class PDENonStatioBatch(eqx.Module):
    domain_batch: Float[Array, "batch_size 1+dimension"]
    border_batch: Float[Array, "batch_size dimension n_facets"]
    initial_batch: Float[Array, "batch_size dimension"]
    param_batch_dict: dict = eqx.field(default=None)
    obs_batch_dict: dict = eqx.field(default=None)


class PDEStatioBatch(eqx.Module):
    domain_batch: Float[Array, "batch_size dimension"]
    border_batch: Float[Array, "batch_size dimension n_facets"]
    param_batch_dict: dict = eqx.field(default=None)
    obs_batch_dict: dict = eqx.field(default=None)
