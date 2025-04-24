from typing import Literal, TypedDict
import equinox as eqx
from jaxtyping import Float, Array


class ObsBatchDict(TypedDict):
    """
    Correct way to handle type hints for dict with fixed set of keys
    https://peps.python.org/pep-0589/
    """

    pinn_in: Float[Array, "obs_batch_size input_dim"]
    val: Float[Array, "obs_batch_size output_dim"]
    eq_params: dict[str, Float[Array, "obs_batch_size 1"]]


class ODEBatch(eqx.Module):
    temporal_batch: Float[Array, "batch_size"]
    param_batch_dict: dict[str, Array] = eqx.field(default=None)
    obs_batch_dict: ObsBatchDict = eqx.field(default=None)


class PDENonStatioBatch(eqx.Module):
    domain_batch: Float[Array, "batch_size 1+dimension"]
    border_batch: Float[Array, "batch_size dimension n_facets"]
    initial_batch: Float[Array, "batch_size dimension"]
    param_batch_dict: dict[str, Array] = eqx.field(default=None)
    obs_batch_dict: ObsBatchDict = eqx.field(default=None)


class PDEStatioBatch(eqx.Module):
    domain_batch: Float[Array, "batch_size dimension"]
    border_batch: Float[Array, "batch_size dimension n_facets"]
    param_batch_dict: dict[str, Array] = eqx.field(default=None)
    obs_batch_dict: ObsBatchDict = eqx.field(default=None)
