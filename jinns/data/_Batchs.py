from typing import TypedDict
import equinox as eqx
from jaxtyping import Float, Array


class ObsBatchDict(TypedDict):
    """
    Keys:
    -pinn_in, a mini batch of pinn inputs
    -val, a mini batch of corresponding observations
    -eq_params, a dictionary with entry names found in `params["eq_params"]`
    and values giving the correspond parameter value for the couple (input,
    value) mentioned before).

    A TypedDict is the correct way to handle type hints for dict with fixed set of keys
    https://peps.python.org/pep-0589/
    """

    pinn_in: Float[Array, "  obs_batch_size input_dim"]
    val: Float[Array, "  obs_batch_size output_dim"]
    eq_params: dict[str, Float[Array, "  obs_batch_size 1"]]


class ODEBatch(eqx.Module):
    temporal_batch: Float[Array, "  batch_size"]
    param_batch_dict: dict[str, Array] = eqx.field(default=None)
    obs_batch_dict: ObsBatchDict = eqx.field(default=None)


class PDENonStatioBatch(eqx.Module):
    domain_batch: Float[Array, "  batch_size 1+dimension"]
    border_batch: Float[Array, "  batch_size dimension n_facets"] | None
    initial_batch: Float[Array, "  batch_size dimension"] | None
    param_batch_dict: dict[str, Array] = eqx.field(default=None)
    obs_batch_dict: ObsBatchDict = eqx.field(default=None)


class PDEStatioBatch(eqx.Module):
    domain_batch: Float[Array, "  batch_size dimension"]
    border_batch: Float[Array, "  batch_size dimension n_facets"] | None
    param_batch_dict: dict[str, Array] = eqx.field(default=None)
    obs_batch_dict: ObsBatchDict = eqx.field(default=None)
