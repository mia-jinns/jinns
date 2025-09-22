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
    eq_params: (
        eqx.Module | None
    )  # None cause sometime user don't provide observed params


class ODEBatch(eqx.Module):
    temporal_batch: Float[Array, "  batch_size"]
    param_batch_dict: eqx.Module | None = eqx.field(default=None)
    obs_batch_dict: ObsBatchDict | None = eqx.field(default=None)


class PDEStatioBatch(eqx.Module):
    domain_batch: Float[Array, "  batch_size dimension"]
    border_batch: Float[Array, "  batch_size dimension n_facets"] | None
    param_batch_dict: eqx.Module | None
    obs_batch_dict: ObsBatchDict | None

    # rewrite __init__ to be able to use inheritance for the NonStatio case
    # below. That way PDENonStatioBatch is a subtype of PDEStatioBatch which
    # 1) makes more sense and 2) CubicMeshPDENonStatio.get_batch passes pyright.
    def __init__(
        self,
        *,
        domain_batch: Float[Array, "  batch_size dimension"],
        border_batch: Float[Array, "  batch_size dimension n_facets"] | None,
        param_batch_dict: eqx.Module | None = None,
        obs_batch_dict: ObsBatchDict | None = None,
    ):
        # TODO: document this ?
        self.domain_batch = domain_batch
        self.border_batch = border_batch
        self.param_batch_dict = param_batch_dict
        self.obs_batch_dict = obs_batch_dict


class PDENonStatioBatch(PDEStatioBatch):
    # TODO: document this ?
    domain_batch: Float[Array, "  batch_size 1+dimension"]  # Override type
    initial_batch: (
        Float[Array, "  batch_size dimension"] | None
    )  # why can it be None ? Examples?

    def __init__(
        self,
        *,
        domain_batch: Float[Array, "  batch_size 1+dimension"],
        border_batch: Float[Array, "  batch_size dimension n_facets"] | None,
        initial_batch: Float[Array, "  batch_size dimension"] | None,
        param_batch_dict: eqx.Module | None = None,
        obs_batch_dict: ObsBatchDict | None = None,
    ):
        self.domain_batch = domain_batch
        self.border_batch = border_batch
        self.initial_batch = initial_batch
        self.param_batch_dict = param_batch_dict
        self.obs_batch_dict = obs_batch_dict
