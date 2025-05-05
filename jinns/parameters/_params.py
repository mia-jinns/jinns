"""
Formalize the data structure for the parameters
"""

from typing import Generic, TypeVar
import jax
import equinox as eqx
from jaxtyping import Array, PyTree, Float

T = TypeVar("T")  # the generic type for what is in the Params PyTree because we
# have possibly Params of Arrays, boolean, ...


class Params(eqx.Module, Generic[T]):
    """
    The equinox module for the parameters

    Parameters
    ----------
    nn_params : PyTree[T]
        A PyTree of the non-static part of the PINN eqx.Module, i.e., the
        parameters of the PINN
    eq_params : dict[str, T]
        A dictionary of the equation parameters. Keys are the parameter name,
        values are their corresponding value
    """

    nn_params: PyTree[T] = eqx.field(kw_only=True, default=None)
    eq_params: dict[str, T] = eqx.field(kw_only=True, default=None)


def _update_eq_params_dict(
    params: Params[Array],
    param_batch_dict: dict[str, Float[Array, " param_batch_size dim"]],
) -> Params:
    """
    Update params.eq_params with a batch of eq_params for given key(s)
    """

    # artificially "complete" `param_batch_dict` with None to match `params`
    # PyTree  structure
    param_batch_dict_ = param_batch_dict | {
        k: None for k in set(params.eq_params.keys()) - set(param_batch_dict.keys())
    }

    # Replace at non None leafs
    params = eqx.tree_at(
        lambda p: p.eq_params,
        params,
        jax.tree_util.tree_map(
            lambda p, q: q if q is not None else p,
            params.eq_params,
            param_batch_dict_,
        ),
    )

    return params


def _get_vmap_in_axes_params(
    eq_params_batch_dict: dict[str, Array], params: Params[Array]
) -> tuple[Params[int | None] | None]:
    """
    Return the input vmap axes when there is batch(es) of parameters to vmap
    over. The latter are designated by keys in eq_params_batch_dict.
    If eq_params_batch_dict is None (i.e. no additional parameter batch), we
    return (None,).

    Note that we return a Params PyTree with an integer to designate the
    vmapped axis or None if there is not
    """
    if eq_params_batch_dict is None:
        return (None,)
    # We use pytree indexing of vmapped axes and vmap on axis
    # 0 of the eq_parameters for which we have a batch
    # this is for a fine-grained vmaping
    # scheme over the params
    vmap_in_axes_params = (
        Params(
            nn_params=None,
            eq_params={
                k: (0 if k in eq_params_batch_dict.keys() else None)
                for k in params.eq_params.keys()
            },
        ),
    )
    return vmap_in_axes_params
