"""
Formalize the data structure for the parameters
"""

import jax
import equinox as eqx
from jaxtyping import Array, PyTree


class Params(eqx.Module):
    """
    The equinox module for the parameters

    Parameters
    ----------
    nn_params
        A PyTree of the non-static part of the PINN eqx.Module, i.e., the
        parameters of the PINN
    eq_params
        A dictionary of the equation parameters. Keys are the parameter name,
        values are their corresponding value
    """

    nn_params: PyTree = eqx.field(kw_only=True)
    eq_params: dict[str, Array] = eqx.field(kw_only=True)


class ParamsDict(eqx.Module):
    """
    The equinox module for the parameters

    Parameters
    ----------
    nn_params
        XXX
    eq_params
        A dictionary of the equation parameters. Keys are the parameter name,
        values are their corresponding value
    """

    nn_params: dict[str, PyTree] = eqx.field(kw_only=True)
    eq_params: dict[str, Array] = eqx.field(kw_only=True)


def _update_eq_params_dict(params, param_batch_dict):
    """
    update params.eq_params with a batch of eq_params for given key(s)
    """
    param_batch_dict_ = param_batch_dict | {
        k: None for k in set(params.eq_params.keys()) - set(param_batch_dict.keys())
    }
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


def _get_vmap_in_axes_params(eq_params_batch_dict, params):
    """
    Return the input vmap axes when there is batch(es) of parameters to vmap
    over. The latter are designated by keys in eq_params_batch_dict
    If eq_params_batch_dict (ie no additional parameter batch), we return None
    """
    if eq_params_batch_dict is None:
        return (None,)
    # We use pytree indexing of vmapped axes and vmap on axis
    # 0 of the eq_parameters for which we have a batch
    # this is for a fine-grained vmaping
    # scheme over the params
    vmap_in_axes_params = Params(
        nn_params=None,
        eq_params={
            k: (0 if k in eq_params_batch_dict.keys() else None)
            for k in params.eq_params.keys()
        },
    )
    return vmap_in_axes_params


def _extract_nn_params(params_dict: ParamsDict, nn_key: str) -> Params:
    """
    Given a ParamsDict for system loss we extract the
    corresponding `nn_params` for `nn_key` and reform Params
    """
    try:
        return Params(
            nn_params=params_dict.nn_params[nn_key],
            eq_params=params_dict.eq_params[nn_key],
        )
    except (KeyError, IndexError) as e:
        return Params(
            nn_params=params_dict.nn_params[nn_key],
            eq_params=params_dict.eq_params,
        )