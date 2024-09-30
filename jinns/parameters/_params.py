"""
Formalize the data structure for the parameters
"""

import jax
import equinox as eqx
from typing import Dict
from jaxtyping import Array, PyTree


class Params(eqx.Module):
    """
    The equinox module for the parameters

    Parameters
    ----------
    nn_params : Pytree
        A PyTree of the non-static part of the PINN eqx.Module, i.e., the
        parameters of the PINN
    eq_params : Dict[str, Array]
        A dictionary of the equation parameters. Keys are the parameter name,
        values are their corresponding value
    """

    nn_params: PyTree = eqx.field(kw_only=True, default=None)
    eq_params: Dict[str, Array] = eqx.field(kw_only=True, default=None)


class ParamsDict(eqx.Module):
    """
    The equinox module for a dictionnary of parameters with different keys
    corresponding to different equations.

    Parameters
    ----------
    nn_params : Dict[str, PyTree]
        The neural network's parameters. Most of the time, it will be the
        Array part of an `eqx.Module` obtained by
        `eqx.partition(module, eqx.is_inexact_array)`.
    eq_params : Dict[str, Array]
        A dictionary of the equation parameters. Dict keys are the parameter name as defined your custom loss.
    """

    nn_params: Dict[str, PyTree] = eqx.field(kw_only=True, default=None)
    eq_params: Dict[str, Array] = eqx.field(kw_only=True, default=None)

    def extract_params(self, nn_key: str) -> Params:
        """
        Extract the corresponding `nn_params` and `eq_params` for `nn_key` and
        return them in the form of a `Params` object.
        """
        try:
            return Params(
                nn_params=self.nn_params[nn_key],
                eq_params=self.eq_params[nn_key],
            )
        except (KeyError, IndexError) as e:
            return Params(
                nn_params=self.nn_params[nn_key],
                eq_params=self.eq_params,
            )


def _update_eq_params_dict(
    params: Params, param_batch_dict: Dict[str, Array]
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
    eq_params_batch_dict: Dict[str, Array], params: Params | ParamsDict
) -> tuple[Params]:
    """
    Return the input vmap axes when there is batch(es) of parameters to vmap
    over. The latter are designated by keys in eq_params_batch_dict.
    If eq_params_batch_dict is None (i.e. no additional parameter batch), we
    return (None,).
    """
    if eq_params_batch_dict is None:
        return (None,)
    # We use pytree indexing of vmapped axes and vmap on axis
    # 0 of the eq_parameters for which we have a batch
    # this is for a fine-grained vmaping
    # scheme over the params
    vmap_in_axes_params = (
        type(params)(
            nn_params=None,
            eq_params={
                k: (0 if k in eq_params_batch_dict.keys() else None)
                for k in params.eq_params.keys()
            },
        ),
    )
    return vmap_in_axes_params
