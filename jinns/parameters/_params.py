"""
Formalize the data structure for the parameters
"""

from __future__ import annotations
from dataclasses import fields
from typing import Generic, TypeVar
import equinox as eqx
from jaxtyping import Array, PyTree

from jinns.utils._DictToModuleMeta import DictToModuleMeta

T = TypeVar("T")  # the generic type for what is in the Params PyTree because we
# have possibly Params of Arrays, boolean, ...

### NOTE
### We are taking derivatives with respect to Params eqx.Modules.
### This has been shown to behave weirdly if some fields of eqx.Modules have
### been set as `field(init=False)`, we then should never create such fields in
### jinns' Params modules.
### We currently have silenced the warning related to this (see jinns.__init__
### see https://github.com/patrick-kidger/equinox/pull/1043/commits/f88e62ab809140334c2f987ed13eff0d80b8be13


class EqParams(metaclass=DictToModuleMeta):
    """
    Note that this is exposed to the user for the particular case where the
    user, during its work, wants to change the equation parameters. In this
    case, the user must import EqParams and call `EqParams.clear()`
    """

    pass


class Params(eqx.Module, Generic[T]):
    """
    The equinox module for the parameters

    Parameters
    ----------
    nn_params : PyTree[T]
        A PyTree of the non-static part of the PINN eqx.Module, i.e., the
        parameters of the PINN
    eq_params : PyTree[T]
        A PyTree of the equation parameters. For retrocompatibility it us
        provided as a dictionary of the equation parameters where keys are the parameter names, and values are their corresponding values. Internally,
        it will be transformed to a custom instance of `EqParams`.
    """

    nn_params: PyTree[T]
    eq_params: PyTree[T]

    def __init__(
        self,
        nn_params: PyTree[T] | None = None,
        eq_params: dict[str, T] | None = None,
    ):
        self.nn_params = nn_params
        if isinstance(eq_params, dict):
            self.eq_params = EqParams(eq_params, "EqParams")
        else:
            self.eq_params = eq_params

    def partition(self, mask: Params[bool] | None):
        """
        following the boolean mask, partition into two Params
        """
        if mask is not None:
            return eqx.partition(self, mask)
        else:
            return self, None


def update_eq_params(
    params: Params[Array],
    eq_param_batch: PyTree[Array] | None,
) -> Params[Array]:
    """
    Update params.eq_params with a batch of eq_params for given key(s)
    """

    if eq_param_batch is None:
        return params

    param_names_to_update = tuple(f.name for f in fields(eq_param_batch))
    params = eqx.tree_at(
        lambda p: p.eq_params,
        params,
        eqx.tree_at(
            lambda pt: tuple(getattr(pt, f) for f in param_names_to_update),
            params.eq_params,
            tuple(getattr(eq_param_batch, f) for f in param_names_to_update),
            is_leaf=lambda x: x is None or eqx.is_inexact_array(x),
        ),
    )

    return params


def _get_vmap_in_axes_params(
    eq_param_batch: eqx.Module | None, params: Params[Array]
) -> tuple[Params[int | None] | None]:
    """
    Return the input vmap axes when there is batch(es) of parameters to vmap
    over. The latter are designated by keys in eq_params_batch_dict.
    If eq_params_batch_dict is None (i.e. no additional parameter batch), we
    return (None,).

    Note that we return a Params PyTree with an integer to designate the
    vmapped axis or None if there is not
    """
    if eq_param_batch is None:
        return (None,)
    # We use pytree indexing of vmapped axes and vmap on axis
    # 0 of the eq_parameters for which we have a batch
    # this is for a fine-grained vmaping
    # scheme over the params
    param_names_to_vmap = tuple(f.name for f in fields(eq_param_batch))
    vmap_axes_dict = {
        k.name: (0 if k.name in param_names_to_vmap else None)
        for k in fields(params.eq_params)
    }
    eq_param_vmap_axes = type(params.eq_params)(**vmap_axes_dict)
    vmap_in_axes_params = (
        Params(
            nn_params=None,
            eq_params=eq_param_vmap_axes,
        ),
    )
    return vmap_in_axes_params
