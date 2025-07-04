"""
Formalize the loss weights data structure
"""

from __future__ import annotations
from dataclasses import fields

from jaxtyping import Array
import jax.numpy as jnp
import equinox as eqx


def lw_converter(x):
    if x is None:
        return x
    elif isinstance(x, tuple):
        # user might input tuple of scalar loss weights to account for cases
        # when dyn loss is also a tuple of (possibly 1D) dyn_loss
        return tuple(jnp.asarray(x_) for x_ in x)
    else:
        return jnp.asarray(x)


class AbstractLossWeights(eqx.Module):
    """
    An abstract class, currently only useful for type hints

    TODO in the future maybe loss weights could be subclasses of
    XDEComponentsAbstract?
    """

    def items(self):
        """
        For the dataclass to be iterated like a dictionary.
        Practical and retrocompatible with old code when loss components were
        dictionaries

        condition: if it is not a tuple it should not be None. It it is a tuple
        it should not be only Nones
        """
        return {
            field.name: getattr(self, field.name)
            for field in fields(self)
            if (
                (
                    not isinstance(getattr(self, field.name), tuple)
                    and getattr(self, field.name) is not None
                )
                or (
                    isinstance(getattr(self, field.name), tuple)
                    and not all(item is None for item in getattr(self, field.name))
                )
            )
        }.items()


class LossWeightsODE(AbstractLossWeights):
    dyn_loss: Array | float | None = eqx.field(
        kw_only=True, default=None, converter=lw_converter
    )
    initial_condition: Array | float | None = eqx.field(
        kw_only=True, default=None, converter=lw_converter
    )
    observations: Array | float | None = eqx.field(
        kw_only=True, default=None, converter=lw_converter
    )


class LossWeightsPDEStatio(AbstractLossWeights):
    dyn_loss: Array | float | None = eqx.field(
        kw_only=True, default=None, converter=lw_converter
    )
    norm_loss: Array | float | None = eqx.field(
        kw_only=True, default=None, converter=lw_converter
    )
    boundary_loss: Array | float | None = eqx.field(
        kw_only=True, default=None, converter=lw_converter
    )
    observations: Array | float | None = eqx.field(
        kw_only=True, default=None, converter=lw_converter
    )


class LossWeightsPDENonStatio(AbstractLossWeights):
    dyn_loss: Array | float | None = eqx.field(
        kw_only=True, default=None, converter=lw_converter
    )
    norm_loss: Array | float | None = eqx.field(
        kw_only=True, default=None, converter=lw_converter
    )
    boundary_loss: Array | float | None = eqx.field(
        kw_only=True, default=None, converter=lw_converter
    )
    observations: Array | float | None = eqx.field(
        kw_only=True, default=None, converter=lw_converter
    )
    initial_condition: Array | float | None = eqx.field(
        kw_only=True, default=None, converter=lw_converter
    )
