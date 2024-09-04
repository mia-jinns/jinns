"""
Formalize the data structure for the parameters
"""

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
