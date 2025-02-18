"""
Implement abstract class for PINN architectures
"""

import abc
from typing import Literal
from dataclasses import InitVar
import equinox as eqx
from jaxtyping import Float, Array, PyTree
import jax.numpy as jnp

from jinns.parameters._params import Params, ParamsDict


class PINN(eqx.Module):
    r"""
    Abstract class for PINN architectures. This can be seen as wrappers for
    eqx.Modules which actually implement the NN architectures.

    For compatibility with jinns, we require that a PINN architecture:
        1) has an eqx.partition call in __post_init__ in order to store the initial
    parameters and static elements
        2) has an eqx.Module (`eqx_network`) InitVar passed to __post_init__ for the
    operation described in 1)
        3) has a eq_type argument, whose type is given below, for internal
    operation in jinns.
        4) has a slice solution argument. It is a jnp.s\_ object which
    indicates which axis of the PINN output is dedicated to the actual equation
    solution. Default None means that slice_solution = the whole PINN output.
    This argument is useful
    when the PINN is also used to output equation parameters for example
    Note that it must be a slice and not an integer (a preprocessing of the
    user provided argument takes care of it).

    Raises
    ------
    RuntimeError
        If the parameter value for eq_type is not in `["ODE", "statio_PDE",
        "nonstatio_PDE"]`
    """

    slice_solution: slice = eqx.field(
        static=True, kw_only=True, default_factory=lambda: jnp.s_[...]
    )
    eq_type: Literal["ODE", "statio_PDE", "nonstatio_PDE"] = eqx.field(
        static=True, kw_only=True
    )

    eqx_network: InitVar[eqx.Module] = eqx.field(kw_only=True)

    params: PyTree = eqx.field(init=False)
    static: PyTree = eqx.field(init=False, static=True)

    def __post_init__(self, eqx_network):
        if self.eq_type not in ["ODE", "statio_PDE", "nonstatio_PDE"]:
            raise RuntimeError("Wrong parameter value for eq_type")
        self.params, self.static = eqx.partition(eqx_network, eqx.is_inexact_array)

    @property
    def init_params(self) -> PyTree:
        """
        Returns an initial set of parameters
        """
        return self.params

    @abc.abstractmethod
    def __call__(
        self,
        inputs: Float[Array, "input_dim"],
        params: Params | ParamsDict | PyTree,
        *args,
        **kwargs,
    ) -> Float[Array, "output_dim"]:
        """
        A proper __call__ implementation performs an eqx.combine here with
        `params` and `self.static` to recreate the callable eqx.Module
        architecture. The rest of the content of this function is dependent on
        the network.
        """
        raise NotImplementedError("A PINN should have a __call__ method")
