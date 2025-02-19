"""
Implement abstract class for PINN architectures
"""

import abc
from typing import Literal, Callable, Union, Any
from dataclasses import InitVar
import equinox as eqx
from jaxtyping import Float, Array, PyTree
import jax.numpy as jnp
import typing
from jinns.parameters._params import Params, ParamsDict


class PINN(eqx.Module):
    r"""
    Abstract class for PINN objects. This can be seen as wrappers on
    eqx.Modules which actually implement the NN architectures.

    We use the `eqx.partition` and `eqx.combine` strategy of Equinox: a
    `filter_spec` is applied on the PyTree and splits it into two PyTree with
    the same structure: a static one (invisible to JAX transform such as JIT,
    grad, etc.) and dynamic one. By convention, anything not static is
    considered a parameter in Jinns.

    For compatibility with jinns, we require that a PINN architecture:

        1) has an eqx.Module (`eqx_network`) InitVar passed to __post_init__
        representing the network architecture.
        2) calls `eqx.partition` in __post_init__ in order to store the
        static part of the model and the initial parameters.
        3) has a `eq_type` argument, used for handling internal operations in
        jinns.
        4) has a `slice_solution` argument. It is a `jnp.s\_` object which
    indicates which axis of the PINN output is dedicated to the actual equation
    solution. Default None means that slice_solution = the whole PINN output.
    For example, this argument is useful when the PINN is also used to output
    equation parameters. Note that it must be a slice and not an integer (a
    preprocessing of the user provided argument takes care of it).

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
    filter_spec: PyTree[Union[bool, Callable[[Any], bool]]] = eqx.field(
        kw_only=True, default=eqx.is_inexact_array
    )

    _init_params: PyTree = eqx.field(init=False)
    static: PyTree = eqx.field(init=False, static=True)

    def __post_init__(self, eqx_network):
        if self.eq_type not in ["ODE", "statio_PDE", "nonstatio_PDE"]:
            raise RuntimeError("Wrong parameter value for eq_type")
        # saving the static part of the model and initial parameters

        self._init_params, self.static = eqx.partition(eqx_network, self.filter_spec)

    @property
    def init_params(self) -> PyTree:
        """
        Returns an initial set of parameters
        """
        return self._init_params

    def combine(self, params: PyTree) -> PyTree:
        return eqx.combine(self.static, params)

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
        # recreate the whole PyTree of the network
        nn = self.combine(params)
        # ... do stuff
        raise NotImplementedError("A PINN should have a __call__ method")
