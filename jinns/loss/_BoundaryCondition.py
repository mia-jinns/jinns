"""
Implements some common boundary conditions
"""

from __future__ import (
    annotations,
)  # https://docs.python.org/3/library/typing.html#constant

from typing import TYPE_CHECKING
from jaxtyping import Float, Array
import jax.numpy as jnp
from jinns.loss._BoundaryConditionAbstract import BoundaryCondition
from jinns.loss._loss_utils import equation_on_all_facets_equal

if TYPE_CHECKING:
    from jinns.parameters import Params
    from jinns.nn._abstract_pinn import AbstractPINN


class Dirichlet(BoundaryCondition):
    r"""
    Implements

    $$
    u(x) = 0, \forall x\in\delta\Omega
    $$
    """

    @equation_on_all_facets_equal
    def equation_u(
        self, inputs: Float[Array, " InputDim"], u: AbstractPINN, params: Params[Array]
    ) -> Float[Array, " eq_dim"]:
        print("inside equation_u", inputs.shape)
        return u(inputs, params)

    @equation_on_all_facets_equal
    def equation_f(
        self, inputs: Float[Array, " InputDim"], params: Params[Array]
    ) -> Float[Array, " eq_dim"]:
        return jnp.array([0.0])
