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
from jinns.utils._utils import get_grid

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
        self, inputs: Float[Array, " dim"], u: AbstractPINN, params: Params[Array]
    ) -> Float[Array, " eq_dim"]:
        """
        Note that we write the body for a single facet here, the decorator
        takes care of applying the same condition for all the facets
        """
        return u(inputs, params)

    @equation_on_all_facets_equal
    def equation_f(
        self, inputs: Float[Array, " dim"], params: Params[Array], gridify: bool = False
    ) -> Float[Array, " eq_dim"]:
        """
        Note that we write the body for a single facet here, the decorator
        takes care of applying the same condition for all the facets

        Note the gridification needed for SPINNs is done here because in this
        body function, either via `@equation_on_all_facets_equal` or via a
        manual handling, the inputs array does not contain the facet axis which
        is required for `get_grid`
        """
        if gridify:  # to handle SPINN, ignore otherwise
            inputs_ = get_grid(inputs)
            # inputs.shape[-1] indicates the number of dimensions of the pb
            # thus we get the correct grid of zeros
            return jnp.zeros(inputs_.shape[: inputs.shape[-1]])[..., None]
        else:
            return jnp.zeros((1,))
