"""
Implements abstract classes for boundary condition
"""

from __future__ import (
    annotations,
)  # https://docs.python.org/3/library/typing.html#constant

import warnings
import abc
from typing import TYPE_CHECKING
import equinox as eqx
from jaxtyping import Float, Array
from jinns.nn import SPINN, PINN
from jinns.utils._utils import _subtract_with_check


# See : https://docs.kidger.site/equinox/api/module/advanced_fields/#equinox.AbstractClassVar--known-issues
if TYPE_CHECKING:
    from jinns.parameters import Params
    from jinns.nn._abstract_pinn import AbstractPINN
else:
    pass

InputDim = Float[Array, " dim n_facet"] | Float[Array, " dim+1 n_facet"]


class BoundaryCondition(eqx.Module):
    r""" """

    def evaluate(
        self,
        inputs: InputDim,
        u: AbstractPINN,
        params: Params[Array],
    ) -> Float[Array, " eq_dim n_facet"]:
        eval_u = self.equation_u(inputs, u, params)

        if eval_u.ndim == 0:
            raise ValueError(
                "The output of loss must be vectorial, i.e. of shape (d,) with d >= 1"
            )
        if eval_u.ndim > 2 and not isinstance(u, SPINN):
            warnings.warn(
                "Return value from BoundaryCondition' equation has more "
                "than two dimensions. This is in general a mistake (probably from "
                "an unfortunate broadcast in jnp.array computations) resulting in "
                "bad reduction operations in losses."
            )

        if isinstance(u, PINN):
            eval_f = self.equation_f(inputs, params, gridify=False)
        elif isinstance(u, SPINN):
            eval_f = self.equation_f(inputs, params, gridify=True)
        else:
            raise ValueError(f"Bad type for u. Got {type(u)}, expected PINN or SPINN")

        # TODO add check on shape

        residual = _subtract_with_check(
            eval_f,
            eval_u,
            cause="boundary condition fun",
        )
        return residual

    @abc.abstractmethod
    def equation_u(
        self,
        inputs: InputDim,
        u: AbstractPINN,
        params: Params[Array],
    ) -> Float[Array, " eq_dim n_facet"]:
        r"""The differential operator on the boundaries defining the stationary PDE.

        !!! warning

            This is an abstract method to be implemented by users.

        Parameters
        ----------
        x : InputDim
            A `d` dimensional jnp.array representing a point in in each
            element (facet) of $\delta\Omega$.
        u : AbstractPINN
            The neural network.
        params : Params[Array]
            The parameters of the equation and the networks, $\theta$ and $\nu$ respectively.

        Returns
        -------
        Float[Array, "eq_dim facet"]
            The residual, *i.e.* the differential operator
            $\mathcal{B}_\theta[u_\nu](x)$ evaluated at each point `x`.

        Raises
        ------
        NotImplementedError
            This is an abstract method to be implemented.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def equation_f(
        self, inputs: InputDim, params: Params[Array], gridify: bool = False
    ) -> Float[Array, " eq_dim n_facet"]:
        r"""The values that the solution should match on the boundaries

        !!! warning

            This is an abstract method to be implemented by users.

        Parameters
        ----------
        x : InputDim
            A `d` dimensional jnp.array representing a point in in each
            element (facet) of $\delta\Omega$.
        params : Params[Array]
            The parameters of the equation and the networks, $\theta$ and $\nu$ respectively.
        gridify : bool
            Whether the inputs should be transformed into a grid
            (`jinns.utils.get_grid`) before calling `f`. This is useful for
            SPINN.

        Returns
        -------
        Float[Array, "eq_dim facet"]
            The residual, *i.e.* the differential operator
            $\mathcal{B}_\theta[u_\nu](x)$ evaluated at each point `x`.

        Raises
        ------
        NotImplementedError
            This is an abstract method to be implemented.
        """
        raise NotImplementedError
