"""
Implements abstract classes for boundary condition
"""

from __future__ import (
    annotations,
)  # https://docs.python.org/3/library/typing.html#constant

from functools import partial
import warnings
import abc
from typing import TYPE_CHECKING
import jax
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


class BoundaryConditionAbstract(eqx.Module):
    r""" """

    def evaluate(
        self,
        inputs: InputDim,
        u: AbstractPINN,
        params: Params[Array],
    ) -> tuple[Float[Array, " eq_dim"], ...]:
        """
        The length of the returned tuple is equal to the number of active
        boundary condition equations (often equal to the number of facets)
        """
        eval_u = self.equation_u(inputs, u, params)

        if any(tuple(map(lambda arr: arr.ndim == 0, eval_u))):
            raise ValueError(
                "The output of loss must be vectorial, i.e. of shape (d,) with"
                " d >= 1. At least one of the return value from BoundaryCondition"
                " breaks the condition."
            )
        if any(tuple(map(lambda arr: arr.ndim > 1, eval_u))) and not isinstance(
            u, SPINN
        ):
            warnings.warn(
                "At least one of the return value from BoundaryCondition' equation has more "
                "than one dimension. This is in general a mistake (probably from "
                "an unfortunate broadcast in jnp.array computations) resulting in "
                "bad reduction operations in losses."
            )

        if isinstance(u, PINN):
            eval_f = self.equation_f(inputs, params, gridify=False)
        elif isinstance(u, SPINN):
            eval_f = self.equation_f(inputs, params, gridify=True)
        else:
            raise ValueError(f"Bad type for u. Got {type(u)}, expected PINN or SPINN")

        # next compute differences between what should match
        residual = jax.tree.map(
            partial(_subtract_with_check, cause="boundary condition fun"),
            eval_f,
            eval_u,
        )
        return residual

    @abc.abstractmethod
    def equation_u(
        self,
        inputs: InputDim,
        u: AbstractPINN,
        params: Params[Array],
    ) -> tuple[Float[Array, " eq_dim"], ...]:
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
        tuple[Float[Array, "eq_dim"]]
            The residual, *i.e.* the differential operator
            $\mathcal{B}_\theta[u_\nu](x)$ evaluated at each facet (last dim of
            x)

        Raises
        ------
        NotImplementedError
            This is an abstract method to be implemented.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def equation_f(
        self, inputs: InputDim, params: Params[Array], gridify: bool = False
    ) -> tuple[Float[Array, " eq_dim"], ...]:
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
        tuple[Float[Array, "eq_dim"]]
            The residual, *i.e.* the differential operator
            $\mathcal{B}_\theta[u_\nu](x)$ evaluated at each facet (last dim of
            x)

        Raises
        ------
        NotImplementedError
            This is an abstract method to be implemented.
        """
        raise NotImplementedError
