"""
Implements some common boundary conditions
"""

from __future__ import (
    annotations,
)  # https://docs.python.org/3/library/typing.html#constant

from typing import TYPE_CHECKING
from jaxtyping import Float, Array
import jax
import jax.numpy as jnp
from jinns.loss._BoundaryConditionAbstract import BoundaryConditionAbstract
from jinns.loss._loss_utils import equation_on_all_facets_equal
from jinns.utils._utils import get_grid
from jinns.nn import SPINN, PINN

if TYPE_CHECKING:
    from jinns.parameters import Params
    from jinns.nn._abstract_pinn import AbstractPINN


class Dirichlet(BoundaryConditionAbstract):
    r"""
    Implements Dirichlet boundary condition

    $$
    u(x) = 0, \forall x\in\delta\Omega
    $$

    (also usable for $x\in\Omega\times I$ (non-stationary case))
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


class Neumann(BoundaryConditionAbstract):
    r"""
    Implements Neumann boundary condition

    $$
    \nabla u(x)\cdot n = 0, \forall x\in\delta\Omega
    $$

    where $n$ is the unitary outgoing vector normal at the border.
    (also usable for $x\in\Omega\times I$ (non-stationary case))
    """

    def equation_u(
        self, inputs: Float[Array, " dim n_facet"], u: AbstractPINN, params: Params[Array]
    ) -> tuple[Float[Array, " eq_dim"], ...]:
        """
        Note that we write the body for all facets explicitly because we need
        to handle the normal vector
        """
        # We resort to the shape of the border_batch to determine the dimension as
        # described in the border_batch function
        if jnp.squeeze(inputs).ndim == 0:  # case 1D borders (just a scalar)
            n = jnp.array([1, -1])  # the unit vectors normal to the two borders
            n_facets = 2

        else:  # case 2D borders (because 3D borders are not supported yet)
            # they are in the order: left, right, bottom, top so we give the normal
            # outgoing vectors accordingly with shape in concordance with
            # border_batch shape (batch_size, ndim, nfacets)
            n = jnp.array([[-1, 1, 0, 0], [0, 0, -1, 1]])
            n_facets = 4
        if isinstance(u, PINN):
            if u.eq_type == "PDEStatio":
                return tuple(
                    jnp.dot(
                        jax.grad(u, 0)(inputs[..., facet], params),
                        n[..., facet]
                    ) for facet in range(n_facets)
                )
            elif u.eq_type == "PDENonStatio":
                return tuple(
                    jnp.dot(
                        jax.grad(u, 0)(inputs[..., facet], params)[1:],
                        n[..., facet]
                    ) for facet in range(n_facets)
                )
            else:
                raise ValueError("Wrong u.eq_type")
        elif isinstance(u, SPINN):
            # the gradient we see in the PINN case can get gradients wrt to x
            # dimensions at once. But it would be very inefficient in SPINN because
            # of the high dim output of u. So we do 2 explicit forward AD, handling all the
            # high dim output at once
            if n_facets == 2:
                du_dx_fun = lambda facet: jax.jvp(
                    lambda inputs_: u(inputs_, params),
                    (inputs[..., facet],),
                    (jnp.ones_like(inputs[..., facet]),),
                )[1]
                if u.eq_type == "PDEStatio":
                    return tuple(
                        du_dx_fun(facet) * n[facet]
                        for facet in range(n_facets)
                    )
                if u.eq_type == "PDENonStatio":
                    return tuple(
                        du_dx_fun(facet)[..., 1] * n[facet]
                        for facet in range(n_facets)
                    )
            elif n_facets == 4:
                du_dx_fun = lambda tangent_vec, facet: jax.jvp(
                    lambda inputs_: u(inputs_, params),
                    (inputs[..., facet],),
                    (tangent_vec,),
                )[1]
                if u.eq_type == "PDEStatio":
                    return tuple(
                        du_dx_fun(
                            jnp.repeat(
                                jnp.array([1.0, 0.0])[None],
                                inputs[..., facet].shape[0],
                                axis=0
                            ),
                            facet
                        ) * n[0, facet] + # this sum is the dot product
                        du_dx_fun(
                            jnp.repeat(
                                jnp.array([0.0, 1.0])[None],
                                inputs[..., facet].shape[0],
                                axis=0
                            ),
                            facet
                        ) * n[1, facet]
                        for facet in range(n_facets)
                    )
                if u.eq_type == "PDENonStatio":
                    print("HERE")
                    return tuple(
                        du_dx_fun(
                            jnp.repeat(
                                jnp.array([0., 1.0, 0.0])[None],
                                inputs[..., facet].shape[0],
                                axis=0
                            ),
                            facet
                        ).squeeze() * n[0, facet]
                        + # this sum is the dot product
                        du_dx_fun(
                            jnp.repeat(
                                jnp.array([0., 0.0, 1.0])[None],
                                inputs[..., facet].shape[0],
                                axis=0
                            ),
                            facet
                        ).squeeze() * n[1, facet]
                        for facet in range(n_facets)
                    )
            else:
                raise ValueError("Not implemented")

    def equation_f(
        self, inputs: Float[Array, " dim n_facet"], params: Params[Array], gridify: bool = False
    ) -> tuple[Float[Array, " eq_dim"], ...]:
        """
        Note that we write the body for all facets explicitly because we need
        to handle the normal vector

        """
        # We resort to the shape of the border_batch to determine the dimension as
        # described in the border_batch function
        if jnp.squeeze(inputs).ndim == 0:  # case 1D borders (just a scalar)
            n_facets = 2

        else:  # case 2D borders (because 3D borders are not supported yet)
            n_facets = 4
        if gridify:  # to handle SPINN, ignore otherwise
            # inputs.shape[-1] indicates the number of dimensions of the pb
            # thus we get the correct grid of zeros
            return tuple(
                jnp.zeros(
                    get_grid(inputs[..., facet]).shape[: inputs.shape[-1]])[..., None]
                for facet in range(n_facets)
            )
        else:
            return tuple(jnp.zeros((1,)) for _ in range(n_facets))
