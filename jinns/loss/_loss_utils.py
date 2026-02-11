"""
Interface for diverse loss functions to factorize code
"""

from __future__ import (
    annotations,
)  # https://docs.python.org/3/library/typing.html#constant

from typing import TYPE_CHECKING, Callable
from types import EllipsisType
import jax
import jax.numpy as jnp
from jax import vmap
from jaxtyping import Float, Array

from jinns.utils._utils import _subtract_with_check, get_grid
from jinns.data._utils import make_cartesian_product
from jinns.parameters._params import _get_vmap_in_axes_params
from jinns.nn._pinn import PINN
from jinns.nn._spinn import SPINN
from jinns.nn._hyperpinn import HyperPINN
from jinns.data._Batchs import PDEStatioBatch, PDENonStatioBatch
from jinns.parameters._params import Params

if TYPE_CHECKING:
    from jinns.loss._BoundaryConditionAbstract import BoundaryConditionAbstract
    from jinns.utils._types import (
        BoundaryEquationUOnFacet,
        BoundaryEquationFOnFacet,
        BoundaryEquationU,
        BoundaryEquationF,
    )
    from jinns.nn._abstract_pinn import AbstractPINN


def dynamic_loss_apply(
    dyn_loss: Callable,
    u: AbstractPINN,
    batch: (
        Float[Array, " batch_size 1"]
        | Float[Array, " batch_size dim"]
        | Float[Array, " batch_size 1+dim"]
    ),
    params: Params[Array],
    # vmap_axes: tuple[int, Params[int | None] | None],
    u_type: PINN | HyperPINN | None = None,
    *,
    no_reduction: bool = False,
) -> Float[Array, " "] | Float[Array, " n_samples eq_dim"]:
    """
    Sometimes when u is a lambda function a or dict we do not have access to
    its type here, hence the last argument
    """
    if u_type == PINN or u_type == HyperPINN or isinstance(u, (PINN, HyperPINN)):
        # v_dyn_loss = vmap(
        #    lambda batch, params: dyn_loss(
        #        batch,
        #        u,
        #        params,  # we must place the params at the end
        #    ),
        #    vmap_axes,
        #    0,
        # )
        residuals = dyn_loss(batch, u, params)
    elif u_type == SPINN or isinstance(u, SPINN):
        residuals = dyn_loss(batch, u, params)
    else:
        raise ValueError(f"Bad type for u. Got {type(u)}, expected PINN or SPINN")
    return residuals


# if no_reduction:
#        return residuals
#    mse_dyn_loss = jnp.mean(jnp.sum(residuals**2, axis=-1))
#
#    return mse_dyn_loss


def normalization_loss_apply(
    u: AbstractPINN,
    batches: (
        tuple[Float[Array, " nb_norm_samples dim"]]
        | tuple[
            Float[Array, " nb_norm_time_slices 1"], Float[Array, " nb_norm_samples dim"]
        ]
    ),
    params: Params[Array],
    vmap_axes_params: tuple[Params[int | None] | None],
    norm_weights: Float[Array, " nb_norm_samples"],
) -> Float[Array, " "]:
    """
    Note the squeezing on each result. We expect unidimensional *PINN since
    they represent probability distributions
    """
    if isinstance(u, (PINN, HyperPINN)):
        if len(batches) == 1:
            v_u = vmap(
                lambda *b: u(*b)[u.slice_solution],
                (0,) + vmap_axes_params,
                0,
            )
            res = v_u(*batches, params)
            assert res.shape[-1] == 1, "norm loss expects unidimensional *PINN"
            # Monte-Carlo integration using importance sampling
            mse_norm_loss = jnp.abs(jnp.mean(res.squeeze() * norm_weights) - 1) ** 2
        else:
            # NOTE this cartesian product is costly
            batch_cart_prod = make_cartesian_product(
                batches[0],
                batches[1],
            ).reshape(batches[0].shape[0], batches[1].shape[0], -1)
            v_u = vmap(
                vmap(
                    lambda t_x, params_: u(t_x, params_),
                    in_axes=(0,) + vmap_axes_params,
                ),
                in_axes=(0,) + vmap_axes_params,
            )
            res = v_u(batch_cart_prod, params)
            assert res.shape[-1] == 1, "norm loss expects unidimensional *PINN"
            # For all times t, we perform an integration. Then we average the
            # losses over times.
            mse_norm_loss = jnp.mean(
                jnp.abs(jnp.mean(res.squeeze() * norm_weights, axis=-1) - 1) ** 2
            )
    elif isinstance(u, SPINN):
        if len(batches) == 1:
            res = u(*batches, params)
            assert res.shape[-1] == 1, "norm loss expects unidimensional *SPINN"
            mse_norm_loss = (
                jnp.abs(
                    jnp.mean(
                        res.squeeze(),
                    )
                    * norm_weights
                    - 1
                )
                ** 2
            )
        else:
            assert batches[1].shape[0] % batches[0].shape[0] == 0
            rep_t = batches[1].shape[0] // batches[0].shape[0]
            res = u(
                jnp.concatenate(
                    [jnp.repeat(batches[0], rep_t, axis=0), batches[1]], axis=-1
                ),
                params,
            )
            assert res.shape[-1] == 1, "norm loss expects unidimensional *SPINN"
            # the outer mean() below is for the times stamps
            mse_norm_loss = jnp.mean(
                jnp.abs(
                    jnp.mean(
                        res.squeeze(),
                        axis=list(d + 1 for d in range(res.ndim - 2)),
                    )
                    * norm_weights
                    - 1
                )
                ** 2
            )
    else:
        raise ValueError(f"Bad type for u. Got {type(u)}, expected PINN or SPINN")

    return mse_norm_loss


def boundary_condition_apply(
    boundary_condition: BoundaryConditionAbstract,
    u: AbstractPINN,
    batch: PDEStatioBatch | PDENonStatioBatch,
    params: Params[Array],
    *,
    no_reduction: bool = False,
) -> Float[Array, " "] | tuple[Float[Array, " n_samples eq_dim"], ...]:
    assert batch.border_batch is not None
    vmap_in_axes = (0,) + _get_vmap_in_axes_params(batch.param_batch_dict, params)

    if isinstance(u, PINN):
        # Note that facets are on the last axis as specified by
        # `BoundaryCondition` function type hints
        v_boundary_condition = vmap(
            lambda inputs, params: boundary_condition.evaluate(inputs, u, params),
            vmap_in_axes,
            0,
        )
        residual = v_boundary_condition(
            batch.border_batch,
            params,
        )
    elif isinstance(u, SPINN):
        residual = boundary_condition.evaluate(batch.border_batch, u, params)
    else:
        raise ValueError(f"Bad type for u. Got {type(u)}, expected PINN or SPINN")
    if no_reduction:
        return residual
    # next square the differences and reduce over the dimensions of the
    # residuals (sum) and reduce over the samples (mean)
    # we get a tree with a mse for each facet
    mse_by_facet = jax.tree.map(lambda r: jnp.mean(jnp.sum(r**2, axis=-1)), residual)
    # next compute the final whole mse by reducing the pytree over the facets
    return jax.tree.reduce(jnp.add, mse_by_facet, jnp.array(0.0))


def equation_on_all_facets_equal(
    equation: BoundaryEquationUOnFacet | BoundaryEquationFOnFacet,
) -> BoundaryEquationU | BoundaryEquationF:
    """
    Decorator to be used around `BoundaryCondition.equation_u` or
    `BoundaryCondition.equation_f` if all the facets should be treated
    identically.
    This means that from `equation_u` or `equation_f` defined on a single facet
    we automatically vectorize the computations on all the facets which extends
    the initial function to work with a trailing `n_facet` dimension for their
    `inputs` and return arguments. See type hinting for another look on what's
    happening.

    The wrapper vectorizes the computations over the facet axis
    with a jax.tree.map which is almost always the best solution.
    The user can draw inspiration from this code for
    more specific situations.
    """

    def wrapper(*args, **kwargs):
        """
        We handle kwargs for `gridify` e.g.
        """
        equation_by_facet = jax.tree.map(
            lambda facet: equation(
                args[0],
                facet.squeeze(),  # note the squeeze to make the trailing axis
                # disappear because the wrapper function does not handle with it
                *args[2:],
                **kwargs,
            ),
            jnp.split(args[1], args[1].shape[-1], axis=-1),  # create a list
            # of array for each facet to vmap over
        )

        return tuple(equation_by_facet)

    return wrapper


def observations_loss_apply(
    u: AbstractPINN,
    batch: Float[Array, " obs_batch_size input_dim"],
    params: Params[Array],
    # vmap_axes: tuple[int, Params[int | None] | None],
    observed_values: Float[Array, " obs_batch_size observation_dim"],
    obs_slice: EllipsisType | slice | None,
) -> Float[Array, " "]:
    if isinstance(u, (PINN, HyperPINN)):
        u_ = lambda *args: u(*args)[u.slice_solution]
        # v_u = vmap(
        #    lambda *args: u(*args)[u.slice_solution],
        #    vmap_axes,
        #    0,
        # )
        val = u_(batch, params)[:, obs_slice]
        residuals = _subtract_with_check(
            observed_values, val, cause="user defined observed_values"
        )
        # mse_observation_loss = jnp.mean(
        #    jnp.sum(
        #        _subtract_with_check(
        #            observed_values, val, cause="user defined observed_values"
        #        )
        #        ** 2,
        #        axis=-1,
        #    )
        # )
    elif isinstance(u, SPINN):
        raise RuntimeError("observation loss term not yet implemented for SPINNs")
    else:
        raise ValueError(f"Bad type for u. Got {type(u)}, expected PINN or SPINN")
    return residuals


def initial_condition_apply(
    u: AbstractPINN,
    omega_batch: Float[Array, " dimension"],
    params: Params[Array],
    vmap_axes: tuple[int, Params[int | None] | None],
    initial_condition_fun: Callable,
    t0: Float[Array, " 1"],
) -> Float[Array, " "]:
    n = omega_batch.shape[0]
    t0_omega_batch = jnp.concatenate([t0 * jnp.ones((n, 1)), omega_batch], axis=1)
    if isinstance(u, (PINN, HyperPINN)):
        v_u_t0 = vmap(
            lambda t0_x, params: _subtract_with_check(
                initial_condition_fun(t0_x[1:]),
                u(t0_x, params),
                cause="Output of initial_condition_fun",
            ),
            vmap_axes,
            0,
        )
        res = v_u_t0(t0_omega_batch, params)  # NOTE take the tiled
        # omega_batch (ie omega_batch_) to have the same batch
        # dimension as params to be able to vmap.
        # Recall that by convention:
        # param_batch_dict = times_batch_size * omega_batch_size
        mse_initial_condition = jnp.mean(jnp.sum(res**2, axis=-1))
    elif isinstance(u, SPINN):
        values = lambda t_x: u(
            t_x,
            params,
        )[0]
        omega_batch_grid = get_grid(omega_batch)
        v_ini = values(t0_omega_batch)
        res = _subtract_with_check(
            initial_condition_fun(omega_batch_grid),
            v_ini,
            cause="Output of initial_condition_fun",
        )
        mse_initial_condition = jnp.mean(jnp.sum(res**2, axis=-1))
    else:
        raise ValueError(f"Bad type for u. Got {type(u)}, expected PINN or SPINN")
    return mse_initial_condition


def initial_condition_check(x, dim_size=None):
    """
    Make a (dim_size,) jnp array from an int, a float or a 0D jnp array

    """
    if isinstance(x, Array):
        if not x.shape:  # e.g. user input: jnp.array(0.)
            x = jnp.array([x])
        if dim_size is not None:  # we check for the required dims_ize
            if x.shape != (dim_size,):
                raise ValueError(
                    f"Wrong dim_size. It should be({dim_size},). Got shape: {x.shape}"
                )

    elif isinstance(x, float):  # e.g. user input: 0.
        x = jnp.array([x])
    elif isinstance(x, int):  # e.g. user input: 0
        x = jnp.array([float(x)])
    else:
        raise ValueError(f"Wrong value, expected Array, float or int, got {type(x)}")
    return x
