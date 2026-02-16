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
from jaxtyping import Float, Array

from jinns.utils._utils import _subtract_with_check, get_grid
from jinns.nn._pinn import PINN
from jinns.nn._spinn import SPINN
from jinns.nn._hyperpinn import HyperPINN
from jinns.parameters._params import Params

if TYPE_CHECKING:
    from jinns.loss._DynamicLossAbstract import DynamicLoss
    from jinns.loss._BoundaryConditionAbstract import BoundaryConditionAbstract
    from jinns.utils._types import (
        BoundaryEquationUOnFacet,
        BoundaryEquationFOnFacet,
        BoundaryEquationU,
        BoundaryEquationF,
    )
    from jinns.nn._abstract_pinn import AbstractPINN


def mean_sum_reduction(residuals: Array | None) -> Array | None:
    """
    Sum over the solution dimensions then average over the samples
    """
    if residuals is not None:
        residuals = jnp.atleast_2d(residuals)
        res = jnp.mean(jnp.sum(residuals**2, axis=-1))
    else:
        res = None
    return res


def dynamic_loss_apply(
    dyn_loss: DynamicLoss,
    u: AbstractPINN,
    batch: (
        Float[Array, " batch_size 1"]
        | Float[Array, " batch_size dim"]
        | Float[Array, " batch_size 1+dim"]
    ),
    params: Params[Array],
) -> Float[Array, " "] | Float[Array, " n_samples eq_dim"]:
    """
    Sometimes when u is a lambda function a or dict we do not have access to
    its type here, hence the last argument
    """
    return dyn_loss.evaluate(batch, u, params)


def normalization_loss_apply(
    u: AbstractPINN,
    x_and_norm_weight: tuple[Array, Array],
    params: Params[Array],
    norm_weights: Float[Array, " nb_norm_samples"],
) -> Float[Array, " "]:
    """
    Note the squeezing on each result. We expect unidimensional *PINN since
    they represent probability distributions
    """
    if isinstance(u, (PINN, HyperPINN)):
        res = u(x_and_norm_weight[0], params)
        assert res.shape[-1] == 1, "norm loss expects unidimensional *PINN"
        # Monte-Carlo integration using importance sampling
        res = res.squeeze() * x_and_norm_weight[1]
        # else:
        #    # NOTE this cartesian product is costly
        #    batch_cart_prod = make_cartesian_product(
        #        batches[0],
        #        batches[1],
        #    ).reshape(batches[0].shape[0], batches[1].shape[0], -1)
        #    v_u = vmap(
        #        vmap(
        #            lambda t_x, params_: u(t_x, params_),
        #            in_axes=(0,) + vmap_axes_params,
        #        ),
        #        in_axes=(0,) + vmap_axes_params,
        #    )
        #    res = v_u(batch_cart_prod, params)
        #    assert res.shape[-1] == 1, "norm loss expects unidimensional *PINN"
        #    # For all times t, we perform an integration. Then we average the
        #    # losses over times.
        #    mse_norm_loss = jnp.mean(
        #        jnp.abs(jnp.mean(res.squeeze() * norm_weights, axis=-1) - 1) ** 2
        #    )
    elif isinstance(u, SPINN):
        # NOTE norm_weight must be scalar here
        res = u(x_and_norm_weight[0], params)
        assert res.shape[-1] == 1, "norm loss expects unidimensional *SPINN"
        res = res.squeeze() * x_and_norm_weight[1]
        #    jnp.abs(
        #        jnp.mean(
        #            res.squeeze(),
        #        )
        #        * norm_weights
        #        - 1
        #    )
        #    ** 2
        # )
        # mse_norm_loss = jnp.mean(
        #    jnp.abs(
        #        jnp.mean(
        #            res.squeeze(),
        #            axis=list(d + 1 for d in range(res.ndim - 2)),
        #        )
        #        * norm_weights
        #        - 1
        #    )
        #    ** 2
        # )
    else:
        raise ValueError(f"Bad type for u. Got {type(u)}, expected PINN or SPINN")

    return res


def boundary_condition_apply(
    boundary_condition: BoundaryConditionAbstract,
    u: AbstractPINN,
    border_batch: Array,
    params: Params[Array],
) -> Float[Array, " "] | tuple[Float[Array, " n_samples eq_dim"], ...]:
    residuals = boundary_condition.evaluate(border_batch, u, params)
    return residuals


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
    batch: tuple[
        Float[Array, " obs_batch_size input_dim"],
        Float[Array, "obs_batch_size observation_dim"],
    ],
    params: Params[Array],
    # vmap_axes: tuple[int, Params[int | None] | None],
    obs_slice: EllipsisType | slice | None,
) -> Float[Array, " "]:
    if isinstance(u, (PINN, HyperPINN)):
        u_ = lambda *args: u(*args)[u.slice_solution]
        val = u_(batch[0], params)[:, obs_slice]
        residuals = _subtract_with_check(
            batch[1], val, cause="user defined observed_values"
        )
    elif isinstance(u, SPINN):
        raise RuntimeError("observation loss term not yet implemented for SPINNs")
    else:
        raise ValueError(f"Bad type for u. Got {type(u)}, expected PINN or SPINN")
    return residuals


def initial_condition_apply(
    u: AbstractPINN,
    omega_batch: Float[Array, " dimension"],
    params: Params[Array],
    initial_condition_fun: Callable,
    t0: Float[Array, " 1"],
) -> Float[Array, " "]:
    # t0_omega_batch = jnp.concatenate([t0 * jnp.ones((n, 1)), omega_batch], axis=1)
    t0_x = jnp.concatenate([t0, omega_batch])  # not a batch anymoer
    if isinstance(u, (PINN, HyperPINN)):
        residuals = _subtract_with_check(
            initial_condition_fun(t0_x[1:]),
            u(t0_x, params),
            cause="Output of initial_condition_fun",
        )
    elif isinstance(u, SPINN):
        omega_batch_grid = get_grid(omega_batch)
        residuals = _subtract_with_check(
            initial_condition_fun(omega_batch_grid),
            u(t0_x, params)[0],
            cause="Output of initial_condition_fun",
        )
    else:
        raise ValueError(f"Bad type for u. Got {type(u)}, expected PINN or SPINN")
    return residuals


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
