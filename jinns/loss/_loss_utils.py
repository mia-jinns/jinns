"""
Interface for diverse loss functions to factorize code
"""

from __future__ import (
    annotations,
)  # https://docs.python.org/3/library/typing.html#constant

from typing import TYPE_CHECKING, Callable, Concatenate, ParamSpec, TypeVar, cast
from types import EllipsisType
import jax
import jax.numpy as jnp
from jaxtyping import Float, Array, PyTree
import equinox as eqx

from jinns.utils._utils import _subtract_with_check, get_grid
from jinns.nn._pinn import PINN
from jinns.nn._spinn import SPINN
from jinns.nn._hyperpinn import HyperPINN
from jinns.parameters._params import Params

if TYPE_CHECKING:
    from jinns.loss._DynamicLossAbstract import DynamicLoss
    from jinns.loss._BoundaryConditionAbstract import BoundaryConditionAbstract
    from jinns.nn._abstract_pinn import AbstractPINN

    P = ParamSpec("P")
    BC = TypeVar(
        "BC", bound=BoundaryConditionAbstract
    )  # https://stackoverflow.com/a/71441339
    # we want a function that takes a class argument that's an [instance of]
    # a subclass of
    # BoundaryConditionAbstract and returns an instance of the corresponding class


def vmap_loss_fun_classical(
    *,
    fun: Callable,
    batch: Array,
    params: Params[Array],
    vmap_in_axes_params: tuple[Params[int | None] | None],
    in_axes: tuple[int, ...] = (0,),
    jacrev: bool = False,
):
    """
    Typically used for vmapping dynamic loss functions of type:
    `Callable[[Array, Params[Array]], tuple[Array, ...]] | None`. But also
    normalization loss (with in_axes=(0, 0) !), initial condition loss (PDE)
    and boundary condition loss
    """
    if fun is None:
        return None
    if jacrev:
        fun = jax.jacrev(fun, argnums=1)
    return jax.vmap(fun, in_axes + vmap_in_axes_params)(batch, params)


def vmap_vmap_fun_normalization(
    *,
    fun: Callable,
    batch: Array,
    params: Params[Array],
    vmap_in_axes_params: tuple[Params[int | None] | None],
    in_axes: tuple[tuple[int, int | None], ...] = ((0, 0),),
    jacrev: bool = False,
    **_,
):
    """
    Specific to t+x (spatio temporal cases)!
    This function is the recipe to vmap normalization_loss_apply
    It goes with the reduction function specific to norm_loss in t+x

    We have a cartesian product of t and samples points (`b[0]`, with 3
    dimensions) in order to
    perform a numerical integration for each t. We have a scalar or array of
    integration weights (`b[1]` broadcastable to the second dimension of
    `b[0]`)

    The outer vmap means we vmap for each time step the
    normalization_loss_apply that is itself vmapped (inner vmap)over the x
    samples (or their weights if weights are arrays, hence the two possible
    values for in_axes_norm_weights).
    """
    if fun is None:
        return None
    if jacrev:
        fun = jax.jacrev(fun, argnums=1)

    cart_prod_t_x = batch[0]
    norm_weights_for_x = batch[1]
    v_u = jax.vmap(  # outer vmap over t
        jax.vmap(  # inner vmap over x and norm_weights if it is an array
            fun,
            in_axes=in_axes + vmap_in_axes_params,
        ),
        in_axes=(((0,) + (None,)),) + vmap_in_axes_params,
    )
    return v_u((cart_prod_t_x, norm_weights_for_x), params)


def vmap_loss_fun_observations(
    *,
    fun: Callable,
    batch: Array,
    params: Params[Array],
    vmap_in_axes_params: tuple[Params[int | None] | None],
    in_axes: tuple[tuple[int, ...]] = (
        (
            0,
            0,
        ),
    ),
    jacrev: bool = False,
):
    """
    Typically made for vmapping observation loss function of type:
    `Callable[[tuple[Array, Array], Params[Array], Array, EllipsisType],
    Array]`

    NOTE as opposed to get_dyn_loss_fun, it is not possible to vmap over the
    tree map because each obs_batch might have different lengths. Of course
    this function _get_obs_loss_fun is more complex than
    _get_dyn_loss_fun because of the next paragraph but what really forbids the
    vmap over tree map (and a similar signature between _get_dyn_loss and
    _get_obs_loss_fun) really is the batch sizes.  Specific for observations
    where we have a tuple of tuples b=(obs_batch_dict["pinn_in"],
    obs_batch_dict["val"]) and a tuple of obs_batch_dict["eq_params"] which
    need to be tree mapped together but the vmap is only on each element of b
    because the vmap on each element of obs_batch_dict["eq_params"] is done
    through p
    """
    if fun is None:
        return None
    if jacrev:
        fun = jax.jacrev(fun, argnums=1)
    return jax.tree.map(
        lambda _b: jax.vmap(
            lambda __b, __p: fun(__b, __p, _b[2], _b[3]),
            in_axes + vmap_in_axes_params,
        )(_b[:2], params),
        batch,
        is_leaf=lambda x: (
            isinstance(x, tuple) and eqx.is_inexact_array(x[0])
        ),  # stop at lowest level of
        # tuples as leaves
    )


def vmap_loss_fun_only_params(
    fun: Callable,
    params: Params[Array],
    vmap_in_axes_params: tuple[Params[int | None] | None],
    jacrev=False,
    **_,
):
    """
    Typically for initial_condition of LossODE of type:
    `Callable[[Params[Array]], Array]`

    NOTE we simulate a vmap axis
    for the reduction to be always correct with the outer
    jnp.mean (here _b is None). Hence the [None] via a PyTree in order to work
    in standard gradient (only an Array) and in NGD (a whole Params[Array])
    """
    if fun is None:
        return None
    if jacrev:
        fun = jax.jacrev(fun, argnums=0)
    if vmap_in_axes_params != (None,):
        # Note that here we use the reduction as defined in
        # self._reduction_functions
        fun = jax.vmap(fun, vmap_in_axes_params)
    return jax.tree.map(
        lambda array: array[None], fun(params), is_leaf=eqx.is_inexact_array
    )


def mean_sum_reduction_pytree(residuals: PyTree[Array | None]) -> PyTree[Array | None]:
    """
    Sum over the solution dimensions then average over the samples for each
    leaf of a pytree
    """
    return jax.tree.map(mean_sum_reduction, residuals)


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
    point: (Float[Array, " dim"] | Float[Array, " batch_size dim"]),
    params: Params[Array],
) -> Float[Array, " "] | Float[Array, " n_samples eq_dim"]:
    """
    `point` is not batched in the case of PINN or HyperPINN
    `point` is batched in the case of SPINN
    """
    return dyn_loss.evaluate(point, u, params)


def normalization_loss_apply(
    u: AbstractPINN,
    x: (Float[Array, " batch_size dim"] | Float[Array, " dim"]),
    norm_weight: Array,
    params: Params[Array],
) -> Float[Array, " "]:
    """
    Note the squeezing on each result. We expect unidimensional *PINN since
    they represent probability distributions.

    `x` is not batched in the case of PINN or HyperPINN
    `x` is batched in the case of SPINN
    """
    if isinstance(u, (PINN, HyperPINN)):
        res = u(x, params)
        assert res.shape[-1] == 1, "norm loss expects unidimensional *PINN"
        assert norm_weight.squeeze().ndim == 0, "normalization weight must be a scalar"
        # Monte-Carlo integration using importance sampling
        res = res.squeeze() * norm_weight.squeeze()
    elif isinstance(u, SPINN):
        # NOTE norm_weight must be scalar here
        res = u(x, params)
        assert res.shape[-1] == 1, "norm loss expects unidimensional *SPINN"
        res = res.squeeze() * norm_weight
    else:
        raise ValueError(f"Bad type for u. Got {type(u)}, expected PINN or SPINN")

    return res


def boundary_condition_apply(
    boundary_condition: BoundaryConditionAbstract,
    u: AbstractPINN,
    border_point: Array,
    params: Params[Array],
) -> Float[Array, " "] | tuple[Float[Array, " n_samples eq_dim"], ...]:
    """
    `border_point` is not batched in the case of PINN or HyperPINN
    `border_point` is batched in the case of SPINN
    """
    residuals = boundary_condition.evaluate(border_point, u, params)
    return residuals


def equation_on_all_facets_equal(
    equation: Callable[
        Concatenate[BC, Float[Array, " InputDim"], P],
        Float[Array, " InputDim"],
    ],
) -> Callable[
    Concatenate[BC, Float[Array, " InputDim n_facet"], P],
    tuple[Float[Array, " InputDim"], ...],
]:
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
    obs_point: tuple[
        Float[Array, " input_dim"],
        Float[Array, " observation_dim"],
    ],
    params: Params[Array],
    # vmap_axes: tuple[int, Params[int | None] | None],
    obs_slice: EllipsisType | slice | None,
) -> Float[Array, " "]:
    if isinstance(u, (PINN, HyperPINN)):
        u_ = lambda *args: u(*args)[u.slice_solution]
        val = u_(obs_point[0], params)[obs_slice]
        residuals = cast(
            Array,
            _subtract_with_check(
                obs_point[1], val, cause="user defined observed_values"
            ),
        )
    elif isinstance(u, SPINN):
        raise RuntimeError("observation loss term not yet implemented for SPINNs")
    else:
        raise ValueError(f"Bad type for u. Got {type(u)}, expected PINN or SPINN")
    return residuals


def initial_condition_apply(
    u: AbstractPINN,
    omega_initial_point: (Float[Array, " dim"] | Float[Array, " batch_size dim"]),
    params: Params[Array],
    initial_condition_fun: Callable,
    t0: Float[Array, " 1"],
) -> Float[Array, " "]:
    """
    `omega_initial_point` is not batched in the case of PINN or HyperPINN
    `omega_initial_point` is batched in the case of SPINN
    """
    if isinstance(u, (PINN, HyperPINN)):
        t0_x = jnp.concatenate([t0, omega_initial_point])  # not a batch anymoer
        residuals = cast(
            Array,
            _subtract_with_check(
                initial_condition_fun(t0_x[1:]),
                u(t0_x, params),
                cause="Output of initial_condition_fun",
            ),
        )
    elif isinstance(u, SPINN):
        n = omega_initial_point.shape[0]
        t0_omega_initial_point = jnp.concatenate(
            [t0 * jnp.ones((n, 1)), omega_initial_point], axis=1
        )
        omega_initial_point_grid = get_grid(omega_initial_point)
        residuals = cast(
            Array,
            _subtract_with_check(
                initial_condition_fun(omega_initial_point_grid),
                u(t0_omega_initial_point, params)[0],
                cause="Output of initial_condition_fun",
            ),
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


# def _rar_step_true_norm_samples(operands: RAROperands) -> RARReturns:
#     loss, params, data, param_data, batch, key, it = operands
#     assert data.rar_parameters is not None

#     if isinstance(loss.u, HyperPINN) or isinstance(loss.u, SPINN):
#         raise NotImplementedError("RAR not implemented for hyperPINN and SPINN")

#     # the signature we get from tree.reduce is Array | int
#     # we are sure this is Array so we use the cast to get rid of int
#     res = cast(
#         Array,
#         jax.tree.reduce(
#             jnp.add,
#             loss.values_and_grad_per_sample(params, loss.norm_samples)[0].dyn_loss,
#             0,
#         ),
#     )
#     res_abs = jnp.abs(res.squeeze())
#     norm_samples_size = res_abs.shape[
#         0
#     ]  # get the batch_size this way so that we are indepedent
#     # of which DG subclass we work with

#     # Here we create the novelty that be incorporated to the norm samples
#     novelty_sample_size = round(norm_samples_size * data.rar_parameters.novelty_proportion)
#     if isinstance(data, CubicMeshPDEStatio) and not isinstance(
#         data, CubicMeshPDENonStatio
#     ):
#         key, *subkeys = jax.random.split(key, data.dim + 1)
#         new_samples = data.sample_in_omega_domain(subkeys, novelty_sample_size)
#     elif isinstance(data, CubicMeshPDENonStatio):
#         key, subkey = jax.random.split(key)
#         new_samples_times = data.sample_in_time_domain(subkey, novelty_sample_size)
#         if data.dim == 1:
#             key, subkeys = jax.random.split(key, 2)
#         else:
#             key, *subkeys = jax.random.split(key, data.dim + 1)
#         new_samples_omega = data.sample_in_omega_domain(subkeys, novelty_sample_size)
#         new_samples = jnp.concatenate([new_samples_times, new_samples_omega], axis=1)
#     else:
#         raise ValueError("Wrong DataGenerator type")
#     if param_data is not None:
#         key, subkey = jax.random.split(key)
#         _, _param_n_samples = param_data.generate_data(subkey, batch_size)
#         new_param_samples = DGParams(_param_n_samples, "DGParams")

#     # RAR-G
#     ## Select the m points with higher dynamic loss, they will be conserved
#     if data.rar_parameters.method == "G":
#         keep_idx = jnp.argsort(res_abs, descending=True)[
#             : round(batch_size * (1 - data.rar_parameters.novelty_proportion))
#         ]
#     # RAR-D
#     elif data.rar_parameters.method == "D":
#         assert data.rar_parameters.k is not None
#         assert data.rar_parameters.c is not None

#         res_normalized = res_abs / (jnp.max(res_abs) + 1e-6)
#         prop_weights = res_normalized**data.rar_parameters.k + data.rar_parameters.c
#         weights = prop_weights / jnp.sum(prop_weights)
#         keep_idx = jax.random.choice(
#             key,
#             a=jnp.arange(res_abs.shape[0]),
#             shape=(round(batch_size * (1 - data.rar_parameters.novelty_proportion)),),
#             replace=False,
#             p=weights.flatten(),
#         )
#     else:
#         raise ValueError("Unknown RAR method")

#     ## Introduce novelty samples with the novelty_sample_size samples that have been sampled
#     ## Begin (Update the batch with the novelty)
#     ### for each param with a jax.tree.map
#     if param_data is not None:
#         param_batch = jax.tree.map(
#             lambda b, new_b: jnp.concatenate([b[keep_idx], new_b], axis=0),
#             batch.param_batch_dict,
#             new_param_samples,
#         )
#     else:
#         param_batch = None
#     if isinstance(batch, ODEBatch) and isinstance(data, DataGeneratorODE):
#         arr = jnp.concatenate([batch.temporal_batch[keep_idx], new_samples], axis=0)
#         batch = eqx.tree_at(lambda pt: pt.temporal_batch, batch, arr)

#         # Also add the new points ie update the fixed datasets of the DGs
#         if data.temporal_batch_size is not None:
#             new_times = data.times.at[
#                 data.curr_time_idx : data.curr_time_idx * data.temporal_batch_size
#             ].set(  # type: ignore
#                 batch.temporal_batch
#             )
#         else:
#             new_times = batch.temporal_batch
#         data = eqx.tree_at(lambda m: m.times, data, new_times)
#     elif isinstance(batch, PDEStatioBatch) or isinstance(batch, PDENonStatioBatch):
#         arr = jnp.concatenate([batch.domain_batch[keep_idx], new_samples], axis=0)
#         batch = eqx.tree_at(lambda pt: pt.domain_batch, batch, arr)
#         # Also add the new points ie update the fixed datasets of the DGs
#         if isinstance(data, CubicMeshPDEStatio) and not isinstance(
#             data, CubicMeshPDENonStatio
#         ):
#             if data.omega_batch_size is not None:
#                 new_omega = data.omega.at[
#                     data.curr_omega_idx : data.curr_omega_idx * data.omega_batch_size
#                 ].set(  # type: ignore
#                     batch.domain_batch
#                 )
#             else:
#                 new_omega = batch.domain_batch
#             data = eqx.tree_at(lambda m: m.omega, data, new_omega)
#         elif isinstance(data, CubicMeshPDENonStatio):
#             if data.domain_batch_size is not None:
#                 new_domain = data.domain.at[
#                     data.curr_domain_idx : data.curr_domain_idx * data.domain_batch_size
#                 ].set(  # type: ignore
#                     batch.domain_batch
#                 )
#             else:
#                 new_domain = batch.domain_batch
#             data = eqx.tree_at(lambda m: m.domain, data, new_domain)
#     else:
#         raise ValueError
#     ## Here is the batch we will return
#     batch = append_param_batch(batch, param_batch)
#     ## End (Update the batch with the novelty)

#     # update RAR parameters for all cases
#     data = eqx.tree_at(lambda m: m.rar_parameters._rar_iter_from_last_sampling, data, 0)

#     # NOTE must return data to be correctly updated because we cannot
#     # have side effects in this function that will be jitted
#     return data, param_data, batch


# def _rar_step_false(operands: RAROperands) -> RARReturns:
#     _, _, data, param_data, batch, _, i = operands

#     assert data.rar_parameters is not None  # for type checker

#     # Add 1 only if we are after the burn in period
#     increment = jax.lax.cond(
#         i <= data.rar_parameters.start_iter,
#         lambda: 0,
#         lambda: 1,
#     )

#     new__rar_iter_from_last_sampling = (
#         data.rar_parameters._rar_iter_from_last_sampling + increment
#     )
#     if isinstance(data, eqx.Module):
#         data = eqx.tree_at(
#             lambda m: m.rar_parameters._rar_iter_from_last_sampling,
#             data,
#             new__rar_iter_from_last_sampling,
#         )
#     else:
#         data._rar_iter_from_last_sampling = new__rar_iter_from_last_sampling
#     return data, param_data, batch
