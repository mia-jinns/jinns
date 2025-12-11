"""
Main module to implement a PDE loss in jinns
"""

from __future__ import (
    annotations,
)  # https://docs.python.org/3/library/typing.html#constant

import abc
from dataclasses import InitVar
from typing import TYPE_CHECKING, Callable, cast, Any, TypeVar, Generic
from types import EllipsisType
import warnings
import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import PRNGKeyArray, Float, Array
from jinns.loss._loss_utils import (
    dynamic_loss_apply,
    boundary_condition_apply,
    normalization_loss_apply,
    observations_loss_apply,
    initial_condition_apply,
    initial_condition_check,
)
from jinns.parameters._params import (
    _get_vmap_in_axes_params,
    update_eq_params,
)
from jinns.parameters._derivative_keys import (
    _set_derivatives,
    DerivativeKeysPDEStatio,
    DerivativeKeysPDENonStatio,
)
from jinns.loss._abstract_loss import AbstractLoss
from jinns.loss._loss_components import PDEStatioComponents, PDENonStatioComponents
from jinns.loss._loss_weights import (
    LossWeightsPDEStatio,
    LossWeightsPDENonStatio,
)
from jinns.data._Batchs import PDEStatioBatch, PDENonStatioBatch
from jinns.parameters._params import Params
from jinns.loss import PDENonStatio, PDEStatio


if TYPE_CHECKING:
    # imports for type hints only
    from jinns.nn._abstract_pinn import AbstractPINN
    from jinns.utils._types import BoundaryConditionFun

_IMPLEMENTED_BOUNDARY_CONDITIONS = [
    "dirichlet",
    "von neumann",
    "vonneumann",
]


# For the same reason that we have the TypeVar in _abstract_loss.py, we have them
# here, because _LossPDEAbstract is abtract and we cannot decide for several
# types between their statio and non-statio version.
# Assigning the type where it can be decide seems a better practice than
# assigning a type at a higher level depending on a child class type. This is
# why we now assign LossWeights and DerivativeKeys in the child class where
# they really can be decided.

L = TypeVar("L", bound=LossWeightsPDEStatio | LossWeightsPDENonStatio)
B = TypeVar("B", bound=PDEStatioBatch | PDENonStatioBatch)
C = TypeVar(
    "C", bound=PDEStatioComponents[Array | None] | PDENonStatioComponents[Array | None]
)
DKPDE = TypeVar("DKPDE", bound=DerivativeKeysPDEStatio | DerivativeKeysPDENonStatio)
Y = TypeVar("Y", bound=PDEStatio | PDENonStatio | None)


class _LossPDEAbstract(
    AbstractLoss[L, B, C, DKPDE],
    Generic[L, B, C, DKPDE, Y],
):
    r"""
    Parameters
    ----------

    loss_weights : LossWeightsPDEStatio | LossWeightsPDENonStatio, default=None
        The loss weights for the differents term : dynamic loss,
        initial condition (if LossWeightsPDENonStatio), boundary conditions if
        any, normalization loss if any and observations if any.
        Can be updated according to a specific algorithm. See
        `update_weight_method`
    update_weight_method : Literal['soft_adapt', 'lr_annealing', 'ReLoBRaLo'], default=None
        Default is None meaning no update for loss weights. Otherwise a string
    omega_boundary_fun : BoundaryConditionFun | dict[str, BoundaryConditionFun], default=None
         The function to be matched in the border condition (can be None) or a
         dictionary of such functions as values and keys as described
         in `omega_boundary_condition`.
    omega_boundary_condition : str | dict[str, str | None], default=None
        Either None (no condition, by default), or a string defining
        the boundary condition (Dirichlet or Von Neumann),
        or a dictionary with such strings as values. In this case,
        the keys are the facets and must be in the following order:
        1D -> [“xmin”, “xmax”], 2D -> [“xmin”, “xmax”, “ymin”, “ymax”].
        Note that high order boundaries are currently not implemented.
        A value in the dict can be None, this means we do not enforce
        a particular boundary condition on this facet.
        The facet called “xmin”, resp. “xmax” etc., in 2D,
        refers to the set of 2D points with fixed “xmin”, resp. “xmax”, etc.
    omega_boundary_dim : int | slice | dict[str, slice], default=None
        Either None, or a slice object or a dictionary of slice objects as
        values and keys as described in `omega_boundary_condition`.
        `omega_boundary_dim` indicates which dimension(s) of the PINN
        will be forced to match the boundary condition.
        Note that it must be a slice and not an integer
        (but a preprocessing of the user provided argument takes care of it)
    norm_samples : Float[Array, " nb_norm_samples dimension"], default=None
        Monte-Carlo sample points for computing the
        normalization constant. Default is None.
    norm_weights : Float[Array, " nb_norm_samples"] | float | int, default=None
        The importance sampling weights for Monte-Carlo integration of the
        normalization constant. Must be provided if `norm_samples` is provided.
        `norm_weights` should be broadcastble to
        `norm_samples`.
        Alternatively, the user can pass a float or an integer that will be
        made broadcastable to `norm_samples`.
        These corresponds to the weights $w_k = \frac{1}{q(x_k)}$ where
        $q(\cdot)$ is the proposal p.d.f. and $x_k$ are the Monte-Carlo samples.
    obs_slice : EllipsisType | slice, default=None
        slice object specifying the begininning/ending of the PINN output
        that is observed (this is then useful for multidim PINN). Default is None.
    key : Key | None
        A JAX PRNG Key for the loss class treated as an attribute. Default is
        None. This field is provided for future developments and additional
        losses that might need some randomness. Note that special care must be
        taken when splitting the key because in-place updates are forbidden in
        eqx.Modules.
    """

    # NOTE static=True only for leaf attributes that are not valid JAX types
    # (ie. jax.Array cannot be static) and that we do not expect to change
    u: eqx.AbstractVar[AbstractPINN]
    dynamic_loss: eqx.AbstractVar[Y]
    omega_boundary_fun: (
        BoundaryConditionFun | dict[str, BoundaryConditionFun] | None
    ) = eqx.field(static=True)
    omega_boundary_condition: str | dict[str, str | None] | None = eqx.field(
        static=True
    )
    omega_boundary_dim: slice | dict[str, slice] = eqx.field(static=True)
    norm_samples: Float[Array, " nb_norm_samples dimension"] | None
    norm_weights: Float[Array, " nb_norm_samples"] | None
    obs_slice: EllipsisType | slice = eqx.field(static=True)
    key: PRNGKeyArray | None

    def __init__(
        self,
        *,
        omega_boundary_fun: BoundaryConditionFun
        | dict[str, BoundaryConditionFun]
        | None = None,
        omega_boundary_condition: str | dict[str, str | None] | None = None,
        omega_boundary_dim: int | slice | dict[str, int | slice] | None = None,
        norm_samples: Float[Array, " nb_norm_samples dimension"] | None = None,
        norm_weights: Float[Array, " nb_norm_samples"] | float | int | None = None,
        obs_slice: EllipsisType | slice | None = None,
        key: PRNGKeyArray | None = None,
        derivative_keys: DKPDE,
        **kwargs: Any,  # for arguments for super()
    ):
        super().__init__(
            loss_weights=self.loss_weights,
            derivative_keys=derivative_keys,
            **kwargs,
        )

        if self.update_weight_method is not None and jnp.any(
            jnp.array(jax.tree.leaves(self.loss_weights)) == 0
        ):
            warnings.warn(
                "self.update_weight_method is activated while some loss "
                "weights are zero. The update weight method will likely "
                "update the zero weight to some non-zero value. Check that "
                "this is the desired behaviour."
            )

        if obs_slice is None:
            self.obs_slice = jnp.s_[...]
        else:
            self.obs_slice = obs_slice

        if (
            isinstance(omega_boundary_fun, dict)
            and not isinstance(omega_boundary_condition, dict)
        ) or (
            not isinstance(omega_boundary_fun, dict)
            and isinstance(omega_boundary_condition, dict)
        ):
            raise ValueError(
                "if one of omega_boundary_fun or "
                "omega_boundary_condition is dict, the other should be too."
            )

        if omega_boundary_condition is None or omega_boundary_fun is None:
            warnings.warn(
                "Missing boundary function or no boundary condition."
                "Boundary function is thus ignored."
            )
        else:
            if isinstance(omega_boundary_condition, dict):
                for _, v in omega_boundary_condition.items():
                    if v is not None and not any(
                        v.lower() in s for s in _IMPLEMENTED_BOUNDARY_CONDITIONS
                    ):
                        raise NotImplementedError(
                            f"The boundary condition {omega_boundary_condition} is not"
                            f"implemented yet. Try one of :"
                            f"{_IMPLEMENTED_BOUNDARY_CONDITIONS}."
                        )
            else:
                if not any(
                    omega_boundary_condition.lower() in s
                    for s in _IMPLEMENTED_BOUNDARY_CONDITIONS
                ):
                    raise NotImplementedError(
                        f"The boundary condition {omega_boundary_condition} is not"
                        f"implemented yet. Try one of :"
                        f"{_IMPLEMENTED_BOUNDARY_CONDITIONS}."
                    )
            if isinstance(omega_boundary_fun, dict) and isinstance(
                omega_boundary_condition, dict
            ):
                keys_omega_boundary_fun = cast(str, omega_boundary_fun.keys())
                if (
                    not (
                        list(keys_omega_boundary_fun) == ["xmin", "xmax"]
                        and list(omega_boundary_condition.keys()) == ["xmin", "xmax"]
                    )
                ) and (
                    not (
                        list(keys_omega_boundary_fun)
                        == ["xmin", "xmax", "ymin", "ymax"]
                        and list(omega_boundary_condition.keys())
                        == ["xmin", "xmax", "ymin", "ymax"]
                    )
                ):
                    raise ValueError(
                        "The key order (facet order) in the "
                        "boundary condition dictionaries is incorrect"
                    )

        self.omega_boundary_fun = omega_boundary_fun
        self.omega_boundary_condition = omega_boundary_condition

        if isinstance(omega_boundary_fun, dict):
            keys_omega_boundary_fun: str = cast(str, omega_boundary_fun.keys())
            if omega_boundary_dim is None:
                self.omega_boundary_dim = {
                    k: jnp.s_[::] for k in keys_omega_boundary_fun
                }
            if not isinstance(omega_boundary_dim, dict):
                raise ValueError(
                    "If omega_boundary_fun is a dict then"
                    " omega_boundary_dim should also be a dict"
                )
            if list(omega_boundary_dim.keys()) != list(keys_omega_boundary_fun):
                raise ValueError(
                    "If omega_boundary_fun is a dict,"
                    " omega_boundary_dim should be a dict with the same keys"
                )
            self.omega_boundary_dim = {}
            for k, v in omega_boundary_dim.items():
                if isinstance(v, int):
                    # rewrite it as a slice to ensure that axis does not disappear when
                    # indexing
                    self.omega_boundary_dim[k] = jnp.s_[v : v + 1]
                else:
                    self.omega_boundary_dim[k] = v

        else:
            assert not isinstance(omega_boundary_dim, dict)
            if omega_boundary_dim is None:
                self.omega_boundary_dim = jnp.s_[::]
            elif isinstance(omega_boundary_dim, int):
                # rewrite it as a slice to ensure that axis does not disappear when
                # indexing
                self.omega_boundary_dim = jnp.s_[
                    omega_boundary_dim : omega_boundary_dim + 1
                ]
            else:
                assert isinstance(omega_boundary_dim, slice)
                self.omega_boundary_dim = omega_boundary_dim

        if norm_samples is not None:
            self.norm_samples = norm_samples
            if norm_weights is None:
                raise ValueError(
                    "`norm_weights` must be provided when `norm_samples` is used!"
                )
            if isinstance(norm_weights, (int, float)):
                self.norm_weights = norm_weights * jnp.ones(
                    (self.norm_samples.shape[0],)
                )
            else:
                assert isinstance(norm_weights, Array)
                if not (norm_weights.shape[0] == norm_samples.shape[0]):
                    raise ValueError(
                        "norm_weights and "
                        "norm_samples must have the same leading dimension"
                    )
                self.norm_weights = norm_weights
        else:
            self.norm_samples = norm_samples
            self.norm_weights = None

        self.key = key

    @abc.abstractmethod
    def _get_dynamic_loss_batch(self, batch: B) -> Array:
        pass

    @abc.abstractmethod
    def _get_normalization_loss_batch(self, batch: B) -> tuple[Array | None, ...]:
        pass

    def _get_dyn_loss_fun(
        self, batch: B, vmap_in_axes_params: tuple[Params[int | None] | None]
    ) -> Callable[[Params[Array]], Array] | None:
        if self.dynamic_loss is not None:
            dyn_loss_eval = self.dynamic_loss.evaluate
            dyn_loss_fun: Callable[[Params[Array]], Array] | None = (
                lambda p: dynamic_loss_apply(
                    dyn_loss_eval,
                    self.u,
                    self._get_dynamic_loss_batch(batch),
                    _set_derivatives(p, self.derivative_keys.dyn_loss),
                    self.vmap_in_axes + vmap_in_axes_params,
                )
            )
        else:
            dyn_loss_fun = None

        return dyn_loss_fun

    def _get_norm_loss_fun(
        self, batch: B, vmap_in_axes_params: tuple[Params[int | None] | None]
    ) -> Callable[[Params[Array]], Array] | None:
        if self.norm_samples is not None:
            norm_loss_fun: Callable[[Params[Array]], Array] | None = (
                lambda p: normalization_loss_apply(
                    self.u,
                    cast(
                        tuple[Array, Array], self._get_normalization_loss_batch(batch)
                    ),
                    _set_derivatives(p, self.derivative_keys.norm_loss),
                    vmap_in_axes_params,
                    self.norm_weights,  # type: ignore -> can't get the __post_init__ narrowing here
                )
            )
        else:
            norm_loss_fun = None
        return norm_loss_fun

    def _get_boundary_loss_fun(
        self, batch: B
    ) -> Callable[[Params[Array]], Array] | None:
        if (
            self.omega_boundary_condition is not None
            and self.omega_boundary_fun is not None
        ):
            boundary_loss_fun: Callable[[Params[Array]], Array] | None = (
                lambda p: boundary_condition_apply(
                    self.u,
                    batch,
                    _set_derivatives(p, self.derivative_keys.boundary_loss),
                    self.omega_boundary_fun,  # type: ignore (we are in lambda)
                    self.omega_boundary_condition,  # type: ignore
                    self.omega_boundary_dim,  # type: ignore
                )
            )
        else:
            boundary_loss_fun = None

        return boundary_loss_fun

    def _get_obs_params_and_obs_loss_fun(
        self,
        batch: B,
        vmap_in_axes_params: tuple[Params[int | None] | None],
        params: Params[Array],
    ) -> tuple[Params[Array] | None, Callable[[Params[Array]], Array] | None]:
        if batch.obs_batch_dict is not None:
            # update params with the batches of observed params
            params_obs = update_eq_params(params, batch.obs_batch_dict["eq_params"])

            pinn_in, val = (
                batch.obs_batch_dict["pinn_in"],
                batch.obs_batch_dict["val"],
            )

            obs_loss_fun: Callable[[Params[Array]], Array] | None = (
                lambda po: observations_loss_apply(
                    self.u,
                    pinn_in,
                    _set_derivatives(po, self.derivative_keys.observations),
                    self.vmap_in_axes + vmap_in_axes_params,
                    val,
                    self.obs_slice,
                )
            )
        else:
            params_obs = None
            obs_loss_fun = None

        return params_obs, obs_loss_fun


class LossPDEStatio(
    _LossPDEAbstract[
        LossWeightsPDEStatio,
        PDEStatioBatch,
        PDEStatioComponents[Array | None],
        DerivativeKeysPDEStatio,
        PDEStatio | None,
    ]
):
    r"""Loss object for a stationary partial differential equation

    $$
        \mathcal{N}[u](x) = 0, \forall x  \in \Omega
    $$

    where $\mathcal{N}[\cdot]$ is a differential operator and the
    boundary condition is $u(x)=u_b(x)$ The additional condition of
    integrating to 1 can be included, i.e. $\int u(x)\mathrm{d}x=1$.

    Parameters
    ----------
    u : AbstractPINN
        the PINN
    dynamic_loss : PDEStatio | None
        the stationary PDE dynamic part of the loss, basically the differential
        operator $\mathcal{N}[u](x)$. Should implement a method
        `dynamic_loss.evaluate(x, u, params)`.
        Can be None in order to access only some part of the evaluate call
        results.
    key : PRNGKeyArray
        A JAX PRNG Key for the loss class treated as an attribute. Default is
        None. This field is provided for future developments and additional
        losses that might need some randomness. Note that special care must be
        taken when splitting the key because in-place updates are forbidden in
        eqx.Modules.
    loss_weights : LossWeightsPDEStatio, default=None
        The loss weights for the differents term : dynamic loss,
        boundary conditions if any, normalization loss if any and
        observations if any.
        Can be updated according to a specific algorithm. See
        `update_weight_method`
    derivative_keys : DerivativeKeysPDEStatio, default=None
        Specify which field of `params` should be differentiated for each
        composant of the total loss. Particularily useful for inverse problems.
        Fields can be "nn_params", "eq_params" or "both". Those that should not
        be updated will have a `jax.lax.stop_gradient` called on them. Default
        is `"nn_params"` for each composant of the loss.
    params : InitVar[Params[Array]], default=None
        The main Params object of the problem needed to instanciate the
        DerivativeKeysODE if the latter is not specified.

    update_weight_method : Literal['soft_adapt', 'lr_annealing', 'ReLoBRaLo'], default=None
        Default is None meaning no update for loss weights. Otherwise a string
    omega_boundary_fun : BoundaryConditionFun | dict[str, BoundaryConditionFun], default=None
         The function to be matched in the border condition (can be None) or a
         dictionary of such functions as values and keys as described
         in `omega_boundary_condition`.
    omega_boundary_condition : str | dict[str, str], default=None
        Either None (no condition, by default), or a string defining
        the boundary condition (Dirichlet or Von Neumann),
        or a dictionary with such strings as values. In this case,
        the keys are the facets and must be in the following order:
        1D -> [“xmin”, “xmax”], 2D -> [“xmin”, “xmax”, “ymin”, “ymax”].
        Note that high order boundaries are currently not implemented.
        A value in the dict can be None, this means we do not enforce
        a particular boundary condition on this facet.
        The facet called “xmin”, resp. “xmax” etc., in 2D,
        refers to the set of 2D points with fixed “xmin”, resp. “xmax”, etc.
    omega_boundary_dim : int | slice | dict[str, slice], default=None
        Either None, or a slice object or a dictionary of slice objects as
        values and keys as described in `omega_boundary_condition`.
        `omega_boundary_dim` indicates which dimension(s) of the PINN
        will be forced to match the boundary condition.
        Note that it must be a slice and not an integer
        (but a preprocessing of the user provided argument takes care of it)
    norm_samples : Float[Array, " nb_norm_samples dimension"], default=None
        Monte-Carlo sample points for computing the
        normalization constant. Default is None.
    norm_weights : Float[Array, " nb_norm_samples"] | float | int, default=None
        The importance sampling weights for Monte-Carlo integration of the
        normalization constant. Must be provided if `norm_samples` is provided.
        `norm_weights` should have the same leading dimension as
        `norm_samples`.
        Alternatively, the user can pass a float or an integer.
        These corresponds to the weights $w_k = \frac{1}{q(x_k)}$ where
        $q(\cdot)$ is the proposal p.d.f. and $x_k$ are the Monte-Carlo samples.
    obs_slice : slice, default=None
        slice object specifying the begininning/ending of the PINN output
        that is observed (this is then useful for multidim PINN). Default is None.

    Raises
    ------
    ValueError
        If conditions on omega_boundary_condition and omega_boundary_fun
        are not respected
    """

    # NOTE static=True only for leaf attributes that are not valid JAX types
    # (ie. jax.Array cannot be static) and that we do not expect to change

    u: AbstractPINN
    dynamic_loss: PDEStatio | None
    loss_weights: LossWeightsPDEStatio
    derivative_keys: DerivativeKeysPDEStatio

    params: InitVar[Params[Array] | None]

    def __init__(
        self,
        *,
        u: AbstractPINN,
        dynamic_loss: PDEStatio | None,
        loss_weights: LossWeightsPDEStatio | None = None,
        derivative_keys: DerivativeKeysPDEStatio | None = None,
        params: Params[Array] | None = None,
        **kwargs: Any,
    ):
        self.u = u
        if loss_weights is None:
            self.loss_weights = LossWeightsPDEStatio()
        else:
            self.loss_weights = loss_weights

        if derivative_keys is None:
            # be default we only take gradient wrt nn_params
            try:
                derivative_keys = DerivativeKeysPDEStatio(params=params)
            except ValueError as exc:
                raise ValueError(
                    "Problem at derivative_keys initialization "
                    f"received {derivative_keys=} and {params=}"
                ) from exc
        else:
            derivative_keys = derivative_keys

        super().__init__(
            derivative_keys=derivative_keys,
            vmap_in_axes=(0,),
            **kwargs,
        )
        self.dynamic_loss = dynamic_loss

    def _get_dynamic_loss_batch(
        self, batch: PDEStatioBatch
    ) -> Float[Array, " batch_size dimension"]:
        return batch.domain_batch

    def _get_normalization_loss_batch(
        self, batch: PDEStatioBatch
    ) -> tuple[Float[Array, " nb_norm_samples dimension"] | None,]:
        return (self.norm_samples,)

    # we could have used typing.cast though

    def evaluate_by_terms(
        self,
        opt_params: Params[Array],
        batch: PDEStatioBatch,
        *,
        non_opt_params: Params[Array] | None = None,
    ) -> tuple[PDEStatioComponents[Array | None], PDEStatioComponents[Array | None]]:
        """
        Evaluate the loss function at a batch of points for given parameters.

        We retrieve two PyTrees with loss values and gradients for each term

        Parameters
        ---------
        opt_params
            Parameters, which are optimized, at which the loss is evaluated
        batch
            Composed of a batch of points in the
            domain, a batch of points in the domain
            border and an optional additional batch of parameters (eg. for
            metamodeling) and an optional additional batch of observed
            inputs/outputs/parameters
        non_opt_params
            Parameters, which are non optimized, at which the loss is evaluated
        """
        if non_opt_params is not None:
            params = eqx.combine(opt_params, non_opt_params)
        else:
            params = opt_params

        # Retrieve the optional eq_params_batch
        # and update eq_params with the latter
        # and update vmap_in_axes
        if batch.param_batch_dict is not None:
            # update eq_params with the batches of generated params
            params = update_eq_params(params, batch.param_batch_dict)

        vmap_in_axes_params = _get_vmap_in_axes_params(batch.param_batch_dict, params)

        # dynamic part
        dyn_loss_fun = self._get_dyn_loss_fun(batch, vmap_in_axes_params)

        # normalization part
        norm_loss_fun = self._get_norm_loss_fun(batch, vmap_in_axes_params)

        # boundary part
        boundary_loss_fun = self._get_boundary_loss_fun(batch)

        # Observation mse
        params_obs, obs_loss_fun = self._get_obs_params_and_obs_loss_fun(
            batch, vmap_in_axes_params, params
        )

        # get the unweighted mses for each loss term as well as the gradients
        all_funs: PDEStatioComponents[Callable[[Params[Array]], Array] | None] = (
            PDEStatioComponents(
                dyn_loss_fun, norm_loss_fun, boundary_loss_fun, obs_loss_fun
            )
        )
        all_params: PDEStatioComponents[Params[Array] | None] = PDEStatioComponents(
            params, params, params, params_obs
        )
        mses_grads = jax.tree.map(
            self.get_gradients,
            all_funs,
            all_params,
            is_leaf=lambda x: x is None,
        )
        mses = jax.tree.map(
            lambda leaf: leaf[0],  # type: ignore
            mses_grads,
            is_leaf=lambda x: isinstance(x, tuple),
        )
        grads = jax.tree.map(
            lambda leaf: leaf[1],  # type: ignore
            mses_grads,
            is_leaf=lambda x: isinstance(x, tuple),
        )

        return mses, grads


class LossPDENonStatio(
    _LossPDEAbstract[
        LossWeightsPDENonStatio,
        PDENonStatioBatch,
        PDENonStatioComponents[Array | None],
        DerivativeKeysPDENonStatio,
        PDENonStatio | None,
    ]
):
    r"""Loss object for a stationary partial differential equation

    $$
        \mathcal{N}[u](t, x) = 0, \forall t \in I, \forall x \in \Omega
    $$

    where $\mathcal{N}[\cdot]$ is a differential operator.
    The boundary condition is $u(t, x)=u_b(t, x),\forall
    x\in\delta\Omega, \forall t$.
    The initial condition is $u(0, x)=u_0(x), \forall x\in\Omega$
    The additional condition of
    integrating to 1 can be included, i.e., $\int u(t, x)\mathrm{d}x=1$.

    Parameters
    ----------
    u : AbstractPINN
        the PINN
    dynamic_loss : PDENonStatio
        the non stationary PDE dynamic part of the loss, basically the differential
        operator $\mathcal{N}[u](t, x)$. Should implement a method
        `dynamic_loss.evaluate(t, x, u, params)`.
        Can be None in order to access only some part of the evaluate call
        results.
    key : PRNGKeyArray
        A JAX PRNG Key for the loss class treated as an attribute. Default is
        None. This field is provided for future developments and additional
        losses that might need some randomness. Note that special care must be
        taken when splitting the key because in-place updates are forbidden in
        eqx.Modules.
        reason
    loss_weights : LossWeightsPDENonStatio, default=None
        The loss weights for the differents term : dynamic loss,
        boundary conditions if any, initial condition, normalization loss if any and
        observations if any.
        Can be updated according to a specific algorithm. See
        `update_weight_method`
    derivative_keys : DerivativeKeysPDENonStatio, default=None
        Specify which field of `params` should be differentiated for each
        composant of the total loss. Particularily useful for inverse problems.
        Fields can be "nn_params", "eq_params" or "both". Those that should not
        be updated will have a `jax.lax.stop_gradient` called on them. Default
        is `"nn_params"` for each composant of the loss.
    initial_condition_fun : Callable, default=None
        A function representing the initial condition at `t0`. If None
        (default) then no initial condition is applied.
    t0 : float | Float[Array, " 1"], default=None
        The time at which to apply the initial condition. If None, the time
        is set to `0` by default.
    max_norm_time_slices : int, default=100
        The maximum number of time points in the Cartesian product with the
        omega points to create the set of collocation points upon which the
        normalization constant is computed.
    max_norm_samples_omega : int, default=1000
        The maximum number of omega points in the Cartesian product with the
        time points to create the set of collocation points upon which the
        normalization constant is computed.
    params : InitVar[Params[Array]], default=None
        The main `Params` object of the problem needed to instanciate the
        `DerivativeKeysODE` if the latter is not specified.
    update_weight_method : Literal['soft_adapt', 'lr_annealing', 'ReLoBRaLo'], default=None
        Default is None meaning no update for loss weights. Otherwise a string
    omega_boundary_fun : BoundaryConditionFun | dict[str, BoundaryConditionFun], default=None
         The function to be matched in the border condition (can be None) or a
         dictionary of such functions as values and keys as described
         in `omega_boundary_condition`.
    omega_boundary_condition : str | dict[str, str], default=None
        Either None (no condition, by default), or a string defining
        the boundary condition (Dirichlet or Von Neumann),
        or a dictionary with such strings as values. In this case,
        the keys are the facets and must be in the following order:
        1D -> [“xmin”, “xmax”], 2D -> [“xmin”, “xmax”, “ymin”, “ymax”].
        Note that high order boundaries are currently not implemented.
        A value in the dict can be None, this means we do not enforce
        a particular boundary condition on this facet.
        The facet called “xmin”, resp. “xmax” etc., in 2D,
        refers to the set of 2D points with fixed “xmin”, resp. “xmax”, etc.
    omega_boundary_dim : slice | dict[str, slice], default=None
        Either None, or a slice object or a dictionary of slice objects as
        values and keys as described in `omega_boundary_condition`.
        `omega_boundary_dim` indicates which dimension(s) of the PINN
        will be forced to match the boundary condition.
        Note that it must be a slice and not an integer
        (but a preprocessing of the user provided argument takes care of it)
    norm_samples : Float[Array, " nb_norm_samples dimension"], default=None
        Monte-Carlo sample points for computing the
        normalization constant. Default is None.
    norm_weights : Float[Array, " nb_norm_samples"] | float | int, default=None
        The importance sampling weights for Monte-Carlo integration of the
        normalization constant. Must be provided if `norm_samples` is provided.
        `norm_weights` should have the same leading dimension as
        `norm_samples`.
        Alternatively, the user can pass a float or an integer.
        These corresponds to the weights $w_k = \frac{1}{q(x_k)}$ where
        $q(\cdot)$ is the proposal p.d.f. and $x_k$ are the Monte-Carlo samples.
    obs_slice : slice, default=None
        slice object specifying the begininning/ending of the PINN output
        that is observed (this is then useful for multidim PINN). Default is None.

    """

    u: AbstractPINN
    dynamic_loss: PDENonStatio | None
    loss_weights: LossWeightsPDENonStatio
    derivative_keys: DerivativeKeysPDENonStatio
    params: InitVar[Params[Array] | None]
    t0: Float[Array, " "]
    initial_condition_fun: Callable[[Float[Array, " dimension"]], Array] | None = (
        eqx.field(static=True)
    )
    max_norm_samples_omega: int = eqx.field(static=True)
    max_norm_time_slices: int = eqx.field(static=True)

    params: InitVar[Params[Array] | None]

    def __init__(
        self,
        *,
        u: AbstractPINN,
        dynamic_loss: PDENonStatio | None,
        loss_weights: LossWeightsPDENonStatio | None = None,
        derivative_keys: DerivativeKeysPDENonStatio | None = None,
        initial_condition_fun: Callable[[Float[Array, " dimension"]], Array]
        | None = None,
        t0: int | float | Float[Array, " "] | None = None,
        max_norm_time_slices: int = 100,
        max_norm_samples_omega: int = 1000,
        params: Params[Array] | None = None,
        **kwargs: Any,
    ):
        self.u = u
        if loss_weights is None:
            self.loss_weights = LossWeightsPDENonStatio()
        else:
            self.loss_weights = loss_weights

        if derivative_keys is None:
            # be default we only take gradient wrt nn_params
            try:
                derivative_keys = DerivativeKeysPDENonStatio(params=params)
            except ValueError as exc:
                raise ValueError(
                    "Problem at derivative_keys initialization "
                    f"received {derivative_keys=} and {params=}"
                ) from exc
        else:
            derivative_keys = derivative_keys

        super().__init__(
            derivative_keys=derivative_keys,
            vmap_in_axes=(0,),  # for t_x
            **kwargs,
        )

        self.dynamic_loss = dynamic_loss

        if initial_condition_fun is None:
            warnings.warn(
                "Initial condition wasn't provided. Be sure to cover for that"
                "case (e.g by. hardcoding it into the PINN output)."
            )
        # some checks for t0
        if t0 is None:
            self.t0 = jnp.array([0])
        else:
            self.t0 = initial_condition_check(t0, dim_size=1)

        self.initial_condition_fun = initial_condition_fun

        # with the variables below we avoid memory overflow since a cartesian
        # product is taken
        self.max_norm_time_slices = max_norm_time_slices
        self.max_norm_samples_omega = max_norm_samples_omega

    def _get_dynamic_loss_batch(
        self, batch: PDENonStatioBatch
    ) -> Float[Array, " batch_size 1+dimension"]:
        return batch.domain_batch

    def _get_normalization_loss_batch(
        self, batch: PDENonStatioBatch
    ) -> tuple[
        Float[Array, " nb_norm_time_slices 1"], Float[Array, " nb_norm_samples dim"]
    ]:
        return (
            batch.domain_batch[: self.max_norm_time_slices, 0:1],
            self.norm_samples[: self.max_norm_samples_omega],  # type: ignore -> cannot narrow a class attr
        )

    def evaluate_by_terms(
        self,
        opt_params: Params[Array],
        batch: PDENonStatioBatch,
        *,
        non_opt_params: Params[Array] | None = None,
    ) -> tuple[
        PDENonStatioComponents[Array | None], PDENonStatioComponents[Array | None]
    ]:
        """
        Evaluate the loss function at a batch of points for given parameters.

        We retrieve two PyTrees with loss values and gradients for each term

        Parameters
        ---------
        opt_params
            Parameters, which are optimized, at which the loss is evaluated
        batch
            Composed of a batch of points in the
            domain, a batch of points in the domain
            border and an optional additional batch of parameters (eg. for
            metamodeling) and an optional additional batch of observed
            inputs/outputs/parameters
        non_opt_params
            Parameters, which are non optimized, at which the loss is evaluated
        """
        if non_opt_params is not None:
            params = eqx.combine(opt_params, non_opt_params)
        else:
            params = opt_params

        omega_initial_batch = batch.initial_batch
        assert omega_initial_batch is not None

        # Retrieve the optional eq_params_batch
        # and update eq_params with the latter
        # and update vmap_in_axes
        if batch.param_batch_dict is not None:
            # update eq_params with the batches of generated params
            params = update_eq_params(params, batch.param_batch_dict)

        vmap_in_axes_params = _get_vmap_in_axes_params(batch.param_batch_dict, params)

        # dynamic part
        dyn_loss_fun = self._get_dyn_loss_fun(batch, vmap_in_axes_params)

        # normalization part
        norm_loss_fun = self._get_norm_loss_fun(batch, vmap_in_axes_params)

        # boundary part
        boundary_loss_fun = self._get_boundary_loss_fun(batch)

        # Observation mse
        params_obs, obs_loss_fun = self._get_obs_params_and_obs_loss_fun(
            batch, vmap_in_axes_params, params
        )

        # initial condition
        if self.initial_condition_fun is not None:
            mse_initial_condition_fun: Callable[[Params[Array]], Array] | None = (
                lambda p: initial_condition_apply(
                    self.u,
                    omega_initial_batch,
                    _set_derivatives(p, self.derivative_keys.initial_condition),
                    (0,) + vmap_in_axes_params,
                    self.initial_condition_fun,  # type: ignore
                    self.t0,
                )
            )
        else:
            mse_initial_condition_fun = None

        # get the unweighted mses for each loss term as well as the gradients
        all_funs: PDENonStatioComponents[Callable[[Params[Array]], Array] | None] = (
            PDENonStatioComponents(
                dyn_loss_fun,
                norm_loss_fun,
                boundary_loss_fun,
                obs_loss_fun,
                mse_initial_condition_fun,
            )
        )
        all_params: PDENonStatioComponents[Params[Array] | None] = (
            PDENonStatioComponents(params, params, params, params_obs, params)
        )
        mses_grads = jax.tree.map(
            self.get_gradients,
            all_funs,
            all_params,
            is_leaf=lambda x: x is None,
        )
        mses = jax.tree.map(
            lambda leaf: leaf[0],  # type: ignore
            mses_grads,
            is_leaf=lambda x: isinstance(x, tuple),
        )
        grads = jax.tree.map(
            lambda leaf: leaf[1],  # type: ignore
            mses_grads,
            is_leaf=lambda x: isinstance(x, tuple),
        )

        return mses, grads
