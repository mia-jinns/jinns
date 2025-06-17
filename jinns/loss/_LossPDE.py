"""
Main module to implement a PDE loss in jinns
"""

from __future__ import (
    annotations,
)  # https://docs.python.org/3/library/typing.html#constant

import abc
from dataclasses import InitVar
from typing import TYPE_CHECKING, Callable, TypedDict
from types import EllipsisType
import warnings
import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float, Array, Key, Int
from jinns.loss._loss_utils import (
    dynamic_loss_apply,
    boundary_condition_apply,
    normalization_loss_apply,
    observations_loss_apply,
    initial_condition_apply,
)
from jinns.parameters._params import (
    _get_vmap_in_axes_params,
    _update_eq_params_dict,
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


if TYPE_CHECKING:
    # imports for type hints only
    from jinns.nn._abstract_pinn import AbstractPINN
    from jinns.loss import PDENonStatio, PDEStatio
    from jinns.utils._types import BoundaryConditionFun

    class LossDictPDEStatio(TypedDict):
        dyn_loss: Float[Array, " "]
        norm_loss: Float[Array, " "]
        boundary_loss: Float[Array, " "]
        observations: Float[Array, " "]

    class LossDictPDENonStatio(LossDictPDEStatio):
        initial_condition: Float[Array, " "]


_IMPLEMENTED_BOUNDARY_CONDITIONS = [
    "dirichlet",
    "von neumann",
    "vonneumann",
]


class _LossPDEAbstract(AbstractLoss):
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
    derivative_keys : DerivativeKeysPDEStatio | DerivativeKeysPDENonStatio, default=None
        Specify which field of `params` should be differentiated for each
        composant of the total loss. Particularily useful for inverse problems.
        Fields can be "nn_params", "eq_params" or "both". Those that should not
        be updated will have a `jax.lax.stop_gradient` called on them. Default
        is `"nn_params"` for each composant of the loss.
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
        `norm_weights` should be broadcastble to
        `norm_samples`.
        Alternatively, the user can pass a float or an integer that will be
        made broadcastable to `norm_samples`.
        These corresponds to the weights $w_k = \frac{1}{q(x_k)}$ where
        $q(\cdot)$ is the proposal p.d.f. and $x_k$ are the Monte-Carlo samples.
    obs_slice : EllipsisType | slice, default=None
        slice object specifying the begininning/ending of the PINN output
        that is observed (this is then useful for multidim PINN). Default is None.
    params : InitVar[Params[Array]], default=None
        The main Params object of the problem needed to instanciate the
        DerivativeKeysODE if the latter is not specified.
    """

    # NOTE static=True only for leaf attributes that are not valid JAX types
    # (ie. jax.Array cannot be static) and that we do not expect to change
    # kw_only in base class is motivated here: https://stackoverflow.com/a/69822584
    derivative_keys: DerivativeKeysPDEStatio | DerivativeKeysPDENonStatio | None = (
        eqx.field(kw_only=True, default=None)
    )
    loss_weights: LossWeightsPDEStatio | LossWeightsPDENonStatio | None = eqx.field(
        kw_only=True, default=None
    )
    omega_boundary_fun: (
        BoundaryConditionFun | dict[str, BoundaryConditionFun] | None
    ) = eqx.field(kw_only=True, default=None, static=True)
    omega_boundary_condition: str | dict[str, str] | None = eqx.field(
        kw_only=True, default=None, static=True
    )
    omega_boundary_dim: slice | dict[str, slice] | None = eqx.field(
        kw_only=True, default=None, static=True
    )
    norm_samples: Float[Array, " nb_norm_samples dimension"] | None = eqx.field(
        kw_only=True, default=None
    )
    norm_weights: Float[Array, " nb_norm_samples"] | float | int | None = eqx.field(
        kw_only=True, default=None
    )
    obs_slice: EllipsisType | slice | None = eqx.field(
        kw_only=True, default=None, static=True
    )

    params: InitVar[Params[Array]] = eqx.field(kw_only=True, default=None)

    def __post_init__(self, params: Params[Array] | None = None):
        """
        Note that neither __init__ or __post_init__ are called when udating a
        Module with eqx.tree_at
        """
        if self.derivative_keys is None:
            # be default we only take gradient wrt nn_params
            try:
                self.derivative_keys = (
                    DerivativeKeysPDENonStatio(params=params)
                    if isinstance(self, LossPDENonStatio)
                    else DerivativeKeysPDEStatio(params=params)
                )
            except ValueError as exc:
                raise ValueError(
                    "Problem at self.derivative_keys initialization "
                    f"received {self.derivative_keys=} and {params=}"
                ) from exc

        if self.loss_weights is None:
            self.loss_weights = (
                LossWeightsPDENonStatio()
                if isinstance(self, LossPDENonStatio)
                else LossWeightsPDEStatio()
            )

        if self.obs_slice is None:
            self.obs_slice = jnp.s_[...]

        if (
            isinstance(self.omega_boundary_fun, dict)
            and not isinstance(self.omega_boundary_condition, dict)
        ) or (
            not isinstance(self.omega_boundary_fun, dict)
            and isinstance(self.omega_boundary_condition, dict)
        ):
            raise ValueError(
                "if one of self.omega_boundary_fun or "
                "self.omega_boundary_condition is dict, the other should be too."
            )

        if self.omega_boundary_condition is None or self.omega_boundary_fun is None:
            warnings.warn(
                "Missing boundary function or no boundary condition."
                "Boundary function is thus ignored."
            )
        else:
            if isinstance(self.omega_boundary_condition, dict):
                for _, v in self.omega_boundary_condition.items():
                    if v is not None and not any(
                        v.lower() in s for s in _IMPLEMENTED_BOUNDARY_CONDITIONS
                    ):
                        raise NotImplementedError(
                            f"The boundary condition {self.omega_boundary_condition} is not"
                            f"implemented yet. Try one of :"
                            f"{_IMPLEMENTED_BOUNDARY_CONDITIONS}."
                        )
            else:
                if not any(
                    self.omega_boundary_condition.lower() in s
                    for s in _IMPLEMENTED_BOUNDARY_CONDITIONS
                ):
                    raise NotImplementedError(
                        f"The boundary condition {self.omega_boundary_condition} is not"
                        f"implemented yet. Try one of :"
                        f"{_IMPLEMENTED_BOUNDARY_CONDITIONS}."
                    )
                if isinstance(self.omega_boundary_fun, dict) and isinstance(
                    self.omega_boundary_condition, dict
                ):
                    if (
                        not (
                            list(self.omega_boundary_fun.keys()) == ["xmin", "xmax"]
                            and list(self.omega_boundary_condition.keys())
                            == ["xmin", "xmax"]
                        )
                    ) or (
                        not (
                            list(self.omega_boundary_fun.keys())
                            == ["xmin", "xmax", "ymin", "ymax"]
                            and list(self.omega_boundary_condition.keys())
                            == ["xmin", "xmax", "ymin", "ymax"]
                        )
                    ):
                        raise ValueError(
                            "The key order (facet order) in the "
                            "boundary condition dictionaries is incorrect"
                        )

        if isinstance(self.omega_boundary_fun, dict):
            if not isinstance(self.omega_boundary_dim, dict):
                raise ValueError(
                    "If omega_boundary_fun is a dict then"
                    " omega_boundary_dim should also be a dict"
                )
            if self.omega_boundary_dim is None:
                self.omega_boundary_dim = {
                    k: jnp.s_[::] for k in self.omega_boundary_fun.keys()
                }
            if list(self.omega_boundary_dim.keys()) != list(
                self.omega_boundary_fun.keys()
            ):
                raise ValueError(
                    "If omega_boundary_fun is a dict,"
                    " omega_boundary_dim should be a dict with the same keys"
                )
            for k, v in self.omega_boundary_dim.items():
                if isinstance(v, int):
                    # rewrite it as a slice to ensure that axis does not disappear when
                    # indexing
                    self.omega_boundary_dim[k] = jnp.s_[v : v + 1]

        else:
            if self.omega_boundary_dim is None:
                self.omega_boundary_dim = jnp.s_[::]
            if isinstance(self.omega_boundary_dim, int):
                # rewrite it as a slice to ensure that axis does not disappear when
                # indexing
                self.omega_boundary_dim = jnp.s_[
                    self.omega_boundary_dim : self.omega_boundary_dim + 1
                ]
            if not isinstance(self.omega_boundary_dim, slice):
                raise ValueError("self.omega_boundary_dim must be a jnp.s_ object")

        if self.norm_samples is not None:
            if self.norm_weights is None:
                raise ValueError(
                    "`norm_weights` must be provided when `norm_samples` is used!"
                )
            if isinstance(self.norm_weights, (int, float)):
                self.norm_weights = self.norm_weights * jnp.ones(
                    (self.norm_samples.shape[0],)
                )
            if isinstance(self.norm_weights, Array):
                if not (self.norm_weights.shape[0] == self.norm_samples.shape[0]):
                    raise ValueError(
                        "self.norm_weights and "
                        "self.norm_samples must have the same leading dimension"
                    )
            else:
                raise ValueError("Wrong type for self.norm_weights")

    @abc.abstractmethod
    def __call__(self, *_, **__):
        pass

    @abc.abstractmethod
    def evaluate(
        self: eqx.Module,
        params: Params[Array],
        batch: PDEStatioBatch | PDENonStatioBatch,
    ) -> tuple[Float[Array, " "], LossDictPDEStatio | LossDictPDENonStatio]:
        raise NotImplementedError


class LossPDEStatio(_LossPDEAbstract):
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
    dynamic_loss : PDEStatio
        the stationary PDE dynamic part of the loss, basically the differential
        operator $\mathcal{N}[u](x)$. Should implement a method
        `dynamic_loss.evaluate(x, u, params)`.
        Can be None in order to access only some part of the evaluate call
        results.
    key : Key
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
    update_weight_method : Literal['soft_adapt', 'lr_annealing', 'ReLoBRaLo'], default=None
        Default is None meaning no update for loss weights. Otherwise a string
    derivative_keys : DerivativeKeysPDEStatio, default=None
        Specify which field of `params` should be differentiated for each
        composant of the total loss. Particularily useful for inverse problems.
        Fields can be "nn_params", "eq_params" or "both". Those that should not
        be updated will have a `jax.lax.stop_gradient` called on them. Default
        is `"nn_params"` for each composant of the loss.
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
    params : InitVar[Params[Array]], default=None
        The main Params object of the problem needed to instanciate the
        DerivativeKeysODE if the latter is not specified.


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
    key: Key | None = eqx.field(kw_only=True, default=None)

    vmap_in_axes: tuple[Int] = eqx.field(init=False, static=True)

    def __post_init__(self, params: Params[Array] | None = None):
        """
        Note that neither __init__ or __post_init__ are called when udating a
        Module with eqx.tree_at!
        """
        super().__post_init__(
            params=params
        )  # because __init__ or __post_init__ of Base
        # class is not automatically called

        self.vmap_in_axes = (0,)  # for x only here

    def _get_dynamic_loss_batch(
        self, batch: PDEStatioBatch
    ) -> Float[Array, " batch_size dimension"]:
        return batch.domain_batch

    def _get_normalization_loss_batch(
        self, _
    ) -> tuple[Float[Array, " nb_norm_samples dimension"]]:
        return (self.norm_samples,)  # type: ignore -> cannot narrow a class attr

    # we could have used typing.cast though

    def _get_observations_loss_batch(
        self, batch: PDEStatioBatch
    ) -> Float[Array, " batch_size obs_dim"]:
        return batch.obs_batch_dict["pinn_in"]

    def __call__(self, *args, **kwargs):
        return self.evaluate(*args, **kwargs)

    def evaluate_by_terms(
        self, params: Params[Array], batch: PDEStatioBatch
    ) -> tuple[PDEStatioComponents[Array | None], PDEStatioComponents[Array | None]]:
        """
        Evaluate the loss function at a batch of points for given parameters.

        We retrieve two PyTrees with loss values and gradients for each term

        Parameters
        ---------
        params
            Parameters at which the loss is evaluated
        batch
            Composed of a batch of points in the
            domain, a batch of points in the domain
            border and an optional additional batch of parameters (eg. for
            metamodeling) and an optional additional batch of observed
            inputs/outputs/parameters
        """
        # Retrieve the optional eq_params_batch
        # and update eq_params with the latter
        # and update vmap_in_axes
        if batch.param_batch_dict is not None:
            # update eq_params with the batches of generated params
            params = _update_eq_params_dict(params, batch.param_batch_dict)

        vmap_in_axes_params = _get_vmap_in_axes_params(batch.param_batch_dict, params)

        # dynamic part
        if self.dynamic_loss is not None:
            dyn_loss_fun = lambda p: dynamic_loss_apply(
                self.dynamic_loss.evaluate,  # type: ignore
                self.u,
                self._get_dynamic_loss_batch(batch),
                _set_derivatives(p, self.derivative_keys.dyn_loss),  # type: ignore
                self.vmap_in_axes + vmap_in_axes_params,
            )
        else:
            dyn_loss_fun = None

        # normalization part
        if self.norm_samples is not None:
            norm_loss_fun = lambda p: normalization_loss_apply(
                self.u,
                self._get_normalization_loss_batch(batch),
                _set_derivatives(p, self.derivative_keys.norm_loss),  # type: ignore
                vmap_in_axes_params,
                self.norm_weights,  # type: ignore -> can't get the __post_init__ narrowing here
            )
        else:
            norm_loss_fun = None

        # boundary part
        if (
            self.omega_boundary_condition is not None
            and self.omega_boundary_dim is not None
            and self.omega_boundary_fun is not None
        ):  # pyright cannot narrow down the three None otherwise as it is class attribute
            boundary_loss_fun = lambda p: boundary_condition_apply(
                self.u,
                batch,
                _set_derivatives(p, self.derivative_keys.boundary_loss),  # type: ignore
                self.omega_boundary_fun,  # type: ignore
                self.omega_boundary_condition,  # type: ignore
                self.omega_boundary_dim,  # type: ignore
            )
        else:
            boundary_loss_fun = None

        # Observation mse
        if batch.obs_batch_dict is not None:
            # update params with the batches of observed params
            params_obs = _update_eq_params_dict(
                params, batch.obs_batch_dict["eq_params"]
            )

            obs_loss_fun = lambda po: observations_loss_apply(
                self.u,
                self._get_observations_loss_batch(batch),
                _set_derivatives(po, self.derivative_keys.observations),  # type: ignore
                self.vmap_in_axes + vmap_in_axes_params,
                batch.obs_batch_dict["val"],
                self.obs_slice,
            )
        else:
            params_obs = None
            obs_loss_fun = None

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
            lambda fun, params: self.get_gradients(fun, params),
            all_funs,
            all_params,
            is_leaf=lambda x: x is None,
        )
        mses = jax.tree.map(
            lambda leaf: leaf[0], mses_grads, is_leaf=lambda x: isinstance(x, tuple)
        )
        grads = jax.tree.map(
            lambda leaf: leaf[1], mses_grads, is_leaf=lambda x: isinstance(x, tuple)
        )

        return mses, grads

    def evaluate(
        self, params: Params[Array], batch: PDEStatioBatch
    ) -> tuple[Float[Array, " "], PDEStatioComponents[Float[Array, " "] | None]]:
        """
        Evaluate the loss function at a batch of points for given parameters.

        We retrieve the total value itself and a PyTree with loss values for each term

        Parameters
        ---------
        params
            Parameters at which the loss is evaluated
        batch
            Composed of a batch of points in the
            domain, a batch of points in the domain
            border and an optional additional batch of parameters (eg. for
            metamodeling) and an optional additional batch of observed
            inputs/outputs/parameters
        """
        loss_terms, _ = self.evaluate_by_terms(params, batch)

        loss_val = self.ponderate_and_sum_loss(loss_terms)

        return loss_val, loss_terms


class LossPDENonStatio(LossPDEStatio):
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
    key : Key
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
    update_weight_method : Literal['soft_adapt', 'lr_annealing', 'ReLoBRaLo'], default=None
        Default is None meaning no update for loss weights. Otherwise a string
    derivative_keys : DerivativeKeysPDENonStatio, default=None
        Specify which field of `params` should be differentiated for each
        composant of the total loss. Particularily useful for inverse problems.
        Fields can be "nn_params", "eq_params" or "both". Those that should not
        be updated will have a `jax.lax.stop_gradient` called on them. Default
        is `"nn_params"` for each composant of the loss.
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
    t0 : float | Float[Array, " 1"], default=None
        The time at which to apply the initial condition. If None, the time
        is set to `0` by default.
    initial_condition_fun : Callable, default=None
        A function representing the initial condition at `t0`. If None
        (default) then no initial condition is applied.
    params : InitVar[Params[Array]], default=None
        The main `Params` object of the problem needed to instanciate the
        `DerivativeKeysODE` if the latter is not specified.

    """

    dynamic_loss: PDENonStatio | None
    # NOTE static=True only for leaf attributes that are not valid JAX types
    # (ie. jax.Array cannot be static) and that we do not expect to change
    initial_condition_fun: Callable | None = eqx.field(
        kw_only=True, default=None, static=True
    )
    t0: float | Float[Array, " 1"] | None = eqx.field(kw_only=True, default=None)

    _max_norm_samples_omega: Int = eqx.field(init=False, static=True)
    _max_norm_time_slices: Int = eqx.field(init=False, static=True)

    def __post_init__(self, params=None):
        """
        Note that neither __init__ or __post_init__ are called when udating a
        Module with eqx.tree_at!
        """
        super().__post_init__(
            params=params
        )  # because __init__ or __post_init__ of Base
        # class is not automatically called

        self.vmap_in_axes = (0,)  # for t_x

        if self.initial_condition_fun is None:
            warnings.warn(
                "Initial condition wasn't provided. Be sure to cover for that"
                "case (e.g by. hardcoding it into the PINN output)."
            )
        # some checks for t0
        if isinstance(self.t0, Array):
            if not self.t0.shape:  # e.g. user input: jnp.array(0.)
                self.t0 = jnp.array([self.t0])
            elif self.t0.shape != (1,):
                raise ValueError(
                    f"Wrong self.t0 input. It should be"
                    f"a float or an array of shape (1,). Got shape: {self.t0.shape}"
                )
        elif isinstance(self.t0, float):  # e.g. user input: 0.
            self.t0 = jnp.array([self.t0])
        elif isinstance(self.t0, int):  # e.g. user input: 0
            self.t0 = jnp.array([float(self.t0)])
        elif self.t0 is None:
            self.t0 = jnp.array([0])
        else:
            raise ValueError("Wrong value for t0")

        # witht the variables below we avoid memory overflow since a cartesian
        # product is taken
        self._max_norm_time_slices = 100
        self._max_norm_samples_omega = 1000

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
            batch.domain_batch[: self._max_norm_time_slices, 0:1],
            self.norm_samples[: self._max_norm_samples_omega],  # type: ignore -> cannot narrow a class attr
        )

    def _get_observations_loss_batch(
        self, batch: PDENonStatioBatch
    ) -> Float[Array, " batch_size 1+dim"]:
        return batch.obs_batch_dict["pinn_in"]

    def __call__(self, *args, **kwargs):
        return self.evaluate(*args, **kwargs)

    def evaluate_by_terms(
        self, params: Params[Array], batch: PDENonStatioBatch
    ) -> tuple[
        PDENonStatioComponents[Array | None], PDENonStatioComponents[Array | None]
    ]:
        """
        Evaluate the loss function at a batch of points for given parameters.

        We retrieve two PyTrees with loss values and gradients for each term

        Parameters
        ---------
        params
            Parameters at which the loss is evaluated
        batch
            Composed of a batch of points in the
            domain, a batch of points in the domain
            border and an optional additional batch of parameters (eg. for
            metamodeling) and an optional additional batch of observed
            inputs/outputs/parameters
        """
        omega_batch = batch.initial_batch
        assert omega_batch is not None

        # Retrieve the optional eq_params_batch
        # and update eq_params with the latter
        # and update vmap_in_axes
        if batch.param_batch_dict is not None:
            # update eq_params with the batches of generated params
            params = _update_eq_params_dict(params, batch.param_batch_dict)

        vmap_in_axes_params = _get_vmap_in_axes_params(batch.param_batch_dict, params)

        # For mse_dyn_loss, mse_norm_loss, mse_boundary_loss,
        # mse_observation_loss we use the evaluate from parent class
        # As well as for their gradients
        partial_mses, partial_grads = super().evaluate_by_terms(params, batch)  # type: ignore
        # ignore because batch is not PDEStatioBatch. We could use typing.cast though

        # initial condition
        if self.initial_condition_fun is not None:
            mse_initial_condition_fun = lambda p: initial_condition_apply(
                self.u,
                omega_batch,
                _set_derivatives(p, self.derivative_keys.initial_condition),  # type: ignore
                (0,) + vmap_in_axes_params,
                self.initial_condition_fun,  # type: ignore
                self.t0,  # type: ignore can't get the narrowing in __post_init__
            )
            mse_initial_condition, grad_initial_condition = self.get_gradients(
                mse_initial_condition_fun, params
            )
        else:
            mse_initial_condition = None
            grad_initial_condition = None

        mses = PDENonStatioComponents(
            partial_mses.dyn_loss,
            partial_mses.norm_loss,
            partial_mses.boundary_loss,
            partial_mses.observations,
            mse_initial_condition,
        )

        grads = PDENonStatioComponents(
            partial_grads.dyn_loss,
            partial_grads.norm_loss,
            partial_grads.boundary_loss,
            partial_grads.observations,
            grad_initial_condition,
        )

        return mses, grads

    def evaluate(
        self, params: Params[Array], batch: PDENonStatioBatch
    ) -> tuple[Float[Array, " "], PDENonStatioComponents[Float[Array, " "] | None]]:
        """
        Evaluate the loss function at a batch of points for given parameters.
        We retrieve the total value itself and a PyTree with loss values for each term


        Parameters
        ---------
        params
            Parameters at which the loss is evaluated
        batch
            Composed of a batch of points in
            the domain, a batch of points in the domain
            border, a batch of time points and an optional additional batch
            of parameters (eg. for metamodeling) and an optional additional batch of observed
            inputs/outputs/parameters
        """
        return super().evaluate(params, batch)  # type: ignore
