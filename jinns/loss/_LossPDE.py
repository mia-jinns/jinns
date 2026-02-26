"""
Main module to implement a PDE loss in jinns
"""

from __future__ import (
    annotations,
)  # https://docs.python.org/3/library/typing.html#constant

from dataclasses import InitVar
from typing import TYPE_CHECKING, Callable, Any, TypeVar, Generic, ClassVar
import warnings
import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import PRNGKeyArray, Float, Array
from jinns.loss._loss_utils import (
    normalization_loss_apply,
    boundary_condition_apply,
    initial_condition_apply,
    initial_condition_check,
    mean_sum_reduction,
    mean_sum_reduction_pytree,
    vmap_loss_fun_classical,
    vmap_loss_fun_observations,
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
from jinns.loss import PDENonStatio, PDEStatio
from jinns.data._Batchs import PDEStatioBatch, PDENonStatioBatch
from jinns.data._utils import make_cartesian_product
from jinns.parameters._params import Params
from jinns.nn._pinn import PINN
from jinns.nn._spinn import SPINN
from jinns.nn._hyperpinn import HyperPINN


if TYPE_CHECKING:
    # imports for type hints only
    from jinns.nn._abstract_pinn import AbstractPINN
    from jinns.loss._BoundaryConditionAbstract import BoundaryConditionAbstract


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
    boundary_condition : BoundaryConditionAbstract | None
        A BoundaryCondition object implementing
        operator $\mathcal{B}[u](inputs)=f(x)$.
        Can be None
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
        **If using SPINN, `norm_weights` must be a scalar**.
    obs_slice : tuple[EllipsisType | slice, ...] | EllipsisType | slice | None, default=None
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
    dynamic_loss: tuple[eqx.AbstractVar[Y] | None, ...]
    boundary_condition: BoundaryConditionAbstract | None
    norm_samples: Float[Array, " nb_norm_samples dimension"] | None
    norm_weights: Float[Array, " nb_norm_samples"] | None
    key: PRNGKeyArray | None

    def __init__(
        self,
        *,
        boundary_condition: BoundaryConditionAbstract | None = None,
        norm_samples: Float[Array, " nb_norm_samples dimension"] | None = None,
        norm_weights: Float[Array, " nb_norm_samples"] | float | int | None = None,
        key: PRNGKeyArray | None = None,
        **kwargs: Any,  # for arguments for super()
    ):
        super().__init__(
            loss_weights=self.loss_weights,
            u=self.u,
            **kwargs,
        )
        self.boundary_condition = boundary_condition
        if self.boundary_condition is None:
            warnings.warn("Missing boundary condition.")

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
                if isinstance(self.u, SPINN):
                    if not isinstance(norm_weights, (int, float)):
                        raise ValueError("norm_weights must be scalar when using SPINN")
                    elif isinstance(norm_weights, Array):
                        if norm_weights.shape != (1,):
                            raise ValueError(
                                "norm_weights must be scalar when using SPINN"
                            )
        else:
            self.norm_samples = norm_samples
            self.norm_weights = None

        self.key = key

    def _get_norm_loss_fun(
        self,
    ) -> Callable[[tuple[Array, Array], Params[Array]], Array] | None:
        if self.norm_samples is not None:
            norm_loss_fun: (
                Callable[[tuple[Array, Array], Params[Array]], Array] | None
            ) = lambda b, p: normalization_loss_apply(
                self.u,
                b,
                _set_derivatives(p, self.derivative_keys.norm_loss),
                self.norm_weights,  # type: ignore -> can't get the __post_init__ narrowing here
            )
        else:
            norm_loss_fun = None
        return norm_loss_fun

    def _get_boundary_loss_fun(
        self,
    ) -> Callable[[Array, Params[Array]], tuple[Array, ...]] | None:
        if self.boundary_condition is not None:
            boundary_loss_fun: (
                Callable[[Array, Params[Array]], tuple[Array, ...]] | None
            ) = lambda b, p: boundary_condition_apply(
                self.boundary_condition,  # type: ignore # we are in lambda...
                self.u,
                b,
                _set_derivatives(p, self.derivative_keys.boundary_loss),
            )
        else:
            boundary_loss_fun = None

        return boundary_loss_fun


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
    dynamic_loss : tuple[PDEStatio, ...] | PDEStatio | None
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
    boundary_condition : BoundaryConditionAbstract | None
        A BoundaryCondition object implementing
        operator $\mathcal{B}[u](inputs)=f(x)$.
        Can be None
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
        **If using SPINN, `norm_weights` must be a scalar**.
    obs_slice : tuple[EllipsisType | slice, ...] | EllipsisType | slice | None, default=None
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
    dynamic_loss: tuple[PDEStatio | None, ...]
    loss_weights: LossWeightsPDEStatio
    derivative_keys: DerivativeKeysPDEStatio

    params: InitVar[Params[Array] | None]

    reduction_functions: ClassVar[PDEStatioComponents[Callable]] = eqx.field(
        static=True,
        default=PDEStatioComponents(
            dyn_loss=mean_sum_reduction_pytree,
            boundary_loss=lambda residual_all_facets: jax.tree.reduce(
                jnp.add,
                jax.tree.map(
                    mean_sum_reduction,
                    residual_all_facets,
                ),
                0.0,
            )  # NOTE this boundary component changes ! Outer sum is for the facets
            if residual_all_facets is not None
            else None,
            norm_loss=lambda res: jnp.abs(jnp.mean(res) - 1.0) ** 2
            if res is not None
            else None,
            observations=mean_sum_reduction_pytree,
        ),
    )
    vmap_loss_fun: ClassVar[PDEStatioComponents[Callable]] = eqx.field(
        static=True,
        default=PDEStatioComponents(
            dyn_loss=vmap_loss_fun_classical,
            boundary_loss=vmap_loss_fun_classical,
            norm_loss=vmap_loss_fun_classical,
            observations=vmap_loss_fun_observations,
        ),
    )

    def __init__(
        self,
        *,
        u: AbstractPINN,
        dynamic_loss: tuple[PDEStatio, ...] | PDEStatio | None,
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
                self.derivative_keys = DerivativeKeysPDEStatio(params=params)
            except ValueError as exc:
                raise ValueError(
                    "Problem at derivative_keys initialization "
                    f"received {derivative_keys=} and {params=}"
                ) from exc
        else:
            self.derivative_keys = derivative_keys
        if not isinstance(dynamic_loss, tuple):
            self.dynamic_loss = (dynamic_loss,)
        else:
            self.dynamic_loss = dynamic_loss
        super().__init__(
            dynamic_loss=self.dynamic_loss,
            derivative_keys=self.derivative_keys,
            **kwargs,
        )

    def _get_normalization_loss_batch(
        self, _
    ) -> tuple[
        Float[Array, " nb_norm_samples dimension"] | None,
        Float[Array, " nb_norm_samples"] | None,
    ]:
        return (self.norm_samples, self.norm_weights)

    def evaluate_by_terms(
        self,
        batch: PDEStatioBatch,
        params: Params[Array],
    ) -> PDEStatioComponents[
        tuple[
            Callable | None,
            tuple[tuple[Array, Array] | None, ...]
            | tuple[Array | None, ...]
            | Array
            | None,
            Params[Array] | None,
            tuple[tuple[int, ...]] | tuple[int, ...] | None,
        ]
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
        # dynamic part
        domain_batch = batch.domain_batch
        dyn_loss_fun = self._get_dyn_loss_fun()

        # normalization part
        norm_batch = self._get_normalization_loss_batch(batch)
        norm_loss_fun = self._get_norm_loss_fun()

        # boundary part
        border_batch = batch.border_batch
        boundary_loss_fun = self._get_boundary_loss_fun()

        # Observation mse
        if batch.obs_batch_dict is not None:
            obs_loss_fun = self._get_obs_loss_fun()
            obs_batch = tuple((b["pinn_in"], b["val"]) for b in batch.obs_batch_dict)
        else:
            obs_loss_fun = None
            obs_batch = None

        all_funs_and_params: PDEStatioComponents[
            tuple[
                Callable | None,
                tuple[tuple[Array, Array] | None, ...]
                | tuple[Array | None, ...]
                | Array
                | None,
                Params[Array] | None,
                tuple[tuple[int, ...]] | tuple[int, ...] | None,
            ]
        ] = PDEStatioComponents(
            dyn_loss=(dyn_loss_fun, domain_batch, params, (0,)),
            norm_loss=(norm_loss_fun, norm_batch, params, (0,)),
            boundary_loss=(boundary_loss_fun, border_batch, params, (0,)),
            observations=(
                obs_loss_fun,
                obs_batch,
                params,
                ((0, 0),),
                batch.obs_batch_dict,
                self.obs_slice,
            ),
        )
        return all_funs_and_params


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
    dynamic_loss : tuple[PDENonStatio, ...] | PDENonStatio | None
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
    boundary_condition : BoundaryConditionAbstract | None
        A BoundaryCondition object implementing
        operator $\mathcal{B}[u](inputs)=f(x)$.
        Can be None
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
        **If using SPINN, `norm_weights` must be a scalar**.
    obs_slice : tuple[EllipsisType | slice, ...] | EllipsisType | slice | None, default=None
        slice object specifying the begininning/ending of the PINN output
        that is observed (this is then useful for multidim PINN). Default is None.
    """

    u: AbstractPINN
    dynamic_loss: tuple[PDENonStatio | None, ...]
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

    reduction_functions: ClassVar[PDENonStatioComponents[Callable]] = eqx.field(
        static=True,
        default=PDENonStatioComponents(
            dyn_loss=mean_sum_reduction_pytree,
            initial_condition=mean_sum_reduction,
            boundary_loss=lambda residual_all_facets: jax.tree.reduce(
                jnp.add,
                jax.tree.map(
                    mean_sum_reduction,
                    residual_all_facets,
                ),
                0.0,
            )  # NOTE this boundary component changes ! Outer sum is for the facets
            if residual_all_facets is not None
            else None,
            # TODO here compute mean only on subarrays of axis 1 (ie for a
            # timestamp)
            norm_loss=lambda res: jnp.mean(
                jnp.abs(
                    jnp.mean(
                        res,
                        axis=list(d + 1 for d in range(res.ndim - 1)),
                    )
                    - 1
                )
                ** 2
            )  # the outer mean() below is for the times stamps
            if res is not None
            else None,
            observations=mean_sum_reduction_pytree,
        ),
    )
    vmap_loss_fun: ClassVar[PDENonStatioComponents[Callable]] = eqx.field(
        static=True,
        default=PDENonStatioComponents(
            dyn_loss=vmap_loss_fun_classical,
            initial_condition=vmap_loss_fun_classical,
            boundary_loss=vmap_loss_fun_classical,
            norm_loss=vmap_loss_fun_classical,
            observations=vmap_loss_fun_observations,
        ),
    )

    def __init__(
        self,
        *,
        u: AbstractPINN,
        dynamic_loss: tuple[PDENonStatio, ...] | PDENonStatio | None,
        loss_weights: LossWeightsPDENonStatio | None = None,
        derivative_keys: DerivativeKeysPDENonStatio | None = None,
        initial_condition_fun: Callable[[Float[Array, " dimension"]], Array]
        | None = None,
        t0: int | float | Float[Array, " "] | None = None,
        max_norm_time_slices: int = 100,
        max_norm_samples_omega: int = 1000,
        params: Params[Array] | None = None,
        **kwargs: Any,  # this is arguments for super()
    ):
        self.u = u
        if loss_weights is None:
            self.loss_weights = LossWeightsPDENonStatio()
        else:
            self.loss_weights = loss_weights

        if derivative_keys is None:
            # be default we only take gradient wrt nn_params
            try:
                self.derivative_keys = DerivativeKeysPDENonStatio(params=params)
            except ValueError as exc:
                raise ValueError(
                    "Problem at derivative_keys initialization "
                    f"received {derivative_keys=} and {params=}"
                ) from exc
        else:
            self.derivative_keys = derivative_keys

        if not isinstance(dynamic_loss, tuple):
            self.dynamic_loss = (dynamic_loss,)
        else:
            self.dynamic_loss = dynamic_loss

        super().__init__(
            dynamic_loss=self.dynamic_loss,
            derivative_keys=self.derivative_keys,
            **kwargs,
        )

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

    def _get_normalization_loss_batch(
        self, batch: PDENonStatioBatch
    ) -> tuple[Array, Array]:
        batches = (
            batch.domain_batch[: self.max_norm_time_slices, 0:1],
            self.norm_samples[: self.max_norm_samples_omega],  # type: ignore -> cannot narrow a class attr
        )
        assert self.norm_weights is not None
        if isinstance(self.u, (PINN, HyperPINN)):
            # norm_weights is a array with n_norm_samples elements
            # we will tile it so that the norm_samples match the spatial points
            norm_weights = jnp.tile(
                self.norm_weights,
                reps=(batches[0].shape[0],)
                + tuple(1 for i in self.norm_weights.shape[1:]),
            )
            return (
                make_cartesian_product(
                    batches[0],
                    batches[1],
                ).reshape(batches[0].shape[0] * batches[1].shape[0], -1),
                norm_weights,
            )
        elif isinstance(self.u, SPINN):
            # norm_weights is nec. a scalar as no other case is implemented
            assert batches[1].shape[0] % batches[0].shape[0] == 0
            rep_t = batches[1].shape[0] // batches[0].shape[0]
            return (
                jnp.concatenate(
                    [jnp.repeat(batches[0], rep_t, axis=0), batches[1]], axis=-1
                ),
                self.norm_weights * jnp.ones_like(batches[1]),
            )
        else:
            raise ValueError(
                f"Bad type for u. Got {type(self.u)}, expected PINN or SPINN"
            )

    def evaluate_by_terms(
        self,
        batch: PDENonStatioBatch,
        params: Params[Array],
    ) -> PDENonStatioComponents[
        tuple[
            Callable | None,
            tuple[tuple[Array, Array] | None, ...]
            | tuple[Array | None, ...]
            | Array
            | None,
            Params[Array] | None,
            tuple[tuple[int, ...]] | tuple[int, ...] | None,
        ]
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
        # dynamic part
        domain_batch = batch.domain_batch
        dyn_loss_fun = self._get_dyn_loss_fun()

        # normalization part
        if self.norm_samples is not None:
            norm_batch = self._get_normalization_loss_batch(batch)
            norm_loss_fun = self._get_norm_loss_fun()
        else:
            norm_batch = None
            norm_loss_fun = None

        # boundary part
        border_batch = batch.border_batch
        boundary_loss_fun = self._get_boundary_loss_fun()

        # Observation mse
        if batch.obs_batch_dict is not None:
            obs_loss_fun = self._get_obs_loss_fun()
            obs_batch = tuple((b["pinn_in"], b["val"]) for b in batch.obs_batch_dict)
        else:
            obs_loss_fun = None
            obs_batch = None

        # initial condition
        if self.initial_condition_fun is not None:
            omega_initial_batch = batch.initial_batch
            initial_condition_fun: Callable[[Array, Params[Array]], Array] | None = (
                lambda b, p: initial_condition_apply(
                    self.u,
                    b,
                    _set_derivatives(p, self.derivative_keys.initial_condition),
                    self.initial_condition_fun,  # type: ignore
                    self.t0,
                )
            )
        else:
            omega_initial_batch = None
            initial_condition_fun = None

        all_funs_and_params: PDENonStatioComponents[
            tuple[
                Callable | None,
                tuple[tuple[Array, Array] | None, ...]
                | tuple[Array | None, ...]
                | Array
                | None,
                Params[Array] | None,
                tuple[tuple[int, ...]] | tuple[int, ...] | None,
                Any,
            ]
        ] = PDENonStatioComponents(
            dyn_loss=(dyn_loss_fun, domain_batch, params, (0,)),
            norm_loss=(norm_loss_fun, norm_batch, params, ((0, 0),)),
            boundary_loss=(boundary_loss_fun, border_batch, params, (0,)),
            initial_condition=(
                initial_condition_fun,
                omega_initial_batch,
                params,
                (0,),
            ),
            observations=(
                obs_loss_fun,
                obs_batch,
                params,
                ((0, 0),),
                batch.obs_batch_dict,
                self.obs_slice,
            ),
        )
        return all_funs_and_params
