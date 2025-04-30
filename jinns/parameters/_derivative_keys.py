"""
Formalize the data structure for the derivative keys
"""

from dataclasses import InitVar
from typing import Literal
from jaxtyping import Array
import jax
import equinox as eqx

from jinns.parameters._params import Params


def _get_masked_parameters(
    derivative_mask_str: str, params: Params[Array]
) -> Params[bool]:
    """
    Creates the Params object with True values where we want to differentiate
    """
    # start with a params object with True everywhere. We will update to False
    # for parameters wrt which we do want not to differentiate the loss
    diff_params = jax.tree.map(
        lambda x: True,
        params,
        is_leaf=lambda x: isinstance(x, eqx.Module)
        and not isinstance(x, Params),  # do not travers nn_params, more
        # granularity could be imagined here, in the future
    )
    if derivative_mask_str == "both":
        return diff_params
    if derivative_mask_str == "eq_params":
        return eqx.tree_at(lambda p: p.nn_params, diff_params, False)
    if derivative_mask_str == "nn_params":
        return eqx.tree_at(
            lambda p: p.eq_params,
            diff_params,
            jax.tree.map(lambda x: False, params.eq_params),
        )
    raise ValueError(
        "Bad value for DerivativeKeys. Got "
        f'{derivative_mask_str}, expected "both", "nn_params" or '
        ' "eq_params"'
    )


class DerivativeKeysODE(eqx.Module):
    """
    A class that specifies with repect to which parameter(s) each term of the
    loss is differentiated. For example, you can specify that the
    [`DynamicLoss`][jinns.loss.DynamicLoss] should be differentiated both with
    respect to the neural network parameters *and* the equation parameters, or only some of them.

    To do so, user can either use strings or a `Params` object
    with PyTree structure matching the parameters of the problem at
    hand, and booleans indicating if gradient is to be taken or not. Internally,
    a `jax.lax.stop_gradient()` is appropriately set to each `True` node when
    computing each loss term.

    !!! note

         1. For unspecified loss term, the default is to differentiate with
        respect to `"nn_params"` only.
         2. No granularity inside `Params.nn_params` is currently supported.
         3. Note that the main Params object of the problem is mandatory if initialization via `from_str()`.

    A typical specification is of the form:
    ```python
    Params(
        nn_params=True | False,
        eq_params={
            "alpha":True | False,
            "beta":True | False,
            ...
        }
    )
    ```

    Parameters
    ----------
    dyn_loss : Params[bool] | None, default=None
        Tell wrt which node of `Params` we will differentiate the
        dynamic loss. To do so, the fields of `Params` contain True (if
        differentiation) or False (if no differentiation).
    observations : Params[bool] | None, default=None
        Tell wrt which parameters among Params we will differentiate the
        observation loss. To do so, the fields of Params contain True (if
        differentiation) or False (if no differentiation).
    initial_condition : Params[bool] | None, default=None
        Tell wrt which parameters among Params we will differentiate the
        initial condition loss. To do so, the fields of Params contain True (if
        differentiation) or False (if no differentiation).
    params : InitVar[Params[Array]], default=None
        The main Params object of the problem. It is required
        if some terms are unspecified (None). This is because, jinns cannot
        infer the content of `Params.eq_params`.
    """

    dyn_loss: Params[bool] | None = eqx.field(kw_only=True, default=None)
    observations: Params[bool] | None = eqx.field(kw_only=True, default=None)
    initial_condition: Params[bool] | None = eqx.field(kw_only=True, default=None)

    params: InitVar[Params[Array] | None] = eqx.field(kw_only=True, default=None)

    def __post_init__(self, params: Params[Array] | None = None):
        if params is None and (
            self.dyn_loss is None
            or self.observations is None
            or self.initial_condition is None
        ):
            raise ValueError(
                "params cannot be None since at least one loss "
                "term has an undefined derivative key Params PyTree"
            )
        if self.dyn_loss is None:
            if params is None:
                raise ValueError("self.dyn_loss is None, hence params should be passed")
            self.dyn_loss = _get_masked_parameters("nn_params", params)
        if self.observations is None:
            if params is None:
                raise ValueError(
                    "self.observations is None, hence params should be passed"
                )
            self.observations = _get_masked_parameters("nn_params", params)
        if self.initial_condition is None:
            if params is None:
                raise ValueError(
                    "self.initial_condition is None, hence params should be passed"
                )
            self.initial_condition = _get_masked_parameters("nn_params", params)

    @classmethod
    def from_str(
        cls,
        params: Params[Array],
        dyn_loss: (
            Literal["nn_params", "eq_params", "both"] | Params[bool]
        ) = "nn_params",
        observations: (
            Literal["nn_params", "eq_params", "both"] | Params[bool]
        ) = "nn_params",
        initial_condition: (
            Literal["nn_params", "eq_params", "both"] | Params[bool]
        ) = "nn_params",
    ):
        """
        Construct the DerivativeKeysODE from strings. For each term of the
        loss, specify whether to differentiate wrt the neural network
        parameters, the equation parameters or both. The `Params` object, which
        contains the actual array of parameters must be passed to
        construct the fields with the appropriate PyTree structure.

        !!! note
            You can mix strings and `Params` if you need granularity.

        Parameters
        ----------
        params
            The actual Params object of the problem.
        dyn_loss
            Tell wrt which parameters among `"nn_params"`, `"eq_params"` or
            `"both"` we will differentiate the dynamic loss. Default is
            `"nn_params"`. Specifying a Params is also possible.
        observations
            Tell wrt which parameters among `"nn_params"`, `"eq_params"` or
            `"both"` we will differentiate the observations. Default is
            `"nn_params"`. Specifying a Params is also possible.
        initial_condition
            Tell wrt which parameters among `"nn_params"`, `"eq_params"` or
            `"both"` we will differentiate the initial condition. Default is
            `"nn_params"`. Specifying a Params is also possible.
        """
        return DerivativeKeysODE(
            dyn_loss=(
                _get_masked_parameters(dyn_loss, params)
                if isinstance(dyn_loss, str)
                else dyn_loss
            ),
            observations=(
                _get_masked_parameters(observations, params)
                if isinstance(observations, str)
                else observations
            ),
            initial_condition=(
                _get_masked_parameters(initial_condition, params)
                if isinstance(initial_condition, str)
                else initial_condition
            ),
        )


class DerivativeKeysPDEStatio(eqx.Module):
    """
    See [jinns.parameters.DerivativeKeysODE][].

    Parameters
    ----------
     dyn_loss : Params[bool] | None, default=None
        Tell wrt which parameters among Params we will differentiate the
        dynamic loss. To do so, the fields of Params contain True (if
        differentiation) or False (if no differentiation).
     observations : Params[bool] | None, default=None
        Tell wrt which parameters among Params we will differentiate the
        observation loss. To do so, the fields of Params contain True (if
        differentiation) or False (if no differentiation).
     boundary_loss : Params[bool] | None, default=None
        Tell wrt which parameters among Params we will differentiate the
        boundary loss. To do so, the fields of Params contain True (if
        differentiation) or False (if no differentiation).
     norm_loss : Params[bool] | None, default=None
        Tell wrt which parameters among Params we will differentiate the
        normalization loss. To do so, the fields of Params contain True (if
        differentiation) or False (if no differentiation).
     params : InitVar[Params[Array]], default=None
        The main Params object of the problem. It is required
        if some terms are unspecified (None). This is because, jinns cannot infer the
        content of `Params.eq_params`.
    """

    dyn_loss: Params[bool] | None = eqx.field(kw_only=True, default=None)
    observations: Params[bool] | None = eqx.field(kw_only=True, default=None)
    boundary_loss: Params[bool] | None = eqx.field(kw_only=True, default=None)
    norm_loss: Params[bool] | None = eqx.field(kw_only=True, default=None)

    params: InitVar[Params[Array] | None] = eqx.field(kw_only=True, default=None)

    def __post_init__(self, params: Params[Array] | None = None):
        if self.dyn_loss is None:
            if params is None:
                raise ValueError("self.dyn_loss is None, hence params should be passed")
            self.dyn_loss = _get_masked_parameters("nn_params", params)
        if self.observations is None:
            if params is None:
                raise ValueError(
                    "self.observations is None, hence params should be passed"
                )
            self.observations = _get_masked_parameters("nn_params", params)
        if self.boundary_loss is None:
            if params is None:
                raise ValueError(
                    "self.boundary_loss is None, hence params should be passed"
                )
            self.boundary_loss = _get_masked_parameters("nn_params", params)
        if self.norm_loss is None:
            if params is None:
                raise ValueError(
                    "self.norm_loss is None, hence params should be passed"
                )
            self.norm_loss = _get_masked_parameters("nn_params", params)

    @classmethod
    def from_str(
        cls,
        params: Params[Array],
        dyn_loss: (
            Literal["nn_params", "eq_params", "both"] | Params[bool]
        ) = "nn_params",
        observations: (
            Literal["nn_params", "eq_params", "both"] | Params[bool]
        ) = "nn_params",
        boundary_loss: (
            Literal["nn_params", "eq_params", "both"] | Params[bool]
        ) = "nn_params",
        norm_loss: (
            Literal["nn_params", "eq_params", "both"] | Params[bool]
        ) = "nn_params",
    ):
        """
        See [jinns.parameters.DerivativeKeysODE.from_str][].

        Parameters
        ----------
        params
            The actual Param object of the problem.
        dyn_loss
            Tell wrt which parameters among `"nn_params"`, `"eq_params"` or
            `"both"` we will differentiate the dynamic loss. Default is
            `"nn_params"`. Specifying a Params is also possible.
        observations
            Tell wrt which parameters among `"nn_params"`, `"eq_params"` or
            `"both"` we will differentiate the observations. Default is
            `"nn_params"`. Specifying a Params is also possible.
        boundary_loss
            Tell wrt which parameters among `"nn_params"`, `"eq_params"` or
            `"both"` we will differentiate the boundary loss. Default is
            `"nn_params"`. Specifying a Params is also possible.
        norm_loss
            Tell wrt which parameters among `"nn_params"`, `"eq_params"` or
            `"both"` we will differentiate the normalization loss. Default is
            `"nn_params"`. Specifying a Params is also possible.
        """
        return DerivativeKeysPDEStatio(
            dyn_loss=(
                _get_masked_parameters(dyn_loss, params)
                if isinstance(dyn_loss, str)
                else dyn_loss
            ),
            observations=(
                _get_masked_parameters(observations, params)
                if isinstance(observations, str)
                else observations
            ),
            boundary_loss=(
                _get_masked_parameters(boundary_loss, params)
                if isinstance(boundary_loss, str)
                else boundary_loss
            ),
            norm_loss=(
                _get_masked_parameters(norm_loss, params)
                if isinstance(norm_loss, str)
                else norm_loss
            ),
        )


class DerivativeKeysPDENonStatio(DerivativeKeysPDEStatio):
    """
    See [jinns.parameters.DerivativeKeysODE][].

    Parameters
    ----------
    dyn_loss : Params[bool] | None, default=None
        Tell wrt which parameters among Params we will differentiate the
        dynamic loss. To do so, the fields of Params contain True (if
        differentiation) or False (if no differentiation).
    observations : Params[bool] | None, default=None
        Tell wrt which parameters among Params we will differentiate the
        observation loss. To do so, the fields of Params contain True (if
        differentiation) or False (if no differentiation).
    boundary_loss : Params[bool] | None, default=None
        Tell wrt which parameters among Params we will differentiate the
        boundary loss. To do so, the fields of Params contain True (if
        differentiation) or False (if no differentiation).
    norm_loss : Params[bool] | None, default=None
        Tell wrt which parameters among Params we will differentiate the
        normalization loss. To do so, the fields of Params contain True (if
        differentiation) or False (if no differentiation).
    initial_condition : Params[bool] | None, default=None
        Tell wrt which parameters among Params we will differentiate the
        initial_condition loss. To do so, the fields of Params contain True (if
        differentiation) or False (if no differentiation).
    params : InitVar[Params[Array]], default=None
        The main Params object of the problem. It is required
        if some terms are unspecified (None). This is because, jinns cannot infer the
        content of `Params.eq_params`.
    """

    initial_condition: Params[bool] | None = eqx.field(kw_only=True, default=None)

    def __post_init__(self, params: Params[Array] | None = None):
        super().__post_init__(params=params)
        if self.initial_condition is None:
            if params is None:
                raise ValueError(
                    "self.initial_condition is None, hence params should be passed"
                )
            self.initial_condition = _get_masked_parameters("nn_params", params)

    @classmethod
    def from_str(
        cls,
        params: Params[Array],
        dyn_loss: (
            Literal["nn_params", "eq_params", "both"] | Params[bool]
        ) = "nn_params",
        observations: (
            Literal["nn_params", "eq_params", "both"] | Params[bool]
        ) = "nn_params",
        boundary_loss: (
            Literal["nn_params", "eq_params", "both"] | Params[bool]
        ) = "nn_params",
        norm_loss: (
            Literal["nn_params", "eq_params", "both"] | Params[bool]
        ) = "nn_params",
        initial_condition: (
            Literal["nn_params", "eq_params", "both"] | Params[bool]
        ) = "nn_params",
    ):
        """
        See [jinns.parameters.DerivativeKeysODE.from_str][].

        Parameters
        ----------
        params
            The actual Params object of the problem.
        dyn_loss
            Tell wrt which parameters among `"nn_params"`, `"eq_params"` or
            `"both"` we will differentiate the dynamic loss. Default is
            `"nn_params"`. Specifying a Params is also possible.
        observations
            Tell wrt which parameters among `"nn_params"`, `"eq_params"` or
            `"both"` we will differentiate the observations. Default is
            `"nn_params"`. Specifying a Params is also possible.
        boundary_loss
            Tell wrt which parameters among `"nn_params"`, `"eq_params"` or
            `"both"` we will differentiate the boundary loss. Default is
            `"nn_params"`. Specifying a Params is also possible.
        norm_loss
            Tell wrt which parameters among `"nn_params"`, `"eq_params"` or
            `"both"` we will differentiate the normalization loss. Default is
            `"nn_params"`. Specifying a Params is also possible.
        initial_condition
            Tell wrt which parameters among `"nn_params"`, `"eq_params"` or
            `"both"` we will differentiate the initial_condition loss. Default is
            `"nn_params"`. Specifying a Params is also possible.
        """
        return DerivativeKeysPDENonStatio(
            dyn_loss=(
                _get_masked_parameters(dyn_loss, params)
                if isinstance(dyn_loss, str)
                else dyn_loss
            ),
            observations=(
                _get_masked_parameters(observations, params)
                if isinstance(observations, str)
                else observations
            ),
            boundary_loss=(
                _get_masked_parameters(boundary_loss, params)
                if isinstance(boundary_loss, str)
                else boundary_loss
            ),
            norm_loss=(
                _get_masked_parameters(norm_loss, params)
                if isinstance(norm_loss, str)
                else norm_loss
            ),
            initial_condition=(
                _get_masked_parameters(initial_condition, params)
                if isinstance(initial_condition, str)
                else initial_condition
            ),
        )


def _set_derivatives(params, derivative_keys):
    """
    We construct an eqx.Module with the fields of derivative_keys, each field
    has a copy of the params with appropriate derivatives set
    """

    def _set_derivatives_(params_, derivative_mask):
        """
        The next lines put a stop_gradient around the fields that do not
        differentiate the loss term
        **Note:** **No granularity inside `Params.nn_params` is currently
        supported.**
        This means a typical Params specification is of the form:
        `Params(nn_params=True | False, eq_params={"alpha":True | False,
        "beta":True | False})`.
        """
        return jax.tree.map(
            lambda p, d: jax.lax.cond(d, lambda p: p, jax.lax.stop_gradient, p),
            params_,
            derivative_mask,
            is_leaf=lambda x: isinstance(x, eqx.Module)
            and not isinstance(x, Params),  # do not travers nn_params, more
            # granularity could be imagined here, in the future
        )

    return _set_derivatives_(params, derivative_keys)
