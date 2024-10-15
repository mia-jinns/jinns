"""
Formalize the data structure for the derivative keys
"""

from functools import partial
from dataclasses import fields, InitVar
from typing import Literal
import jax
import equinox as eqx

from jinns.parameters._params import Params, ParamsDict


def _get_masked_parameters(
    derivative_mask_str: str, params: Params | ParamsDict
) -> Params | ParamsDict:
    """
    Creates the Params object with True values where we want to differentiate
    """
    if isinstance(params, Params):
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
    elif isinstance(params, ParamsDict):
        # do not travers nn_params, more
        # granularity could be imagined here, in the future
        diff_params = ParamsDict(
            nn_params=True, eq_params=jax.tree.map(lambda x: True, params.eq_params)
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

    else:
        raise ValueError(
            f"Bad value for params. Got {type(params)}, expected Params "
            " or ParamsDict"
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
         3. Note that the main Params or ParamsDict object of the problem is mandatory if initialization via `from_str()`.

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
    dyn_loss : Params | ParamsDict | None, default=None
        Tell wrt which node of `Params` we will differentiate the
        dynamic loss. To do so, the fields of `Params` contain True (if
        differentiation) or False (if no differentiation).
    observations : Params | ParamsDict | None, default=None
        Tell wrt which parameters among Params we will differentiate the
        observation loss. To do so, the fields of Params contain True (if
        differentiation) or False (if no differentiation).
    initial_condition : Params | ParamsDict | None, default=None
        Tell wrt which parameters among Params we will differentiate the
        initial condition loss. To do so, the fields of Params contain True (if
        differentiation) or False (if no differentiation).
    params : InitVar[Params | ParamsDict], default=None
        The main Params object of the problem. It is required
        if some terms are unspecified (None). This is because, jinns cannot
        infer the content of `Params.eq_params`.
    """

    dyn_loss: Params | ParamsDict | None = eqx.field(kw_only=True, default=None)
    observations: Params | ParamsDict | None = eqx.field(kw_only=True, default=None)
    initial_condition: Params | ParamsDict | None = eqx.field(
        kw_only=True, default=None
    )

    params: InitVar[Params | ParamsDict] = eqx.field(kw_only=True, default=None)

    def __post_init__(self, params=None):
        if self.dyn_loss is None:
            try:
                self.dyn_loss = _get_masked_parameters("nn_params", params)
            except AttributeError:
                raise ValueError(
                    "self.dyn_loss is None, hence params should be " "passed"
                )
        if self.observations is None:
            try:
                self.observations = _get_masked_parameters("nn_params", params)
            except AttributeError:
                raise ValueError(
                    "self.observations is None, hence params should be " "passed"
                )
        if self.initial_condition is None:
            try:
                self.initial_condition = _get_masked_parameters("nn_params", params)
            except AttributeError:
                raise ValueError(
                    "self.initial_condition is None, hence params should be " "passed"
                )

    @classmethod
    def from_str(
        cls,
        params: Params | ParamsDict,
        dyn_loss: (
            Literal["nn_params", "eq_params", "both"] | Params | ParamsDict
        ) = "nn_params",
        observations: (
            Literal["nn_params", "eq_params", "both"] | Params | ParamsDict
        ) = "nn_params",
        initial_condition: (
            Literal["nn_params", "eq_params", "both"] | Params | ParamsDict
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
            The actual Params or ParamsDict object of the problem.
        dyn_loss
            Tell wrt which parameters among `"nn_params"`, `"eq_params"` or
            `"both"` we will differentiate the dynamic loss. Default is
            `"nn_params"`. Specifying a Params or ParamsDict is also possible.
        observations
            Tell wrt which parameters among `"nn_params"`, `"eq_params"` or
            `"both"` we will differentiate the observations. Default is
            `"nn_params"`. Specifying a Params or ParamsDict is also possible.
        initial_condition
            Tell wrt which parameters among `"nn_params"`, `"eq_params"` or
            `"both"` we will differentiate the initial condition. Default is
            `"nn_params"`. Specifying a Params or ParamsDict is also possible.
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
     dyn_loss : Params | ParamsDict | None, default=None
        Tell wrt which parameters among Params we will differentiate the
        dynamic loss. To do so, the fields of Params contain True (if
        differentiation) or False (if no differentiation).
     observations : Params | ParamsDict | None, default=None
        Tell wrt which parameters among Params we will differentiate the
        observation loss. To do so, the fields of Params contain True (if
        differentiation) or False (if no differentiation).
     boundary_loss : Params | ParamsDict | None, default=None
        Tell wrt which parameters among Params we will differentiate the
        boundary loss. To do so, the fields of Params contain True (if
        differentiation) or False (if no differentiation).
     norm_loss : Params | ParamsDict | None, default=None
        Tell wrt which parameters among Params we will differentiate the
        normalization loss. To do so, the fields of Params contain True (if
        differentiation) or False (if no differentiation).
     params : InitVar[Params | ParamsDict], default=None
        The main Params object of the problem. It is required
        if some terms are unspecified (None). This is because, jinns cannot infer the
        content of `Params.eq_params`.
    """

    dyn_loss: Params | ParamsDict | None = eqx.field(kw_only=True, default=None)
    observations: Params | ParamsDict | None = eqx.field(kw_only=True, default=None)
    boundary_loss: Params | ParamsDict | None = eqx.field(kw_only=True, default=None)
    norm_loss: Params | ParamsDict | None = eqx.field(kw_only=True, default=None)

    params: InitVar[Params | ParamsDict] = eqx.field(kw_only=True, default=None)

    def __post_init__(self, params=None):
        if self.dyn_loss is None:
            try:
                self.dyn_loss = _get_masked_parameters("nn_params", params)
            except AttributeError:
                raise ValueError("self.dyn_loss is None, hence params should be passed")
        if self.observations is None:
            try:
                self.observations = _get_masked_parameters("nn_params", params)
            except AttributeError:
                raise ValueError(
                    "self.observations is None, hence params should be passed"
                )
        if self.boundary_loss is None:
            try:
                self.boundary_loss = _get_masked_parameters("nn_params", params)
            except AttributeError:
                raise ValueError(
                    "self.boundary_loss is None, hence params should be passed"
                )
        if self.norm_loss is None:
            try:
                self.norm_loss = _get_masked_parameters("nn_params", params)
            except AttributeError:
                raise ValueError(
                    "self.norm_loss is None, hence params should be passed"
                )

    @classmethod
    def from_str(
        cls,
        params: Params | ParamsDict,
        dyn_loss: (
            Literal["nn_params", "eq_params", "both"] | Params | ParamsDict
        ) = "nn_params",
        observations: (
            Literal["nn_params", "eq_params", "both"] | Params | ParamsDict
        ) = "nn_params",
        boundary_loss: (
            Literal["nn_params", "eq_params", "both"] | Params | ParamsDict
        ) = "nn_params",
        norm_loss: (
            Literal["nn_params", "eq_params", "both"] | Params | ParamsDict
        ) = "nn_params",
    ):
        """
        See [jinns.parameters.DerivativeKeysODE.from_str][].

        Parameters
        ----------
        params
            The actual Param or ParamsDict object of the problem.
        dyn_loss
            Tell wrt which parameters among `"nn_params"`, `"eq_params"` or
            `"both"` we will differentiate the dynamic loss. Default is
            `"nn_params"`. Specifying a Params or ParamsDict is also possible.
        observations
            Tell wrt which parameters among `"nn_params"`, `"eq_params"` or
            `"both"` we will differentiate the observations. Default is
            `"nn_params"`. Specifying a Params or ParamsDict is also possible.
        boundary_loss
            Tell wrt which parameters among `"nn_params"`, `"eq_params"` or
            `"both"` we will differentiate the boundary loss. Default is
            `"nn_params"`. Specifying a Params or ParamsDict is also possible.
        norm_loss
            Tell wrt which parameters among `"nn_params"`, `"eq_params"` or
            `"both"` we will differentiate the normalization loss. Default is
            `"nn_params"`. Specifying a Params or ParamsDict is also possible.
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
    dyn_loss : Params | ParamsDict | None, default=None
        Tell wrt which parameters among Params we will differentiate the
        dynamic loss. To do so, the fields of Params contain True (if
        differentiation) or False (if no differentiation).
    observations : Params | ParamsDict | None, default=None
        Tell wrt which parameters among Params we will differentiate the
        observation loss. To do so, the fields of Params contain True (if
        differentiation) or False (if no differentiation).
    boundary_loss : Params | ParamsDict | None, default=None
        Tell wrt which parameters among Params we will differentiate the
        boundary loss. To do so, the fields of Params contain True (if
        differentiation) or False (if no differentiation).
    norm_loss : Params | ParamsDict | None, default=None
        Tell wrt which parameters among Params we will differentiate the
        normalization loss. To do so, the fields of Params contain True (if
        differentiation) or False (if no differentiation).
    initial_condition : Params | ParamsDict | None, default=None
        Tell wrt which parameters among Params we will differentiate the
        initial_condition loss. To do so, the fields of Params contain True (if
        differentiation) or False (if no differentiation).
    params : InitVar[Params | ParamsDict], default=None
        The main Params object of the problem. It is required
        if some terms are unspecified (None). This is because, jinns cannot infer the
        content of `Params.eq_params`.
    """

    initial_condition: Params | ParamsDict | None = eqx.field(
        kw_only=True, default=None
    )

    def __post_init__(self, params=None):
        super().__post_init__(params=params)
        if self.initial_condition is None:
            try:
                self.initial_condition = _get_masked_parameters("nn_params", params)
            except AttributeError:
                raise ValueError(
                    "self.initial_condition is None, hence params should be passed"
                )

    @classmethod
    def from_str(
        cls,
        params: Params | ParamsDict,
        dyn_loss: (
            Literal["nn_params", "eq_params", "both"] | Params | ParamsDict
        ) = "nn_params",
        observations: (
            Literal["nn_params", "eq_params", "both"] | Params | ParamsDict
        ) = "nn_params",
        boundary_loss: (
            Literal["nn_params", "eq_params", "both"] | Params | ParamsDict
        ) = "nn_params",
        norm_loss: (
            Literal["nn_params", "eq_params", "both"] | Params | ParamsDict
        ) = "nn_params",
        initial_condition: (
            Literal["nn_params", "eq_params", "both"] | Params | ParamsDict
        ) = "nn_params",
    ):
        """
        See [jinns.parameters.DerivativeKeysODE.from_str][].

        Parameters
        ----------
        params
            The actual Params | ParamsDict object of the problem.
        dyn_loss
            Tell wrt which parameters among `"nn_params"`, `"eq_params"` or
            `"both"` we will differentiate the dynamic loss. Default is
            `"nn_params"`. Specifying a Params or ParamsDict is also possible.
        observations
            Tell wrt which parameters among `"nn_params"`, `"eq_params"` or
            `"both"` we will differentiate the observations. Default is
            `"nn_params"`. Specifying a Params or ParamsDict is also possible.
        boundary_loss
            Tell wrt which parameters among `"nn_params"`, `"eq_params"` or
            `"both"` we will differentiate the boundary loss. Default is
            `"nn_params"`. Specifying a Params or ParamsDict is also possible.
        norm_loss
            Tell wrt which parameters among `"nn_params"`, `"eq_params"` or
            `"both"` we will differentiate the normalization loss. Default is
            `"nn_params"`. Specifying a Params or ParamsDict is also possible.
        initial_condition
            Tell wrt which parameters among `"nn_params"`, `"eq_params"` or
            `"both"` we will differentiate the initial_condition loss. Default is
            `"nn_params"`. Specifying a Params or ParamsDict is also possible.
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

    def _set_derivatives_ParamsDict(params_, derivative_mask):
        """
        The next lines put a stop_gradient around the fields that do not
        differentiate the loss term
        **Note:** **No granularity inside `ParamsDict.nn_params` is currently
        supported.**
        This means a typical Params specification is of the form:
        `ParamsDict(nn_params=True | False, eq_params={"0":{"alpha":True | False,
        "beta":True | False}}, "1":{"alpha":True | False, "beta":True | False}})`.
        """
        # a ParamsDict object is reconstructed by hand since we do not want to
        # traverse nn_params, for now...
        return ParamsDict(
            nn_params=jax.lax.cond(
                derivative_mask.nn_params,
                lambda p: p,
                jax.lax.stop_gradient,
                params_.nn_params,
            ),
            eq_params=jax.tree.map(
                lambda p, d: jax.lax.cond(d, lambda p: p, jax.lax.stop_gradient, p),
                params_.eq_params,
                derivative_mask.eq_params,
            ),
        )

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

    if isinstance(params, ParamsDict):
        return _set_derivatives_ParamsDict(params, derivative_keys)
    return _set_derivatives_(params, derivative_keys)
