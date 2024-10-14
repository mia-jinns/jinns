"""
Formalize the data structure for the derivative keys
"""

from functools import partial
from dataclasses import fields, InitVar
from typing import Literal
import jax
import equinox as eqx

from jinns.parameters._params import Params


def _get_Params(differentiate_wrt_str: str, params: Params) -> Params:
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
    if differentiate_wrt_str == "both":
        return diff_params
    if differentiate_wrt_str == "eq_params":
        return eqx.tree_at(lambda p: p.nn_params, diff_params, False)
    if differentiate_wrt_str == "nn_params":
        return eqx.tree_at(
            lambda p: p.eq_params,
            diff_params,
            jax.tree.map(lambda x: False, params.eq_params),
        )
    raise ValueError(
        "Bad value for DerivativeKeys. Got "
        f'{differentiate_wrt_str}, expected "both", "nn_params" or '
        ' "eq_params"'
    )


class DerivativeKeysODE(eqx.Module):
    """
    A class that specified wrt which parameter(s) each term of the loss is
    differentiated. In the most general case, for each loss term we specifiy a
    Params object with True if the loss is diff wrt that term and False
    otherwise. Another way and more convenient way to initialize is by passing
    strings. See the `DerivativeKeysODE.from_str()` class method.

    In both case of initialization, the user can skip the specification for a
    term, leading to a default differentiate of the latter wrt `"nn_params"`.

    **Note:** **No granularity inside `Params.nn_params` is currently
    supported.**
    This means a typical Params specification is of the form:
    `Params(nn_params=True | False, eq_params={"alpha":True | False,
    "beta":True | False})`.

    Note that the main Params object of the problem is mandatory passed if
    initialization via `from_str()`. It is required in the other initialization
    if some terms are unspecified (None). This is because, jinns cannot infer the
    content of `Params.eq_params`. **Unspecified terms will be initialized with
    the rule `"nn_params"`**.

    Parameters
    ----------
    dyn_loss : Params | None, default=None
        Tell wrt which parameters among Params we will differentiate the
        dynamic loss. To do so, the fields of Params contain True (if
        differentiation) or False (if no differentiation). See note above.
    observations : Params | None, default=None
        Tell wrt which parameters among Params we will differentiate the
        observation loss. To do so, the fields of Params contain True (if
        differentiation) or False (if no differentiation). See note above.
    initial_condition : Params | None, default=None
        Tell wrt which parameters among Params we will differentiate the
        initial condition loss. To do so, the fields of Params contain True (if
        differentiation) or False (if no differentiation). See note above.
    params : InitVar[Params], default=None
        The main Params object of the problem. It is required
        if some terms are unspecified (None). This is because, jinns cannot infer the
        content of `Params.eq_params`.
    """

    # Not that the Params structure below only contain boolean, hence we can
    # and should treat it as static for JIT compilation
    dyn_loss: Params | None = eqx.field(kw_only=True, default=None)
    observations: Params | None = eqx.field(kw_only=True, default=None)
    initial_condition: Params | None = eqx.field(kw_only=True, default=None)

    params: InitVar[Params] = eqx.field(default=None)

    def __post_init__(self, params):
        if self.dyn_loss is None:
            try:
                self.dyn_loss = _get_Params("nn_params", params)
            except AttributeError:
                raise ValueError(
                    "self.dyn_loss is None, hence params should be " "passed"
                )
        if self.observations is None:
            try:
                self.observations = _get_Params("nn_params", params)
            except AttributeError:
                raise ValueError(
                    "self.observations is None, hence params should be " "passed"
                )
        if self.initial_condition is None:
            try:
                self.initial_condition = _get_Params("nn_params", params)
            except AttributeError:
                raise ValueError(
                    "self.initial_condition is None, hence params should be " "passed"
                )

    @classmethod
    def from_str(
        cls,
        params: Params,
        dyn_loss: Literal["nn_params", "eq_params", "both"] = "nn_params",
        observations: Literal["nn_params", "eq_params", "both"] = "nn_params",
        initial_condition: Literal["nn_params", "eq_params", "both"] = "nn_params",
    ):
        """
        Construct the DerivativeKeysODE from strings. For each term of the
        loss, specify whether to differentiate wrt the NN parameters, the equation
        parameters or both. The Params object, which contains the actual array of
        parameters must be passed to help construct the fields.

        Note that to have more granularity over the derivations, the natural
        `__init__` method should be used. The latter accepts Params object with
        boolean values at each of the fields.

        Parameters
        ----------
        params
            The actual Params object of the problem.
        dyn_loss
            Tell wrt which parameters among `"nn_params"`, `"eq_params"` or
            `"both"` we will differentiate the dynamic loss. Default is
            `"nn_params"`
        observations
            Tell wrt which parameters among `"nn_params"`, `"eq_params"` or
            `"both"` we will differentiate the observations. Default is
            `"nn_params"`
        initial_condition
            Tell wrt which parameters among `"nn_params"`, `"eq_params"` or
            `"both"` we will differentiate the initial condition. Default is
            `"nn_params"`
        """
        return DerivativeKeysODE(
            dyn_loss=_get_Params(dyn_loss, params),
            observations=_get_Params(observations, params),
            initial_condition=_get_Params(initial_condition, params),
        )


class DerivativeKeysPDEStatio(eqx.Module):

    dyn_loss: Literal["nn_params", "eq_params", "both"] | None = eqx.field(
        kw_only=True, default="nn_params", static=True
    )
    observations: Literal["nn_params", "eq_params", "both"] | None = eqx.field(
        kw_only=True, default="nn_params", static=True
    )
    boundary_loss: Literal["nn_params", "eq_params", "both"] | None = eqx.field(
        kw_only=True, default="nn_params", static=True
    )
    norm_loss: Literal["nn_params", "eq_params", "both"] | None = eqx.field(
        kw_only=True, default="nn_params", static=True
    )


class DerivativeKeysPDENonStatio(DerivativeKeysPDEStatio):

    initial_condition: Literal["nn_params", "eq_params", "both"] = eqx.field(
        kw_only=True, default="nn_params", static=True
    )


def _set_derivatives(params, derivative_keys):
    """
    We construct an eqx.Module with the fields of derivative_keys, each field
    has a copy of the params with appropriate derivatives set
    """

    @partial(jax.jit, static_argnames=["differentiate_wrt"])
    def _set_derivatives_(params_, differentiate_wrt):
        """
        The next lines put a stop_gradient around the fields that do not
        differentiate the loss term
        **Note:** **No granularity inside `Params.nn_params` is currently
        supported.**
        This means a typical Params specification is of the form:
        `Params(nn_params=True | False, eq_params={"alpha":True | False,
        "beta":True | False})`.
        """
        # return eqx.tree_at(
        #    lambda p: tuple(pi for pi, di in zip(p, differentiate_wrt) if di),
        #    params_,
        #    replace_fn=jax.lax.stop_gradient,
        #    is_leaf=lambda x: isinstance(x, eqx.Module)
        #    and not isinstance(x, Params),  # do not travers nn_params, more
        #    # granularity could be imagined here, in the future
        # )
        return jax.tree.map(
            lambda p, d: jax.lax.cond(
                d, lambda p: p, jax.lax.stop_gradient, p
            ),  # p if d else jax.lax.stop_gradient(p),
            params_,
            differentiate_wrt,
            is_leaf=lambda x: isinstance(x, eqx.Module)
            and not isinstance(x, Params),  # do not travers nn_params, more
            # granularity could be imagined here, in the future
        )

    def _set_derivatives_dict(params_, differentiate_wrt):
        return {
            k: _set_derivatives_(params__, differentiate_wrt) for k, params__ in params_
        }

    if not isinstance(params, dict):
        return _set_derivatives_(params, derivative_keys)
    return _set_derivatives_dict(params, derivative_keys)
