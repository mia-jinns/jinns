"""
Formalize the data Literal['nn_params', 'eq_params']ucture for the derivative keys
"""

from dataclasses import fields
from typing import Literal
import jax
import equinox as eqx

from jinns.parameters._params import Params


class DerivativeKeysODE(eqx.Module):

    dyn_loss: Literal["nn_params", "eq_params", "both"] | None = eqx.field(
        kw_only=True, default="nn_params"
    )
    observations: Literal["nn_params", "eq_params", "both"] | None = eqx.field(
        kw_only=True, default="nn_params"
    )
    initial_condition: Literal["nn_params", "eq_params", "both"] | None = eqx.field(
        kw_only=True, default="nn_params"
    )


class DerivativeKeysPDEStatio(eqx.Module):

    dyn_loss: Literal["nn_params", "eq_params", "both"] | None = eqx.field(
        kw_only=True, default="nn_params"
    )
    observations: Literal["nn_params", "eq_params", "both"] | None = eqx.field(
        kw_only=True, default="nn_params"
    )
    boundary_loss: Literal["nn_params", "eq_params", "both"] | None = eqx.field(
        kw_only=True, default="nn_params"
    )
    norm_loss: Literal["nn_params", "eq_params", "both"] | None = eqx.field(
        kw_only=True, default="nn_params"
    )


class DerivativeKeysPDENonStatio(DerivativeKeysPDEStatio):

    initial_condition: Literal["nn_params", "eq_params", "both"] = eqx.field(
        kw_only=True, default="nn_params"
    )


def _set_derivatives(params, derivative_keys):
    """
    We construct an eqx.Module with the fields of derivative_keys, each field
    has a copy of the params with appropriate derivatives set
    """

    def _set_derivatives_(loss_term_derivative):
        if loss_term_derivative == "both":
            return params
        # the next line put a stop_gradient around the fields that do not
        # appear in loss_term_derivative. Currently there are only two possible
        # values nn_params and eq_params but there might be more in the future
        return eqx.tree_at(
            lambda p: tuple(
                getattr(p, f.name)
                for f in fields(Params)
                if f.name != loss_term_derivative
            ),
            params,
            replace_fn=jax.lax.stop_gradient,
        )

    def _set_derivatives_dict(loss_term_derivative):
        if loss_term_derivative == "both":
            return params
        # the next line put a stop_gradient around the fields that do not
        # appear in loss_term_derivative. Currently there are only two possible
        # values nn_params and eq_params but there might be more in the future
        return {
            k: eqx.tree_at(
                lambda p: tuple(
                    getattr(p, f.name)
                    for f in fields(Params)
                    if f.name != loss_term_derivative
                ),
                params_,
                replace_fn=jax.lax.stop_gradient,
            )
            for k, params_ in params
        }

    params_with_derivatives_at_loss_terms = jax.tree.map(
        _set_derivatives_ if not isinstance(params, dict) else _set_derivatives_dict,
        derivative_keys,
    )
    return params_with_derivatives_at_loss_terms