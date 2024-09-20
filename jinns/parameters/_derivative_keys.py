"""
Formalize the data structure for the derivative keys
"""

from dataclasses import fields
from typing import Literal
import jax
import equinox as eqx

from jinns.parameters._params import Params


class DerivativeKeysODE(eqx.Module):
    # we use static = True because all fields are string, hence should be
    # invisible by JAX transforms (JIT, etc.)
    dyn_loss: Literal["nn_params", "eq_params", "both"] | None = eqx.field(
        kw_only=True, default="nn_params", static=True
    )
    observations: Literal["nn_params", "eq_params", "both"] | None = eqx.field(
        kw_only=True, default="nn_params", static=True
    )
    initial_condition: Literal["nn_params", "eq_params", "both"] | None = eqx.field(
        kw_only=True, default="nn_params", static=True
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

    if not isinstance(params, dict):
        return _set_derivatives_(derivative_keys)
    else:
        return _set_derivatives_dict(derivative_keys)
