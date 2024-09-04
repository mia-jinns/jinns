"""
Formalize the data Literal['nn_params', 'eq_params']ucture for the derivative keys
"""

from typing import Literal
import jax
import equinox as eqx


class DerivativeKeysODE(eqx.Module):

    dyn_loss: Literal["nn_params", "eq_params", "both"] = eqx.field(
        kw_only=True, default="nn_params"
    )
    observations: Literal["nn_params", "eq_params", "both"] = eqx.field(
        kw_only=True, default="nn_params"
    )
    initial_condition: Literal["nn_params", "eq_params", "both"] = eqx.field(
        kw_only=True, default="nn_params"
    )


class DerivativeKeysPDEStatio(eqx.Module):

    dyn_loss: Literal["nn_params", "eq_params", "both"] = eqx.field(
        kw_only=True, default="nn_params"
    )
    observations: Literal["nn_params", "eq_params", "both"] = eqx.field(
        kw_only=True, default="nn_params"
    )
    boundary_loss: Literal["nn_params", "eq_params", "both"] = eqx.field(
        kw_only=True, default="nn_params"
    )
    norm_loss: Literal["nn_params", "eq_params", "both"] = eqx.field(
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
        return eqx.tree_at(
            lambda p: (
                getattr(p, loss_term_derivative)
                if (loss_term_derivative != "both")
                else ((p.nn_params, p.eq_params))
            ),
            params,
            replace_fn=jax.lax.stop_gradient,
        )

    params_with_derivatives_at_loss_terms = jax.tree.map(
        _set_derivatives_, derivative_keys
    )
    return params_with_derivatives_at_loss_terms
