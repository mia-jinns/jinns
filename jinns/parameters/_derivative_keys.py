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

    def _set_derivatives_(params_, loss_term_derivative):
        ## If loss_term_derivative is a string
        if loss_term_derivative == "both":
            return params_
        # the next line put a stop_gradient around the fields that do not
        # appear in loss_term_derivative. Currently there are only two possible
        # values "nn_params" and "eq_params" but there might be more in the future
        if isinstance(loss_term_derivative, str):
            return eqx.tree_at(
                lambda p: tuple(
                    getattr(p, f.name)
                    for f in fields(Params)
                    if f.name != loss_term_derivative
                ),
                params_,
                replace_fn=jax.lax.stop_gradient,
            )
        ## If loss_term_derivative is a Params, to give maximal granularity to
        # the user to control which term is differentiated
        if isinstance(loss_term_derivative, Params):
            return jax.tree.map(
                lambda p, d: p if d else jax.lax.stop_gradient(p),
                params_,
                loss_term_derivative,
                is_leaf=lambda x: isinstance(x, eqx.Module)
                and not isinstance(x, Params),
            )
        raise ValueError(
            "Wrong value for loss_term_derivative, it should "
            "be either a string or a Params object with the same "
            "structure as Params with True values where derivatives are"
            " taken and False otherwise"
        )

    def _set_derivatives_dict(params_, loss_term_derivative):
        if loss_term_derivative == "both":
            return params_
        return {
            k: _set_derivatives_(params__, loss_term_derivative)
            for k, params__ in params_
        }

    if not isinstance(params, dict):
        return _set_derivatives_(params, derivative_keys)
    else:
        return _set_derivatives_dict(params, derivative_keys)
