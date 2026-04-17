"""ssbfgs and ssbroyden implementation."""

# from collections.abc import Callable
# import functools
# from typing import Any, Optional, Union
from typing import Callable, NamedTuple

# import warnings
import jax
import jax.numpy as jnp
import equinox as eqx
import optax.tree
from optax._src import base, numerics

# from optax._src import factorized
from optax._src import linesearch as _linesearch

from jinns.parameters import Params
from jinns.nn._hyperpinn import _get_param_nb

# from optax._src import wrappers
# from optax.transforms import _clipping

# MaskOrFn = Optional[Union[Any, Callable[[base.Params], Any]]]
LINESEARCH_TYPE = base.GradientTransformationExtraArgs | base.GradientTransformation


class ScaleBySSBFGSState(NamedTuple):
    """State for SS-BFGS solver.

    Attributes:
    count: iteration of the algorithm.
    params: current parameters.
    updates: current updates.
    hk: current hessian latriw approximation.
    linesearch_state: current linesearch state.
    """

    count: jax.typing.ArrayLike
    params: optax.Params
    updates: optax.Params
    hk: jax.typing.ArrayLike
    linesearch_state: NamedTuple


def _self_scaled_bfgs_or_broyden(
    linesearch: LINESEARCH_TYPE | None = _linesearch.scale_by_zoom_linesearch(
        max_linesearch_steps=25,
        # stepsize_precision=1e-9,
        # approx_dec_rtol=1e-9,
        initial_guess_strategy="one",
    ),
    broyden: bool = False,
) -> base.GradientTransformationExtraArgs:
    r"""Scales updates by SS-BFGS."""

    def init_fn(params: base.Params) -> ScaleBySSBFGSState:
        # print("params.shape: ", params.shape)
        return ScaleBySSBFGSState(
            count=jnp.asarray(0, dtype=jnp.int32),
            params=optax.tree.zeros_like(params),
            updates=optax.tree.zeros_like(params),
            hk=jnp.eye(params.shape[0]),
            linesearch_state=linesearch.init(params),
        )

    def update_fn(
        grad_k: base.Updates,
        state: ScaleBySSBFGSState,
        theta_k: base.Params,
        value: jax.typing.ArrayLike,
        grad: base.Updates,
        value_fn: Callable[..., tuple[jax.typing.ArrayLike, base.Updates]],
        grad_fn: Callable[..., tuple[jax.typing.ArrayLike, base.Updates]],
        **extra_args_for_fn,
    ) -> tuple[base.Updates, ScaleBySSBFGSState]:
        # TODO PT to array and array to PT should be done here, so that we can
        # easily wrap the value_fn and grad_fn functions with a array to PT.
        # also we would not need the jinns wrapper anymore

        # numel = theta_k.shape[0]
        # jax.debug.print("numel: {}", numel)

        # jax.debug.print("state.hk: {}", state.hk)
        direction = -state.hk @ grad_k
        s_k, linesearch_state = linesearch.update(
            direction,
            state.linesearch_state,
            theta_k,
            value=value,
            grad=grad,
            value_fn=value_fn,
            **extra_args_for_fn,
        )

        # compute some values for next turn:
        alpha_k = linesearch_state.learning_rate

        theta_kp1 = optax.apply_updates(theta_k, s_k)

        # get the gradients at theta_kp1
        grad_kp1 = grad_fn(theta_kp1, **extra_args_for_fn)

        # s_k = theta_kp1 - theta_k
        y_k = grad_kp1 - grad_k
        Hkyk = state.hk @ y_k
        yk_dot_Hkyk = y_k @ Hkyk
        yk_dot_sk = y_k @ s_k
        v_k = jnp.sqrt(yk_dot_Hkyk) * (s_k / (yk_dot_sk) - Hkyk / yk_dot_Hkyk)

        # method ssbfgs
        tau_k = jnp.minimum(1.0, -yk_dot_sk / (alpha_k * (s_k @ grad_k)))
        phi_k = 1.0

        # method ssbroyden
        if broyden:
            numel = theta_k.shape[0]
            b_k = -alpha_k * (s_k @ grad_k) / yk_dot_sk
            h_k = yk_dot_Hkyk / yk_dot_sk
            a_k = h_k * b_k - 1.0
            c_k = jnp.sqrt(a_k / (a_k + 1.0))
            rhom_k = jnp.minimum(1.0, h_k * (1 - c_k))
            thetam_k = (rhom_k - 1) / a_k
            thetap_k = 1.0 / rhom_k
            th_k = jnp.maximum(thetam_k, jnp.minimum(thetap_k, (1.0 - b_k) / b_k))
            sigma_k = 1 + a_k * th_k
            sigma_k_pow = sigma_k ** (-1 / (numel - 1))

            tau_k = jnp.where(
                th_k > 0,
                tau_k * jnp.minimum(sigma_k_pow, 1.0 / th_k),
                jnp.minimum(tau_k * sigma_k_pow, sigma_k),
            )
            # jax.debug.print("th_k: {}", th_k)
            # jax.debug.print("tau_k: {}", tau_k)
            phi_k = (1 - th_k) / (1.0 + a_k * th_k)

        temp1 = (Hkyk[:, None] @ Hkyk[None, :]) / yk_dot_Hkyk
        temp2 = phi_k * (v_k[:, None] @ v_k[None, :])
        temp3 = (s_k[:, None] @ s_k[None, :]) / yk_dot_sk
        H_kp1 = (1.0 / tau_k) * (state.hk - temp1 + temp2) + temp3

        new_state = ScaleBySSBFGSState(
            count=numerics.safe_increment(state.count),
            params=theta_kp1,
            updates=s_k,
            hk=H_kp1,
            linesearch_state=linesearch_state,
        )
        return s_k, new_state

    return base.GradientTransformationExtraArgs(init_fn, update_fn)


# def ssbfgs(
#     linesearch: LINESEARCH_TYPE | None = _linesearch.scale_by_zoom_linesearch(
#         max_linesearch_steps=25,
#         # approx_dec_rtol=1e-9,
#         # stepsize_precision=1e-9,
#         initial_guess_strategy="one",
#     ),
# ) -> base.GradientTransformationExtraArgs:
#     """SS-BFGS optimizer."""
#
#     return scale_by_ssbfgs(linesearch)


def params_array_to_pytree(
    params_array,
    params,
):
    """Helper function for NGD: from matrix to PyTree representation of parameters.
    This function converts the raw matrix representation of the network
    trainable parameters into a `Params` object with the correct PyTree structure to
    be handled by optax for the updates.

    By default, the field `eq_params` is filled with zeros, as NGD is not
    meant for eq_params. If in inverse problem mode, the field `eq_params` must be provided
    by the `eq_params_array` arguments.
    """
    _, params_cumsum = _get_param_nb(params)
    flat = eqx.tree_at(
        jax.tree.leaves,
        params.nn_params,
        jnp.split(params_array, params_cumsum[:-1]),
    )

    return jax.tree.map(
        lambda a, b: a.reshape(b.shape),
        flat,
        params,
        is_leaf=eqx.is_inexact_array,
    )


class SSBFGSState(eqx.Module):
    """
    State for jinns wrapper over ssbfgs from scimba implementation
    """

    tx_state: optax.OptState
    # count: jax.typing.ArrayLike
    # params: Params
    # updates: Params
    # hk: jax.typing.ArrayLike
    # linesearch_state: NamedTuple


def self_scaled_bfgs_or_broyden(
    *,
    linesearch: LINESEARCH_TYPE | None = _linesearch.scale_by_zoom_linesearch(
        max_linesearch_steps=15,
        initial_guess_strategy="one",
    ),
    broyden: bool = False,
):
    optimizer = _self_scaled_bfgs_or_broyden(linesearch, broyden)

    def init(params: Params) -> SSBFGSState:
        params_as_array = jnp.concatenate(
            jax.tree.map(lambda l: l.flatten(), jax.tree.leaves(params)), axis=0
        )
        return SSBFGSState(tx_state=optimizer.init(params_as_array))

    def update(
        grad_k: Params,
        state: SSBFGSState,
        theta_k: Params,
        value: jax.typing.ArrayLike,
        grad: Params,
        value_fn: Callable[..., tuple[jax.typing.ArrayLike, base.Updates]],
        grad_fn: Callable[..., tuple[jax.typing.ArrayLike, base.Updates]],
        **extra_args_for_fn,
    ):
        theta_k_as_array = jnp.concatenate(
            jax.tree.map(lambda l: l.flatten(), jax.tree.leaves(theta_k)), axis=0
        )
        grad_k_as_array = jnp.concatenate(
            jax.tree.map(lambda l: l.flatten(), jax.tree.leaves(grad_k)), axis=0
        )
        grad_as_array = jnp.concatenate(
            jax.tree.map(lambda l: l.flatten(), jax.tree.leaves(grad)), axis=0
        )

        new_params_as_array, new_tx_state = optimizer.update(
            grad_k_as_array,
            state.tx_state,
            theta_k_as_array,
            value=value,
            grad=grad_as_array,
            value_fn=value_fn,
            grad_fn=grad_fn,
            **extra_args_for_fn,
        )

        return (
            params_array_to_pytree(new_params_as_array, theta_k),
            eqx.tree_at(lambda pt: pt.tx_state, state, new_tx_state),
        )

    return base.GradientTransformationExtraArgs(init, update)  # type: ignore
