"""ssbfgs and ssbroyden implementation.
Taken and adapted from scimba v1.3.3
https://gitlab.com/scimba/scimba
"""

from typing import Callable, NamedTuple

# import warnings
import jax
import jax.numpy as jnp
import equinox as eqx
import optax.tree
from optax._src import base, numerics

from optax._src import linesearch as _linesearch

from jinns.parameters import Params
from jinns.nn._hyperpinn import _get_param_nb

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


def self_scaled_bfgs_or_broyden(
    linesearch: LINESEARCH_TYPE | None = None,
    broyden: bool = False,
) -> base.GradientTransformationExtraArgs:
    r"""Scales updates by SS-BFGS."""

    if linesearch is None:
        # the _linesearch instanciation choices below are made by empirical
        # observation, feel free to experiment other combinations
        if broyden:
            linesearch = _linesearch.scale_by_zoom_linesearch(
                max_linesearch_steps=25,
                initial_guess_strategy="one",
            )
        else:
            linesearch = _linesearch.scale_by_backtracking_linesearch(
                max_backtracking_steps=15,
            )

    def init_fn(params_pt: Params) -> ScaleBySSBFGSState:
        params = jnp.concatenate(
            jax.tree.map(lambda l: l.flatten(), jax.tree.leaves(params_pt)), axis=0
        )
        # NOTE that we pass the Params as PyTree to linesearch beacuase it is
        # the unmodified optax linesearch which works for arbitrary PyTree
        # whereas the update_fn of ssBFGS or ssBroyden is written for
        # jnp.array only
        return ScaleBySSBFGSState(
            count=jnp.asarray(0, dtype=jnp.int32),
            params=optax.tree.zeros_like(params),
            updates=optax.tree.zeros_like(params),
            hk=jnp.eye(params.shape[0]),
            linesearch_state=linesearch.init(params_pt),
        )

    def update_fn(
        grad_k_pt: Params,
        state: ScaleBySSBFGSState,
        theta_k_pt: Params,
        value: jax.typing.ArrayLike,
        grad_pt: Params,
        value_fn: Callable[..., tuple[jax.typing.ArrayLike, base.Updates]],
        grad_fn: Callable[..., tuple[jax.typing.ArrayLike, base.Updates]],
        **extra_args_for_fn,
    ) -> tuple[base.Updates, ScaleBySSBFGSState]:
        theta_k = jnp.concatenate(
            jax.tree.map(lambda l: l.flatten(), jax.tree.leaves(theta_k_pt)), axis=0
        )
        grad_k = jnp.concatenate(
            jax.tree.map(lambda l: l.flatten(), jax.tree.leaves(grad_k_pt)), axis=0
        )
        grad = jnp.concatenate(
            jax.tree.map(lambda l: l.flatten(), jax.tree.leaves(grad_pt)), axis=0
        )
        direction = -state.hk @ grad_k
        s_k_pt, linesearch_state = linesearch.update(
            params_array_to_pytree(direction, grad_k_pt),
            state.linesearch_state,
            params_array_to_pytree(theta_k, theta_k_pt),
            value=value,
            grad=params_array_to_pytree(grad, grad_pt),
            value_fn=value_fn,
            **extra_args_for_fn,
        )
        s_k = jnp.concatenate(
            jax.tree.map(lambda l: l.flatten(), jax.tree.leaves(s_k_pt)), axis=0
        )

        # compute some values for next turn:
        alpha_k = linesearch_state.learning_rate

        theta_kp1 = optax.apply_updates(theta_k, s_k)

        # get the gradients at theta_kp1
        grad_kp1_pt = grad_fn(
            params_array_to_pytree(theta_kp1, theta_k_pt), **extra_args_for_fn
        )

        grad_kp1 = jnp.concatenate(
            jax.tree.map(lambda l: l.flatten(), jax.tree.leaves(grad_kp1_pt)), axis=0
        )

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
        return params_array_to_pytree(s_k, grad_k_pt), new_state

    return base.GradientTransformationExtraArgs(init_fn, update_fn)


def params_array_to_pytree(
    params_array,
    params,
):
    """Helper function: from matrix to PyTree representation of parameters.
    This function converts the raw matrix representation of the network
    trainable parameters into a `Params` object with the correct PyTree structure to
    be handled by optax for the updates.

    Same as for NGD but it is applied on the whole Params with the
    treatment for nn_params and eq_params
    """
    _, params_cumsum = _get_param_nb(params)
    flat = eqx.tree_at(
        jax.tree.leaves,
        params,
        jnp.split(params_array, params_cumsum[:-1]),
    )

    return jax.tree.map(
        lambda a, b: a.reshape(b.shape),
        flat,
        params,
        is_leaf=eqx.is_inexact_array,
    )
