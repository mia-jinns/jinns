"""ssbfgs and ssbroyden implementation.
Taken and adapted from scimba v1.3.3
https://gitlab.com/scimba/scimba
"""

import os

os.environ["JAX_PLATFORMS"] = "cpu"

from typing import Callable, NamedTuple, Literal
from jaxtyping import Float, Int, Array

# import warnings
import jax
import jax.numpy as jnp
import equinox as eqx
import lineax as lx
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

    count: Int[Array, " 1"]
    params: optax.Params
    updates: optax.Params
    hk: Float[Array, " n_params"]
    linesearch_state: NamedTuple


def self_scaled_bfgs_or_broyden(
    linesearch: LINESEARCH_TYPE | None = None,
    broyden: bool = False,
) -> base.GradientTransformationExtraArgs:
    r"""
    Scales updates by ssBFGS or ssBroyden.

    The implementation is taken from the [scimba package](https://gitlab.com/scimba/scimba)
    The algorithms are described in [this article](https://arxiv.org/pdf/2405.04230)

    Parameters
    ----------
    linesearch
        It is recommended to use a linesearch method that computes a learning rate,
        a.k.a. stepsize, to satisfy some criterion such as a sufficient decrease of the objective
        by additional calls to the objective
        by default optax.scale_by_zoom_linesearch(max_linesearch_steps=25, initial_guess_strategy="one")
        is used for ssBroyden
        by default optax.scale_by_backtracking_linesearch(max_backtracking_steps=15)
        is used for ssBFGS
    broyden
        If False then ssBFGS updates will be used, else ssBroyden.
    Returns
    -------
    optax.GradientTransformationExtraArgs
        The ssBFGS or ssBroyden optimizer
    """

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
            linesearch_state=linesearch.init(params_pt),  # type: ignore
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
            value=value,  # type: ignore
            grad=params_array_to_pytree(grad, grad_pt),  # type: ignore
            value_fn=value_fn,  # type: ignore
            **extra_args_for_fn,
        )
        s_k = jnp.concatenate(
            jax.tree.map(lambda l: l.flatten(), jax.tree.leaves(s_k_pt)), axis=0
        )

        # compute some values for next turn:
        alpha_k = linesearch_state.learning_rate  # type: ignore

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

        # # method ssbroyden
        # if broyden:
        #     numel = theta_k.shape[0]
        #     b_k = -alpha_k * (s_k @ grad_k) / yk_dot_sk
        #     h_k = yk_dot_Hkyk / yk_dot_sk
        #     a_k = h_k * b_k - 1.0
        #     c_k = jnp.sqrt(a_k / (a_k + 1.0))
        #     rhom_k = jnp.minimum(1.0, h_k * (1 - c_k))
        #     thetam_k = (rhom_k - 1) / a_k
        #     thetap_k = 1.0 / rhom_k
        #     th_k = jnp.maximum(thetam_k, jnp.minimum(thetap_k, (1.0 - b_k) / b_k))
        #     sigma_k = 1 + a_k * th_k
        #     sigma_k_pow = sigma_k ** (-1 / (numel - 1))

        #     tau_k = jnp.where(
        #         th_k > 0,
        #         tau_k * jnp.minimum(sigma_k_pow, 1.0 / th_k),
        #         jnp.minimum(tau_k * sigma_k_pow, sigma_k),
        #     )
        #     phi_k = (1 - th_k) / (1.0 + a_k * th_k)

        # temp1 = (Hkyk[:, None] @ Hkyk[None, :]) / yk_dot_Hkyk
        # temp2 = phi_k * (v_k[:, None] @ v_k[None, :])
        # temp3 = (s_k[:, None] @ s_k[None, :]) / yk_dot_sk
        # H_kp1 = (1.0 / tau_k) * (state.hk - temp1 + temp2) + temp3

        new_state = ScaleBySSBFGSState(
            count=numerics.safe_increment(state.count),  # type: ignore
            params=theta_kp1,
            updates=s_k,
            hk=state.hk,
            linesearch_state=linesearch_state,  # type: ignore
        )
        return params_array_to_pytree(s_k, grad_k_pt), new_state

    return base.GradientTransformationExtraArgs(init_fn, update_fn)  # type: ignore


class ScaleBySSBroydenFamilyState(NamedTuple):
    """State for SS-Broyden class of algorithms.

    Attributes:
    count: iteration of the algorithm.
    params: current parameters.
    updates: current updates.
    hk: current hessian latriw approximation.
    linesearch_state: current linesearch state.
    """

    count: Int[Array, " 1"]
    params: eqx.Module
    updates: eqx.Module
    hk: Float[Array, " n_params"]
    linesearch_state: NamedTuple


def scale_by_SSBroyden_family(
    linesearch: LINESEARCH_TYPE | None = None,
    member: Literal["BFGS", "SSBFGS", "DFP", "SSDFP", "Broyden", "SSBroyden"] = "SSBroyden",
) -> base.GradientTransformationExtraArgs:
    r"""
    Scales updates with one algorithm member from the SSBroyden family

    The algorithms are described in [this article](https://arxiv.org/pdf/2405.04230).
    This implementation closely follows Self-Scaled Broyden Family of Quasi-Newton Methods in JAX, Bioli and Abarrategi, 2026;
    [link](https://arxiv.org/pdf/2603.10599)

    Parameters
    ----------
    linesearch
        It is recommended to use a linesearch method that computes a learning rate,
        a.k.a. stepsize, to satisfy some criterion such as a sufficient decrease of the objective
        by additional calls to the objective
    member
        The type of algorithm used as defined in Table 1 of [this article](https://arxiv.org/pdf/2603.10599)
    Returns
    -------
    optax.GradientTransformationExtraArgs
        The ssBroyden optimizer
    """

    if linesearch is None:
        linesearch = _linesearch.scale_by_zoom_linesearch(
            max_linesearch_steps=25,
            initial_guess_strategy="one",
        )


    def init_fn(params_pt: Params) -> ScaleBySSBroydenFamilyState:
        # params = jnp.concatenate(
        #     jax.tree.map(lambda l: l.flatten(), jax.tree.leaves(params_pt)), axis=0
        # )
        # NOTE that we pass the Params as PyTree to linesearch beacuase it is
        # the unmodified optax linesearch which works for arbitrary PyTree
        # whereas the update_fn of ssBFGS or ssBroyden is written for
        # jnp.array only
        return ScaleBySSBroydenFamilyState(
            count=jnp.asarray(0, dtype=jnp.int32),
            params=optax.tree.zeros_like(params_pt),
            updates=optax.tree.zeros_like(params_pt),
            hk=jax.tree.map(lambda l: jnp.eye(l.shape[0]), params_pt),
            linesearch_state=linesearch.init(params_pt),  # type: ignore
        )

    def update_fn(
        gradk_pt: Params,
        state: ScaleBySSBroydenFamilyState,
        thetak_pt: Params,
        value: jax.typing.ArrayLike,
        grad_pt: Params,
        value_fn: Callable[..., tuple[jax.typing.ArrayLike, base.Updates]],
        grad_fn: Callable[..., tuple[jax.typing.ArrayLike, base.Updates]],
        **extra_args_for_fn,
    ) -> tuple[base.Updates, ScaleBySSBroydenFamilyState]:

        dk = lx.PyTreeLinearOperator(
            -state.hk, jax.eval_shape(lambda: gradk_pt)
        )
        sk_pt, linesearch_state = linesearch.update(
            dk.mv(gradk_pt),
            state.linesearch_state,
            thetak_pt, # type: ignore
            value=value,  # type: ignore
            grad=grad_pt,  # type: ignore
            value_fn=value_fn,  # type: ignore
            **extra_args_for_fn,
        )

        # compute some values for next turn:
        alphak = linesearch_state.learning_rate  # type: ignore

        thetakp1_pt = optax.apply_updates(thetak_pt, sk_pt) # theta is x in the article

        # get the gradients at theta_kp1
        gradkp1_pt = grad_fn(thetakp1_pt, **extra_args_for_fn)

        # sk_pt = jax.tree.map(lambda a, b: a - b, thetakp1_pt, thetak_pt) # no need to recompute
        yk_pt = jax.tree.map(lambda a, b: a - b, gradkp1_pt, gradk_pt)
        yksk = lx.PyTreeLinearOperator(yk_pt, (1,)).mv(sk_pt)  # scalar

        rhok_pt = 1 / yksk

        Hkyk_pt = lx.PyTreeLinearOperator(state.hk, jax.eval_shape(lambda: yk_pt))
        ykHkyk = (lx.IdentityLinearOperator(yk_pt) @ Hkyk_pt).mv(yk_pt)  # scalar

        vk = jax.tree.map(
            lambda a, b: (a / yksk - b / ykHkyk),
            sk_pt,
            Hkyk_pt.mv(yk_pt),
        )
        
        # we use the fact that Bksk = -alphak*grad_k (see Eq. 5 of Optimizing the Optimizers, 2026)
        skgradk = lx.PyTreeLinearOperator(sk_pt, (1,)).mv(gradk_pt)  # scalar
        bk = - alphak / yksk * skgradk  # scalar

        hk = ykHkyk / yksk

        if member == "BFGS":
            thetak = 0.
            tauk = 1.0
        elif member == "SSBFGS":
            thetak = 0.
            sigma_k = 1 + a_k * th_k
            sigma_k_pow = sigma_k ** (-1 / (numel - 1))

            tau_k = jnp.where(
                th_k > 0,
                tau_k * jnp.minimum(sigma_k_pow, 1.0 / th_k),
                jnp.minimum(tau_k * sigma_k_pow, sigma_k),
            )
        elif member == "DFP":
        elif member == "SSDFP":
        elif member == "Broyden":
        elif member == "SSBroyden":
        else:
            raise ValueError("Wrong member value, must be either BFGS, SSBFGS, DFS, SSDFP, Broyden or SSBroyden")

        ak = hk * bk - 1.0
        ck = jnp.sqrt(ak / (ak + 1.0))
        rhok_minus = jnp.minimum(1.0, hk * (1 - ck))
        thetak_minus= (rhok_minus - 1) / ak
        thetak_plus = 1.0 / rhok_minus
        thetak = jnp.maximum(thetak_minus, jnp.minimum(thetak_plus, (1.0 - bk) / bk))

        phik = (1 - thetak) / (1.0 + ak * thetak)


        # Hkyk = state.hk @ y_k
        # yk_dot_Hkyk = y_k @ Hkyk
        # yk_dot_sk = y_k @ s_k
        # v_k = jnp.sqrt(yk_dot_Hkyk) * (s_k / (yk_dot_sk) - Hkyk / yk_dot_Hkyk)

        # method ssbfgs
        tau_k = jnp.minimum(1.0, -yk_dot_sk / (alpha_k * sk_dot_grad_k))
        phi_k = 1.0

        # # method ssbroyden
        # if broyden:
        #     numel = theta_k.shape[0]
        #     b_k = -alpha_k * (s_k @ grad_k) / yk_dot_sk
        #     h_k = yk_dot_Hkyk / yk_dot_sk
        #     a_k = h_k * b_k - 1.0
        #     c_k = jnp.sqrt(a_k / (a_k + 1.0))
        #     rhom_k = jnp.minimum(1.0, h_k * (1 - c_k))
        #     thetam_k = (rhom_k - 1) / a_k
        #     thetap_k = 1.0 / rhom_k
        #     th_k = jnp.maximum(thetam_k, jnp.minimum(thetap_k, (1.0 - b_k) / b_k))
        #     sigma_k = 1 + a_k * th_k
        #     sigma_k_pow = sigma_k ** (-1 / (numel - 1))

        #     tau_k = jnp.where(
        #         th_k > 0,
        #         tau_k * jnp.minimum(sigma_k_pow, 1.0 / th_k),
        #         jnp.minimum(tau_k * sigma_k_pow, sigma_k),
        #     )
        #     phi_k = (1 - th_k) / (1.0 + a_k * th_k)

        # Hkyk_pt_materialised = Hkyk_pt.mv(y_k_pt)
        # Hkyk_pt_materialised_ = jax.tree.map(
        #     lambda a: a[:, None],
        #     Hkyk_pt_materialised
        # )
        # Hkyk_pt_materialised__ = jax.tree.map(
        #     lambda a: a[None, :],
        #     Hkyk_pt_materialised
        # )
        # temp1 = lx.IdentityLinearOperator(
        #     Hkyk_pt_materialised_,
        # ) @

        # temp1 = (Hkyk[:, None] @ Hkyk[None, :]) / yk_dot_Hkyk
        # temp2 = phi_k * (v_k[:, None] @ v_k[None, :])
        # temp3 = (s_k[:, None] @ s_k[None, :]) / yk_dot_sk
        # H_kp1 = (1.0 / tau_k) * (state.hk - temp1 + temp2) + temp3

        new_state = ScaleBySSBFGSState(
            count=numerics.safe_increment(state.count),  # type: ignore
            params=theta_kp1_pt,
            updates=s_k_pt,
            hk=state.hk,
            linesearch_state=linesearch_state,  # type: ignore
        )
        return s_k_pt, new_state

    return base.GradientTransformationExtraArgs(init_fn, update_fn)  # type: ignore


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


if __name__ == "__main__":
    #### Legacy implementation : back and forth between array form and pytree form
    def fun(x, params):
        return params[0] @ x + params[1]

    grad_fun = jax.jacfwd(fun, 1)

    n = 100
    x = jnp.empty((n,))
    params = (jnp.empty((n, n)), jnp.empty((n,)))

    tx = self_scaled_bfgs_or_broyden()
    opt_state = tx.init(params)

    def step(x, params, opt_state):
        grad = grad_fun(x, params)
        value = fun(x, params)
        print(grad, value.shape)
        updates, opt_state = tx.update(
            grad, opt_state, params, value, grad, fun, grad_fun
        )
        params = optax.apply_updates(params, updates)
        return params

    print(jax.jit(step).lower(x, params, opt_state).cost_analysis())

    #### implementation with lineax
