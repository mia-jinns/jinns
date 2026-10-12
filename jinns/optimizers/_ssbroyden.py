"""
SSBroyden family of algorithms
"""

from typing import Callable, NamedTuple, Literal
from jaxtyping import Float, Int, Array

# import warnings
import jax
import jax.numpy as jnp
import equinox as eqx
import optax.tree
from optax._src import base, numerics

from optax._src import linesearch as _linesearch

from jinns.parameters import Params

LINESEARCH_TYPE = base.GradientTransformationExtraArgs | base.GradientTransformation


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
    method: Literal["BFGS", "SSBFGS", "Broyden", "SSBroyden"] = "SSBroyden",
) -> base.GradientTransformationExtraArgs:
    r"""
    Scales updates with one algorithm method from the SSBroyden family

    The algorithms are described in [this article](https://arxiv.org/pdf/2405.04230).
    This implementation closely follows Self-Scaled Broyden Family of Quasi-Newton Methods in JAX, Bioli and Abarrategi, 2026;
    [link](https://arxiv.org/pdf/2603.10599)

    Parameters
    ----------
    linesearch
        It is recommended to use a linesearch method that computes a learning rate,
        a.k.a. stepsize, to satisfy some criterion such as a sufficient decrease of the objective
        by additional calls to the objective
    method
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
        params = jnp.concatenate(
            jax.tree.map(lambda l: l.flatten(), jax.tree.leaves(params_pt)), axis=0
        )
        # NOTE that we pass the Params as PyTree to linesearch beacuase it is
        # the unmodified optax linesearch which works for arbitrary PyTree
        # whereas the update_fn of ssBFGS or ssBroyden is written for
        # jnp.array only
        return ScaleBySSBroydenFamilyState(
            count=jnp.asarray(0, dtype=jnp.int32),
            params=optax.tree.zeros_like(params_pt),
            updates=optax.tree.zeros_like(params_pt),
            hk=jnp.eye(
                params.shape[0]
            ),  # jax.tree.map(lambda l: jnp.eye(l.shape[0]), params_pt),
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
        gradk, unravelfn_params = jax.flatten_util.ravel_pytree(gradk_pt)
        dk = -state.hk @ gradk
        sk_pt, linesearch_state = linesearch.update(
            unravelfn_params(dk),
            state.linesearch_state,
            thetak_pt,  # type: ignore
            value=value,  # type: ignore
            grad=grad_pt,  # type: ignore
            value_fn=value_fn,  # type: ignore
            **extra_args_for_fn,
        )

        # compute some values for next turn:
        alphak = linesearch_state.learning_rate  # type: ignore

        thetakp1_pt = optax.apply_updates(thetak_pt, sk_pt)  # theta is x in the article

        # get the gradients at theta_kp1
        gradkp1_pt = grad_fn(thetakp1_pt, **extra_args_for_fn)

        yk = (
            jax.flatten_util.ravel_pytree(gradkp1_pt)[0] - gradk
        )  # jax.tree.map(lambda a, b: a - b, gradkp1_pt, gradk_pt)

        # Next operation is a scalar product of a PyTree distributed array
        # This can be handled leaf by leaf
        # yk = jax.flatten_util.ravel_pytree(yk_pt)[0]
        sk = jax.flatten_util.ravel_pytree(sk_pt)[0]
        yksk = jnp.dot(yk, sk)

        rhok = 1 / yksk

        Hkyk = state.hk @ yk
        ykHkyk = yk @ Hkyk

        vk = sk / yksk - Hkyk / ykHkyk

        # next operation is a scalar product
        skgradk = jnp.dot(gradk, sk)
        # we use the fact that Bksk = -sk*grad_k (see Eq. 5 of Optimizing the Optimizers, 2026)
        bk = -alphak / yksk * skgradk  # scalar
        hk = ykHkyk / yksk
        ak = hk * bk - 1.0

        # Computation of thetak
        if method == "BFGS" or method == "SSBFGS":
            thetak = 0.0
        elif method == "Broyden" or method == "SSBroyden":
            ck = jnp.sqrt(ak / (ak + 1.0))
            rhok_minus = jnp.minimum(1.0, hk * (1 - ck))
            thetak_minus = (rhok_minus - 1) / ak
            thetak_plus = 1.0 / rhok_minus
            thetak = jnp.maximum(
                thetak_minus, jnp.minimum(thetak_plus, (1.0 - bk) / bk)
            )

        # Computation of tauk
        N = state.hk.shape[0]
        if method == "BFGS" or method == "Broyden":
            tauk = 1.0
        else:
            sigmak = 1 + ak * thetak  # pylint: disable=possibly-used-before-assignment
            sigmak_pow = jnp.abs(sigmak) ** (1 / (1 - N))
            rhopk = jnp.minimum(1.0, 1.0 / bk)
            if method == "SSBFGS":
                # in this case thetak = 0. and the jax.lax.cond or jnp.where raises an error because it cannot
                # handle 1./ thetak, hence this specific branch
                tauk = jnp.minimum(rhopk * sigmak_pow, sigmak)
            elif method == "SSBroyden":
                sigmak = 1 + ak * thetak  # pylint: disable=possibly-used-before-assignment
                sigmak_pow = jnp.abs(sigmak) ** (1 / (1 - N))
                rhopk = jnp.minimum(1.0, 1.0 / bk)

                tauk = jnp.where(  # sensibly faster than lax.cond
                    thetak > 0,
                    rhopk * jnp.minimum(sigmak_pow, 1.0 / thetak),
                    jnp.minimum(rhopk * sigmak_pow, sigmak),
                )

        phik = (1 - thetak) / (1.0 + ak * thetak)

        # outer products of PyTrees distributed arrays
        vkvk = jnp.outer(vk, vk)
        sksk = jnp.outer(sk, sk)

        Hkp1 = (
            1.0
            / tauk  # pylint: disable=possibly-used-before-assignment
            * (state.hk - jnp.outer(Hkyk, Hkyk) / ykHkyk + phik * ykHkyk * vkvk)
            + rhok * sksk
        )  # pylint: disable=possibly-used-before-assignment
        new_state = ScaleBySSBroydenFamilyState(
            count=numerics.safe_increment(state.count),  # type: ignore
            params=thetakp1_pt,  # type: ignore
            updates=sk_pt,  # type: ignore
            hk=Hkp1,
            linesearch_state=linesearch_state,  # type: ignore
        )
        return sk_pt, new_state

    return base.GradientTransformationExtraArgs(init_fn, update_fn)  # type: ignore
