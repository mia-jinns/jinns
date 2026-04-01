from jax._src.basearray import Array
from jax._src.basearray import Array
import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from typing import (
    TYPE_CHECKING,
    Any,
    TypeVar,
    Literal,
)
from jaxtyping import Float, Array
from jinns.utils._types import (
    AnyLossComponents,
)
from jinns.parameters._params import Params
from jinns.nn._hyperpinn import _get_param_nb

# Typing stuff
Component = TypeVar(name="Component", bound=AnyLossComponents[Array | Params | None])


def _reweight_pytree(pt: Component, lw: Component) -> Component:
    """Helper function for NGD: reweight the residual or their gradients.

    Multiply all the arrays at leaves of pt (`leaf`) by the saclar at leaves of
    lw (`w`).
    If `leaf` is again a pytree, multiply each leaf of `leaf` by `w`.
    """
    return jax.tree.map(
        lambda w, leaf: (
            w * leaf
            if eqx.is_inexact_array(leaf)
            else jax.tree.map(  # leaf is a Params when reweighting g
                lambda arr: w * arr, leaf, is_leaf=eqx.is_inexact_array
            )
        ),
        lw,
        pt,
        is_leaf=lambda x: eqx.is_inexact_array(x) or isinstance(x, Params),
    )


def _post_process_pytree_of_grad(
    y: Component, param_type: Literal["nn_params", "eq_params"] = "nn_params"
) -> Component:  # [Float[Array, "n n_equations n_parameters"], ...]
    """Helper function for NGD: reshape neural network weights.

    Restructure a PyTree of gradient per sample (with arbitrary shape) into the appropriate
    shapes for computing total euclidean gradients and gram matrix preconditionner.

    Parameters
    -------
    y: Component
        A PyTree with XDEComponent structure and containing gradient **per samples**
        in arbitrary shape dictated by the Params tree structure.
    param_type: Literal["nn_params", "eq_params"], default="nn_params"
        A selection to designate which type of parameter we should be working on between
        eq_params and nn_params.

    Returns
    -------
    Component
        A PyTree with similar structure as `y` and containing gradient **per samples**.
    """

    # TODO: make the doc clearer, for now it is written for when param_type="nn_params"

    # First, get the PyTree of all trainable parameters for each loss terms (e.g. `dyn_loss`, `init`, `BC` etc.)
    # with list of layers as its leaves.
    l = jax.tree.map(
        lambda pt: jax.tree.leaves(
            getattr(pt, param_type), is_leaf=eqx.is_inexact_array
        ),
        y,
        is_leaf=lambda x: isinstance(x, Params),
    )

    # Then, flatten each layer of trainable parameters into shape
    # -> (nb, n_equation, *)
    # where
    #  - `n_equation` is the number of  equations for vector valued system of equations.
    #  - `nb` is the number of point in the (mini-)batch for each loss term.
    #  - `*` means the number of free parameters of the layer.
    l = jax.tree.map(
        lambda leaf: [a.reshape((a.shape[0], a.shape[1], -1)) for a in leaf],
        l,
        is_leaf=lambda x: isinstance(x, list),
    )

    # Then, concatenate all layers into a shape (nb, n_equation, p) where p is the total
    # number of free parameters for ALL layers.
    return jax.tree.map(
        lambda leaf: jnp.concatenate(leaf, axis=2),
        l,
        is_leaf=lambda x: isinstance(x, list),
    )


def params_array_to_pytree(
    nn_params_array: Float[Array, " n_parameters"],
    params: Params,
    eq_params_array=None,
) -> Params:
    """Helper function for NGD: from matrix to PyTree representation of parameters.

    This function converts the raw matrix representation of the network
    trainable parameters into a `Params` object with the correct PyTree structure to
    be handled by optax for the updates.

    By default, the field `eq_params` is filled with zeros, as NGD is not
    meant for eq_params. If in inverse problem mode, the field `eq_params` must be provided
    by the `eq_params_array` arguments.
    """
    _, params_cumsum = _get_param_nb(params.nn_params)
    nn_flat = eqx.tree_at(
        jax.tree.leaves,
        params.nn_params,
        jnp.split(nn_params_array, params_cumsum[:-1]),
    )

    nn_params_pt = jax.tree.map(
        lambda a, b: a.reshape(b.shape),
        nn_flat,
        params.nn_params,
        is_leaf=eqx.is_inexact_array,
    )
    if eq_params_array is not None:
        _, params_cumsum = _get_param_nb(params.eq_params)
        eq_flat = eqx.tree_at(
            jax.tree.leaves,
            params.eq_params,
            jnp.split(eq_params_array, params_cumsum[:-1]),
        )

        eq_params_pt = jax.tree.map(
            lambda a, b: a.reshape(b.shape),
            eq_flat,
            params.eq_params,
            is_leaf=eqx.is_inexact_array,
        )
    else:
        # Wrap everything in a Params() object
        # by default eq_params is filled with Zeros so that additive updates
        # leave them unchanged.
        eq_params_pt = optax.tree.zeros_like(params.eq_params)
    return Params(
        nn_params=nn_params_pt,
        eq_params=eq_params_pt,
    )


def _get_sqrt_weights_per_sample(
    lw: Component, r: Component, batch_norm: bool = True
) -> Component:
    """Helper function for NGD: computes sqrt(weights) in front of each sample

    This return a PyTree of component with term-samples weights. There is an option
    to include the (sqrt of) batch normalization in front of each term.

    Parameters
    ----------
    lw : Component
        Component object with loss weights at its leafs. These will be square-rooted
    batch_norm : bool, optional
        wether to divide each loss weights by 1/sqrt(n_colloc * n_term), by default True
    """
    if batch_norm:
        weights_per_sample = jax.tree.unflatten(
            jax.tree.structure(r),
            jnp.sqrt(jnp.array(jax.tree.leaves(lw)))
            / jnp.sqrt(
                jnp.array(
                    jax.tree.leaves(
                        jax.tree.map(
                            lambda l: (
                                l.shape[0] * l.shape[1]
                            ),  # [0]: nb_point & [1]: nb_equations in system
                            r,
                        )
                    )
                )
            ),
        )
        return weights_per_sample
    else:
        # The following weight (without / sqrt{n} normalization) is useful for the value_fn
        weights_per_sample_no_avg = jax.tree.unflatten(
            jax.tree.structure(r), jnp.sqrt(jnp.array(jax.tree.leaves(lw)))
        )
        return weights_per_sample_no_avg


def assemble_ngd_gram_matrix_and_euclidean_gradient(
    r: Component,
    g: Component,
    sqrt_weights_per_sample: Component,
    with_eq_params_update=False,
) -> tuple[
    Float[Array, "n_parameters n_parameters"],
    Float[Array, " n_parameters"],
    Float[Array, " n_eq_parameters"] | None,
]:
    """Compute the euclidean gradient of the loss wrt nn_params and the preconditionning matrix

    Returns
    -------
    tuple[Float[Array, "n_parameters n_parameters"], Float[Array, " n_parameters"]]
        tuple(G, eucliden_grad)
    """
    # --
    # Compute weights (sqrt{\lambda_{term} / n_{term}}) in front of each sample of each `term`
    # Store these as a PyTree with the same tree structure as r and g.

    # --
    # Reweight each sample
    reweighted_r = _reweight_pytree(pt=r, lw=sqrt_weights_per_sample)
    reweighted_g = _reweight_pytree(pt=g, lw=sqrt_weights_per_sample)

    # ---
    # Flatten the pytree of params gradient as a tuple of (n, n_equations, p)
    # arrays
    Ms = _post_process_pytree_of_grad(reweighted_g, "nn_params")

    Rs = reweighted_r

    # --
    # Form the complete euclidean gradient

    # NOTE: this is subtile but beware that euclidean gradient (might) differs from
    # jax.grad(loss.evaluate)(params) here.
    # Indeed jinns takes the sum(mean(loss_type)) while here we compute mean(sum(all_loss_types).
    # These might differs when different number of samples per terms are used.
    # Equality can be matched by changing the jinns reduction function internally.
    # See tests/optimizer_tests/test_euclidean_gradient_equality.py

    euclidean_grad_pytree = jax.tree.map(
        lambda M, R: jnp.einsum("ijk,ij->k", M, R), Ms, Rs
    )
    # which is equivalent to
    # >>> euclidean_grad_array = jnp.sum(
    #    M.transpose((0, 2, 1)) @ R[..., None],
    #    axis=0,
    # ).squeeze()  # shape (n_params,)
    # # NOTE jnp.sum because averaging is done via the (1 / sqrt(n)) reweighting above
    euclidean_grad_array: Array = jax.tree.reduce(
        function=jnp.add, tree=euclidean_grad_pytree
    )

    # --
    # Then assemble the Gram Matrix
    #   1. Do the mean over the `n` collocation points -> get a `(n_equation, p, p)` array.
    #      NOTE: jnp.einsum instead of a mean here because averaging is in reweighting above.
    #   2. Then, do the sum over `n_equation` (axis=0)
    gram_mat_pytree = jax.tree.map(
        f=lambda M: jnp.einsum("ijk,ijl->kl", M, M),
        # <=> lambda M: (M.transpose((1, 2, 0)) @ M.transpose((1, 0, 2))).sum(axis=0),
        tree=Ms,
    )

    gram_mat: Array = jax.tree.reduce(function=jnp.add, tree=gram_mat_pytree)

    if not with_eq_params_update:
        euclidean_grad_array_eq_params = None
    else:
        # --
        # If we want to update eq_params (inverse problem), this is done using
        # the euclidean gradients wrt eq_params. We can form it from M and R the same way as
        # for nn_params.
        Ms_eq_params = _post_process_pytree_of_grad(reweighted_g, "eq_params")
        euclidean_grad_pytree_eq_params = jax.tree.map(
            lambda M, R: jnp.einsum("ijk,ij->k", M, R), Ms_eq_params, Rs
        )
        euclidean_grad_array_eq_params = jax.tree.reduce(
            jnp.add, euclidean_grad_pytree_eq_params
        )

    return gram_mat, euclidean_grad_array, euclidean_grad_array_eq_params
