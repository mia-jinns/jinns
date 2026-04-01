# Wrapper on `optax.GradientTransformationExtraArgs` to carry info about
# wether to use NGD gradient_step() in solve (instead of regular)

import equinox as eqx
import optax
import jinns
from typing import Optional
import jax
import jax.numpy as jnp


from jinns.optimizers.utils_ngd import Component, _reweight_pytree


class NGDState(eqx.Module):
    """A custom state for Natural Gradient Descent.

    Useful internally: it tells `jinns` when to trigger NGD. All custom NGD optimizer state
    should inherit this.
    """

    tx_state: optax.OptState  # optimizer state


class VanillaNGDState(NGDState):
    """State for vanilla Natural Gradient Descent with ridge regularization."""

    sgd_learning_rate: float = 1.0
    gram_reg: float = 1e-5  # small ridge regularization on diag(G) when inverting
    with_eq_params_update: bool = eqx.field(static=True, default=False)


def vanilla_ngd(
    *,
    sgd_learning_rate: float = 1.0,
    gram_reg: float = 1e-5,
    linesearch: Optional[
        optax.GradientTransformationExtraArgs
    ] = optax.scale_by_backtracking_linesearch(max_backtracking_steps=15, verbose=True),
    eq_params_tx: dict | None = None,
) -> optax.GradientTransformationExtraArgs:
    r"""jinns implementation of vanilla Natural Gradient Descent (NGD).

    See e.g. [Johannes Müller, Marius Zeinhofer - Achieving High Accuracy with PINNs via Energy Natural Gradient Descent](https://proceedings.mlr.press/v202/muller23b/muller23b.pdf)

    This vanilla implementation uses a ridge regularization on the diagonal of $G$ before solving
    the linear system.

    $$
        \eta = (\hat{G} + \lambda I_p)^{-1} \nabla_{\nu} \mathcal{L}(\nu).
    $$

    !!! note

        For ease of PyTree manipulation internally, in jinns, the gram matrix $G$ and $\eta$
        are computed internally in  `solver/_utils._loss_evaluate_and_natural_gradient_step`.
        This optax optimizer simply takes care of the additive updates and linesearch (recommended).

    Parameters
    ----------
    sgd_learning_rate : float, optional
        the starting learning rate multiply the NGD update, by default 1.0
    gram_reg : float, optional
        the ridge regularization used before inverting the Gram matrix, by default 1e-5
    linesearch : Optional[base.GradientTransformationExtraArgs], optional
        it is recommended to use a linesearch method that computes a learning rate,
        a.k.a. stepsize, to satisfy some criterion such as a sufficient decrease of the objective
        by additional calls to the objective
        by default optax.scale_by_backtracking_linesearch(max_backtracking_steps=15, verbose=True)
    eq_params_tx : dict | None, optional
        optional dictionnary of optax optimizers for each eq_params, by default None which means eq_params are not updated (forward problem)

    Returns
    -------
    optax.GradientTransformationExtraArgs
        the vanilla ngd optimizer
    """
    if linesearch is None:
        linesearch_ = optax.identity()
    else:
        linesearch_ = linesearch

    ngd_optim_ = optax.chain(
        optax.sgd(learning_rate=1.0),
        linesearch_,
    )
    if eq_params_tx is not None:
        # In the line below, parameters that are not updated are assigned the
        # "freeze" label. The "freeze" label is then associated to the null
        # update. We resort to the "freeze" label when a None value is found.
        # see https://github.com/google-deepmind/optax/blob/main/examples/freezing_parameters.ipynb
        param_labels = jinns.parameters.Params(
            nn_params="ngd",
            eq_params={
                key: (key if val is not None else "freeze")
                for key, val in eq_params_tx.items()
            },
        )

        ngd_optim = optax.partition(
            {
                **{
                    "ngd": ngd_optim_,
                },
                **{
                    (key if val is not None else "freeze"): (
                        val if val is not None else optax.set_to_zero()
                    )
                    for key, val in eq_params_tx.items()
                },
            },
            param_labels,
            mask_compatible_extra_args=True,  # https://github.com/google-deepmind/optax/issues/1649
        )
        with_eq_params_update = True
    else:
        ngd_optim = ngd_optim_
        with_eq_params_update = False

    def init(params: optax.Params) -> VanillaNGDState:
        return VanillaNGDState(
            tx_state=ngd_optim.init(params),
            sgd_learning_rate=sgd_learning_rate,
            gram_reg=gram_reg,
            with_eq_params_update=with_eq_params_update,
        )

    def update(
        r_g_sw: tuple[Component, Component, Component],
        ngd_state: VanillaNGDState,
        params,
        loss,
        batch,
        loss_value,
        non_opt_params,  # TODO: I'd like to avoid passing this
        **kwargs,  # should not be used imo
    ) -> tuple[optax.Updates, VanillaNGDState]:
        # NOTE: here params shoudl be thought as an opt_params in jinns
        # thus with possible None field for eq_params

        del kwargs

        # -- Compute the necessary quantities from r, g
        from jinns.optimizers.utils_ngd import (
            assemble_ngd_gram_matrix_and_euclidean_gradient,
            params_array_to_pytree,
        )

        r, g, sqrt_weights_per_sample = r_g_sw
        gram_mat, euclidean_grad_array_nn, euclidean_grad_array_eq = (
            assemble_ngd_gram_matrix_and_euclidean_gradient(
                r=r,
                g=g,
                sqrt_weights_per_sample=sqrt_weights_per_sample,
                with_eq_params_update=with_eq_params_update,
            )
        )

        # --
        # Solve the linear system (G + reg * I) @ natural_grad = eucl_grad to get
        # the nn_params natural gradient.
        reg: float = ngd_state.gram_reg
        n_param = gram_mat.shape[0]
        natural_grad_array_nn = jax.scipy.linalg.solve(
            gram_mat + reg * jnp.eye(n_param), euclidean_grad_array_nn, assume_a="sym"
        )

        # --
        # Final step : restructure the natural gradient as a Params PyTree
        # NOTE: if in inverse problem mode, the leaf at eq_params is filled with standard euclidean
        # gradients, and zeros if in forward problem mode.

        ngd_updates = params_array_to_pytree(
            natural_grad_array_nn, params, euclidean_grad_array_eq
        )

        # ---
        # In case of linesearch, we need euclidean grad + value_fn
        euclidean_grads = params_array_to_pytree(
            euclidean_grad_array_nn, params, euclidean_grad_array_eq
        )

        # TODO: this is where things get complicated
        # I would like to avoid passing `non_opt_params` or `params` to update() if possible,
        # as it obfuscate code comprehension imo.
        # But the signature of loss.values_and_grad_per_sample() makes it complicated to do so.
        # With solve() we could just forget about it and set non_opt_params to None (not a good idea)
        # but with solve_alternate() this becomes a crucial issue as eq_params will be missing.

        # I don't fully understand the intrication between optax, the opt_params vs params paradigm and jinns
        # But to put the following piece of code from solver/_utils.py inside the optax optimizer,
        # we need access to params, not just opt_params. Can we find a way to do everything with opt_params only ?
        # It would be very ugly to pass params (or even just params.eq_params) to the optimizer.update
        # function
        def ngd_value_fn(params):
            """
            non_opt_params is passed to update as is needed to reform the correct
            params (with non None values at non optimized params - this is done in
            loss.evaluate) in order to be able to evaluate
            """
            # Not using loss.evaluate here cause of the mean(sum()) vs sum(mean)
            # remark. This fn computes the loss we are truly minimizing with NGD.
            new_r, _ = loss.values_and_grad_per_sample(
                params,
                batch,
                non_opt_params=non_opt_params,
            )
            total_loss = jnp.sum(
                jnp.concatenate(
                    jax.tree.leaves(
                        jax.tree.map(
                            lambda arr: jnp.sum(arr**2, axis=-1),
                            _reweight_pytree(
                                new_r, sqrt_weights_per_sample
                            ),  # `r` changes !
                        ),
                    ),
                    axis=0,
                )
            )
            return total_loss

        if with_eq_params_update:
            # Following https://github.com/google-deepmind/optax/issues/1649
            def fill_eq_params_value_fn(
                ngd_value_fn,
                params,  # this is where it get tricky, should we use non_opt_param ?
            ):
                """Reconstructs the full parameter tree from the masked one.
                Specific case: this is always eq_params that will be masked


                We need to pass the full params (`params`) because opt_params might contain
                None at eq_params, which would fail the last instruction
                """

                def wrapper(masked_params):  # this is what will be called by the
                    # backtracking line search callback
                    # ie., it will contain masked params that we need to fill in
                    full_params = eqx.tree_at(
                        lambda pt: pt.eq_params, masked_params, params.eq_params
                    )
                    return ngd_value_fn(full_params, non_opt_params=non_opt_params)

                return wrapper

            value_fn = fill_eq_params_value_fn(ngd_value_fn, params)
        else:
            # if we are not doing an inverse problem, there is no optax.partition
            # to handle
            value_fn = ngd_value_fn

        tx_state = ngd_state.tx_state
        ngd_updates, new_tx_state = ngd_optim.update(
            ngd_updates,  # type: ignore
            tx_state,
            params,
            # extra kwargs passed to backtracking line search `update()` method
            value=loss_value,
            grad=euclidean_grads,
            value_fn=value_fn,
        )

        return (
            ngd_updates,
            eqx.tree_at(lambda pt: pt.tx_state, ngd_state, new_tx_state),
        )

    return optax.GradientTransformationExtraArgs(
        init, update
    )  # TODO: is there an easy way to make the typing conform here ?
