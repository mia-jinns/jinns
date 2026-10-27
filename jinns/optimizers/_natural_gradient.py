# Wrapper on `optax.GradientTransformationExtraArgs` to carry info about
# wether to use NGD gradient_step() in solve (instead of regular)

import equinox as eqx
import optax
import jinns
import jax
import jax.numpy as jnp

from jinns.optimizers._utils_ngd import (
    _loss_weights_to_components,
    _get_sqrt_weights_per_sample,
    _reweight_pytree,
    unravel_array_to_pytree,
    weighted_jac_and_residuals,
    full_gramian_from_jacobian,
    get_euclidean_grads,
)
from jaxtyping import Float, Array


def _process_solver_kwargs(solver, **kwargs):
    if solver not in NGD_SOLVERS.keys():
        raise ValueError(
            f"Solver name must be one of {NGD_SOLVERS.keys()}, got {solver}."
        )
    # Below we could check stuff depending on the solver.


class NGDState(eqx.Module):
    """A custom state for Natural Gradient Descent.

    Useful internally: it tells `jinns` when to trigger NGD. All custom NGD optimizer state
    should inherit this.

    Parameters
    ----------

    with_eq_params_update: bool, default = True
        A boolean specifying if the updates for `params.eq_params` should be applied or not.
        If True, we are in inverse problem mode and euclidean gradient for eq_params are passed
        to the optimizer. If False, we are in forward problem mode and they are set to zero.
        This is useful in `solve_alternate`, as users may want to perform NGD on nn_params
        while leaving the eq_params untouched during a few steps.
    """

    tx_state: optax.OptState  # optimizer state
    with_eq_params_update: bool = eqx.field(static=True, default=False)
    sgd_learning_rate: float = 1.0
    gram_reg: float = 1e-5  # small ridge regularization on diag(G) when inverting


def vanilla_ngd(
    *,
    sgd_learning_rate: float = 1.0,
    gram_reg: float = 1e-5,
    linesearch: optax.GradientTransformationExtraArgs = optax.scale_by_backtracking_linesearch(
        max_backtracking_steps=15, verbose=True
    ),
    solver: str = "vanilla",
    eq_params_tx: dict | None = None,
    **kwargs,
) -> optax.GradientTransformationExtraArgs:
    r"""jinns implementation of vanilla Natural Gradient Descent (NGD).

    See e.g. [Johannes Müller, Marius Zeinhofer - Achieving High Accuracy with PINNs via Energy Natural Gradient Descent](https://proceedings.mlr.press/v202/muller23b/muller23b.pdf)

    This vanilla implementation uses a ridge regularization on the diagonal of $G$ before solving
    the linear system.

    $$
        \eta = (\hat{G} + \lambda I_p)^{-1} \nabla_{\nu} \mathcal{L}(\nu).
    $$


    Parameters
    ----------
    sgd_learning_rate
        the starting learning rate multiply the NGD update, by default 1.0
    gram_reg
        the ridge regularization used before inverting the Gram matrix, by default 1e-5
    linesearch
        it is recommended to use a linesearch method that computes a learning rate,
        a.k.a. stepsize, to satisfy some criterion such as a sufficient decrease of the objective
        by additional calls to the objective
        by default optax.scale_by_backtracking_linesearch(max_backtracking_steps=15, verbose=True)
    solver:
        which method solver to invert the system. From the full Gram matrix (default) or matrix-free method.
    eq_params_tx
        optional dictionnary of optax optimizers for each eq_params, by default None which means eq_params are not updated (forward problem)

    Returns
    -------
    optax.GradientTransformationExtraArgs
        the vanilla ngd optimizer with a special signature for the `update(r_g_sw, ngd_state, params, loss, batch,
        loss_value)` function. Here

         * the update `r_g_sw` is a 3-tuple with the 1) residuals $r(x_i)$ *per-sample*, 2) the gradient $\partial_\nu \rho_{\nu}(x_i)$ *per-sample*, and 3) the sample weights $w_i$
         * the `loss` is a `jinns.loss.LossXDE` object
         * `batch` is the current batch of point
    """
    if linesearch is None:
        linesearch_ = optax.identity()
    else:
        linesearch_ = linesearch

    ngd_optim_ = optax.chain(
        optax.sgd(learning_rate=sgd_learning_rate),
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

    _process_solver_kwargs(solver, **kwargs)
    ngd_solver: AbstractNGDSolver = NGD_SOLVERS[solver](**kwargs)

    def init(params: optax.Params) -> NGDState:
        return NGDState(
            tx_state=ngd_optim.init(params),
            sgd_learning_rate=sgd_learning_rate,
            gram_reg=gram_reg,
            with_eq_params_update=with_eq_params_update,
        )

    def update(
        loss,
        ngd_state: NGDState,
        params,
        batch,
        **_,
    ) -> tuple[optax.Updates, NGDState]:
        # --
        # Final step : restructure the natural gradient as a Params PyTree
        # NOTE: if in inverse problem mode, the leaf at eq_params is filled with standard euclidean
        # gradients, and zeros if in forward problem mode.

        (
            natural_grad_array_nn,
            euclidean_grad_array_nn,
            euclidean_grad_array_eq,
            train_loss_value,
            loss_terms,
            r,  # Would be nice to avoid to return `w_r` here. But need it for lw_ below in `value_fn`
        ) = ngd_solver.compute_ng(loss, params, batch, ngd_state)
        ngd_pytree = unravel_array_to_pytree(
            natural_grad_array_nn, params, euclidean_grad_array_eq
        )

        # ---
        # In case of linesearch, we need euclidean grad + value_fn
        lw_ = _loss_weights_to_components(loss.loss_weights, r)
        sqrt_weights_per_sample = _get_sqrt_weights_per_sample(lw_, r)
        euclidean_grads = unravel_array_to_pytree(
            euclidean_grad_array_nn, params, euclidean_grad_array_eq
        )

        def ngd_value_fn(params):
            """"""
            # Not using loss.evaluate here cause of the mean(sum()) vs sum(mean)
            # remark. This fn computes the loss we are truly minimizing with NGD.
            new_r = loss.evaluate_per_sample(
                params,
                batch,
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

        if ngd_state.with_eq_params_update:
            # Following https://github.com/google-deepmind/optax/issues/1649
            # to handle the optax partition doing some masking:
            # -> We need to pass the full params (`params`) because opt_params might contain
            # None at eq_params, which would fail the last instruction
            # NOTE that this is a specific manipulation when using optax
            # partition for linesearch and this is NOT linked with jinns masking induced by
            # jinns.solve_alternate()
            def fill_eq_params_value_fn(ngd_value_fn, params):
                """Reconstructs the full parameter tree from the masked one.
                Specific case: this is always eq_params that will be masked
                """

                def wrapper(
                    masked_params_for_linesearch,
                ):  # this is what will be called by the
                    # backtracking line search callback
                    # ie., it will contain masked params that we need to fill in
                    full_params = eqx.tree_at(
                        lambda pt: pt.eq_params,
                        masked_params_for_linesearch,
                        params.eq_params,
                    )
                    return ngd_value_fn(full_params)

                return wrapper

            value_fn = fill_eq_params_value_fn(ngd_value_fn, params)
        else:
            # if we are not doing an inverse problem, there is no optax.partition
            # to handle
            value_fn = ngd_value_fn

        tx_state = ngd_state.tx_state
        ngd_pytree, new_tx_state = ngd_optim.update(
            ngd_pytree,  # type: ignore
            tx_state,
            params,
            # extra kwargs passed to backtracking line search `update()` method
            value=train_loss_value,
            grad=euclidean_grads,
            value_fn=value_fn,
        )

        return (
            ngd_pytree,
            eqx.tree_at(lambda pt: pt.tx_state, ngd_state, new_tx_state),
            train_loss_value,
            loss_terms,
        )

    return optax.GradientTransformationExtraArgs(
        init,  # type: ignore
        update,  # type: ignore
    )


def compute_ng_full_gramian(
    loss, params, batch, ngd_state, with_eq_params_update
) -> tuple[
    Float[Array, " n_params"],
    Float[Array, " n_params"],
    Float[Array, " n_params"] | None,
]:
    # Get the unreduced residuals and their gradient (for each sample)
    # for each loss term
    r, g = loss.values_and_grad_per_sample(
        params,
        batch,
    )

    # --
    # We can also assemble the total loss and loss_terms from reweighted residuals `r`
    # NOTE: subtility here, we don't reweight by * 1/sqrt{n} because the averaging is done
    # inside loss._reduction_functions so we don't want to account twice for it.
    lw_ = _loss_weights_to_components(loss.loss_weights, r)
    sqrt_weights_per_sample = _get_sqrt_weights_per_sample(lw=lw_, r=r, batch_norm=True)

    # --- Also compute loss per terms for jinns
    sqrt_weights_per_sample_no_avg = _get_sqrt_weights_per_sample(
        lw=lw_, r=r, batch_norm=False
    )
    loss_terms = jax.tree.map(
        lambda red_fun, r_: red_fun(r_),
        loss._reduction_functions,
        _reweight_pytree(pt=r, lw=sqrt_weights_per_sample_no_avg),
    )
    # ---

    train_loss_value = jnp.sum(
        jnp.concatenate(
            jax.tree.leaves(
                jax.tree.map(
                    lambda arr: jnp.sum(arr**2, axis=-1),
                    _reweight_pytree(pt=r, lw=sqrt_weights_per_sample),
                ),
            ),
            axis=0,
        )
    )

    # -- Compute the necessary quantities from r, g
    jacobians_nn, weighted_r, jacobians_eq = weighted_jac_and_residuals(
        r, g, sqrt_weights_per_sample, with_eq_params_update
    )

    # -- Build the euclidean grads
    euclidean_grad_array_nn_params, euclidean_grad_array_eq_params = (
        get_euclidean_grads(
            jacobians_nn, weighted_r, jacobians_eq, with_eq_params_update
        )
    )

    # -- Build the gram matrix
    gram_mat = full_gramian_from_jacobian(jacobians_nn)

    # --
    # Solve the linear system (G + reg * I) @ natural_grad = eucl_grad to get
    # the nn_params natural gradient.
    reg: float = ngd_state.gram_reg
    n_param = gram_mat.shape[0]
    natural_grad_array_nn = jax.scipy.linalg.solve(
        gram_mat + reg * jnp.eye(n_param),
        euclidean_grad_array_nn_params,
        assume_a="sym",
    )
    return (
        natural_grad_array_nn,
        euclidean_grad_array_nn_params,
        euclidean_grad_array_eq_params,
        train_loss_value,
        loss_terms,
        r,
    )


def compute_ng_matrix_free(
    loss,
    params,
    batch,
    ngd_state,
    with_eq_params_update,
    maxit,
    rtol,
) -> tuple[
    Float[Array, " n_params"],
    Float[Array, " n_params"],
    Float[Array, " n_params"] | None,
]:
    from jinns.optimizers._conjugate_gradient import conjugate_gradient

    # Get gradient for each term as in jinns.solve
    train_loss_terms, euclidean_grad_terms = loss.values_and_grads(
        params,
        batch,
    )

    # Get the full gradient as in jinns
    euclidean_grad_pytree = loss.ponderate_and_sum_gradient(euclidean_grad_terms)

    # Flatten param pytree to shape (p,)
    flat_nn, unravel_nn = jax.flatten_util.ravel_pytree(params.nn_params)
    euclidean_grad_array_nn_params, _ = jax.flatten_util.ravel_pytree(
        euclidean_grad_pytree.nn_params
    )

    def scaled_residuals(flat_nn_p):
        nn_p = unravel_nn(flat_nn_p)
        r_per_term = loss.evaluate_per_sample(
            batch=batch,
            params=jinns.parameters.Params(nn_params=nn_p, eq_params=params.eq_params),
        )

        # Ponderate each residual for each term (dyn, bc, init, ...)
        lw_ = _loss_weights_to_components(loss.loss_weights, r_per_term)
        sqrt_weights_per_sample = _get_sqrt_weights_per_sample(
            lw=lw_, r=r_per_term, batch_norm=True
        )

        weighted_r_per_term = jax.tree.map(
            lambda w, res: w * res,
            sqrt_weights_per_sample,
            r_per_term,
        )

        # Stack each term of the whole contribution
        stacked_r, _ = jax.flatten_util.ravel_pytree(weighted_r_per_term)
        return stacked_r.squeeze(), r_per_term  # shape (n_tot,)

    # See doc of jax.linearize
    # -> useful when several call to jvp() at fixed parameter flat_nn.
    stacked_r, jvp, r = jax.linearize(scaled_residuals, flat_nn, has_aux=True)
    vjp = jax.linear_transpose(jvp, flat_nn)

    def mf_gram_product(v: Array[Float, " n_params"]) -> Array[Float, " n_params"]:
        """Matrix-free version of v -> J^T (J v) + reg v"""
        return vjp(jvp(v))[0] + ngd_state.gram_reg * v

    natural_grad_array_nn, exit_code = conjugate_gradient(
        mf_gram_product,
        euclidean_grad_array_nn_params,
        rtol=rtol,
        max_iter=maxit,
        pginv=None,
    )

    # Compute auxiliary quantities such as full loss and loss per term
    train_loss_value = jnp.mean(jax.flatten_util.ravel_pytree(stacked_r**2)[0])
    lw_ = _loss_weights_to_components(loss.loss_weights, r)
    sqrt_weights_per_sample_no_avg = _get_sqrt_weights_per_sample(
        lw=lw_, r=r, batch_norm=False
    )
    loss_terms = jax.tree.map(
        lambda red_fun, r_: red_fun(r_),
        loss._reduction_functions,
        _reweight_pytree(pt=r, lw=sqrt_weights_per_sample_no_avg),
    )

    if ngd_state.with_eq_params_update:
        # Question (to sort out): when eq_params is a vector we need to ensure that
        # JAX ravel/unravel is the same as our custom one in jinns
        euclidean_grad_array_eq_params, _ = jax.flatten_util.ravel_pytree(
            euclidean_grad_pytree.eq_params
        )
    else:
        euclidean_grad_array_eq_params = None

    return (
        natural_grad_array_nn,
        euclidean_grad_array_nn_params,
        euclidean_grad_array_eq_params,
        train_loss_value,
        loss_terms,
        r,
    )


class AbstractNGDSolver(eqx.Module):
    def compute_ng(
        self, loss, params, batch, ngd_state: NGDState
    ) -> tuple[
        Float[Array, " n_params"],
        Float[Array, " n_params"],
        Float[Array, " n_params"] | None,
    ]:
        """ """
        raise NotImplementedError
        pass


class VanillaNGDSolver(AbstractNGDSolver):
    def compute_ng(self, loss, params, batch, ngd_state: NGDState):
        return compute_ng_full_gramian(
            loss, params, batch, ngd_state, ngd_state.with_eq_params_update
        )


class MatrixFreeNGDSolver(AbstractNGDSolver):
    maxit: int = 50
    rtol: float = 1e-5

    def compute_ng(self, loss, params, batch, ngd_state: NGDState):
        return compute_ng_matrix_free(
            loss,
            params,
            batch,
            ngd_state,
            ngd_state.with_eq_params_update,
            maxit=self.maxit,
            rtol=self.rtol,
        )


NGD_SOLVERS = {
    "vanilla": VanillaNGDSolver,
    "matrix-free": MatrixFreeNGDSolver,
}
