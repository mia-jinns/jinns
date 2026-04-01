# Wrapper on `optax.GradientTransformationExtraArgs` to carry info about
# wether to use NGD gradient_step() in solve (instead of regular)

import equinox as eqx
import optax
import jinns
from typing import Optional


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
    """jinns implementation of vanilla Natural Gradient Descent (NGD)

    Parameters
    ----------
    sgd_learning_rate : float, optional
        the starting learning rate multiply the NGD update, by default 1.0
    gram_reg : float, optional
        the ridge regularization used before inverting the Gram matrix, by default 1e-5
    linesearch : Optional[base.GradientTransformationExtraArgs], optional
        the linesearch method that computes a learning rate, a.k.a. stepsize, to satisfy some criterion such as a sufficient decrease of the objective by additional calls to the objective
        by default optax.scale_by_backtracking_linesearch(max_backtracking_steps=15, verbose=True)
    eq_params_tx : dict | None, optional
        optional dictionnary of optax optimizers for each eq_params, by default None which means eq_params are not updated (forward problem)

    Returns
    -------
    optax.GradientTransformationExtraArgs
        the vanilla ngd optimizer
    """
    if linesearch is None:
        linesearch = optax.identity()

    ngd_optim_ = optax.chain(
        optax.sgd(learning_rate=1.0),
        linesearch,
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
        updates, ngd_state, params=None, **extra_kwargs
    ) -> tuple[optax.Updates, VanillaNGDState]:
        tx_state = ngd_state.tx_state
        updates, new_tx_state = ngd_optim.update(
            updates, tx_state, params, **extra_kwargs
        )
        return (updates, eqx.tree_at(lambda pt: pt.tx_state, ngd_state, new_tx_state))

    return optax.GradientTransformationExtraArgs(
        init, update
    )  # TODO: is there an easy way to make the typing conform here ?
