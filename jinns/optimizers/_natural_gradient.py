# Wrapper on `optax.GradientTransformationExtraArgs` to carry info about
# wether to use NGD gradient_step() in solve (instead of regular)

from typing import NamedTuple
import optax


class NGDState(NamedTuple):
    tx_state: optax.OptState
    sgd_learning_rate: float = 1.0
    gram_reg: float = 1e-5  # small ridge regularization on diag(G) when inverting
    max_backtracking_steps: int = 15
    verbose_linesearch: bool = True


def vanilla_ngd(
    *,
    sgd_learning_rate=1.0,
    gram_reg=1e-5,
    max_backtracking_steps=15,
    verbose_linesearch=True,
) -> optax.GradientTransformationExtraArgs:
    """
    An optax optimizer for Natural Gradient Descent in its vanilla version.
    For now we force to internally use optax vanilla additive gradient update
    since it makes no sense to use other optimizers
    """
    ngd_optim = optax.chain(
        optax.sgd(learning_rate=1.0),
        optax.scale_by_backtracking_linesearch(max_backtracking_steps=15, verbose=True),
    )

    def init(params: optax.Params) -> NGDState:
        return NGDState(
            tx_state=ngd_optim.init(params),
            sgd_learning_rate=sgd_learning_rate,
            gram_reg=gram_reg,
            max_backtracking_steps=max_backtracking_steps,
            verbose_linesearch=verbose_linesearch,
        )

    def update(
        updates, ngd_state, params=None, **extra_kwargs
    ) -> tuple[optax.Updates, NGDState]:
        tx_state = ngd_state.tx_state
        updates, new_tx_state = ngd_optim.update(
            updates, tx_state, params, **extra_kwargs
        )
        return (updates, ngd_state._replace(tx_state=new_tx_state))

    return optax.GradientTransformationExtraArgs(
        init, update
    )  # TODO: is there an easy way to make the typing conform here ?
