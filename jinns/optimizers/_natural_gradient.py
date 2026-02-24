# Wrapper on `optax.GradientTransformationExtraArgs` to carry info about
# wether to use NGD gradient_step() in solve (instead of regular)

import optax
import equinox as eqx


class NGDState(eqx.Module):
    tx_state: optax.OptState
    is_ngd: bool = True  # useful to


def vanilla_ngd(
    tx: optax.GradientTransformationExtraArgs,
) -> optax.GradientTransformationExtraArgs:
    """
    An optax optimizer for Natural Gradient Descent in its vanilla version.
    It is simply a wrapper on `tx` (the real optimizer) with an extra state
    """

    def init(params: optax.Params) -> NGDState:
        tx_state = tx.init(params)
        return NGDState(tx_state=tx_state, is_ngd=True)

    def update(
        updates, state, params=None, **extra_args
    ) -> tuple[optax.Updates, NGDState]:
        tx_state = state.tx_state
        updates, new_tx_state = tx.update(updates, tx_state, params, **extra_args)
        return (updates, NGDState(tx_state=new_tx_state, is_ngd=True))

    return optax.GradientTransformationExtraArgs(
        init, update
    )  # TODO: is there an easy way to make the typing conform here ?
