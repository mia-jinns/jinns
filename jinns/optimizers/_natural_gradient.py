# Wrapper on `optax.GradientTransformationExtraArgs` to carry info about
# wether to use NGD gradient_step() in solve (instead of regular)

from typing import NamedTuple
import optax
import jinns


class NGDState(NamedTuple):
    tx_state: optax.OptState
    sgd_learning_rate: float = 1.0
    gram_reg: float = 1e-5  # small ridge regularization on diag(G) when inverting
    max_backtracking_steps: int = 15
    verbose_linesearch: bool = True
    with_eq_params_update: bool = True


def vanilla_ngd(
    *,
    sgd_learning_rate=1.0,
    gram_reg=1e-5,
    max_backtracking_steps=15,
    verbose_linesearch=True,
    eq_params_tx=None,
) -> optax.GradientTransformationExtraArgs:
    """
    An optax optimizer for Natural Gradient Descent in its vanilla version.
    For now we force to internally use optax vanilla additive gradient update
    since it makes no sense to use other optimizers
    """
    ngd_optim_ = optax.chain(
        optax.sgd(learning_rate=1.0),
        optax.scale_by_backtracking_linesearch(max_backtracking_steps=15, verbose=True),
    )
    if eq_params_tx is not None:
        param_labels = jinns.parameters.Params(
            nn_params="ngd", eq_params={key: key for key, _ in eq_params_tx.items()}
        )

        ngd_optim = optax.partition(
            {
                **{
                    "ngd": ngd_optim_,
                },
                **eq_params_tx,
            },
            param_labels,
            mask_compatible_extra_args=True,  # https://github.com/google-deepmind/optax/issues/1649
        )
        with_eq_params_update = True
    else:
        ngd_optim = ngd_optim_
        with_eq_params_update = False

    def init(params: optax.Params) -> NGDState:
        return NGDState(
            tx_state=ngd_optim.init(params),
            sgd_learning_rate=sgd_learning_rate,
            gram_reg=gram_reg,
            max_backtracking_steps=max_backtracking_steps,
            verbose_linesearch=verbose_linesearch,
            with_eq_params_update=with_eq_params_update,
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
