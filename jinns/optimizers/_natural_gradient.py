# Wrapper on `optax.GradientTransformationExtraArgs` to carry info about
# wether to use NGD gradient_step() in solve (instead of regular)

import equinox as eqx
import optax
import jinns


class NGDState(eqx.Module):
    tx_state: optax.OptState
    sgd_learning_rate: float = 1.0
    gram_reg: float = 1e-5  # small ridge regularization on diag(G) when inverting
    max_backtracking_steps: int = 15
    verbose_linesearch: bool = True
    with_eq_params_update: bool = eqx.field(static=True, default=False)


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

    eq_params_tx is a dictionnary with keys corresponding to eq_params and
    values being optax optimizer. For parameters that are not updatd, values
    should be None
    """
    ngd_optim_ = optax.chain(
        optax.sgd(learning_rate=1.0),
        optax.scale_by_backtracking_linesearch(max_backtracking_steps=15, verbose=True),
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
        return (updates, eqx.tree_at(lambda pt: pt.tx_state, ngd_state, new_tx_state))

    return optax.GradientTransformationExtraArgs(
        init, update
    )  # TODO: is there an easy way to make the typing conform here ?
