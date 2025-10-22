import jax.numpy as jnp
import optax

import jinns
from jinns.parameters._params import Params


# @pytest.fixture
def init_Params_objects():
    n_iter = Params(nn_params=100, eq_params={"theta": 200, "kappa": 300})

    init_params = Params(
        nn_params=jnp.array([10.0, 10.0]),
        eq_params={"theta": jnp.array([1.0]), "kappa": jnp.array([2.0])},
    )

    optimizers = Params(
        nn_params=optax.adam(learning_rate=1e-3),
        eq_params={
            "theta": optax.radam(learning_rate=1e-3),
            "kappa": optax.adamw(learning_rate=1e-3),
        },
    )

    return n_iter, init_params, optimizers


n_iter, init_params, optimizers = init_Params_objects()

loss_kwargs = {
    "u": None,
    "loss_weights": None,
    "dynamic_loss": None,
    "derivative_keys": None,  # defaults to nn_params everywhere
    "params": init_params,
}

loss = jinns.loss.LossODE(**loss_kwargs)

# jinns.solve_alternate(n_iter, init_params, None, loss, optimizers)
