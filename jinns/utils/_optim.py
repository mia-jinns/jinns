"""
Implements utility functions for optimization
"""

import optax
import jax
import jax.numpy as jnp


def alternate_optimizer(list_first_params, list_second_params, n_iter, evry, tx1, tx2):
    """
    Alternatively optimize on two sets of parameters for equal number of steps

    Parameters
    ----------
    list_first_params
        The first set of parameter to optimize on. A list of leaves from the `params` dict
    list_second_params
        The second set of parameter to optimize on. A list of leaves from the `params` dict
    n_iter
        The total number of iterations
    evry
        The number of iterations we spend optimizing a set of parameters before switching
    tx1
        An optax optimizer of the set of parameters 1
    tx2
        An optax optimizer of the set of parameters 1

    Returns
    -------
    tx
        An optax optimizer object
    """

    def map_nested_fn(fn):
        """Recursively apply `fn` to the key-value pairs of a nested dict"""

        def map_fn(nested_dict):
            return {
                k: (map_fn(v) if isinstance(v, dict) else fn(k, v))
                for k, v in nested_dict.items()
            }

        return map_fn

    label_fn = map_nested_fn(lambda k, _: k)

    def should_update_1(step):
        return jax.tree_util.tree_reduce(
            lambda x, y: jnp.logical_or(x, y),
            [
                jnp.logical_and((step > i * evry), (step < (i + 1) * evry))
                for i in range(1, n_iter // evry, 2)
            ],
        )

    def should_update_2(step):
        return jax.tree_util.tree_reduce(
            lambda x, y: jnp.logical_or(x, y),
            [
                jnp.logical_and((step > i * evry), (step < (i + 1) * evry))
                for i in range(
                    0, n_iter // evry, 2
                )  # starts at 0 since this one is blocked first
            ],
        )

    first_adam = optax.chain(
        tx1,
        optax.maybe_update(
            optax.scale(0.0), should_update_1
        ),  # We add an update (a GradientTransform if should_update is True) i.e. we mult the update by 0.
        # not to take a step
    )
    second_adam = optax.chain(
        tx2,
        optax.maybe_update(
            optax.scale(0.0), should_update_2
        ),  # We add an update (a GradientTransform if should_update is True) i.e. we mult the update by 0.
        # not to take a step
    )

    return optax.multi_transform(
        {k: first_adam for k in list_first_params}
        | {
            k: second_adam for k in list_second_params
        },  # those gradient transforms must correspond to leaves of parameter pytree
        label_fn,
    )


def delayed_optimizer(list_first_params, list_second_params, delay_steps, tx1, tx2):
    """
    Optimize on two sets of parameters, the optimization on the second set of
    parameters start after `delay_steps` of freezing

    Parameters
    ----------
    list_first_params
        The first set of parameter to optimize on. A list of leaves from the `params` dict
    list_second_params
        The second set of parameter to optimize on. A list of leaves from the `params` dict
    delay_steps
        The number of steps we wait before starting the optimization on the
        second set of parameters
    tx1
        An optax optimizer of the set of parameters 1
    tx2
        An optax optimizer of the set of parameters 1

    Returns
    -------
    tx
        An optax optimizer object
    """

    def map_nested_fn(fn):
        """Recursively apply `fn` to the key-value pairs of a nested dict"""

        def map_fn(nested_dict):
            return {
                k: (map_fn(v) if isinstance(v, dict) else fn(k, v))
                for k, v in nested_dict.items()
            }

        return map_fn

    label_fn = map_nested_fn(lambda k, _: k)

    def should_update_2(step):
        return step < delay_steps

    delayed_tx2 = optax.chain(
        tx2,
        optax.maybe_update(
            optax.scale(0.0), should_update_2
        ),  # We add an update (a GradientTransform if should_update is True) i.e. we mult the update by 0.
        # not to take a step
    )

    return optax.multi_transform(
        {k: tx1 for k in list_first_params}
        | {
            k: delayed_tx2 for k in list_second_params
        },  # those gradient transforms must correspond to leaves of parameter pytree
        label_fn,
    )
