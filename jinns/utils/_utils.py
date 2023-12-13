import numpy as np
import jax
import jax.numpy as jnp
import optax
import equinox as eqx
from functools import reduce
from operator import getitem


def _check_nan_in_pytree(pytree):
    """
    Check if there is a NaN value anywhere is the pytree

    Parameters
    ----------
    pytree
        A pytree

    Returns
    -------
    res
        A boolean. True if any of the pytree content is NaN
    """
    return jnp.any(
        jnp.array(
            [
                value
                for value in jax.tree_util.tree_leaves(
                    jax.tree_util.tree_map(lambda x: jnp.any(jnp.isnan(x)), pytree)
                )
            ]
        )
    )


def _tracked_parameters(params, tracked_params_key_list):
    """
    Returns a pytree with the same structure as params with True is the
    parameter is tracked False otherwise
    """

    def set_nested_item(dataDict, mapList, val):
        """
        Set item in nested dictionary
        https://stackoverflow.com/questions/54137991/how-to-update-values-in-nested-dictionary-if-keys-are-in-a-list
        """
        reduce(getitem, mapList[:-1], dataDict)[mapList[-1]] = val
        return dataDict

    tracked_params = jax.tree_util.tree_map(
        lambda x: False, params
    )  # init with all False

    for key_list in tracked_params_key_list:
        tracked_params = set_nested_item(tracked_params, key_list, True)

    return tracked_params


def _get_grid(in_array):
    """
    From an array of shape (B, D), D > 1, get the grid array, i.e., an array of
    shape (B, B, ...(D times)..., B, D): along the last axis we have the array
    of values
    """
    if in_array.shape[-1] > 1 or in_array.ndim > 1:
        return jnp.stack(
            jnp.meshgrid(
                *(in_array[..., d] for d in range(in_array.shape[-1])), indexing="ij"
            ),
            axis=-1,
        )
    else:
        return in_array


def _get_vmap_in_axes_params(eq_params_batch_dict, params):
    """
    Return the input vmap axes when there is batch(es) of parameters to vmap
    over. The latter are designated by keys in eq_params_batch_dict
    If eq_params_batch_dict (ie no additional parameter batch), we return None
    """
    if eq_params_batch_dict is None:
        return (None,)
    else:
        # We use pytree indexing of vmapped axes and vmap on axis
        # 0 of the eq_parameters for which we have a batch
        # this is for a fine-grained vmaping
        # scheme over the params
        vmap_in_axes_params = (
            {
                "eq_params": {
                    k: (0 if k in eq_params_batch_dict.keys() else None)
                    for k in params["eq_params"].keys()
                },
                "nn_params": None,
            },
        )
        return vmap_in_axes_params


def _check_user_func_return(r, shape):
    """
    Correctly handles the result from a user defined function (eg a boundary
    condition) to get the correct broadcast
    """
    if isinstance(r, int) or isinstance(r, float):
        # if we have a scalar cast it to float
        return float(r)
    if r.shape == () or len(r.shape) == 1:
        # if we have a scalar (or a vector, but no batch dim) inside an array
        return r.astype(float)
    else:
        # if we have an array of the shape of the batch dimension(s) check that
        # we have the correct broadcast
        # the reshape below avoids a missing (1,) ending dimension
        # depending on how the user has coded the inital function
        return r.reshape(shape)


def alternate_optax_solver(
    steps, parameters_set1, parameters_set2, lr_set1, lr_set2, label_fn=None
):
    """
    This function creates an optax optimizer that alternates the optimization
    between two set of parameters (ie. when some parameters are update to a
    given learning rates, others are not updated (learning rate = 0)
    The optimizers are scaled by adam parameters.

    __Note:__ The alternating pattern relies on
    `optax.piecewise_constant_schedule` which __multiplies__ learning rates of
    previous steps (current included) to set the new learning rate. Hence, our
    strategy used here is to relying on potentially cancelling power of tens to
    create the alternating scheme.

    Parameters
    ----------
    steps
        An array which describes the epochis number at which we alternate the
        optimization: the parameter_set that is being updated now stops
        updating, the other parameter_set starts updating.
        __Note:__ The step 0 should not be included
    parameters_set1
        A list of leaf level keys which must be found in the general `params` dict. The
        parameters in this `set1` will be the parameters which are updated
        first in the alternating scheme.
    parameters_set2
        A list of leaf level keys which must be found in the general `params` dict. The
        parameters in this `set2` will be the parameters which are not updated
        first in the alternating scheme.
    lr_set1
        A float. The learning rate of updates for set1.
    lr_set2
        A float. The learning rate of updates for set2.
    label_fn
        The same function as the label_fn function passed in an optax
        `multi_transform`
        [https://optax.readthedocs.io/en/latest/api.html#optax.multi_transform](see
        here)
        Default None, ie, we already internally provide the default one (as
        proposed in the optax documentation) which may suit many use cases

    Returns
    -------
    tx
        The optax optimizer object
    """

    def map_nested_fn(fn):
        """
        Recursively apply `fn` to the key-value pairs of a nested dict
        We follow the example from
        https://optax.readthedocs.io/en/latest/api.html#optax.multi_transform
        for different learning rates
        """

        def map_fn(nested_dict):
            return {
                k: (map_fn(v) if isinstance(v, dict) else fn(k, v))
                for k, v in nested_dict.items()
            }

        return map_fn

    label_fn = map_nested_fn(lambda k, _: k)

    power_to_0 = 1e-25  # power of ten used to force a learning rate to 0
    power_to_lr = 1 / power_to_0  # power of ten used to force a learning rate to lr
    nn_params_scheduler = optax.piecewise_constant_schedule(
        init_value=lr_set1,
        boundaries_and_scales={
            k: (
                power_to_0
                if even_odd % 2 == 0  # set lr to 0 eg if even_odd is even ie at
                # first step
                else power_to_lr
            )
            for even_odd, k in enumerate(steps)
        },
    )
    eq_params_scheduler = optax.piecewise_constant_schedule(
        init_value=power_to_0 * lr_set2,  # so normal learning rate is 1e-3
        boundaries_and_scales={
            k: (power_to_lr if even_odd % 2 == 0 else power_to_0)
            for even_odd, k in enumerate(steps)
        },
    )

    # the scheduler for set1 is called nn_chain because we usually start by
    # updating the NN parameters
    nn_chain = optax.chain(
        optax.scale_by_adam(),
        optax.scale_by_schedule(nn_params_scheduler),
        optax.scale(-1.0),
    )
    eq_chain = optax.chain(
        optax.scale_by_adam(),
        optax.scale_by_schedule(eq_params_scheduler),
        optax.scale(-1.0),
    )
    dict_params_set1 = {p: nn_chain for p in parameters_set1}
    dict_params_set2 = {p: eq_chain for p in parameters_set2}
    tx = optax.multi_transform(
        {**dict_params_set1, **dict_params_set2},
        label_fn,
    )

    return tx


def euler_maruyama_density(t, x, s, y, params, Tmax=1):
    eps = 1e-6
    delta = jnp.abs(t - s) * Tmax
    mu = params["alpha_sde"] * (params["mu_sde"] - y) * delta
    var = params["sigma_sde"] ** 2 * delta
    return (
        1 / jnp.sqrt(2 * jnp.pi * var) * jnp.exp(-0.5 * ((x - y) - mu) ** 2 / var) + eps
    )


def log_euler_maruyama_density(t, x, s, y, params):
    eps = 1e-6
    delta = jnp.abs(t - s)
    mu = params["alpha_sde"] * (params["mu_sde"] - y) * delta
    logvar = params["logvar_sde"]
    return (
        -0.5
        * (jnp.log(2 * jnp.pi * delta) + logvar + ((x - y) - mu) ** 2 / jnp.exp(logvar))
        + eps
    )


def euler_maruyama(x0, alpha, mu, sigma, T, N):
    """
    Simulate 1D diffusion process with simple parametrization using the Euler
    Maruyama method in the interval [0, T]
    """
    path = [np.array([x0])]

    time_steps, step_size = np.linspace(0, T, N, retstep=True)
    for i in time_steps[1:]:
        path.append(
            path[-1]
            + step_size * (alpha * (mu - path[-1]))
            + sigma * np.random.normal(loc=0.0, scale=np.sqrt(step_size))
        )

    return time_steps, np.stack(path)
