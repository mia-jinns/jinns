"""
Implements various utility functions
"""

from functools import reduce
from operator import getitem
import numpy as np
import jax
import jax.numpy as jnp
import optax


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
            list(
                jax.tree_util.tree_leaves(
                    jax.tree_util.tree_map(lambda x: jnp.any(jnp.isnan(x)), pytree)
                )
            )
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
    return in_array


def _get_vmap_in_axes_params(eq_params_batch_dict, params):
    """
    Return the input vmap axes when there is batch(es) of parameters to vmap
    over. The latter are designated by keys in eq_params_batch_dict
    If eq_params_batch_dict (ie no additional parameter batch), we return None
    """
    if eq_params_batch_dict is None:
        return (None,)
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
    if isinstance(r, (int, float)):
        # if we have a scalar cast it to float
        return float(r)
    if r.shape == () or len(r.shape) == 1:
        # if we have a scalar (or a vector)
        return r.astype(float)
    # if we have an array of the shape of the batch dimension(s) check that
    # we have the correct broadcast
    # the reshape below avoids a missing (1,) ending dimension
    # depending on how the user has coded the inital function
    return r.reshape(shape)


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
    for _ in time_steps[1:]:
        path.append(
            path[-1]
            + step_size * (alpha * (mu - path[-1]))
            + sigma * np.random.normal(loc=0.0, scale=np.sqrt(step_size))
        )

    return time_steps, np.stack(path)
