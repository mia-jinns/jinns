"""
Implements various utility functions
"""

from math import prod
import warnings
import jax
import jax.numpy as jnp
from jaxtyping import PyTree, Array

from jinns.data._DataGenerators import (
    DataGeneratorODE,
    CubicMeshPDEStatio,
    CubicMeshPDENonStatio,
)


def _check_nan_in_pytree(pytree: PyTree) -> bool:
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
            jax.tree_util.tree_leaves(
                jax.tree_util.tree_map(lambda x: jnp.any(jnp.isnan(x)), pytree)
            )
        )
    )


def get_grid(in_array: Array) -> Array:
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


def _check_user_func_return(
    r: Array | int, shape: tuple, cause: str = ""
) -> Array | float:
    """
    Correctly handles the result from a user defined function (eg a boundary
    condition) to get the correct broadcast
    """
    if isinstance(r, (int, float)):
        # if we have a scalar cast it to float
        return float(r)
    if r.shape == ():
        # if we have a scalar inside a ndarray
        return r.astype(float)
    if r.shape[-1] == shape[-1]:
        # the broadcast will be OK
        return r.astype(float)
    # the reshape below avoids a missing (1,) ending dimension
    # depending on how the user has coded the inital function
    if r.shape != shape:
        warnings.warn(
            f"[{cause}] Performing a operation between arrays"
            f" of different shapes: got {r.shape} and {shape}."
            f" This can cause unexpected broadcast!"
            f" Reshaping {r.shape} into {shape}"
        )
    return r.reshape(shape)


def _subtract_with_check(
    a: Array | int, b: Array | int, cause: str = ""
) -> Array | float:
    a = _check_user_func_return(a, b.shape, cause=cause)
    return a - b
