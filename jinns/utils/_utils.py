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


def _check_shape_and_type(
    r: Array | int, expected_shape: tuple, cause: str = "", binop: str = ""
) -> Array | float:
    """
    Ensures float type and correct shapes for broadcasting when performing a
    binary operation (like -, + or *) between two arrays.
    First array is a custom user (observation data or output of initial/BC
    functions), the expected shape is the same as the PINN's.
    """
    if isinstance(r, (int, float)):
        # if we have a scalar cast it to float
        return float(r)
    if r.shape == ():
        # if we have a scalar inside a ndarray
        return r.astype(float)
    if r.shape[-1] == expected_shape[-1]:
        # broadcasting will be OK
        return r.astype(float)

    if r.shape != expected_shape:
        # Usually, the reshape below  adds a missing (1,) final axis to ensure # the PINN output and the other function (initial/boundary condition)
        # have the correct shape, depending on how the user has coded the
        # initial/boundary condition.
        warnings.warn(
            f"[{cause}] Performing operation `{binop}` between arrays"
            f" of different shapes: got {r.shape} for the custom array and"
            f" {expected_shape} for the PINN."
            f" This can cause unexpected and wrong broadcasting."
            f" Reshaping {r.shape} into {expected_shape}. Reshape your"
            f" custom array to math the {expected_shape=} to prevent this"
            f" warning."
        )
    return r.reshape(expected_shape)


def _subtract_with_check(
    a: Array | int, b: Array | int, cause: str = ""
) -> Array | float:
    a = _check_shape_and_type(a, b.shape, cause=cause, binop="-")
    return a - b
