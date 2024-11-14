"""
Implements various utility functions
"""

import jax
import jax.numpy as jnp
from jaxtyping import PyTree, Array

from jinns.data._DataGenerators import (
    DataGeneratorODE,
    CubicMeshPDEStatio,
    CubicMeshPDENonStatio,
)


def _check_batch_size(other_data, main_data, attr_name):
    if isinstance(main_data, DataGeneratorODE):
        if main_data.temporal_batch_size is not None:
            if getattr(other_data, attr_name) != main_data.temporal_batch_size:
                raise ValueError(
                    f"{other_data.__class__}.{attr_name} must be equal"
                    f" to {main_data.__class__}.temporal_batch_size for correct"
                    " vectorization"
                )
        else:
            if main_data.nt is not None:
                if getattr(other_data, attr_name) != main_data.nt:
                    raise ValueError(
                        f"{other_data.__class__}.{attr_name} must be equal"
                        f" to {main_data.__class__}.nt for correct"
                        " vectorization"
                    )
    if isinstance(main_data, CubicMeshPDEStatio) and not isinstance(
        main_data, CubicMeshPDENonStatio
    ):
        if main_data.omega_batch_size is not None:
            if getattr(other_data, attr_name) != main_data.omega_batch_size:
                raise ValueError(
                    f"{other_data.__class__}.{attr_name} must be equal"
                    f" to {main_data.__class__}.omega_batch_size for correct"
                    " vectorization"
                )
        else:
            if main_data.n is not None:
                if getattr(other_data, attr_name) != main_data.n:
                    raise ValueError(
                        f"{other_data.__class__}.{attr_name} must be equal"
                        f" to {main_data.__class__}.n for correct"
                        " vectorization"
                    )
        if main_data.omega_border_batch_size is not None:
            if getattr(other_data, attr_name) != main_data.omega_border_batch_size:
                raise ValueError(
                    f"{other_data.__class__}.{attr_name} must be equal"
                    f" to {main_data.__class__}.omega_border_batch_size for correct"
                    " vectorization"
                )
        else:
            if main_data.nb is not None:
                if getattr(other_data, attr_name) != main_data.nb:
                    raise ValueError(
                        f"{other_data.__class__}.{attr_name} must be equal"
                        f" to {main_data.__class__}.nb for correct"
                        " vectorization"
                    )
    if isinstance(main_data, CubicMeshPDENonStatio):
        if main_data.domain_batch_size is not None:
            if getattr(other_data, attr_name) != main_data.domain_batch_size:
                raise ValueError(
                    f"{other_data.__class__}.{attr_name} must be equal"
                    f" to {main_data.__class__}.domain_batch_size for correct"
                    " vectorization"
                )
        else:
            if main_data.n is not None:
                if getattr(other_data, attr_name) != main_data.n:
                    raise ValueError(
                        f"{other_data.__class__}.{attr_name} must be equal"
                        f" to {main_data.__class__}.n for correct"
                        " vectorization"
                    )
        if main_data.border_batch_size is not None:
            if getattr(other_data, attr_name) != main_data.border_batch_size:
                raise ValueError(
                    f"{other_data.__class__}.{attr_name} must be equal"
                    f" to {main_data.__class__}.border_batch_size for correct"
                    " vectorization"
                )
        else:
            if main_data.nb is not None:
                if getattr(other_data, attr_name) != (main_data.nb // 2**main_data.dim):
                    raise ValueError(
                        f"{other_data.__class__}.{attr_name} must be equal"
                        f" to ({main_data.__class__}.nb // 2**{main_data.__class__}.dim)"
                        " for correct vectorization"
                    )
        if main_data.initial_batch_size is not None:
            if getattr(other_data, attr_name) != main_data.initial_batch_size:
                raise ValueError(
                    f"{other_data.__class__}.{attr_name} must be equal"
                    f" to {main_data.__class__}.initial_batch_size for correct"
                    " vectorization"
                )
        else:
            if main_data.ni is not None:
                if getattr(other_data, attr_name) != main_data.ni:
                    raise ValueError(
                        f"{other_data.__class__}.{attr_name} must be equal"
                        f" to {main_data.__class__}.ni for correct"
                        " vectorization"
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


def _get_grid(in_array: Array) -> Array:
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


def _check_user_func_return(r: Array | int, shape: tuple) -> Array | int:
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
    return r.reshape(shape)
