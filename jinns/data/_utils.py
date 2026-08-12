"""
Utility functions for DataGenerators
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray, Array, Float

if TYPE_CHECKING:
    from jinns.utils._types import AnyBatch
    from jinns.data._Batchs import ObsBatchDict
    from jinns.data._RARParameters import RARParameters


def append_param_batch(
    batch: AnyBatch, param_batch_dict: eqx.Module | None
) -> AnyBatch:
    """
    Utility function that fills the field `batch.param_batch_dict` of a batch object.
    """
    return eqx.tree_at(
        lambda m: m.param_batch_dict,
        batch,
        param_batch_dict,
        is_leaf=lambda x: x is None,
    )


def append_obs_batch(
    batch: AnyBatch, obs_batch_dict: tuple[ObsBatchDict, ...]
) -> AnyBatch:
    """
    Utility function that fills the field `batch.obs_batch_dict` of a batch object
    """
    return eqx.tree_at(
        lambda m: m.obs_batch_dict, batch, obs_batch_dict, is_leaf=lambda x: x is None
    )


def make_cartesian_product(
    b1: Float[Array, " batch_size dim1"], b2: Float[Array, " batch_size dim2"]
) -> Float[Array, " rows=batch_size*batch_size (dim1+dim2)"]:
    # rows= serves to disable jaxtyping wish for runtime check since it does not like the star
    # operator, we wish use not as expected
    """
    Create the cartesian product of a time and a border omega batches
    by tiling and repeating

    ```{python}
    >>> b = jnp.array([4, 5, 6])[..., None]
    >>> a = jnp.array([1, 2, 3])[..., None]
    >>> make_cartesian_product(a, b)
    Array([[1, 4],
        [1, 5],
        [1, 6],
        [2, 4],
        [2, 5],
        [2, 6],
        [3, 4],
        [3, 5],
        [3, 6]], dtype=int32)
    ```
    """
    n1 = b1.shape[0]
    n2 = b2.shape[0]
    b1 = jnp.repeat(b1, n2, axis=0)
    b2 = jnp.tile(b2, reps=(n1,) + tuple(1 for i in b2.shape[1:]))
    return jnp.concatenate([b1, b2], axis=1)


def _reset_batch_idx_and_permute(
    operands: tuple[PRNGKeyArray, Float[Array, " n dimension"], int, None],
) -> tuple[PRNGKeyArray, Float[Array, " n dimension"], int]:
    key, domain, curr_idx, _ = operands
    # resetting counter
    curr_idx = 0
    # reshuffling
    key, subkey = jax.random.split(key)
    domain = jax.random.permutation(subkey, domain, axis=0, independent=False)
    # return updated
    return (key, domain, curr_idx)


def _increment_batch_idx(
    operands: tuple[PRNGKeyArray, Float[Array, " n dimension"], int, int],
) -> tuple[PRNGKeyArray, Float[Array, " n dimension"], int]:
    key, domain, curr_idx, batch_size = operands
    # simply increases counter and get the batch
    curr_idx += batch_size
    return (key, domain, curr_idx)


def _reset_or_increment(
    bend: int,
    n_eff: int,
    operands: tuple[PRNGKeyArray, Float[Array, " n dimension"], int, int],
) -> tuple[PRNGKeyArray, Float[Array, " n dimension"], int]:
    """
    Factorize the code of the jax.lax.cond which checks if we have seen all the
    batches in an epoch
    If bend > n_eff (ie n when no RAR sampling) we reshuffle and start from 0
    again. Otherwise, if bend < n_eff, this means there are still *_batch_size
    samples at least that have not been seen and we can take a new batch

    Parameters
    ----------
    bend
        An integer. The new hypothetical index for the starting of the batch
    n_eff
        An integer. The number of points to see to complete an epoch
    operands
        A tuple. As passed to _reset_batch_idx_and_permute and
        _increment_batch_idx

    Returns
    -------
    res
        A tuple as returned by _reset_batch_idx_and_permute or
        _increment_batch_idx
    """
    return jax.lax.cond(
        bend > n_eff, _reset_batch_idx_and_permute, _increment_batch_idx, operands
    )


def _check_and_set_rar_parameters(
    rar_parameters: RARParameters | None, n: int
) -> int | None:
    if rar_parameters is not None:
        # set internal counter for the number of gradient steps since the
        # last new collocation points have been added
        # It is not 0 to ensure the first iteration of RAR happens just
        # after start_iter. See the _proceed_to_rar() function in _rar.py
        _rar_iter_from_last_sampling = rar_parameters.update_every - 1
        # set iternal counter for the number of times collocation points
        # have been added
    else:
        _rar_iter_from_last_sampling = None

    return _rar_iter_from_last_sampling
