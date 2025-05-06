"""
Utility functions for DataGenerators
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Key, Array, Float

if TYPE_CHECKING:
    from jinns.utils._types import AnyBatch
    from jinns.data._Batchs import ObsBatchDict


def append_param_batch(batch: AnyBatch, param_batch_dict: dict[str, Array]) -> AnyBatch:
    """
    Utility function that fills the field `batch.param_batch_dict` of a batch object.
    """
    return eqx.tree_at(
        lambda m: m.param_batch_dict,
        batch,
        param_batch_dict,
        is_leaf=lambda x: x is None,
    )


def append_obs_batch(batch: AnyBatch, obs_batch_dict: ObsBatchDict) -> AnyBatch:
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
    """
    n1 = b1.shape[0]
    n2 = b2.shape[0]
    b1 = jnp.repeat(b1, n2, axis=0)
    b2 = jnp.tile(b2, reps=(n1,) + tuple(1 for i in b2.shape[1:]))
    return jnp.concatenate([b1, b2], axis=1)


def _reset_batch_idx_and_permute(
    operands: tuple[Key, Float[Array, " n dimension"], int, None, Float[Array, " n"]],
) -> tuple[Key, Float[Array, " n dimension"], int]:
    key, domain, curr_idx, _, p = operands
    # resetting counter
    curr_idx = 0
    # reshuffling
    key, subkey = jax.random.split(key)
    if p is None:
        domain = jax.random.permutation(subkey, domain, axis=0, independent=False)
    else:
        # otherwise p is used to avoid collocation points not in n_start
        # NOTE that replace=True to avoid undefined behaviour but then, the
        # domain.shape[0] does not really grow as in the original RAR. instead,
        # it always comprises the same number of points, but the points are
        # updated
        domain = jax.random.choice(
            subkey, domain, shape=(domain.shape[0],), replace=True, p=p
        )

    # return updated
    return (key, domain, curr_idx)


def _increment_batch_idx(
    operands: tuple[Key, Float[Array, " n dimension"], int, int, Float[Array, " n"]],
) -> tuple[Key, Float[Array, " n dimension"], int]:
    key, domain, curr_idx, batch_size, _ = operands
    # simply increases counter and get the batch
    curr_idx += batch_size
    return (key, domain, curr_idx)


def _reset_or_increment(
    bend: int,
    n_eff: int,
    operands: tuple[Key, Float[Array, " n dimension"], int, int, Float[Array, " n"]],
) -> tuple[Key, Float[Array, " n dimension"], int]:
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
    rar_parameters: dict, n: int, n_start: int
) -> tuple[int, Float[Array, " n"] | None, int | None, int | None]:
    if rar_parameters is not None and n_start is None:
        raise ValueError(
            "n_start must be provided in the context of RAR sampling scheme"
        )

    if rar_parameters is not None:
        # Default p is None. However, in the RAR sampling scheme we use 0
        # probability to specify non-used collocation points (i.e. points
        # above n_start). Thus, p is a vector of probability of shape (nt, 1).
        p = jnp.zeros((n,))
        p = p.at[:n_start].set(1 / n_start)
        # set internal counter for the number of gradient steps since the
        # last new collocation points have been added
        # It is not 0 to ensure the first iteration of RAR happens just
        # after start_iter. See the _proceed_to_rar() function in _rar.py
        rar_iter_from_last_sampling = rar_parameters["update_every"] - 1
        # set iternal counter for the number of times collocation points
        # have been added
        rar_iter_nb = 0
    else:
        n_start = n
        p = None
        rar_iter_from_last_sampling = None
        rar_iter_nb = None

    return n_start, p, rar_iter_from_last_sampling, rar_iter_nb
