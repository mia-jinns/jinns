"""
Utility functions for DataGenerators
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Literal
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray, Array, Float

if TYPE_CHECKING:
    from jinns.utils._types import AnyBatch
    from jinns.data._Batchs import ObsBatchDict


class RARParameters(eqx.Module):
    """
    Dataclass to specify the Residual Adaptative Resampling procedure

    RAR methods as inspired from https://arxiv.org/pdf/2207.10289
    However the critical difference is that the dataset size is fixed. So new points replace others


    Parameters
    ----------
    update_every : int
        the number of gradient steps taken between
        each update of collocation points in the RAR algo.
    novelty_proportion : float
        the proportion of the batchsize which is replaced
        by new samples at each RAR step
    method : Literal['G', 'D']
        - either "G" for RAR-G, ie, new points replace the batch points with the lowest dynamic loss
        - either "D" for RAR-D, ie, new points replace the batch points that have not been selected when
        `resampling batch_size * (1 - novelty_proportion)` points among the batch points with weigths given
        by the formula (2) in the article below. In this case, RARParameters must specify the keys 'k' and 'c'
        with float values as defined in the formula.
    start_iter : int, default=0
        the iteration at which we start the RAR sampling scheme (we first have a "burn-in" period).
    k : Array
        the float value of k, only for RAR-D. When no prior information the article recommends k=2.0
    c : Array
        the float value of c, only for RAR-D. When no prior information the article recommends c=0.0
    """

    update_every: int = eqx.field(static=True, kw_only=True)
    novelty_proportion: float = eqx.field(static=True, kw_only=True)
    method: Literal["G", "D"] = eqx.field(static=True, kw_only=True)
    start_iter: int = eqx.field(default=0, static=True, kw_only=True)
    k: Array | None = eqx.field(default=None, static=True, kw_only=True)
    c: Array | None = eqx.field(default=None, static=True, kw_only=True)

    _rar_iter_from_last_sampling: int = eqx.field(init=False)

    def __post_init__(self):
        if self.method == "D" and (self.k is None or self.c is None):
            raise ValueError("k and c must be specified for RAR-D")
        if self.novelty_proportion < 0 or self.novelty_proportion > 1:
            raise ValueError("novelty_proportion must be in [0; 1]")
        if self.k is not None and self.c is not None:
            self.k = jnp.array(self.k)
            self.c = jnp.array(self.c)

        self._rar_iter_from_last_sampling = 0


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
