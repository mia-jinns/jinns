"""
Util functions for DataGenerators
"""

import abc
import equinox as eqx
import jax
import jax.numpy as jnp
from typing import Union
from jaxtyping import Key, Int, Bool, PyTree, Array


class ODEBatch(eqx.Module):
    temporal_batch: Array
    param_batch_dict: dict = None
    obs_batch_dict: dict = None


class PDENonStatioBatch(eqx.Module):
    inside_batch: Array
    border_batch: Array
    temporal_batch: Array
    param_batch_dict: dict = None
    obs_batch_dict: dict = None


class PDEStatioBatch(eqx.Module):
    inside_batch: Array
    border_batch: Array
    param_batch_dict: dict = None
    obs_batch_dict: dict = None


class AbstractDataGenerator(eqx.Module):
    # no eqx.AbstractVar because we do not want to deal with those variables in
    # the child class
    curr_idx: Int = eqx.field(init=False)
    batch_size: Int = eqx.field(init=False)
    p: Int = eqx.field(init=False)
    domain: Array = eqx.field(init=False)

    @abc.abstractmethod
    def get_batch(
        self,
    ) -> tuple[
        "AbstractDataGenerator", Union[ODEBatch, PDEStatioBatch, PDENonStatioBatch]
    ]:
        raise NotImplementedError


def _reset_batch_idx_and_permute(
    data_generator: "AbstractDataGenerator",
) -> "AbstractDataGenerator":
    # resetting counter
    # reshuffling
    # we want that permutation = choice when p=None
    # otherwise p is used to avoid collocation points not in nt_start
    key, subkey = jax.random.split(data_generator.key)
    new = eqx.tree_at(
        lambda m: (m.curr_idx, m.key, m.domain),
        data_generator,
        (
            0,
            key,
            jax.random.choice(
                subkey,
                data_generator.domain,
                shape=(data_generator.domain.shape[0],),
                replace=False,
                p=data_generator.p,
            ),
        ),
    )
    return new


def _increment_batch_idx(
    data_generator: "AbstractDataGenerator",
) -> "AbstractDataGenerator":
    return eqx.tree_at(
        lambda m: m.curr_idx,
        data_generator,
        data_generator.curr_idx + data_generator.batch_size,
    )


def _reset_or_increment(
    data_generator: "AbstractDataGenerator", bend: Int, n_eff: Int
) -> "AbstractDataGenerator":
    """
    Factorize the code of the jax.lax.cond which checks if we have seen all the
    batches in an epoch
    If bend > n_eff (ie n when no RAR sampling) we reshuffle and start from 0
    again. Otherwise, if bend < n_eff, this means there are still *_batch_size
    samples at least that have not been seen and we can take a new batch

    Parameters
    ----------
    data_generator
        A DataGenerator
    bend
        An integer. The new hypothetical index for the starting of the batch
    n_eff
        An integer. The number of points to see to complete an epoch

    Returns
    -------
    res
        A DataGenerator with updated attributes
    """
    return jax.lax.cond(
        bend > n_eff, _reset_batch_idx_and_permute, _increment_batch_idx, data_generator
    )
