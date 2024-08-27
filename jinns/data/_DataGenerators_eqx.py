"""
Define the DataGeneratorODE equinox module
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Key, Int, PyTree, Array
from typing import Union, NamedTuple


# TODO ? change to eqx.Module, but this require an update of all the _replace()
# calls
class ODEBatch(NamedTuple):  # eqx.Module):
    temporal_batch: Array
    param_batch_dict: dict = None
    obs_batch_dict: dict = None


class PDENonStatioBatch(NamedTuple):  # eqx.Module):
    inside_batch: Array
    border_batch: Array
    temporal_batch: Array
    param_batch_dict: dict = None
    obs_batch_dict: dict = None


class PDEStatioBatch(NamedTuple):  # eqx.Module):
    inside_batch: Array
    border_batch: Array
    param_batch_dict: dict = None
    obs_batch_dict: dict = None


def _reset_batch_idx_and_permute(operands):
    key, domain, curr_idx, _, p = operands
    # resetting counter
    curr_idx = 0
    # reshuffling
    key, subkey = jax.random.split(key)
    # domain = random.permutation(subkey, domain, axis=0, independent=False)
    # we want that permutation = choice when p=None
    # otherwise p is used to avoid collocation points not in nt_start
    domain = jax.random.choice(
        subkey, domain, shape=(domain.shape[0],), replace=False, p=p
    )

    # return updated
    return (key, domain, curr_idx)


def _increment_batch_idx(operands):
    key, domain, curr_idx, batch_size, _ = operands
    # simply increases counter and get the batch
    curr_idx += batch_size
    return (key, domain, curr_idx)


def _reset_or_increment(bend, n_eff, operands):
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


class DataGeneratorODE_eqx(eqx.Module):
    """
    A class implementing data generator object for ordinary differential equations.

    Parameters
    ----------
    key
        Jax random key to sample new time points and to shuffle batches
    nt
        An integer. The number of total time points that will be divided in
        batches. Batches are made so that each data point is seen only
        once during 1 epoch.
    tmin
        A float. The minimum value of the time domain to consider
    tmax
        A float. The maximum value of the time domain to consider
    temporal_batch_size
        An integer. The size of the batch of randomly selected points among
        the `nt` points.
    method
        Either `grid` or `uniform`, default is `uniform`.
        The method that generates the `nt` time points. `grid` means
        regularly spaced points over the domain. `uniform` means uniformly
        sampled points over the domain
    rar_parameters
        Default to None: do not use Residual Adaptative Resampling.
        Otherwise a dictionary with keys. `start_iter`: the iteration at
        which we start the RAR sampling scheme (we first have a burn in
        period). `update_rate`: the number of gradient steps taken between
        each appending of collocation points in the RAR algo.
        `sample_size`: the size of the sample from which we will select new
        collocation points. `selected_sample_size`: the number of selected
        points from the sample to be added to the current collocation
        points
        "DeepXDE: A deep learning library for solving differential
        equations", L. Lu, SIAM Review, 2021
    nt_start
        Defaults to None. The effective size of nt used at start time.
        This value must be
        provided when rar_parameters is not None. Otherwise we set internally
        nt_start = nt and this is hidden from the user.
        In RAR, nt_start
        then corresponds to the initial number of points we train the PINN.
    """

    key: Key
    nt: Int
    tmin: Int
    tmax: Int
    temporal_batch_size: Int = eqx.field(static=True)  # static cause used as a
    # shape in jax.lax.dynamic_slice
    method: str = eqx.field(static=True, default_factory=lambda: "uniform")
    rar_parameters: PyTree = None
    nt_start: int = None

    # all the init=False fields are set in __post_init__, even after a _replace
    # or eqx.tree_at __post_init__ is called
    p_times: Array = eqx.field(init=False)
    rar_iter_from_last_sampling: Array = eqx.field(init=False)
    rar_iter_nb: Array = eqx.field(init=False)
    curr_time_idx: Int = eqx.field(init=False)
    times: Array = eqx.field(init=False)

    def __post_init__(self):
        if self.rar_parameters is not None and self.nt_start is None:
            raise ValueError(
                "nt_start must be provided in the context of RAR sampling scheme"
            )

        if self.rar_parameters is not None:
            # Default p is None. However, in the RAR sampling scheme we use 0
            # probability to specify non-used collocation points (i.e. points
            # above nt_start). Thus, p is a vector of probability of shape (nt, 1).
            self.p_times = jnp.zeros((self.nt,))
            self.p_times = self.p_times.at[: self.nt_start].set(1 / self.nt_start)
            # set internal counter for the number of gradient steps since the
            # last new collocation points have been added
            self.rar_iter_from_last_sampling = 0
            # set iternal counter for the number of times collocation points
            # have been added
            self.rar_iter_nb = 0

        if self.rar_parameters is None or self.nt_start is None:
            self.nt_start = self.nt
            self.p_times = None
            self.rar_iter_from_last_sampling = None
            self.rar_iter_nb = None

        self.curr_time_idx = jnp.iinfo(jnp.int32).max - self.temporal_batch_size - 1
        # to be sure there is a
        # shuffling at first get_batch() we do not call
        # _reset_batch_idx_and_permute in __init__ or __post_init__ because it
        # would return a copy of self and we have not investigate what would
        # happen
        # NOTE the (- self.temporal_batch_size - 1) because otherwise when computing
        # `bend` we overflow the max int32 with unwanted behaviour
        key, subkey = jax.random.split(self.key)
        self.key = key
        self.times = self.generate_time_data(subkey)

        # Note that, here, in __init__ (and __post_init__), this is the
        # only place where self assignment are authorized so we do the
        # above way for the key. But note that a method returning a copy
        # for an out-of-place key update is not possible because we would
        # replace the self inside its __init__ (by the new); that is, the
        # following lines break on the error "times attributes not
        # initialized" :D
        # new, key = self._get_key()
        # new.times = self.generate_time_data(subkey)

    def sample_in_time_domain(self, key) -> Array:
        return jax.random.uniform(key, (self.nt,), minval=self.tmin, maxval=self.tmax)

    def generate_time_data(self, key) -> Array:
        """
        Construct a complete set of `self.nt` time points according to the
        specified `self.method`

        Note that self.times has always size self.nt and not self.nt_start, even
        in RAR scheme, we must allocate all the collocation points
        """
        if self.method == "grid":
            partial_times = (self.tmax - self.tmin) / self.nt
            return jnp.arange(self.tmin, self.tmax, partial_times)
        if self.method == "uniform":
            return self.sample_in_time_domain(key)
        raise ValueError("Method " + self.method + " is not implemented.")

    def _get_time_operands(self):
        return (
            self.key,
            self.times,
            self.curr_time_idx,
            self.temporal_batch_size,
            self.p_times,
        )

    def temporal_batch(self) -> tuple["DataGeneratorODE", Array]:
        """
        Return a batch of time points. If all the batches have been seen, we
        reshuffle them, otherwise we just return the next unseen batch.
        """
        bstart = self.curr_time_idx
        bend = bstart + self.temporal_batch_size

        # Compute the effective number of used collocation points
        if self.rar_parameters is not None:
            nt_eff = (
                self.nt_start
                + self.rar_iter_nb * self.rar_parameters["selected_sample_size"]
            )
        else:
            nt_eff = self.nt

        new_attributes = _reset_or_increment(bend, nt_eff, self._get_time_operands())
        new = eqx.tree_at(
            lambda m: (m.key, m.times, m.curr_time_idx), self, new_attributes
        )

        # commands below are equivalent to
        # return self.times[i:(i+t_batch_size)]
        # start indices can be dynamic be the slice shape is fixed
        return new, jax.lax.dynamic_slice(
            new.times,
            start_indices=(new.curr_time_idx,),
            slice_sizes=(new.temporal_batch_size,),
        )

    def get_batch(self) -> tuple["DataGeneratorODE", ODEBatch]:
        """
        Generic method to return a batch. Here we call `self.temporal_batch()`
        """
        new, temporal_batch = self.temporal_batch()
        return new, ODEBatch(temporal_batch=temporal_batch)
