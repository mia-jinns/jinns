"""
Define the DataGeneratorODE equinox module
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Key, Int, PyTree, Array
from jinns.data._AbstractDataGenerator import (
    ODEBatch,
    AbstractDataGenerator,
    _reset_or_increment,
)


class DataGeneratorODE_eqx(AbstractDataGenerator):
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
    p: Array = eqx.field(init=False)
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
            self.p = jnp.zeros((self.nt,))
            self.p = self.p.at[: self.nt_start].set(1 / self.nt_start)
            # set internal counter for the number of gradient steps since the
            # last new collocation points have been added
            self.rar_iter_from_last_sampling = 0
            # set iternal counter for the number of times collocation points
            # have been added
            self.rar_iter_nb = 0

        if self.rar_parameters is None or self.nt_start is None:
            self.nt_start = self.nt
            self.p = None
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

    # Next we set some @properties to interact with the abstract base class
    # attribute names
    # Note that the setters are only (for validity of prperty attribute) and
    # for initialization (the only place where a
    # self assignement is authorized in eqx.Module). Moreover the attributes
    # the properties point to are set in the __post_init__
    # Otherwise all the following modifications of the attributes will be done
    # out-of-place by eqx.tree_at (no use of setters). Trying to use the
    # setters later on will anyway raise an error since we cannot modify
    # attributes
    # BUT eqx.tree_at seems to take care of properties since
    # eqx.tree_at(lambda m:m.curr_time_idx, train_data, 0) does update both
    # curr_time_idx and curr_idx
    @property
    def curr_time_idx(self):
        return self.curr_idx

    @curr_time_idx.setter
    def curr_time_idx(self, val):
        self.curr_idx = val

    @property
    def times(self):
        return self.domain

    @times.setter
    def times(self, val):
        self.domain = val

    @property
    def temporal_batch_size(self):
        return self.batch_size

    @temporal_batch_size.setter
    def temporal_batch_size(self, val):
        self.batch_size = val
        # Note
        # the following with walrus assignement notation could be possible?
        # but this would be a little bit more obscure. Eg.:
        # return eqx.tree_at(lambda m: m.batch_size, self, val)
        # and assign like new := temporal_batch_size(val)

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

        new = _reset_or_increment(self, bend, nt_eff)

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


if __name__ == "__main__":
    timeDG = DataGeneratorODE(jax.random.PRNGKey(0), 1024, 0, 1, 32)
    print(timeDG.rar_iter_nb)
    timeDG = eqx.tree_at(
        lambda t: t.rar_iter_nb, timeDG, 1000, is_leaf=lambda x: x is None
    )
    print(timeDG.rar_iter_nb)
    vals, treedef = jax.tree.flatten(timeDG)
    timeDG = jax.tree.unflatten(treedef, vals)
    print(timeDG.rar_iter_nb)  # we see that we do not go again in the __post_init__
    timeDG.get_batch()
