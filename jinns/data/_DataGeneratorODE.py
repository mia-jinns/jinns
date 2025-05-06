"""
Define the DataGenerators modules
"""

from __future__ import (
    annotations,
)  # https://docs.python.org/3/library/typing.html#constant
from typing import TYPE_CHECKING
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Key, Array, Float
from jinns.data._Batchs import ODEBatch
from jinns.data._utils import _check_and_set_rar_parameters, _reset_or_increment
from jinns.data._AbstractDataGenerator import AbstractDataGenerator

if TYPE_CHECKING:
    pass


class DataGeneratorODE(AbstractDataGenerator):
    """
    A class implementing data generator object for ordinary differential equations.

    Parameters
    ----------
    key : Key
        Jax random key to sample new time points and to shuffle batches
    nt : int
        The number of total time points that will be divided in
        batches. Batches are made so that each data point is seen only
        once during 1 epoch.
    tmin : float
        The minimum value of the time domain to consider
    tmax : float
        The maximum value of the time domain to consider
    temporal_batch_size : int | None, default=None
        The size of the batch of randomly selected points among
        the `nt` points. If None, no minibatches are used.
    method : str, default="uniform"
        Either `grid` or `uniform`, default is `uniform`.
        The method that generates the `nt` time points. `grid` means
        regularly spaced points over the domain. `uniform` means uniformly
        sampled points over the domain
    rar_parameters : RarParameterDict, default=None
       A TypedDict to specify the Residual Adaptative Resampling procedure. See
       the docstring from RarParameterDict
    n_start : int, default=None
        Defaults to None. The effective size of nt used at start time.
        This value must be
        provided when rar_parameters is not None. Otherwise we set internally
        n_start = nt and this is hidden from the user.
        In RAR, n_start
        then corresponds to the initial number of points we train the PINN.
    """

    key: Key = eqx.field(kw_only=True)
    nt: int = eqx.field(kw_only=True, static=True)
    tmin: Float = eqx.field(kw_only=True)
    tmax: Float = eqx.field(kw_only=True)
    temporal_batch_size: int | None = eqx.field(static=True, default=None, kw_only=True)
    method: str = eqx.field(
        static=True, kw_only=True, default_factory=lambda: "uniform"
    )
    rar_parameters: dict[str, int] = eqx.field(default=None, kw_only=True)
    n_start: int = eqx.field(static=True, default=None, kw_only=True)

    # all the init=False fields are set in __post_init__
    p: Float[Array, " nt 1"] | None = eqx.field(init=False)
    rar_iter_from_last_sampling: int | None = eqx.field(init=False)
    rar_iter_nb: int | None = eqx.field(init=False)
    curr_time_idx: int = eqx.field(init=False)
    times: Float[Array, " nt 1"] = eqx.field(init=False)

    def __post_init__(self):
        (
            self.n_start,
            self.p,
            self.rar_iter_from_last_sampling,
            self.rar_iter_nb,
        ) = _check_and_set_rar_parameters(self.rar_parameters, self.nt, self.n_start)

        if self.temporal_batch_size is not None:
            self.curr_time_idx = self.nt + self.temporal_batch_size
            # to be sure there is a shuffling at first get_batch()
            # NOTE in the extreme case we could do:
            # self.curr_time_idx=jnp.iinfo(jnp.int32).max - self.temporal_batch_size - 1
            # but we do not test for such extreme values. Where we subtract
            # self.temporal_batch_size - 1 because otherwise when computing
            # `bend` we do not want to overflow the max int32 with unwanted behaviour
        else:
            self.curr_time_idx = 0

        self.key, self.times = self.generate_time_data(self.key)
        # Note that, here, in __init__ (and __post_init__), this is the
        # only place where self assignment are authorized so we do the
        # above way for the key.

    def sample_in_time_domain(
        self, key: Key, sample_size: int | None = None
    ) -> Float[Array, " nt 1"]:
        return jax.random.uniform(
            key,
            (self.nt if sample_size is None else sample_size, 1),
            minval=self.tmin,
            maxval=self.tmax,
        )

    def generate_time_data(self, key: Key) -> tuple[Key, Float[Array, " nt"]]:
        """
        Construct a complete set of `self.nt` time points according to the
        specified `self.method`

        Note that self.times has always size self.nt and not self.n_start, even
        in RAR scheme, we must allocate all the collocation points
        """
        key, subkey = jax.random.split(self.key)
        if self.method == "grid":
            partial_times = (self.tmax - self.tmin) / self.nt
            return key, jnp.arange(self.tmin, self.tmax, partial_times)[:, None]
        if self.method == "uniform":
            return key, self.sample_in_time_domain(subkey)
        raise ValueError("Method " + self.method + " is not implemented.")

    def _get_time_operands(
        self,
    ) -> tuple[
        Key, Float[Array, " nt 1"], int, int | None, Float[Array, " nt 1"] | None
    ]:
        return (
            self.key,
            self.times,
            self.curr_time_idx,
            self.temporal_batch_size,
            self.p,
        )

    def temporal_batch(
        self,
    ) -> tuple[DataGeneratorODE, Float[Array, " temporal_batch_size"]]:
        """
        Return a batch of time points. If all the batches have been seen, we
        reshuffle them, otherwise we just return the next unseen batch.
        """
        if self.temporal_batch_size is None or self.temporal_batch_size == self.nt:
            # Avoid unnecessary reshuffling
            return self, self.times

        bstart = self.curr_time_idx
        bend = bstart + self.temporal_batch_size

        # Compute the effective number of used collocation points
        if self.rar_parameters is not None:
            nt_eff = (
                self.n_start
                + self.rar_iter_nb  # type: ignore
                * self.rar_parameters["selected_sample_size"]
            )
        else:
            nt_eff = self.nt

        new_attributes = _reset_or_increment(
            bend,
            nt_eff,
            self._get_time_operands(),  # type: ignore
            # ignore since the case self.temporal_batch_size is None has been
            # handled above
        )
        new = eqx.tree_at(
            lambda m: (m.key, m.times, m.curr_time_idx), self, new_attributes
        )

        # commands below are equivalent to
        # return self.times[i:(i+t_batch_size)]
        # start indices can be dynamic but the slice shape is fixed
        return new, jax.lax.dynamic_slice(
            new.times,
            start_indices=(new.curr_time_idx, 0),
            slice_sizes=(new.temporal_batch_size, 1),
        )

    def get_batch(self) -> tuple[DataGeneratorODE, ODEBatch]:
        """
        Generic method to return a batch. Here we call `self.temporal_batch()`
        """
        new, temporal_batch = self.temporal_batch()
        return new, ODEBatch(temporal_batch=temporal_batch)
