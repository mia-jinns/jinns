# pylint: disable=unsubscriptable-object
"""
Define the DataGeneratorODE equinox module
"""
from __future__ import (
    annotations,
)  # https://docs.python.org/3/library/typing.html#constant

from typing import TYPE_CHECKING, Dict
from dataclasses import InitVar
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Key, Int, PyTree, Array, Float, Bool
from jinns.data._Batchs import *

if TYPE_CHECKING:
    from jinns.utils._types import *


def append_param_batch(batch: AnyBatch, param_batch_dict: dict) -> AnyBatch:
    """
    Utility function that fill the param_batch_dict of a batch object with a
    param_batch_dict
    """
    return eqx.tree_at(
        lambda m: m.param_batch_dict,
        batch,
        param_batch_dict,
        is_leaf=lambda x: x is None,
    )


def append_obs_batch(batch: AnyBatch, obs_batch_dict: dict) -> AnyBatch:
    """
    Utility function that fill the obs_batch_dict of a batch object with a
    obs_batch_dict
    """
    return eqx.tree_at(
        lambda m: m.obs_batch_dict, batch, obs_batch_dict, is_leaf=lambda x: x is None
    )


def make_cartesian_product(
    b1: Float[Array, "batch_size dim1"], b2: Float[Array, "batch_size dim2"]
) -> Float[Array, "(batch_size*batch_size) (dim1+dim2)"]:
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
    operands: tuple[Key, Float[Array, "n dimension"], Int, None, Float[Array, "n"]]
) -> tuple[Key, Float[Array, "n dimension"], Int]:
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


def _increment_batch_idx(
    operands: tuple[Key, Float[Array, "n dimension"], Int, None, Float[Array, "n"]]
) -> tuple[Key, Float[Array, "n dimension"], Int]:
    key, domain, curr_idx, batch_size, _ = operands
    # simply increases counter and get the batch
    curr_idx += batch_size
    return (key, domain, curr_idx)


def _reset_or_increment(
    bend: Int,
    n_eff: Int,
    operands: tuple[Key, Float[Array, "n dimension"], Int, None, Float[Array, "n"]],
) -> tuple[Key, Float[Array, "n dimension"], Int]:
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
    rar_parameters: dict, n: Int, n_start: Int
) -> tuple[Int, Float[Array, "n"], Int, Int]:
    if rar_parameters is not None and n_start is None:
        raise ValueError(
            "nt_start must be provided in the context of RAR sampling scheme"
        )

    if rar_parameters is not None:
        # Default p is None. However, in the RAR sampling scheme we use 0
        # probability to specify non-used collocation points (i.e. points
        # above nt_start). Thus, p is a vector of probability of shape (nt, 1).
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


class DataGeneratorODE(eqx.Module):
    """
    A class implementing data generator object for ordinary differential equations.

    Parameters
    ----------
    key : Key
        Jax random key to sample new time points and to shuffle batches
    nt : Int
        The number of total time points that will be divided in
        batches. Batches are made so that each data point is seen only
        once during 1 epoch.
    tmin : float
        The minimum value of the time domain to consider
    tmax : float
        The maximum value of the time domain to consider
    temporal_batch_size : int
        The size of the batch of randomly selected points among
        the `nt` points.
    method : str, default="uniform"
        Either `grid` or `uniform`, default is `uniform`.
        The method that generates the `nt` time points. `grid` means
        regularly spaced points over the domain. `uniform` means uniformly
        sampled points over the domain
    rar_parameters : Dict[str, Int], default=None
        Default to None: do not use Residual Adaptative Resampling.
        Otherwise a dictionary with keys. `start_iter`: the iteration at
        which we start the RAR sampling scheme (we first have a burn in
        period). `update_rate`: the number of gradient steps taken between
        each appending of collocation points in the RAR algo.
        `sample_size`: the size of the sample from which we will select new
        collocation points. `selected_sample_size_times`: the number of selected
        points from the sample to be added to the current collocation
        points
    nt_start : Int, default=None
        Defaults to None. The effective size of nt used at start time.
        This value must be
        provided when rar_parameters is not None. Otherwise we set internally
        nt_start = nt and this is hidden from the user.
        In RAR, nt_start
        then corresponds to the initial number of points we train the PINN.
    """

    key: Key
    nt: Int
    tmin: Float
    tmax: Float
    temporal_batch_size: Int = eqx.field(static=True)  # static cause used as a
    # shape in jax.lax.dynamic_slice
    method: str = eqx.field(static=True, default_factory=lambda: "uniform")
    rar_parameters: Dict[str, Int] = None
    nt_start: Int = eqx.field(static=True, default=None)

    # all the init=False fields are set in __post_init__, even after a _replace
    # or eqx.tree_at __post_init__ is called
    p_times: Float[Array, "nt"] = eqx.field(init=False)
    rar_iter_from_last_sampling: Int = eqx.field(init=False)
    rar_iter_nb: Int = eqx.field(init=False)
    curr_time_idx: Int = eqx.field(init=False)
    times: Float[Array, "nt"] = eqx.field(init=False)

    def __post_init__(self):
        (
            self.nt_start,
            self.p_times,
            self.rar_iter_from_last_sampling,
            self.rar_iter_nb,
        ) = _check_and_set_rar_parameters(self.rar_parameters, self.nt, self.nt_start)

        self.curr_time_idx = jnp.iinfo(jnp.int32).max - self.temporal_batch_size - 1
        # to be sure there is a
        # shuffling at first get_batch() we do not call
        # _reset_batch_idx_and_permute in __init__ or __post_init__ because it
        # would return a copy of self and we have not investigate what would
        # happen
        # NOTE the (- self.temporal_batch_size - 1) because otherwise when computing
        # `bend` we overflow the max int32 with unwanted behaviour

        self.key, self.times = self.generate_time_data(self.key)
        # Note that, here, in __init__ (and __post_init__), this is the
        # only place where self assignment are authorized so we do the
        # above way for the key. Note that one of the motivation to return the
        # key from generate_*_data is to easily align key with legacy
        # DataGenerators to use same unit tests

    def sample_in_time_domain(
        self, key: Key, sample_size: Int = None
    ) -> Float[Array, "nt"]:
        return jax.random.uniform(
            key,
            (self.nt if sample_size is None else sample_size,),
            minval=self.tmin,
            maxval=self.tmax,
        )

    def generate_time_data(self, key: Key) -> tuple[Key, Float[Array, "nt"]]:
        """
        Construct a complete set of `self.nt` time points according to the
        specified `self.method`

        Note that self.times has always size self.nt and not self.nt_start, even
        in RAR scheme, we must allocate all the collocation points
        """
        key, subkey = jax.random.split(self.key)
        if self.method == "grid":
            partial_times = (self.tmax - self.tmin) / self.nt
            return key, jnp.arange(self.tmin, self.tmax, partial_times)
        if self.method == "uniform":
            return key, self.sample_in_time_domain(subkey)
        raise ValueError("Method " + self.method + " is not implemented.")

    def _get_time_operands(
        self,
    ) -> tuple[Key, Float[Array, "nt"], Int, Int, Float[Array, "nt"]]:
        return (
            self.key,
            self.times,
            self.curr_time_idx,
            self.temporal_batch_size,
            self.p_times,
        )

    def temporal_batch(
        self,
    ) -> tuple["DataGeneratorODE", Float[Array, "temporal_batch_size"]]:
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
                + self.rar_iter_nb * self.rar_parameters["selected_sample_size_times"]
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


class CubicMeshPDEStatio(eqx.Module):
    r"""
    A class implementing data generator object for stationary partial
    differential equations.

    Parameters
    ----------
    key : Key
        Jax random key to sample new time points and to shuffle batches
    n : Int
        The number of total $\Omega$ points that will be divided in
        batches. Batches are made so that each data point is seen only
        once during 1 epoch.
    nb : Int | None
        The total number of points in $\partial\Omega$.
        Can be `None` not to lose performance generating the border
        batch if they are not used
    omega_batch_size : Int
        The size of the batch of randomly selected points among
        the `n` points.
    omega_border_batch_size : Int | None
        The size of the batch of points randomly selected
        among the `nb` points.
        Can be `None` not to lose performance generating the border
        batch if they are not used
    dim : Int
        Dimension of $\Omega$ domain
    min_pts : tuple[tuple[Float, Float], ...]
        A tuple of minimum values of the domain along each dimension. For a sampling
        in `n` dimension, this represents $(x_{1, min}, x_{2,min}, ...,
        x_{n, min})$
    max_pts : tuple[tuple[Float, Float], ...]
        A tuple of maximum values of the domain along each dimension. For a sampling
        in `n` dimension, this represents $(x_{1, max}, x_{2,max}, ...,
        x_{n,max})$
    method : str, default="uniform"
        Either `grid` or `uniform`, default is `uniform`.
        The method that generates the `nt` time points. `grid` means
        regularly spaced points over the domain. `uniform` means uniformly
        sampled points over the domain
    rar_parameters : Dict[str, Int], default=None
        Default to None: do not use Residual Adaptative Resampling.
        Otherwise a dictionary with keys. `start_iter`: the iteration at
        which we start the RAR sampling scheme (we first have a burn in
        period). `update_every`: the number of gradient steps taken between
        each appending of collocation points in the RAR algo.
        `sample_size_omega`: the size of the sample from which we will select new
        collocation points. `selected_sample_size_omega`: the number of selected
        points from the sample to be added to the current collocation
        points
    n_start : Int, default=None
        Defaults to None. The effective size of n used at start time.
        This value must be
        provided when rar_parameters is not None. Otherwise we set internally
        n_start = n and this is hidden from the user.
        In RAR, n_start
        then corresponds to the initial number of points we train the PINN.
    """

    # kw_only in base class is motivated here: https://stackoverflow.com/a/69822584
    key: Key = eqx.field(kw_only=True)
    n: Int = eqx.field(kw_only=True)
    nb: Int | None = eqx.field(kw_only=True)
    omega_batch_size: Int = eqx.field(
        kw_only=True, static=True
    )  # static cause used as a
    # shape in jax.lax.dynamic_slice
    omega_border_batch_size: Int | None = eqx.field(
        kw_only=True, static=True
    )  # static cause used as a
    # shape in jax.lax.dynamic_slice
    dim: Int = eqx.field(kw_only=True, static=True)  # static cause used as a
    # shape in jax.lax.dynamic_slice
    min_pts: tuple[tuple[Float, Float], ...] = eqx.field(kw_only=True)
    max_pts: tuple[tuple[Float, Float], ...] = eqx.field(kw_only=True)
    method: str = eqx.field(
        kw_only=True, static=True, default_factory=lambda: "uniform"
    )
    rar_parameters: Dict[str, Int] = eqx.field(kw_only=True, default=None)
    n_start: Int = eqx.field(kw_only=True, default=None, static=True)

    # all the init=False fields are set in __post_init__, even after a _replace
    # or eqx.tree_at __post_init__ is called
    p_omega: Float[Array, "n"] = eqx.field(init=False)
    p_border: None = eqx.field(init=False)
    rar_iter_from_last_sampling: Int = eqx.field(init=False)
    rar_iter_nb: Int = eqx.field(init=False)
    curr_omega_idx: Int = eqx.field(init=False)
    curr_omega_border_idx: Int = eqx.field(init=False)
    omega: Float[Array, "n dim"] = eqx.field(init=False)
    omega_border: Float[Array, "1 2"] | Float[Array, "(nb//4) 2 4"] | None = eqx.field(
        init=False
    )

    def __post_init__(self):
        assert self.dim == len(self.min_pts) and isinstance(self.min_pts, tuple)
        assert self.dim == len(self.max_pts) and isinstance(self.max_pts, tuple)

        (
            self.n_start,
            self.p_omega,
            self.rar_iter_from_last_sampling,
            self.rar_iter_nb,
        ) = _check_and_set_rar_parameters(self.rar_parameters, self.n, self.n_start)

        self.p_border = None  # no RAR sampling for border for now

        # Special handling for the border batch
        if self.omega_border_batch_size is None:
            self.nb = None
            self.omega_border_batch_size = None
        elif self.dim == 1:
            # 1-D case : the arguments `nb` and `omega_border_batch_size` are
            # ignored but kept for backward stability. The attributes are
            # always set to 2.
            self.nb = 2
            self.omega_border_batch_size = 2
            # We are in 1-D case => omega_border_batch_size is
            # ignored since borders of Omega are singletons.
            #  self.border_batch() will return [xmin, xmax]
        else:
            if self.nb % (2 * self.dim) != 0 or self.nb < 2 * self.dim:
                raise ValueError(
                    "number of border point must be"
                    " a multiple of 2xd (the # of faces of a d-dimensional cube)"
                )
            if self.nb // (2 * self.dim) < self.omega_border_batch_size:
                raise ValueError(
                    "number of points per facets (nb//2*self.dim)"
                    " cannot be lower than border batch size"
                )
            self.nb = int((2 * self.dim) * (self.nb // (2 * self.dim)))

        self.curr_omega_idx = jnp.iinfo(jnp.int32).max - self.omega_batch_size - 1
        # see explaination in DataGeneratorODE
        if self.omega_border_batch_size is None:
            self.curr_omega_border_idx = None
        else:
            self.curr_omega_border_idx = (
                jnp.iinfo(jnp.int32).max - self.omega_border_batch_size - 1
            )
        # key, subkey = jax.random.split(self.key)
        # self.key = key
        self.key, self.omega, self.omega_border = self.generate_data(self.key)
        # see explaination in DataGeneratorODE for the key

    def sample_in_omega_domain(
        self, keys: Key, sample_size: Int = None
    ) -> Float[Array, "n dim"]:
        sample_size = self.n if sample_size is None else sample_size
        if self.dim == 1:
            xmin, xmax = self.min_pts[0], self.max_pts[0]
            return jax.random.uniform(
                keys, shape=(sample_size, 1), minval=xmin, maxval=xmax
            )
        # keys = jax.random.split(key, self.dim)
        return jnp.concatenate(
            [
                jax.random.uniform(
                    keys[i],
                    (sample_size, 1),
                    minval=self.min_pts[i],
                    maxval=self.max_pts[i],
                )
                for i in range(self.dim)
            ],
            axis=-1,
        )

    def sample_in_omega_border_domain(
        self, keys: Key
    ) -> Float[Array, "1 2"] | Float[Array, "(nb//4) 2 4"] | None:
        if self.omega_border_batch_size is None:
            return None
        if self.dim == 1:
            xmin = self.min_pts[0]
            xmax = self.max_pts[0]
            return jnp.array([xmin, xmax]).astype(float)
        if self.dim == 2:
            # currently hard-coded the 4 edges for d==2
            # TODO : find a general & efficient way to sample from the border
            # (facets) of the hypercube in general dim.

            facet_n = self.nb // (2 * self.dim)
            xmin = jnp.hstack(
                [
                    self.min_pts[0] * jnp.ones((facet_n, 1)),
                    jax.random.uniform(
                        keys[0],
                        (facet_n, 1),
                        minval=self.min_pts[1],
                        maxval=self.max_pts[1],
                    ),
                ]
            )
            xmax = jnp.hstack(
                [
                    self.max_pts[0] * jnp.ones((facet_n, 1)),
                    jax.random.uniform(
                        keys[1],
                        (facet_n, 1),
                        minval=self.min_pts[1],
                        maxval=self.max_pts[1],
                    ),
                ]
            )
            ymin = jnp.hstack(
                [
                    jax.random.uniform(
                        keys[2],
                        (facet_n, 1),
                        minval=self.min_pts[0],
                        maxval=self.max_pts[0],
                    ),
                    self.min_pts[1] * jnp.ones((facet_n, 1)),
                ]
            )
            ymax = jnp.hstack(
                [
                    jax.random.uniform(
                        keys[3],
                        (facet_n, 1),
                        minval=self.min_pts[0],
                        maxval=self.max_pts[0],
                    ),
                    self.max_pts[1] * jnp.ones((facet_n, 1)),
                ]
            )
            return jnp.stack([xmin, xmax, ymin, ymax], axis=-1)
        raise NotImplementedError(
            "Generation of the border of a cube in dimension > 2 is not "
            + f"implemented yet. You are asking for generation in dimension d={self.dim}."
        )

    def generate_data(self, key: Key) -> tuple[
        Key,
        Float[Array, "n dim"],
        Float[Array, "1 2"] | Float[Array, "(nb//4) 2 4"] | None,
    ]:
        r"""
        Construct a complete set of `self.n` $\Omega$ points according to the
        specified `self.method`. Also constructs a complete set of `self.nb`
        $\partial\Omega$ points if `self.omega_border_batch_size` is not
        `None`. If the latter is `None` we set `self.omega_border` to `None`.
        """
        # Generate Omega
        if self.method == "grid":
            if self.dim == 1:
                xmin, xmax = self.min_pts[0], self.max_pts[0]
                partial = (xmax - xmin) / self.n
                # shape (n, 1)
                omega = jnp.arange(xmin, xmax, partial)[:, None]
            else:
                partials = [
                    (self.max_pts[i] - self.min_pts[i]) / jnp.sqrt(self.n)
                    for i in range(self.dim)
                ]
                xyz_ = jnp.meshgrid(
                    *[
                        jnp.arange(self.min_pts[i], self.max_pts[i], partials[i])
                        for i in range(self.dim)
                    ]
                )
                xyz_ = [a.reshape((self.n, 1)) for a in xyz_]
                omega = jnp.concatenate(xyz_, axis=-1)
        elif self.method == "uniform":
            if self.dim == 1:
                key, subkeys = jax.random.split(key, 2)
            else:
                key, *subkeys = jax.random.split(key, self.dim + 1)
            omega = self.sample_in_omega_domain(subkeys)
        else:
            raise ValueError("Method " + self.method + " is not implemented.")

        # Generate border of omega
        if self.dim == 2 and self.omega_border_batch_size is not None:
            key, *subkeys = jax.random.split(key, 5)
        else:
            subkeys = None
        omega_border = self.sample_in_omega_border_domain(subkeys)

        return key, omega, omega_border

    def _get_omega_operands(
        self,
    ) -> tuple[Key, Float[Array, "n dim"], Int, Int, Float[Array, "n"]]:
        return (
            self.key,
            self.omega,
            self.curr_omega_idx,
            self.omega_batch_size,
            self.p_omega,
        )

    def inside_batch(
        self,
    ) -> tuple["CubicMeshPDEStatio", Float[Array, "omega_batch_size dim"]]:
        r"""
        Return a batch of points in $\Omega$.
        If all the batches have been seen, we reshuffle them,
        otherwise we just return the next unseen batch.
        """
        # Compute the effective number of used collocation points
        if self.rar_parameters is not None:
            n_eff = (
                self.n_start
                + self.rar_iter_nb * self.rar_parameters["selected_sample_size_omega"]
            )
        else:
            n_eff = self.n

        bstart = self.curr_omega_idx
        bend = bstart + self.omega_batch_size

        new_attributes = _reset_or_increment(bend, n_eff, self._get_omega_operands())
        new = eqx.tree_at(
            lambda m: (m.key, m.omega, m.curr_omega_idx), self, new_attributes
        )

        return new, jax.lax.dynamic_slice(
            new.omega,
            start_indices=(new.curr_omega_idx, 0),
            slice_sizes=(new.omega_batch_size, new.dim),
        )

    def _get_omega_border_operands(
        self,
    ) -> tuple[
        Key, Float[Array, "1 2"] | Float[Array, "(nb//4) 2 4"] | None, Int, Int, None
    ]:
        return (
            self.key,
            self.omega_border,
            self.curr_omega_border_idx,
            self.omega_border_batch_size,
            self.p_border,
        )

    def border_batch(
        self,
    ) -> tuple[
        "CubicMeshPDEStatio",
        Float[Array, "1 1 2"] | Float[Array, "omega_border_batch_size 2 4"] | None,
    ]:
        r"""
        Return

        - The value `None` if `self.omega_border_batch_size` is `None`.

        - a jnp array with two fixed values $(x_{min}, x_{max})$ if
          `self.dim` = 1. There is no sampling here, we return the entire
          $\partial\Omega$

        - a batch of points in $\partial\Omega$ otherwise, stacked by
          facet on the last axis.
          If all the batches have been seen, we reshuffle them,
          otherwise we just return the next unseen batch.


        """
        if self.omega_border_batch_size is None:
            return self, None
        if self.dim == 1:
            # 1-D case, no randomness : we always return the whole omega border,
            # i.e. (1, 1, 2) shape jnp.array([[[xmin], [xmax]]]).
            return self, self.omega_border[None, None]  # shape is (1, 1, 2)
        bstart = self.curr_omega_border_idx
        bend = bstart + self.omega_border_batch_size

        new_attributes = _reset_or_increment(
            bend, self.nb, self._get_omega_border_operands()
        )
        new = eqx.tree_at(
            lambda m: (m.key, m.omega_border, m.curr_omega_border_idx),
            self,
            new_attributes,
        )

        return new, jax.lax.dynamic_slice(
            new.omega_border,
            start_indices=(new.curr_omega_border_idx, 0, 0),
            slice_sizes=(new.omega_border_batch_size, new.dim, 2 * new.dim),
        )

    def get_batch(self) -> tuple["CubicMeshPDEStatio", PDEStatioBatch]:
        """
        Generic method to return a batch. Here we call `self.inside_batch()`
        and `self.border_batch()`
        """
        new, inside_batch = self.inside_batch()
        new, border_batch = new.border_batch()
        return new, PDEStatioBatch(inside_batch=inside_batch, border_batch=border_batch)


class CubicMeshPDENonStatio(CubicMeshPDEStatio):
    r"""
    A class implementing data generator object for non stationary partial
    differential equations. Formally, it extends `CubicMeshPDEStatio`
    to include a temporal batch.

    Parameters
    ----------
    key : Key
        Jax random key to sample new time points and to shuffle batches
    n : Int
        The number of total $\Omega$ points that will be divided in
        batches. Batches are made so that each data point is seen only
        once during 1 epoch.
    nb : Int | None
        The total number of points in $\partial\Omega$.
        Can be `None` not to lose performance generating the border
        batch if they are not used
    nt : Int
        The number of total time points that will be divided in
        batches. Batches are made so that each data point is seen only
        once during 1 epoch.
    omega_batch_size : Int
        The size of the batch of randomly selected points among
        the `n` points.
    omega_border_batch_size : Int | None
        The size of the batch of points randomly selected
        among the `nb` points.
        Can be `None` not to lose performance generating the border
        batch if they are not used
    temporal_batch_size : Int
        The size of the batch of randomly selected points among
        the `nt` points.
    dim : Int
        An integer. dimension of $\Omega$ domain
    min_pts : tuple[tuple[Float, Float], ...]
        A tuple of minimum values of the domain along each dimension. For a sampling
        in `n` dimension, this represents $(x_{1, min}, x_{2,min}, ...,
        x_{n, min})$
    max_pts : tuple[tuple[Float, Float], ...]
        A tuple of maximum values of the domain along each dimension. For a sampling
        in `n` dimension, this represents $(x_{1, max}, x_{2,max}, ...,
        x_{n,max})$
    tmin : float
        The minimum value of the time domain to consider
    tmax : float
        The maximum value of the time domain to consider
    method : str, default="uniform"
        Either `grid` or `uniform`, default is `uniform`.
        The method that generates the `nt` time points. `grid` means
        regularly spaced points over the domain. `uniform` means uniformly
        sampled points over the domain
    rar_parameters : Dict[str, Int], default=None
        Default to None: do not use Residual Adaptative Resampling.
        Otherwise a dictionary with keys. `start_iter`: the iteration at
        which we start the RAR sampling scheme (we first have a burn in
        period). `update_every`: the number of gradient steps taken between
        each appending of collocation points in the RAR algo.
        `sample_size_omega`: the size of the sample from which we will select new
        collocation points. `selected_sample_size_omega`: the number of selected
        points from the sample to be added to the current collocation
        points.
    n_start : Int, default=None
        Defaults to None. The effective size of n used at start time.
        This value must be
        provided when rar_parameters is not None. Otherwise we set internally
        n_start = n and this is hidden from the user.
        In RAR, n_start
        then corresponds to the initial number of omega points we train the PINN.
    nt_start : Int, default=None
        Defaults to None. A RAR hyper-parameter. Same as ``n_start`` but
        for times collocation point. See also ``DataGeneratorODE``
        documentation.
    cartesian_product : Bool, default=True
        Defaults to True. Whether we return the cartesian product of the
        temporal batch with the inside and border batches. If False we just
        return their concatenation.
    """

    temporal_batch_size: Int = eqx.field(kw_only=True)
    tmin: Float = eqx.field(kw_only=True)
    tmax: Float = eqx.field(kw_only=True)
    nt: Int = eqx.field(kw_only=True)
    temporal_batch_size: Int = eqx.field(kw_only=True, static=True)
    cartesian_product: Bool = eqx.field(kw_only=True, default=True, static=True)
    nt_start: int = eqx.field(kw_only=True, default=None, static=True)

    p_times: Array = eqx.field(init=False)
    curr_time_idx: Int = eqx.field(init=False)
    times: Array = eqx.field(init=False)

    def __post_init__(self):
        """
        Note that neither __init__ or __post_init__ are called when udating a
        Module with eqx.tree_at!
        """
        super().__post_init__()  # because __init__ or __post_init__ of Base
        # class is not automatically called

        if not self.cartesian_product:
            if self.temporal_batch_size != self.omega_batch_size:
                raise ValueError(
                    "If stacking is requested between the time and "
                    "inside batches of collocation points, self.temporal_batch_size "
                    "must then be equal to self.omega_batch_size"
                )
            if (
                self.dim > 1
                and self.omega_border_batch_size is not None
                and self.temporal_batch_size != self.omega_border_batch_size
            ):
                raise ValueError(
                    "If dim > 1 and stacking is requested between the time and "
                    "inside batches of collocation points, self.temporal_batch_size "
                    "must then be equal to self.omega_border_batch_size"
                )
            # Note if self.dim == 1:
            #    print(
            #        "Cartesian product is not requested but will be "
            #        "executed anyway since dim=1"
            #    )

        # Set-up for timewise RAR (some quantity are already set-up by super())
        (
            self.nt_start,
            self.p_times,
            _,
            _,
        ) = _check_and_set_rar_parameters(self.rar_parameters, self.nt, self.nt_start)

        self.curr_time_idx = jnp.iinfo(jnp.int32).max - self.temporal_batch_size - 1
        self.key, _ = jax.random.split(self.key, 2)  # to make it equivalent to
        # the call to _reset_batch_idx_and_permute in legacy DG
        self.key, self.times = self.generate_time_data(self.key)
        # see explaination in DataGeneratorODE for the key

    def sample_in_time_domain(
        self, key: Key, sample_size: Int = None
    ) -> Float[Array, "nt"]:
        return jax.random.uniform(
            key,
            (self.nt if sample_size is None else sample_size,),
            minval=self.tmin,
            maxval=self.tmax,
        )

    def _get_time_operands(
        self,
    ) -> tuple[Key, Float[Array, "nt"], Int, Int, Float[Array, "nt"]]:
        return (
            self.key,
            self.times,
            self.curr_time_idx,
            self.temporal_batch_size,
            self.p_times,
        )

    def generate_time_data(self, key: Key) -> tuple[Key, Float[Array, "nt"]]:
        """
        Construct a complete set of `self.nt` time points according to the
        specified `self.method`

        Note that self.times has always size self.nt and not self.nt_start, even
        in RAR scheme, we must allocate all the collocation points
        """
        key, subkey = jax.random.split(key, 2)
        if self.method == "grid":
            partial_times = (self.tmax - self.tmin) / self.nt
            return key, jnp.arange(self.tmin, self.tmax, partial_times)
        if self.method == "uniform":
            return key, self.sample_in_time_domain(subkey)
        raise ValueError("Method " + self.method + " is not implemented.")

    def temporal_batch(
        self,
    ) -> tuple["CubicMeshPDENonStatio", Float[Array, "temporal_batch_size"]]:
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
                + self.rar_iter_nb * self.rar_parameters["selected_sample_size_times"]
            )
        else:
            nt_eff = self.nt

        new_attributes = _reset_or_increment(bend, nt_eff, self._get_time_operands())
        new = eqx.tree_at(
            lambda m: (m.key, m.times, m.curr_time_idx), self, new_attributes
        )

        return new, jax.lax.dynamic_slice(
            new.times,
            start_indices=(new.curr_time_idx,),
            slice_sizes=(new.temporal_batch_size,),
        )

    def get_batch(self) -> tuple["CubicMeshPDENonStatio", PDENonStatioBatch]:
        """
        Generic method to return a batch. Here we call `self.inside_batch()`,
        `self.border_batch()` and `self.temporal_batch()`
        """
        new, x = self.inside_batch()
        new, dx = new.border_batch()
        new, t = new.temporal_batch()
        t = t.reshape(new.temporal_batch_size, 1)

        if new.cartesian_product:
            t_x = make_cartesian_product(t, x)
        else:
            t_x = jnp.concatenate([t, x], axis=1)

        if dx is not None:
            t_ = t.reshape(new.temporal_batch_size, 1, 1)
            t_ = jnp.repeat(t_, dx.shape[-1], axis=2)
            if new.cartesian_product or new.dim == 1:
                t_dx = make_cartesian_product(t_, dx)
            else:
                t_dx = jnp.concatenate([t_, dx], axis=1)
        else:
            t_dx = None

        return new, PDENonStatioBatch(
            times_x_inside_batch=t_x, times_x_border_batch=t_dx
        )


class DataGeneratorObservations(eqx.Module):
    r"""
    Despite the class name, it is rather a dataloader from user provided
    observations that will be used for the observations loss

    Parameters
    ----------
    key : Key
        Jax random key to shuffle batches
    obs_batch_size : Int
        The size of the batch of randomly selected points among
        the `n` points. `obs_batch_size` will be the same for all
        elements of the return observation dict batch.
        NOTE: no check is done BUT users should be careful that
        `obs_batch_size` must be equal to `temporal_batch_size` or
        `omega_batch_size` or the product of both. In the first case, the
        present DataGeneratorObservations instance complements an ODEBatch,
        PDEStatioBatch or a PDENonStatioBatch (with self.cartesian_product
        = False). In the second case, `obs_batch_size` =
        `temporal_batch_size * omega_batch_size` if the present
        DataGeneratorParameter complements a PDENonStatioBatch
        with self.cartesian_product = True
    observed_pinn_in : Float[Array, "n_obs nb_pinn_in"]
        Observed values corresponding to the input of the PINN
        (eg. the time at which we recorded the observations). The first
        dimension must corresponds to the number of observed_values.
        The second dimension depends on the input dimension of the PINN,
        that is `1` for ODE, `n_dim_x` for stationnary PDE and `n_dim_x + 1`
        for non-stationnary PDE.
    observed_values : Float[Array, "n_obs, nb_pinn_out"]
        Observed values that the PINN should learn to fit. The first
        dimension must be aligned with observed_pinn_in.
    observed_eq_params : Dict[str, Float[Array, "n_obs 1"]], default={}
        A dict with keys corresponding to
        the parameter name. The keys must match the keys in
        `params["eq_params"]`. The values are jnp.array with 2 dimensions
        with values corresponding to the parameter value for which we also
        have observed_pinn_in and observed_values. Hence the first
        dimension must be aligned with observed_pinn_in and observed_values.
        Optional argument.
    sharding_device : jax.sharding.Sharding, default=None
        Default None. An optional sharding object to constraint the storage
        of observed inputs, values and parameters. Typically, a
        SingleDeviceSharding(cpu_device) to avoid loading on GPU huge
        datasets of observations. Note that computations for **batches**
        can still be performed on other devices (*e.g.* GPU, TPU or
        any pre-defined Sharding) thanks to the `obs_batch_sharding`
        arguments of `jinns.solve()`. Read the docs for more info.
    """

    key: Key
    obs_batch_size: Int = eqx.field(static=True)
    observed_pinn_in: Float[Array, "n_obs nb_pinn_in"]
    observed_values: Float[Array, "n_obs nb_pinn_out"]
    observed_eq_params: Dict[str, Float[Array, "n_obs 1"]] = eqx.field(
        static=True, default_factory=lambda: {}
    )
    sharding_device: jax.sharding.Sharding = eqx.field(static=True, default=None)

    n: Int = eqx.field(init=False)
    curr_idx: Int = eqx.field(init=False)
    indices: Array = eqx.field(init=False)

    def __post_init__(self):
        if self.observed_pinn_in.shape[0] != self.observed_values.shape[0]:
            raise ValueError(
                "self.observed_pinn_in and self.observed_values must have same first axis"
            )
        for _, v in self.observed_eq_params.items():
            if v.shape[0] != self.observed_pinn_in.shape[0]:
                raise ValueError(
                    "self.observed_pinn_in and the values of"
                    " self.observed_eq_params must have the same first axis"
                )
        if len(self.observed_pinn_in.shape) == 1:
            self.observed_pinn_in = self.observed_pinn_in[:, None]
        if len(self.observed_pinn_in.shape) > 2:
            raise ValueError("self.observed_pinn_in must have 2 dimensions")
        if len(self.observed_values.shape) == 1:
            self.observed_values = self.observed_values[:, None]
        if len(self.observed_values.shape) > 2:
            raise ValueError("self.observed_values must have 2 dimensions")
        for k, v in self.observed_eq_params.items():
            if len(v.shape) == 1:
                self.observed_eq_params[k] = v[:, None]
            if len(v.shape) > 2:
                raise ValueError(
                    "Each value of observed_eq_params must have 2 dimensions"
                )

        self.n = self.observed_pinn_in.shape[0]

        if self.sharding_device is not None:
            self.observed_pinn_in = jax.lax.with_sharding_constraint(
                self.observed_pinn_in, self.sharding_device
            )
            self.observed_values = jax.lax.with_sharding_constraint(
                self.observed_values, self.sharding_device
            )
            self.observed_eq_params = jax.lax.with_sharding_constraint(
                self.observed_eq_params, self.sharding_device
            )

        self.curr_idx = jnp.iinfo(jnp.int32).max - self.obs_batch_size - 1
        # For speed and to avoid duplicating data what is really
        # shuffled is a vector of indices
        if self.sharding_device is not None:
            self.indices = jax.lax.with_sharding_constraint(
                jnp.arange(self.n), self.sharding_device
            )
        else:
            self.indices = jnp.arange(self.n)

        # recall post_init is the only place with _init_ where we can set
        # self attribute in a in-place way
        self.key, _ = jax.random.split(self.key, 2)  # to make it equivalent to
        # the call to _reset_batch_idx_and_permute in legacy DG

    def _get_operands(self) -> tuple[Key, Int[Array, "n"], Int, Int, None]:
        return (
            self.key,
            self.indices,
            self.curr_idx,
            self.obs_batch_size,
            None,
        )

    def obs_batch(
        self,
    ) -> tuple[
        "DataGeneratorObservations", Dict[str, Float[Array, "obs_batch_size dim"]]
    ]:
        """
        Return a dictionary with (keys, values): (pinn_in, a mini batch of pinn
        inputs), (obs, a mini batch of corresponding observations), (eq_params,
        a dictionary with entry names found in `params["eq_params"]` and values
        giving the correspond parameter value for the couple
        (input, observation) mentioned before).
        It can also be a dictionary of dictionaries as described above if
        observed_pinn_in, observed_values, etc. are dictionaries with keys
        representing the PINNs.
        """

        new_attributes = _reset_or_increment(
            self.curr_idx + self.obs_batch_size, self.n, self._get_operands()
        )
        new = eqx.tree_at(
            lambda m: (m.key, m.indices, m.curr_idx), self, new_attributes
        )

        minib_indices = jax.lax.dynamic_slice(
            new.indices,
            start_indices=(new.curr_idx,),
            slice_sizes=(new.obs_batch_size,),
        )

        obs_batch = {
            "pinn_in": jnp.take(
                new.observed_pinn_in, minib_indices, unique_indices=True, axis=0
            ),
            "val": jnp.take(
                new.observed_values, minib_indices, unique_indices=True, axis=0
            ),
            "eq_params": jax.tree_util.tree_map(
                lambda a: jnp.take(a, minib_indices, unique_indices=True, axis=0),
                new.observed_eq_params,
            ),
        }
        return new, obs_batch

    def get_batch(
        self,
    ) -> tuple[
        "DataGeneratorObservations", Dict[str, Float[Array, "obs_batch_size dim"]]
    ]:
        """
        Generic method to return a batch
        """
        return self.obs_batch()


class DataGeneratorParameter(eqx.Module):
    r"""
    A data generator for additional unidimensional parameter(s)

    Parameters
    ----------
    keys : Key | Dict[str, Key]
        Jax random key to sample new time points and to shuffle batches
        or a dict of Jax random keys with key entries from param_ranges
    n : Int
        The number of total points that will be divided in
        batches. Batches are made so that each data point is seen only
        once during 1 epoch.
    param_batch_size : Int
        The size of the batch of randomly selected points among
        the `n` points. `param_batch_size` will be the same for all
        additional batch of parameter.
        NOTE: no check is done BUT users should be careful that
        `param_batch_size` must be equal to `temporal_batch_size` or
        `omega_batch_size` or the product of both. In the first case, the
        present DataGeneratorParameter instance complements an ODEBatch, a
        PDEStatioBatch or a PDENonStatioBatch (with self.cartesian_product
        = False). In the second case, `param_batch_size` =
        `temporal_batch_size * omega_batch_size` if the present
        DataGeneratorParameter complements a PDENonStatioBatch
        with self.cartesian_product = True
    param_ranges : Dict[str, tuple[Float, Float] | None, default={}
        A dict. A dict of tuples (min, max), which
        reprensents the range of real numbers where to sample batches (of
        length `param_batch_size` among `n` points).
        The key corresponds to the parameter name. The keys must match the
        keys in `params["eq_params"]`.
        By providing several entries in this dictionary we can sample
        an arbitrary number of parameters.
        **Note** that we currently only support unidimensional parameters.
        This argument can be done if we only use `user_data`.
    method : str, default="uniform"
        Either `grid` or `uniform`, default is `uniform`. `grid` means
        regularly spaced points over the domain. `uniform` means uniformly
        sampled points over the domain
    user_data : Dict[str, Float[Array, "n"]] | None, default={}
        A dictionary containing user-provided data for parameters.
        As for `param_ranges`, the key corresponds to the parameter name,
        the keys must match the keys in `params["eq_params"]` and only
        unidimensional arrays are supported. Therefore, the jnp arrays
        found at `user_data[k]` must have shape `(n, 1)` or `(n,)`.
        Note that if the same key appears in `param_ranges` and `user_data`
        priority goes for the content in `user_data`.
        Defaults to None.
    """

    keys: Key | Dict[str, Key]
    n: Int
    param_batch_size: Int = eqx.field(static=True)
    param_ranges: Dict[str, tuple[Float, Float]] = eqx.field(
        static=True, default_factory=lambda: {}
    )
    method: str = eqx.field(static=True, default="uniform")
    user_data: Dict[str, Float[Array, "n"]] | None = eqx.field(
        static=True, default_factory=lambda: {}
    )

    curr_param_idx: Dict[str, Int] = eqx.field(init=False)
    param_n_samples: Dict[str, Array] = eqx.field(init=False)

    def __post_init__(self):
        if self.user_data is None:
            self.user_data = {}
        if self.param_ranges is None:
            self.param_ranges = {}
        if self.n < self.param_batch_size:
            raise ValueError(
                f"Number of data points ({self.n}) is smaller than the"
                f"number of batch points ({self.param_batch_size})."
            )
        if not isinstance(self.keys, dict):
            all_keys = set().union(self.param_ranges, self.user_data)
            self.keys = dict(zip(all_keys, jax.random.split(self.keys, len(all_keys))))

        self.curr_param_idx = {}
        for k in self.keys.keys():
            self.curr_param_idx[k] = (
                jnp.iinfo(jnp.int32).max - self.param_batch_size - 1
            )

        # The call to self.generate_data() creates
        # the dict self.param_n_samples and then we will only use this one
        # because it merges the scattered data between `user_data` and
        # `param_ranges`
        self.keys, self.param_n_samples = self.generate_data(self.keys)

    def generate_data(
        self, keys: Dict[str, Key]
    ) -> tuple[Dict[str, Key], Dict[str, Float[Array, "n"]]]:
        """
        Generate parameter samples, either through generation
        or using user-provided data.
        """
        param_n_samples = {}

        all_keys = set().union(self.param_ranges, self.user_data)
        for k in all_keys:
            if (
                self.user_data
                and k in self.user_data.keys()  # pylint: disable=no-member
            ):
                if self.user_data[k].shape == (self.n, 1):
                    param_n_samples[k] = self.user_data[k]
                if self.user_data[k].shape == (self.n,):
                    param_n_samples[k] = self.user_data[k][:, None]
                else:
                    raise ValueError(
                        "Wrong shape for user provided parameters"
                        f" in user_data dictionary at key='{k}'"
                    )
            else:
                if self.method == "grid":
                    xmin, xmax = self.param_ranges[k][0], self.param_ranges[k][1]
                    partial = (xmax - xmin) / self.n
                    # shape (n, 1)
                    param_n_samples[k] = jnp.arange(xmin, xmax, partial)[:, None]
                elif self.method == "uniform":
                    xmin, xmax = self.param_ranges[k][0], self.param_ranges[k][1]
                    keys[k], subkey = jax.random.split(keys[k], 2)
                    param_n_samples[k] = jax.random.uniform(
                        subkey, shape=(self.n, 1), minval=xmin, maxval=xmax
                    )
                else:
                    raise ValueError("Method " + self.method + " is not implemented.")

        return keys, param_n_samples

    def _get_param_operands(
        self, k: str
    ) -> tuple[Key, Float[Array, "n"], Int, Int, None]:
        return (
            self.keys[k],
            self.param_n_samples[k],
            self.curr_param_idx[k],
            self.param_batch_size,
            None,
        )

    def param_batch(self):
        """
        Return a dictionary with batches of parameters
        If all the batches have been seen, we reshuffle them,
        otherwise we just return the next unseen batch.
        """

        def _reset_or_increment_wrapper(param_k, idx_k, key_k):
            return _reset_or_increment(
                idx_k + self.param_batch_size,
                self.n,
                (key_k, param_k, idx_k, self.param_batch_size, None),
            )

        res = jax.tree_util.tree_map(
            _reset_or_increment_wrapper,
            self.param_n_samples,
            self.curr_param_idx,
            self.keys,
        )
        # we must transpose the pytrees because keys are merged in res
        # https://jax.readthedocs.io/en/latest/jax-101/05.1-pytrees.html#transposing-trees
        new_attributes = jax.tree_util.tree_transpose(
            jax.tree_util.tree_structure(self.keys),
            jax.tree_util.tree_structure([0, 0, 0]),
            res,
        )

        new = eqx.tree_at(
            lambda m: (m.keys, m.param_n_samples, m.curr_param_idx),
            self,
            new_attributes,
        )

        return new, jax.tree_util.tree_map(
            lambda p, q: jax.lax.dynamic_slice(
                p, start_indices=(q, 0), slice_sizes=(new.param_batch_size, 1)
            ),
            new.param_n_samples,
            new.curr_param_idx,
        )

    def get_batch(self):
        """
        Generic method to return a batch
        """
        return self.param_batch()


class DataGeneratorObservationsMultiPINNs(eqx.Module):
    r"""
    Despite the class name, it is rather a dataloader from user provided
    observations that will be used for the observations loss.
    This is the DataGenerator to use when dealing with multiple PINNs
    (`u_dict`) in SystemLossODE/SystemLossPDE

    Technically, the constraint on the observations in SystemLossXDE are
    applied in `constraints_system_loss_apply` and in this case the
    batch.obs_batch_dict is a dict of obs_batch_dict over which the tree_map
    applies (we select the obs_batch_dict corresponding to its `u_dict` entry)

    Parameters
    ----------
    obs_batch_size : Int
        The size of the batch of randomly selected observations
        `obs_batch_size` will be the same for all the
        elements of the obs dict.
        NOTE: no check is done BUT users should be careful that
        `obs_batch_size` must be equal to `temporal_batch_size` or
        `omega_batch_size` or the product of both. In the first case, the
        present DataGeneratorObservations instance complements an ODEBatch,
        PDEStatioBatch or a PDENonStatioBatch (with self.cartesian_product
        = False). In the second case, `obs_batch_size` =
        `temporal_batch_size * omega_batch_size` if the present
        DataGeneratorParameter complements a PDENonStatioBatch
        with self.cartesian_product = True
    observed_pinn_in_dict : Dict[str, Float[Array, "n_obs nb_pinn_in"] | None]
        A dict of observed_pinn_in as defined in DataGeneratorObservations.
        Keys must be that of `u_dict`.
        If no observation exists for a particular entry of `u_dict` the
        corresponding key must still exist in observed_pinn_in_dict with
        value None
    observed_values_dict : Dict[str, Float[Array, "n_obs, nb_pinn_out"] | None]
        A dict of observed_values as defined in DataGeneratorObservations.
        Keys must be that of `u_dict`.
        If no observation exists for a particular entry of `u_dict` the
        corresponding key must still exist in observed_values_dict with
        value None
    observed_eq_params_dict : Dict[str, Dict[str, Float[Array, "n_obs 1"]]]
        A dict of observed_eq_params as defined in DataGeneratorObservations.
        Keys must be that of `u_dict`.
        **Note**: if no observation exists for a particular entry of `u_dict` the
        corresponding key must still exist in observed_eq_params_dict with
        value `{}` (empty dictionnary).
    key
        Jax random key to shuffle batches.
    """

    obs_batch_size: Int
    observed_pinn_in_dict: Dict[str, Float[Array, "n_obs nb_pinn_in"] | None]
    observed_values_dict: Dict[str, Float[Array, "n_obs nb_pinn_out"] | None]
    observed_eq_params_dict: Dict[str, Dict[str, Float[Array, "n_obs 1"]]] = eqx.field(
        default=None, kw_only=True
    )
    key: InitVar[Key]

    data_gen_obs: Dict[str, "DataGeneratorObservations"] = eqx.field(init=False)

    def __post_init__(self, key):
        if self.observed_pinn_in_dict is None or self.observed_values_dict is None:
            raise ValueError(
                "observed_pinn_in_dict and observed_values_dict " "must be provided"
            )
        if self.observed_pinn_in_dict.keys() != self.observed_values_dict.keys():
            raise ValueError(
                "Keys must be the same in observed_pinn_in_dict"
                " and observed_values_dict"
            )

        if self.observed_eq_params_dict is None:
            self.observed_eq_params_dict = {
                k: {} for k in self.observed_pinn_in_dict.keys()
            }
        elif self.observed_pinn_in_dict.keys() != self.observed_eq_params_dict.keys():
            raise ValueError(
                f"Keys must be the same in observed_eq_params_dict"
                f" and observed_pinn_in_dict and observed_values_dict"
            )

        keys = dict(
            zip(
                self.observed_pinn_in_dict.keys(),
                jax.random.split(key, len(self.observed_pinn_in_dict)),
            )
        )
        self.data_gen_obs = jax.tree_util.tree_map(
            lambda k, pinn_in, val, eq_params: (
                DataGeneratorObservations(
                    k, self.obs_batch_size, pinn_in, val, eq_params
                )
                if pinn_in is not None
                else None
            ),
            keys,
            self.observed_pinn_in_dict,
            self.observed_values_dict,
            self.observed_eq_params_dict,
        )

    def obs_batch(self) -> tuple["DataGeneratorObservationsMultiPINNs", PyTree]:
        """
        Returns a dictionary of DataGeneratorObservations.obs_batch with keys
        from `u_dict`
        """
        data_gen_and_batch_pytree = jax.tree_util.tree_map(
            lambda a: a.get_batch() if a is not None else {},
            self.data_gen_obs,
            is_leaf=lambda x: isinstance(x, DataGeneratorObservations),
        )  # note the is_leaf note to traverse the DataGeneratorObservations and
        # thus to be able to call the method on the element(s) of
        # self.data_gen_obs which are not None
        new_attribute = jax.tree_util.tree_map(
            lambda a: a[0],
            data_gen_and_batch_pytree,
            is_leaf=lambda x: isinstance(x, tuple),
        )
        new = eqx.tree_at(lambda m: m.data_gen_obs, self, new_attribute)
        batches = jax.tree_util.tree_map(
            lambda a: a[1],
            data_gen_and_batch_pytree,
            is_leaf=lambda x: isinstance(x, tuple),
        )

        return new, batches

    def get_batch(self) -> tuple["DataGeneratorObservationsMultiPINNs", PyTree]:
        """
        Generic method to return a batch
        """
        return self.obs_batch()
