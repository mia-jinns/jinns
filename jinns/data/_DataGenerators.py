# pylint: disable=unsubscriptable-object
"""
Define the DataGenerators modules
"""
from __future__ import (
    annotations,
)  # https://docs.python.org/3/library/typing.html#constant
import warnings
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
    Utility function that fills the field `batch.param_batch_dict` of a batch object.
    """
    return eqx.tree_at(
        lambda m: m.param_batch_dict,
        batch,
        param_batch_dict,
        is_leaf=lambda x: x is None,
    )


def append_obs_batch(batch: AnyBatch, obs_batch_dict: dict) -> AnyBatch:
    """
    Utility function that fills the field `batch.obs_batch_dict` of a batch object
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
    operands: tuple[Key, Float[Array, "n dimension"], Int, None, Float[Array, "n"]],
) -> tuple[Key, Float[Array, "n dimension"], Int]:
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
    operands: tuple[Key, Float[Array, "n dimension"], Int, None, Float[Array, "n"]],
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
    temporal_batch_size : int | None, default=None
        The size of the batch of randomly selected points among
        the `nt` points. If None, no minibatches are used.
    method : str, default="uniform"
        Either `grid` or `uniform`, default is `uniform`.
        The method that generates the `nt` time points. `grid` means
        regularly spaced points over the domain. `uniform` means uniformly
        sampled points over the domain
    rar_parameters : Dict[str, Int], default=None
        Defaults to None: do not use Residual Adaptative Resampling.
        Otherwise a dictionary with keys

        - `start_iter`: the iteration at which we start the RAR sampling scheme (we first have a "burn-in" period).
        - `update_every`: the number of gradient steps taken between
        each update of collocation points in the RAR algo.
        - `sample_size`: the size of the sample from which we will select new
        collocation points.
        - `selected_sample_size`: the number of selected
        points from the sample to be added to the current collocation
        points.
    n_start : Int, default=None
        Defaults to None. The effective size of nt used at start time.
        This value must be
        provided when rar_parameters is not None. Otherwise we set internally
        n_start = nt and this is hidden from the user.
        In RAR, n_start
        then corresponds to the initial number of points we train the PINN.
    """

    key: Key = eqx.field(kw_only=True)
    nt: Int = eqx.field(kw_only=True, static=True)
    tmin: Float = eqx.field(kw_only=True)
    tmax: Float = eqx.field(kw_only=True)
    temporal_batch_size: Int | None = eqx.field(static=True, default=None, kw_only=True)
    method: str = eqx.field(
        static=True, kw_only=True, default_factory=lambda: "uniform"
    )
    rar_parameters: Dict[str, Int] = eqx.field(default=None, kw_only=True)
    n_start: Int = eqx.field(static=True, default=None, kw_only=True)

    # all the init=False fields are set in __post_init__
    p: Float[Array, "nt 1"] = eqx.field(init=False)
    rar_iter_from_last_sampling: Int = eqx.field(init=False)
    rar_iter_nb: Int = eqx.field(init=False)
    curr_time_idx: Int = eqx.field(init=False)
    times: Float[Array, "nt 1"] = eqx.field(init=False)

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
        self, key: Key, sample_size: Int = None
    ) -> Float[Array, "nt 1"]:
        return jax.random.uniform(
            key,
            (self.nt if sample_size is None else sample_size, 1),
            minval=self.tmin,
            maxval=self.tmax,
        )

    def generate_time_data(self, key: Key) -> tuple[Key, Float[Array, "nt"]]:
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
    ) -> tuple[Key, Float[Array, "nt 1"], Int, Int, Float[Array, "nt 1"]]:
        return (
            self.key,
            self.times,
            self.curr_time_idx,
            self.temporal_batch_size,
            self.p,
        )

    def temporal_batch(
        self,
    ) -> tuple["DataGeneratorODE", Float[Array, "temporal_batch_size"]]:
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
        # start indices can be dynamic but the slice shape is fixed
        return new, jax.lax.dynamic_slice(
            new.times,
            start_indices=(new.curr_time_idx, 0),
            slice_sizes=(new.temporal_batch_size, 1),
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
        The total number of points in $\partial\Omega$. Can be None if no
        boundary condition is specified.
    omega_batch_size : Int | None, default=None
        The size of the batch of randomly selected points among
        the `n` points. If None no minibatches are used.
    omega_border_batch_size : Int | None, default=None
        The size of the batch of points randomly selected
        among the `nb` points. If None, `omega_border_batch_size`
        no minibatches are used. In dimension 1,
        minibatches are never used since the boundary is composed of two
        singletons.
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
        Defaults to None: do not use Residual Adaptative Resampling.
        Otherwise a dictionary with keys

        - `start_iter`: the iteration at which we start the RAR sampling scheme (we first have a "burn-in" period).
        - `update_every`: the number of gradient steps taken between
        each update of collocation points in the RAR algo.
        - `sample_size`: the size of the sample from which we will select new
        collocation points.
        - `selected_sample_size`: the number of selected
        points from the sample to be added to the current collocation
        points.
    n_start : Int, default=None
        Defaults to None. The effective size of n used at start time.
        This value must be
        provided when rar_parameters is not None. Otherwise we set internally
        n_start = n and this is hidden from the user.
        In RAR, n_start
        then corresponds to the initial number of points we train the PINN on.
    """

    # kw_only in base class is motivated here: https://stackoverflow.com/a/69822584
    key: Key = eqx.field(kw_only=True)
    n: Int = eqx.field(kw_only=True, static=True)
    nb: Int | None = eqx.field(kw_only=True, static=True, default=None)
    omega_batch_size: Int | None = eqx.field(
        kw_only=True,
        static=True,
        default=None,  # can be None as
        # CubicMeshPDENonStatio inherits but also if omega_batch_size=n
    )  # static cause used as a
    # shape in jax.lax.dynamic_slice
    omega_border_batch_size: Int | None = eqx.field(
        kw_only=True, static=True, default=None
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

    # all the init=False fields are set in __post_init__
    p: Float[Array, "n"] = eqx.field(init=False)
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
            self.p,
            self.rar_iter_from_last_sampling,
            self.rar_iter_nb,
        ) = _check_and_set_rar_parameters(self.rar_parameters, self.n, self.n_start)

        if self.method == "grid" and self.dim == 2:
            perfect_sq = int(jnp.round(jnp.sqrt(self.n)) ** 2)
            if self.n != perfect_sq:
                warnings.warn(
                    "Grid sampling is requested in dimension 2 with a non"
                    f" perfect square dataset size (self.n = {self.n})."
                    f" Modifying self.n to self.n = {perfect_sq}."
                )
            self.n = perfect_sq

        if self.nb is not None:
            if self.dim == 1:
                self.omega_border_batch_size = None
                # We are in 1-D case => omega_border_batch_size is
                # ignored since borders of Omega are singletons.
                #  self.border_batch() will return [xmin, xmax]
            else:
                if self.nb % (2 * self.dim) != 0 or self.nb < 2 * self.dim:
                    raise ValueError(
                        f"number of border point must be"
                        f" a multiple of 2xd = {2*self.dim} (the # of faces of"
                        f" a d-dimensional cube). Got {self.nb=}."
                    )
                if (
                    self.omega_border_batch_size is not None
                    and self.nb // (2 * self.dim) < self.omega_border_batch_size
                ):
                    raise ValueError(
                        f"number of points per facets ({self.nb//(2*self.dim)})"
                        f" cannot be lower than border batch size "
                        f" ({self.omega_border_batch_size})."
                    )
                self.nb = int((2 * self.dim) * (self.nb // (2 * self.dim)))

        if self.omega_batch_size is None:
            self.curr_omega_idx = 0
        else:
            self.curr_omega_idx = self.n + self.omega_batch_size
            # to be sure there is a shuffling at first get_batch()

        if self.omega_border_batch_size is None:
            self.curr_omega_border_idx = 0
        else:
            self.curr_omega_border_idx = self.nb + self.omega_border_batch_size
            # to be sure there is a shuffling at first get_batch()

        self.key, self.omega = self.generate_omega_data(self.key)
        self.key, self.omega_border = self.generate_omega_border_data(self.key)

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
        self, keys: Key, sample_size: int = None
    ) -> Float[Array, "1 2"] | Float[Array, "(nb//4) 2 4"] | None:
        sample_size = self.nb if sample_size is None else sample_size
        if sample_size is None:
            return None
        if self.dim == 1:
            xmin = self.min_pts[0]
            xmax = self.max_pts[0]
            return jnp.array([xmin, xmax]).astype(float)
        if self.dim == 2:
            # currently hard-coded the 4 edges for d==2
            # TODO : find a general & efficient way to sample from the border
            # (facets) of the hypercube in general dim.
            facet_n = sample_size // (2 * self.dim)
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

    def generate_omega_data(self, key: Key, data_size: int = None) -> tuple[
        Key,
        Float[Array, "n dim"],
    ]:
        r"""
        Construct a complete set of `self.n` $\Omega$ points according to the
        specified `self.method`.
        """
        data_size = self.n if data_size is None else data_size
        # Generate Omega
        if self.method == "grid":
            if self.dim == 1:
                xmin, xmax = self.min_pts[0], self.max_pts[0]
                ## shape (n, 1)
                omega = jnp.linspace(xmin, xmax, data_size)[:, None]
            else:
                xyz_ = jnp.meshgrid(
                    *[
                        jnp.linspace(
                            self.min_pts[i],
                            self.max_pts[i],
                            int(jnp.round(jnp.sqrt(data_size))),
                        )
                        for i in range(self.dim)
                    ]
                )
                xyz_ = [a.reshape((data_size, 1)) for a in xyz_]
                omega = jnp.concatenate(xyz_, axis=-1)
        elif self.method == "uniform":
            if self.dim == 1:
                key, subkeys = jax.random.split(key, 2)
            else:
                key, *subkeys = jax.random.split(key, self.dim + 1)
            omega = self.sample_in_omega_domain(subkeys, sample_size=data_size)
        else:
            raise ValueError("Method " + self.method + " is not implemented.")
        return key, omega

    def generate_omega_border_data(self, key: Key, data_size: int = None) -> tuple[
        Key,
        Float[Array, "1 2"] | Float[Array, "(nb//4) 2 4"] | None,
    ]:
        r"""
        Also constructs a complete set of `self.nb`
        $\partial\Omega$ points if `self.omega_border_batch_size` is not
        `None`. If the latter is `None` we set `self.omega_border` to `None`.
        """
        # Generate border of omega
        data_size = self.nb if data_size is None else data_size
        if self.dim == 2:
            key, *subkeys = jax.random.split(key, 5)
        else:
            subkeys = None
        omega_border = self.sample_in_omega_border_domain(
            subkeys, sample_size=data_size
        )

        return key, omega_border

    def _get_omega_operands(
        self,
    ) -> tuple[Key, Float[Array, "n dim"], Int, Int, Float[Array, "n"]]:
        return (
            self.key,
            self.omega,
            self.curr_omega_idx,
            self.omega_batch_size,
            self.p,
        )

    def inside_batch(
        self,
    ) -> tuple["CubicMeshPDEStatio", Float[Array, "omega_batch_size dim"]]:
        r"""
        Return a batch of points in $\Omega$.
        If all the batches have been seen, we reshuffle them,
        otherwise we just return the next unseen batch.
        """
        if self.omega_batch_size is None or self.omega_batch_size == self.n:
            # Avoid unnecessary reshuffling
            return self, self.omega

        # Compute the effective number of used collocation points
        if self.rar_parameters is not None:
            n_eff = (
                self.n_start
                + self.rar_iter_nb * self.rar_parameters["selected_sample_size"]
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
            None,
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
        if self.nb is None:
            # Avoid unnecessary reshuffling
            return self, None

        if self.dim == 1:
            # Avoid unnecessary reshuffling
            # 1-D case, no randomness : we always return the whole omega border,
            # i.e. (1, 1, 2) shape jnp.array([[[xmin], [xmax]]]).
            return self, self.omega_border[None, None]  # shape is (1, 1, 2)

        if (
            self.omega_border_batch_size is None
            or self.omega_border_batch_size == self.nb // 2**self.dim
        ):
            # Avoid unnecessary reshuffling
            return self, self.omega_border

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
        return new, PDEStatioBatch(domain_batch=inside_batch, border_batch=border_batch)


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
        The number of total $I\times \Omega$ points that will be divided in
        batches. Batches are made so that each data point is seen only
        once during 1 epoch.
    nb : Int | None
        The total number of points in $\partial\Omega$. Can be None if no
        boundary condition is specified.
    ni : Int
        The number of total $\Omega$ points at $t=0$ that will be divided in
        batches. Batches are made so that each data point is seen only
        once during 1 epoch.
    domain_batch_size : Int | None, default=None
        The size of the batch of randomly selected points among
        the `n` points. If None no mini-batches are used.
    border_batch_size : Int | None, default=None
        The size of the batch of points randomly selected
        among the `nb` points. If None, `domain_batch_size` no
        mini-batches are used.
    initial_batch_size : Int | None, default=None
        The size of the batch of randomly selected points among
        the `ni` points. If None no
        mini-batches are used.
    dim : Int
        An integer. Dimension of $\Omega$ domain.
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
        Defaults to None: do not use Residual Adaptative Resampling.
        Otherwise a dictionary with keys

        - `start_iter`: the iteration at which we start the RAR sampling scheme (we first have a "burn-in" period).
        - `update_every`: the number of gradient steps taken between
        each update of collocation points in the RAR algo.
        - `sample_size`: the size of the sample from which we will select new
        collocation points.
        - `selected_sample_size`: the number of selected
        points from the sample to be added to the current collocation
        points.
    n_start : Int, default=None
        Defaults to None. The effective size of n used at start time.
        This value must be
        provided when rar_parameters is not None. Otherwise we set internally
        n_start = n and this is hidden from the user.
        In RAR, n_start
        then corresponds to the initial number of omega points we train the PINN.
    """

    tmin: Float = eqx.field(kw_only=True)
    tmax: Float = eqx.field(kw_only=True)
    ni: Int = eqx.field(kw_only=True, static=True)
    domain_batch_size: Int | None = eqx.field(kw_only=True, static=True, default=None)
    initial_batch_size: Int | None = eqx.field(kw_only=True, static=True, default=None)
    border_batch_size: Int | None = eqx.field(kw_only=True, static=True, default=None)

    curr_domain_idx: Int = eqx.field(init=False)
    curr_initial_idx: Int = eqx.field(init=False)
    curr_border_idx: Int = eqx.field(init=False)
    domain: Float[Array, "n 1+dim"] = eqx.field(init=False)
    border: Float[Array, "(nb//2) 1+1 2"] | Float[Array, "(nb//4) 2+1 4"] | None = (
        eqx.field(init=False)
    )
    initial: Float[Array, "ni dim"] = eqx.field(init=False)

    def __post_init__(self):
        """
        Note that neither __init__ or __post_init__ are called when udating a
        Module with eqx.tree_at!
        """
        super().__post_init__()  # because __init__ or __post_init__ of Base
        # class is not automatically called

        if self.method == "grid":
            # NOTE we must redo the sampling with the square root number of samples
            # and then take the cartesian product
            self.n = int(jnp.round(jnp.sqrt(self.n)) ** 2)
            if self.dim == 2:
                # in the case of grid sampling in 2D in dim 2 in non-statio,
                # self.n needs to be a perfect ^4, because there is the
                # cartesian product with time domain which is also present
                perfect_4 = int(jnp.round(self.n**0.25) ** 4)
                if self.n != perfect_4:
                    warnings.warn(
                        "Grid sampling is requested in dimension 2 in non"
                        " stationary setting with a non"
                        f" perfect square dataset size (self.n = {self.n})."
                        f" Modifying self.n to self.n = {perfect_4}."
                    )
                self.n = perfect_4
            self.key, half_domain_times = self.generate_time_data(
                self.key, int(jnp.round(jnp.sqrt(self.n)))
            )

            self.key, half_domain_omega = self.generate_omega_data(
                self.key, data_size=int(jnp.round(jnp.sqrt(self.n)))
            )
            self.domain = make_cartesian_product(half_domain_times, half_domain_omega)

            # NOTE
            (
                self.n_start,
                self.p,
                self.rar_iter_from_last_sampling,
                self.rar_iter_nb,
            ) = _check_and_set_rar_parameters(self.rar_parameters, self.n, self.n_start)
        elif self.method == "uniform":
            self.key, domain_times = self.generate_time_data(self.key, self.n)
            self.domain = jnp.concatenate([domain_times, self.omega], axis=1)
        else:
            raise ValueError(
                f"Bad value for method. Got {self.method}, expected"
                ' "grid" or "uniform"'
            )

        if self.domain_batch_size is None:
            self.curr_domain_idx = 0
        else:
            self.curr_domain_idx = self.n + self.domain_batch_size
            # to be sure there is a shuffling at first get_batch()
        if self.nb is not None:
            # the check below has already been done in super.__post_init__ if
            # dim > 1. Here we retest it in whatever dim
            if self.nb % (2 * self.dim) != 0 or self.nb < 2 * self.dim:
                raise ValueError(
                    "number of border point must be"
                    " a multiple of 2xd (the # of faces of a d-dimensional cube)"
                )
            # the check below concern omega_border_batch_size for dim > 1 in
            # super.__post_init__. Here it concerns all dim values since our
            # border_batch is the concatenation or cartesian product with times
            if (
                self.border_batch_size is not None
                and self.nb // (2 * self.dim) < self.border_batch_size
            ):
                raise ValueError(
                    "number of points per facets (nb//2*self.dim)"
                    " cannot be lower than border batch size"
                )
            self.key, boundary_times = self.generate_time_data(
                self.key, self.nb // (2 * self.dim)
            )
            boundary_times = boundary_times.reshape(-1, 1, 1)
            boundary_times = jnp.repeat(
                boundary_times, self.omega_border.shape[-1], axis=2
            )
            if self.dim == 1:
                self.border = make_cartesian_product(
                    boundary_times, self.omega_border[None, None]
                )
            else:
                self.border = jnp.concatenate(
                    [boundary_times, self.omega_border], axis=1
                )
            if self.border_batch_size is None:
                self.curr_border_idx = 0
            else:
                self.curr_border_idx = self.nb + self.border_batch_size
                # to be sure there is a shuffling at first get_batch()

        else:
            self.border = None
            self.curr_border_idx = None
            self.border_batch_size = None

        if self.ni is not None:
            perfect_sq = int(jnp.round(jnp.sqrt(self.ni)) ** 2)
            if self.ni != perfect_sq:
                warnings.warn(
                    "Grid sampling is requested in dimension 2 with a non"
                    f" perfect square dataset size (self.ni = {self.ni})."
                    f" Modifying self.ni to self.ni = {perfect_sq}."
                )
            self.ni = perfect_sq
            self.key, self.initial = self.generate_omega_data(
                self.key, data_size=self.ni
            )

            if self.initial_batch_size is None or self.initial_batch_size == self.ni:
                self.curr_initial_idx = 0
            else:
                self.curr_initial_idx = self.ni + self.initial_batch_size
                # to be sure there is a shuffling at first get_batch()
        else:
            self.initial = None
            self.initial_batch_size = None
            self.curr_initial_idx = None

        # the following attributes will not be used anymore
        self.omega = None
        self.omega_border = None

    def generate_time_data(self, key: Key, nt: Int) -> tuple[Key, Float[Array, "nt 1"]]:
        """
        Construct a complete set of `nt` time points according to the
        specified `self.method`
        """
        key, subkey = jax.random.split(key, 2)
        if self.method == "grid":
            partial_times = (self.tmax - self.tmin) / nt
            return key, jnp.arange(self.tmin, self.tmax, partial_times)[:, None]
        if self.method == "uniform":
            return key, self.sample_in_time_domain(subkey, nt)
        raise ValueError("Method " + self.method + " is not implemented.")

    def sample_in_time_domain(self, key: Key, nt: Int) -> Float[Array, "nt 1"]:
        return jax.random.uniform(
            key,
            (nt, 1),
            minval=self.tmin,
            maxval=self.tmax,
        )

    def _get_domain_operands(
        self,
    ) -> tuple[Key, Float[Array, "n 1+dim"], Int, Int, None]:
        return (
            self.key,
            self.domain,
            self.curr_domain_idx,
            self.domain_batch_size,
            self.p,
        )

    def domain_batch(
        self,
    ) -> tuple["CubicMeshPDEStatio", Float[Array, "domain_batch_size 1+dim"]]:

        if self.domain_batch_size is None or self.domain_batch_size == self.n:
            # Avoid unnecessary reshuffling
            return self, self.domain

        bstart = self.curr_domain_idx
        bend = bstart + self.domain_batch_size

        # Compute the effective number of used collocation points
        if self.rar_parameters is not None:
            n_eff = (
                self.n_start
                + self.rar_iter_nb * self.rar_parameters["selected_sample_size"]
            )
        else:
            n_eff = self.n

        new_attributes = _reset_or_increment(bend, n_eff, self._get_domain_operands())
        new = eqx.tree_at(
            lambda m: (m.key, m.domain, m.curr_domain_idx),
            self,
            new_attributes,
        )
        return new, jax.lax.dynamic_slice(
            new.domain,
            start_indices=(new.curr_domain_idx, 0),
            slice_sizes=(new.domain_batch_size, new.dim + 1),
        )

    def _get_border_operands(
        self,
    ) -> tuple[
        Key, Float[Array, "nb 1+1 2"] | Float[Array, "(nb//4) 2+1 4"], Int, Int, None
    ]:
        return (
            self.key,
            self.border,
            self.curr_border_idx,
            self.border_batch_size,
            None,
        )

    def border_batch(
        self,
    ) -> tuple[
        "CubicMeshPDENonStatio",
        Float[Array, "border_batch_size 1+1 2"]
        | Float[Array, "border_batch_size 2+1 4"]
        | None,
    ]:
        if self.nb is None:
            # Avoid unnecessary reshuffling
            return self, None

        if (
            self.border_batch_size is None
            or self.border_batch_size == self.nb // 2**self.dim
        ):
            # Avoid unnecessary reshuffling
            return self, self.border

        bstart = self.curr_border_idx
        bend = bstart + self.border_batch_size

        n_eff = self.border.shape[0]

        new_attributes = _reset_or_increment(bend, n_eff, self._get_border_operands())
        new = eqx.tree_at(
            lambda m: (m.key, m.border, m.curr_border_idx),
            self,
            new_attributes,
        )

        return new, jax.lax.dynamic_slice(
            new.border,
            start_indices=(new.curr_border_idx, 0, 0),
            slice_sizes=(
                new.border_batch_size,
                new.dim + 1,
                2 * new.dim,
            ),
        )

    def _get_initial_operands(
        self,
    ) -> tuple[Key, Float[Array, "ni dim"], Int, Int, None]:
        return (
            self.key,
            self.initial,
            self.curr_initial_idx,
            self.initial_batch_size,
            None,
        )

    def initial_batch(
        self,
    ) -> tuple["CubicMeshPDEStatio", Float[Array, "initial_batch_size dim"]]:
        if self.initial_batch_size is None or self.initial_batch_size == self.ni:
            # Avoid unnecessary reshuffling
            return self, self.initial

        bstart = self.curr_initial_idx
        bend = bstart + self.initial_batch_size

        n_eff = self.ni

        new_attributes = _reset_or_increment(bend, n_eff, self._get_initial_operands())
        new = eqx.tree_at(
            lambda m: (m.key, m.initial, m.curr_initial_idx),
            self,
            new_attributes,
        )
        return new, jax.lax.dynamic_slice(
            new.initial,
            start_indices=(new.curr_initial_idx, 0),
            slice_sizes=(new.initial_batch_size, new.dim),
        )

    def get_batch(self) -> tuple["CubicMeshPDENonStatio", PDENonStatioBatch]:
        """
        Generic method to return a batch. Here we call `self.domain_batch()`,
        `self.border_batch()` and `self.initial_batch()`
        """
        new, domain = self.domain_batch()
        if self.border is not None:
            new, border = new.border_batch()
        else:
            border = None
        if self.initial is not None:
            new, initial = new.initial_batch()
        else:
            initial = None

        return new, PDENonStatioBatch(
            domain_batch=domain, border_batch=border, initial_batch=initial
        )


class DataGeneratorObservations(eqx.Module):
    r"""
    Despite the class name, it is rather a dataloader for user-provided
    observations which will are used in the observations loss.

    Parameters
    ----------
    key : Key
        Jax random key to shuffle batches
    obs_batch_size : Int | None
        The size of the batch of randomly selected points among
        the `n` points. If None, no minibatch are used.
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
        arguments of `jinns.solve()`. Read `jinns.solve()` doc for more info.
    """

    key: Key
    obs_batch_size: Int | None = eqx.field(static=True)
    observed_pinn_in: Float[Array, "n_obs nb_pinn_in"]
    observed_values: Float[Array, "n_obs nb_pinn_out"]
    observed_eq_params: Dict[str, Float[Array, "n_obs 1"]] = eqx.field(
        static=True, default_factory=lambda: {}
    )
    sharding_device: jax.sharding.Sharding = eqx.field(static=True, default=None)

    n: Int = eqx.field(init=False, static=True)
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

        if self.obs_batch_size is not None:
            self.curr_idx = self.n + self.obs_batch_size
            # to be sure there is a shuffling at first get_batch()
        else:
            self.curr_idx = 0
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
        if self.obs_batch_size is None or self.obs_batch_size == self.n:
            # Avoid unnecessary reshuffling
            return self, {
                "pinn_in": self.observed_pinn_in,
                "val": self.observed_values,
                "eq_params": self.observed_eq_params,
            }

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
    A data generator for additional unidimensional equation parameter(s).
    Mostly useful for metamodeling where batch of `params.eq_params` are fed
    to the network.

    Parameters
    ----------
    keys : Key | Dict[str, Key]
        Jax random key to sample new time points and to shuffle batches
        or a dict of Jax random keys with key entries from param_ranges
    n : Int
        The number of total points that will be divided in
        batches. Batches are made so that each data point is seen only
        once during 1 epoch.
    param_batch_size : Int | None, default=None
        The size of the batch of randomly selected points among
        the `n` points.  **Important**: no check is performed but
        `param_batch_size` must be the same as other collocation points
        batch_size (time, space or timexspace depending on the context). This is because we vmap the network on all its axes at once to compute the MSE. Also, `param_batch_size` will be the same for all parameters. If None, no mini-batches are used.
    param_ranges : Dict[str, tuple[Float, Float] | None, default={}
        A dict. A dict of tuples (min, max), which
        reprensents the range of real numbers where to sample batches (of
        length `param_batch_size` among `n` points).
        The key corresponds to the parameter name. The keys must match the
        keys in `params["eq_params"]`.
        By providing several entries in this dictionary we can sample
        an arbitrary number of parameters.
        **Note** that we currently only support unidimensional parameters.
        This argument can be None if we use `user_data`.
    method : str, default="uniform"
        Either `grid` or `uniform`, default is `uniform`. `grid` means
        regularly spaced points over the domain. `uniform` means uniformly
        sampled points over the domain
    user_data : Dict[str, Float[jnp.ndarray, "n"]] | None, default={}
        A dictionary containing user-provided data for parameters.
        The keys corresponds to the parameter name,
        and must match the keys in `params["eq_params"]`. Only
        unidimensional `jnp.array` are supported. Therefore, the array at
        `user_data[k]` must have shape `(n, 1)` or `(n,)`.
        Note that if the same key appears in `param_ranges` and `user_data`
        priority goes for the content in `user_data`.
        Defaults to None.
    """

    keys: Key | Dict[str, Key]
    n: Int = eqx.field(static=True)
    param_batch_size: Int | None = eqx.field(static=True, default=None)
    param_ranges: Dict[str, tuple[Float, Float]] = eqx.field(
        static=True, default_factory=lambda: {}
    )
    method: str = eqx.field(static=True, default="uniform")
    user_data: Dict[str, Float[onp.Array, "n"]] | None = eqx.field(
        default_factory=lambda: {}
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

        if self.param_batch_size is None:
            self.curr_param_idx = None
        else:
            self.curr_param_idx = {}
            for k in self.keys.keys():
                self.curr_param_idx[k] = self.n + self.param_batch_size
                # to be sure there is a shuffling at first get_batch()

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

        if self.param_batch_size is None or self.param_batch_size == self.n:
            return self, self.param_n_samples

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
    `batch.obs_batch_dict` is a dict of obs_batch_dict over which the tree_map
    applies (we select the obs_batch_dict corresponding to its `u_dict` entry)

    Parameters
    ----------
    obs_batch_size : Int
        The size of the batch of randomly selected observations
        `obs_batch_size` will be the same for all the
        elements of the obs dict.
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
