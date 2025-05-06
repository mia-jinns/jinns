"""
Define the DataGenerators modules
"""

from __future__ import (
    annotations,
)  # https://docs.python.org/3/library/typing.html#constant
import warnings
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Key, Array, Float
from jinns.data._Batchs import PDEStatioBatch
from jinns.data._utils import _check_and_set_rar_parameters, _reset_or_increment
from jinns.data._AbstractDataGenerator import AbstractDataGenerator


class CubicMeshPDEStatio(AbstractDataGenerator):
    r"""
    A class implementing data generator object for stationary partial
    differential equations.

    Parameters
    ----------
    key : Key
        Jax random key to sample new time points and to shuffle batches
    n : int
        The number of total $\Omega$ points that will be divided in
        batches. Batches are made so that each data point is seen only
        once during 1 epoch.
    nb : int | None
        The total number of points in $\partial\Omega$. Can be None if no
        boundary condition is specified.
    omega_batch_size : int | None, default=None
        The size of the batch of randomly selected points among
        the `n` points. If None no minibatches are used.
    omega_border_batch_size : int | None, default=None
        The size of the batch of points randomly selected
        among the `nb` points. If None, `omega_border_batch_size`
        no minibatches are used. In dimension 1,
        minibatches are never used since the boundary is composed of two
        singletons.
    dim : int
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
    rar_parameters : dict[str, int], default=None
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
    n_start : int, default=None
        Defaults to None. The effective size of n used at start time.
        This value must be
        provided when rar_parameters is not None. Otherwise we set internally
        n_start = n and this is hidden from the user.
        In RAR, n_start
        then corresponds to the initial number of points we train the PINN on.
    """

    # kw_only in base class is motivated here: https://stackoverflow.com/a/69822584
    key: Key = eqx.field(kw_only=True)
    n: int = eqx.field(kw_only=True, static=True)
    nb: int | None = eqx.field(kw_only=True, static=True, default=None)
    omega_batch_size: int | None = eqx.field(
        kw_only=True,
        static=True,
        default=None,  # can be None as
        # CubicMeshPDENonStatio inherits but also if omega_batch_size=n
    )  # static cause used as a
    # shape in jax.lax.dynamic_slice
    omega_border_batch_size: int | None = eqx.field(
        kw_only=True, static=True, default=None
    )  # static cause used as a
    # shape in jax.lax.dynamic_slice
    dim: int = eqx.field(kw_only=True, static=True)  # static cause used as a
    # shape in jax.lax.dynamic_slice
    min_pts: tuple[float, ...] = eqx.field(kw_only=True)
    max_pts: tuple[float, ...] = eqx.field(kw_only=True)
    method: str = eqx.field(
        kw_only=True, static=True, default_factory=lambda: "uniform"
    )
    rar_parameters: dict[str, int] = eqx.field(kw_only=True, default=None)
    n_start: int = eqx.field(kw_only=True, default=None, static=True)

    # all the init=False fields are set in __post_init__
    p: Float[Array, " n"] | None = eqx.field(init=False)
    rar_iter_from_last_sampling: int | None = eqx.field(init=False)
    rar_iter_nb: int | None = eqx.field(init=False)
    curr_omega_idx: int = eqx.field(init=False)
    curr_omega_border_idx: int = eqx.field(init=False)
    omega: Float[Array, " n dim"] = eqx.field(init=False)
    omega_border: Float[Array, " 1 2"] | Float[Array, " (nb//4) 2 4"] | None = (
        eqx.field(init=False)
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

        if self.omega_batch_size is None:
            self.curr_omega_idx = 0
        else:
            self.curr_omega_idx = self.n + self.omega_batch_size
            # to be sure there is a shuffling at first get_batch()

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
                        f" a multiple of 2xd = {2 * self.dim} (the # of faces of"
                        f" a d-dimensional cube). Got {self.nb=}."
                    )
                if (
                    self.omega_border_batch_size is not None
                    and self.nb // (2 * self.dim) < self.omega_border_batch_size
                ):
                    raise ValueError(
                        f"number of points per facets ({self.nb // (2 * self.dim)})"
                        f" cannot be lower than border batch size "
                        f" ({self.omega_border_batch_size})."
                    )
                self.nb = int((2 * self.dim) * (self.nb // (2 * self.dim)))

            if self.omega_border_batch_size is None:
                self.curr_omega_border_idx = 0
            else:
                self.curr_omega_border_idx = self.nb + self.omega_border_batch_size
                # to be sure there is a shuffling at first get_batch()
        else:  # self.nb is None
            self.curr_omega_border_idx = 0

        self.key, self.omega = self.generate_omega_data(self.key)
        self.key, self.omega_border = self.generate_omega_border_data(self.key)

    def sample_in_omega_domain(
        self, keys: Key, sample_size: int
    ) -> Float[Array, " n dim"]:
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
        self, keys: Key, sample_size: int | None = None
    ) -> Float[Array, " 1 2"] | Float[Array, " (nb//4) 2 4"] | None:
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

    def generate_omega_data(
        self, key: Key, data_size: int | None = None
    ) -> tuple[
        Key,
        Float[Array, " n dim"],
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

    def generate_omega_border_data(
        self, key: Key, data_size: int | None = None
    ) -> tuple[
        Key,
        Float[Array, " 1 2"] | Float[Array, " (nb//4) 2 4"] | None,
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
    ) -> tuple[Key, Float[Array, " n dim"], int, int | None, Float[Array, " n"] | None]:
        return (
            self.key,
            self.omega,
            self.curr_omega_idx,
            self.omega_batch_size,
            self.p,
        )

    def inside_batch(
        self,
    ) -> tuple[CubicMeshPDEStatio, Float[Array, " omega_batch_size dim"]]:
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
                + self.rar_iter_nb  # type: ignore
                * self.rar_parameters["selected_sample_size"]
            )
        else:
            n_eff = self.n

        bstart = self.curr_omega_idx
        bend = bstart + self.omega_batch_size

        new_attributes = _reset_or_increment(
            bend,
            n_eff,
            self._get_omega_operands(),  # type: ignore
            # ignore since the case self.omega_batch_size is None has been
            # handled above
        )
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
        Key,
        Float[Array, " 1 2"] | Float[Array, " (nb//4) 2 4"] | None,
        int,
        int | None,
        None,
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
        CubicMeshPDEStatio,
        Float[Array, " 1 1 2"] | Float[Array, " omega_border_batch_size 2 4"] | None,
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
        if self.nb is None or self.omega_border is None:
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
            bend,
            self.nb,
            self._get_omega_border_operands(),  # type: ignore
            # ignore since the case self.omega_border_batch_size is None has been
            # handled above
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

    def get_batch(self) -> tuple[CubicMeshPDEStatio, PDEStatioBatch]:
        """
        Generic method to return a batch. Here we call `self.inside_batch()`
        and `self.border_batch()`
        """
        new, inside_batch = self.inside_batch()
        new, border_batch = new.border_batch()
        return new, PDEStatioBatch(domain_batch=inside_batch, border_batch=border_batch)
