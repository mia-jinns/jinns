"""
Define the CubicMeshPDEStatio equinox module
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Key, Int, Bool, PyTree, Array

from jinns.data._utils import PDEStatioBatch, AbstractDataGenerator, _reset_or_increment


class CubicMeshPDEStatio(AbstractDataGenerator):
    r"""
    A class implementing data generator object for stationary partial
    differential equations.

    Parameters
    ----------
    key
        Jax random key to sample new time points and to shuffle batches
    n
        An integer. The number of total :math:`\Omega` points that will be divided in
        batches. Batches are made so that each data point is seen only
        once during 1 epoch.
    nb
        An integer. The total number of points in :math:`\partial\Omega`.
        Can be `None` not to lose performance generating the border
        batch if they are not used
    omega_batch_size
        An integer. The size of the batch of randomly selected points among
        the `n` points.
    omega_border_batch_size
        An integer. The size of the batch of points randomly selected
        among the `nb` points.
        Can be `None` not to lose performance generating the border
        batch if they are not used
    dim
        An integer. dimension of :math:`\Omega` domain
    min_pts
        A tuple of minimum values of the domain along each dimension. For a sampling
        in `n` dimension, this represents :math:`(x_{1, min}, x_{2,min}, ...,
        x_{n, min})`
    max_pts
        A tuple of maximum values of the domain along each dimension. For a sampling
        in `n` dimension, this represents :math:`(x_{1, max}, x_{2,max}, ...,
        x_{n,max})`
    method
        Either `grid` or `uniform`, default is `grid`.
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
    n_start
        Defaults to None. The effective size of n used at start time.
        This value must be
        provided when rar_parameters is not None. Otherwise we set internally
        n_start = n and this is hidden from the user.
        In RAR, n_start
        then corresponds to the initial number of points we train the PINN.
    data_exists
        Must be left to `False` when created by the user. Avoids the
        regeneration of :math:`\Omega`, :math:`\partial\Omega` and
        time points at each pytree flattening and unflattening.
    """

    key: Key
    n: Int
    nb: Int
    omega_batch_size: Int
    omega_border_batch_size: Int
    dim: Int
    min_pts: tuple
    max_pts: tuple
    method: str = eqx.field(static=True, default_factory=lambda: "grid")
    rar_parameters: PyTree = None
    n_start: int = None
    data_exists: Bool = False

    # all the init=False fields are set in __post_init__, even after a _replace
    # or eqx.tree_at __post_init__ is called
    p: Array = eqx.field(init=False)
    rar_iter_from_last_sampling: Array = eqx.field(init=False)
    rar_iter_nb: Array = eqx.field(init=False)
    curr_omega_idx: Int = eqx.field(init=False)
    curr_omega_border_idx: Int = eqx.field(init=False)
    omega: Array = eqx.field(init=False)
    omega_border: Array = eqx.field(init=False)

    def __post_init__(self):
        assert self.dim == len(self.min_pts) and isinstance(self.min_pts, tuple)
        assert self.dim == len(self.max_pts) and isinstance(self.max_pts, tuple)

        if self.rar_parameters is not None and self.n_start is None:
            raise ValueError(
                "n_start must be provided in the context of RAR sampling scheme"
            )
        if not self.data_exists:
            if self.rar_parameters is not None:
                # Default p is None. However, in the RAR sampling scheme we use 0
                # probability to specify non-used collocation points (i.e. points
                # above n_start). Thus, p is a vector of probability of shape (n, 1).
                self.p = jnp.zeros((self.n,))
                self.p = self.p.at[: self.n_start].set(1 / self.n_start)
                # set internal counter for the number of gradient steps since the
                # last new collocation points have been added
                self.rar_iter_from_last_sampling = 0
                # set iternal counter for the number of times collocation points
                # have been added
                self.rar_iter_nb = 0

            if self.rar_parameters is None or self.n_start is None:
                self.n_start = self.n
                self.p = None
                self.rar_iter_from_last_sampling = None
                self.rar_iter_nb = None

            if self.omega_border_batch_size is None:
                self.nb = None
            elif self.dim == 1:
                # 1-D case : the arguments `nb` and `omega_border_batch_size` are
                # ignored but kept for backward stability. The attributes are
                # always set to 2.
                self.nb = 2
                self.omega_border_batch_size = 2

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
                self.omega_border_batch_size = self.omega_border_batch_size

            self.curr_omega_border_idx = jnp.iinfo(jnp.int32).max
            self.curr_omega_idx = jnp.iinfo(jnp.int32).max  # to be sure there is a
            # shuffling at first get_batch() we do not call
            # _reset_batch_idx_and_permute in __init__ or __post_init__ because it
            # would return a copy of self and we have not investigate what would
            # happen

            key, subkey = jax.random.split(self.key)
            self.key = key
            self.omega, self.omega_border = self.generate_data(subkey)

        self.data_exists = True

    def sample_in_omega_domain(self, key) -> Array:
        if self.dim == 1:
            xmin, xmax = self.min_pts[0], self.max_pts[0]
            return jax.random.uniform(key, shape=(self.n, 1), minval=xmin, maxval=xmax)
        keys = jax.random.split(key, self.dim)
        return jnp.concatenate(
            [
                jax.random.uniform(
                    keys[i],
                    (self.n, 1),
                    minval=self.min_pts[i],
                    maxval=self.max_pts[i],
                )
                for i in range(self.dim)
            ],
            axis=-1,
        )

    def sample_in_omega_border_domain(self, key) -> Array:
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
            subkeys = jax.random.split(key, 4)
            xmin = jnp.hstack(
                [
                    self.min_pts[0] * jnp.ones((facet_n, 1)),
                    jax.random.uniform(
                        subkeys[0],
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
                        subkeys[1],
                        (facet_n, 1),
                        minval=self.min_pts[1],
                        maxval=self.max_pts[1],
                    ),
                ]
            )
            ymin = jnp.hstack(
                [
                    jax.random.uniform(
                        subkeys[2],
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
                        subkeys[3],
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

    def generate_data(self, key) -> tuple[Array, Array]:
        r"""
        Construct a complete set of `self.n` :math:`\Omega` points according to the
        specified `self.method`. Also constructs a complete set of `self.nb`
        :math:`\partial\Omega` points if `self.omega_border_batch_size` is not
        `None`. If the latter is `None` we set `self.omega_border` to `None`.
        """
        key1, key2 = jax.random.split(key)
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
            omega = self.sample_in_omega_domain(key1)
        else:
            raise ValueError("Method " + self.method + " is not implemented.")

        # Generate border of omega
        omega_border = self.sample_in_omega_border_domain(key2)
        return omega, omega_border

    # def _get_omega_operands(self):
    #    return (
    #        self._key,
    #        self.omega,
    #        self.curr_omega_idx,
    #        self.omega_batch_size,
    #        self.p,
    #    )

    def inside_batch(self) -> tuple["CubicMeshPDEStatio", Array]:
        r"""
        Return a batch of points in :math:`\Omega`.
        If all the batches have been seen, we reshuffle them,
        otherwise we just return the next unseen batch.
        """
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

        new = _reset_or_increment(self, bend, n_eff)

        # commands below are equivalent to
        # return self.omega[i:(i+batch_size), 0:dim]
        return new, jax.lax.dynamic_slice(
            new.omega,
            start_indices=(new.curr_omega_idx, 0),
            slice_sizes=(new.omega_batch_size, new.dim),
        )

    def _get_omega_border_operands(self):
        return (
            self._key,
            self.omega_border,
            self.curr_omega_border_idx,
            self.omega_border_batch_size,
            self.p_border,
        )

    def border_batch(self):
        r"""
        Return

        - The value `None` if `self.omega_border_batch_size` is `None`.

        - a jnp array with two fixed values :math:`(x_{min}, x_{max})` if
          `self.dim` = 1. There is no sampling here, we return the entire
          :math:`\partial\Omega`

        - a batch of points in :math:`\partial\Omega` otherwise, stacked by
          facet on the last axis.
          If all the batches have been seen, we reshuffle them,
          otherwise we just return the next unseen batch.


        """
        if self.omega_border_batch_size is None:
            return None
        if self.dim == 1:
            # 1-D case, no randomness : we always return the whole omega border,
            # i.e. (1, 1, 2) shape jnp.array([[[xmin], [xmax]]]).
            return self.omega_border[None, None]  # shape is (1, 1, 2)
        bstart = self.curr_omega_border_idx
        bend = bstart + self.omega_border_batch_size

        # (
        #    self._key,
        #    self.omega_border,
        #    self.curr_omega_border_idx,
        # ) = _reset_or_increment(bend, self.nb, self._get_omega_border_operands())
        new = _reset_or_increment(self, bend, self.nb)

        # commands below are equivalent to
        # return self.omega[i:(i+batch_size), 0:dim, 0:nb_facets]
        # and nb_facets = 2 * dimension
        # but JAX prefer the latter
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
        return new, PDEStatioBatch(inside_batch, border_batch)
