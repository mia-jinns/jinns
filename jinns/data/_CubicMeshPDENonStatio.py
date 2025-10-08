"""
Define the DataGenerators modules
"""

from __future__ import (
    annotations,
)  # https://docs.python.org/3/library/typing.html#constant
import warnings
import equinox as eqx
import numpy as np
import jax
import jax.numpy as jnp
from scipy.stats import qmc
from jaxtyping import PRNGKeyArray, Array, Float
from jinns.data._Batchs import PDENonStatioBatch
from jinns.data._utils import (
    make_cartesian_product,
    _check_and_set_rar_parameters,
    _reset_or_increment,
)
from jinns.data._CubicMeshPDEStatio import CubicMeshPDEStatio


class CubicMeshPDENonStatio(CubicMeshPDEStatio):
    r"""
    A class implementing data generator object for non stationary partial
    differential equations. Formally, it extends `CubicMeshPDEStatio`
    to include a temporal batch.

    Parameters
    ----------
    key : PRNGKeyArray
        Jax random key to sample new time points and to shuffle batches
    n : int
        The number of total $I\times \Omega$ points that will be divided in
        batches. Batches are made so that each data point is seen only
        once during 1 epoch.
    nb : int | None
        The total number of points in $\partial\Omega$. Can be None if no
        boundary condition is specified.
    ni : int
        The number of total $\Omega$ points at $t=0$ that will be divided in
        batches. Batches are made so that each data point is seen only
        once during 1 epoch.
    domain_batch_size : int | None, default=None
        The size of the batch of randomly selected points among
        the `n` points. If None no mini-batches are used.
    border_batch_size : int | None, default=None
        The size of the batch of points randomly selected
        among the `nb` points. If None, `domain_batch_size` no
        mini-batches are used.
    initial_batch_size : int | None, default=None
        The number of randomly selected points among the `ni` initial spatial
        points used for initial condition. If None, no mini-batches are used.
    dim : int
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
    method : Literal["uniform", "grid", "sobol", "halton"], default="uniform"
        Either `grid` or `uniform`, default is `uniform`.
        The method that generates the `nt` time points. `grid` means
        regularly spaced points over the domain. `uniform` means uniformly
        sampled points over the domain.
        **Note** that Sobol and Halton approaches use scipy modules and will not
        be JIT compatible.
    rar_parameters : Dict[str, int], default=None
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
        then corresponds to the initial number of omega points we train the PINN.
    """

    tmin: float
    tmax: float
    ni: int = eqx.field(static=True)
    domain_batch_size: int | None = eqx.field(static=True)
    initial_batch_size: int | None = eqx.field(static=True)
    border_batch_size: int | None = eqx.field(static=True)

    # --- Below fields are not passed as arguments to __init__
    curr_domain_idx: int = eqx.field(init=False)
    curr_initial_idx: int = eqx.field(init=False)
    curr_border_idx: int = eqx.field(init=False)
    domain: Float[Array, " n 1+dim"] = eqx.field(init=False)
    border: Float[Array, " (nb//2) 1+1 2"] | Float[Array, " (nb//4) 2+1 4"] | None = (
        eqx.field(init=False)
    )
    initial: Float[Array, " ni dim"] | None = eqx.field(init=False)

    def __init__(
        self,
        tmin: float,
        tmax: float,
        ni: int,
        domain_batch_size: int | None = None,
        initial_batch_size: int | None = None,
        border_batch_size: int | None = None,
        **kwargs,  # kwargs for CubicMeshPDEStatio.__init__
    ):
        """
        Note that neither __init__ or __post_init__ are called when udating a
        Module with eqx.tree_at!
        """
        # sanity check
        if ni is None:
            raise ValueError("`ni` cannot be None.")

        super().__init__(**kwargs)
        self.tmin = tmin
        self.tmax = tmax
        self.ni = ni

        self.domain_batch_size = domain_batch_size
        self.initial_batch_size = initial_batch_size
        self.border_batch_size = border_batch_size

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

            # NOTE below re-do CubicMeshPDE.__init__() ? Maybe useless?
            (
                self.n_start,
                self.p,
                self.rar_iter_from_last_sampling,
                self.rar_iter_nb,
            ) = _check_and_set_rar_parameters(self.rar_parameters, self.n, self.n_start)
        elif self.method == "uniform":
            self.key, domain_times = self.generate_time_data(self.key, self.n)
            self.domain = jnp.concatenate([domain_times, self.omega], axis=1)
        elif self.method in ["sobol", "halton"]:
            self.key, self.domain = self.qmc_in_time_omega_domain(self.key, self.n)
        else:
            raise ValueError(
                f'Bad value for method. Got {self.method}, expected "grid" or "uniform" or "sobol" or "halton"'
            )

        if self.domain_batch_size is None:
            self.curr_domain_idx = 0
        else:
            self.curr_domain_idx = self.n + self.domain_batch_size
            # to be sure there is a shuffling at first get_batch()
        if self.nb is not None:
            assert (
                self.omega_border is not None
            )  # this needs to have been instanciated in super.__post_init__()
            # the check below has already been done in super.__post_init__ if
            # dim > 1. Here we retest it in whatever dim
            if self.nb % (2 * self.dim) != 0 or self.nb < 2 * self.dim:
                raise ValueError(
                    "number of border point must be"
                    " a multiple of 2xd (the # of faces of a d-dimensional cube)"
                )
            # the check below concern omega_border_batch_size for dim > 1 in
            # super.__init__. Here it concerns all dim values since our
            # border_batch is the concatenation or cartesian product with times
            if (
                self.border_batch_size is not None
                and self.nb // (2 * self.dim) < self.border_batch_size
            ):
                raise ValueError(
                    "number of points per facets (nb//2*self.dim)"
                    " cannot be lower than border batch size"
                )
            if self.method in ["grid", "uniform"]:
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
            else:
                self.key, self.border = self.qmc_in_time_omega_border_domain(
                    self.key,
                    self.nb,  # type: ignore (see inside the fun)
                )

            if self.border_batch_size is None:
                self.curr_border_idx = 0
            else:
                self.curr_border_idx = self.nb + self.border_batch_size
                # to be sure there is a shuffling at first get_batch()

        else:
            self.border = None
            self.border_batch_size = None
            self.curr_border_idx = 0

        if ni is not None:
            if self.method == "grid":
                perfect_sq = int(jnp.round(jnp.sqrt(self.ni)) ** 2)
                if self.ni != perfect_sq:
                    warnings.warn(
                        "Grid sampling is requested in dimension 2 with a non"
                        f" perfect square dataset size (self.ni = {self.ni})."
                        f" Modifying self.ni to self.ni = {perfect_sq}."
                    )
                self.ni = perfect_sq
            if self.method in ["sobol", "halton"]:
                log2_n = jnp.log2(self.ni)
                lower_pow = 2 ** jnp.floor(log2_n)
                higher_pow = 2 ** jnp.ceil(log2_n)
                closest_power_of_two = (
                    lower_pow
                    if (self.ni - lower_pow) < (higher_pow - self.ni)
                    else higher_pow
                )
                if self.n != closest_power_of_two:
                    warnings.warn(
                        f"QuasiMonteCarlo sampling with {self.method} requires sample size to be a power fo 2."
                        f"Modfiying self.n from {self.ni} to {closest_power_of_two}.",
                    )
                self.ni = int(closest_power_of_two)
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

        # the following attributes will not be used anymore
        self.omega = None  # type: ignore
        self.omega_border = None

    def generate_time_data(
        self, key: PRNGKeyArray, nt: int
    ) -> tuple[PRNGKeyArray, Float[Array, " nt 1"]]:
        """
        Construct a complete set of `nt` time points according to the
        specified `self.method`
        """
        key, subkey = jax.random.split(key, 2)
        if self.method == "grid":
            partial_times = (self.tmax - self.tmin) / nt
            return key, jnp.arange(self.tmin, self.tmax, partial_times)[:, None]
        elif self.method in ["uniform", "sobol", "halton"]:
            return key, self.sample_in_time_domain(subkey, nt)
        raise ValueError("Method " + self.method + " is not implemented.")

    def sample_in_time_domain(
        self, key: PRNGKeyArray, nt: int
    ) -> Float[Array, " nt 1"]:
        return jax.random.uniform(key, (nt, 1), minval=self.tmin, maxval=self.tmax)

    def qmc_in_time_omega_domain(
        self, key: PRNGKeyArray, sample_size: int
    ) -> tuple[PRNGKeyArray, Float[Array, "n 1+dim"]]:
        """
        Because in Quasi-Monte Carlo sampling we cannot concatenate two vectors generated independently
        We generate time and omega samples jointly
        """
        key, subkey = jax.random.split(key, 2)
        qmc_generator = qmc.Sobol if self.method == "sobol" else qmc.Halton
        sampler = qmc_generator(
            d=self.dim + 1, scramble=True, rng=np.random.default_rng(np.uint32(subkey))
        )
        samples = sampler.random(n=sample_size)
        samples[:, 1:] = qmc.scale(
            samples[:, 1:], l_bounds=self.min_pts, u_bounds=self.max_pts
        )  # We scale omega domain to be in (min_pts, max_pts)
        return key, jnp.array(samples)

    def qmc_in_time_omega_border_domain(
        self, key: PRNGKeyArray, sample_size: int | None = None
    ) -> tuple[PRNGKeyArray, Float[Array, "n 1+dim"]] | None:
        """
        For each facet of the border we generate Quasi-MonteCarlo sequences jointy with time.

        We need to do some type ignore in this function because we have lost
        the type narrowing from post_init,  type checkers only narrow at function level and because we cannot narrow a class attribute.
        """
        qmc_generator = qmc.Sobol if self.method == "sobol" else qmc.Halton
        sample_size = self.nb if sample_size is None else sample_size
        if sample_size is None:
            return None
        if self.dim == 1:
            key, subkey = jax.random.split(key, 2)
            qmc_seq = qmc_generator(
                d=1, scramble=True, rng=np.random.default_rng(np.uint32(subkey))
            )
            boundary_times = jnp.array(
                qmc_seq.random(self.nb // (2 * self.dim))  # type: ignore
            )
            boundary_times = boundary_times.reshape(-1, 1, 1)
            boundary_times = jnp.repeat(
                boundary_times,
                self.omega_border.shape[-1],  # type: ignore
                axis=2,
            )
            return key, make_cartesian_product(
                boundary_times,
                self.omega_border[None, None],  # type: ignore
            )
        if self.dim == 2:
            # currently hard-coded the 4 edges for d==2
            # TODO : find a general & efficient way to sample from the border
            # (facets) of the hypercube in general dim.
            key, *subkeys = jax.random.split(key, 5)
            facet_n = sample_size // (2 * self.dim)

            def generate_qmc_sample(key, min_val, max_val):
                qmc_seq = qmc_generator(
                    d=2,
                    scramble=True,
                    rng=np.random.default_rng(np.uint32(key)),
                )
                u = qmc_seq.random(n=facet_n)
                u[:, 1:2] = qmc.scale(u[:, 1:2], l_bounds=min_val, u_bounds=max_val)
                return jnp.array(u)

            xmin_sample = generate_qmc_sample(
                subkeys[0], self.min_pts[1], self.max_pts[1]
            )  # [t,x,y]
            xmin = jnp.hstack(
                [
                    xmin_sample[:, 0:1],
                    self.min_pts[0] * jnp.ones((facet_n, 1)),
                    xmin_sample[:, 1:2],
                ]
            )
            xmax_sample = generate_qmc_sample(
                subkeys[1], self.min_pts[1], self.max_pts[1]
            )
            xmax = jnp.hstack(
                [
                    xmax_sample[:, 0:1],
                    self.max_pts[0] * jnp.ones((facet_n, 1)),
                    xmax_sample[:, 1:2],
                ]
            )
            ymin = jnp.hstack(
                [
                    generate_qmc_sample(subkeys[2], self.min_pts[0], self.max_pts[0]),
                    self.min_pts[1] * jnp.ones((facet_n, 1)),
                ]
            )
            ymax = jnp.hstack(
                [
                    generate_qmc_sample(subkeys[3], self.min_pts[0], self.max_pts[0]),
                    self.max_pts[1] * jnp.ones((facet_n, 1)),
                ]
            )
            return key, jnp.stack([xmin, xmax, ymin, ymax], axis=-1)
        raise NotImplementedError(
            "Generation of the border of a cube in dimension > 2 is not "
            + f"implemented yet. You are asking for generation in dimension d={self.dim}."
        )

    def _get_domain_operands(
        self,
    ) -> tuple[PRNGKeyArray, Float[Array, " n 1+dim"], int, int | None, Array | None]:
        return (
            self.key,
            self.domain,
            self.curr_domain_idx,
            self.domain_batch_size,
            self.p,
        )

    def domain_batch(
        self,
    ) -> tuple[CubicMeshPDENonStatio, Float[Array, " domain_batch_size 1+dim"]]:
        if self.domain_batch_size is None or self.domain_batch_size == self.n:
            # Avoid unnecessary reshuffling
            return self, self.domain

        bstart = self.curr_domain_idx
        bend = bstart + self.domain_batch_size

        # Compute the effective number of used collocation points
        if self.rar_parameters is not None:
            n_eff = (
                self.n_start
                + self.rar_iter_nb  # type: ignore
                * self.rar_parameters["selected_sample_size"]
            )
        else:
            n_eff = self.n

        new_attributes = _reset_or_increment(
            bend,
            n_eff,
            self._get_domain_operands(),  # type: ignore
            # ignore since the case self.domain_batch_size is None has been
            # handled above
        )
        new = eqx.tree_at(
            lambda m: (m.key, m.domain, m.curr_domain_idx),  # type: ignore
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
        PRNGKeyArray,
        Float[Array, " nb 1+1 2"] | Float[Array, " (nb//4) 2+1 4"] | None,
        int,
        int | None,
        None,
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
        CubicMeshPDENonStatio,
        Float[Array, " border_batch_size 1+1 2"]
        | Float[Array, " border_batch_size 2+1 4"]
        | None,
    ]:
        if self.nb is None or self.border is None:
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

        new_attributes = _reset_or_increment(
            bend,
            n_eff,
            self._get_border_operands(),  # type: ignore
            # ignore since the case self.border_batch_size is None has been
            # handled above
        )
        new = eqx.tree_at(
            lambda m: (m.key, m.border, m.curr_border_idx),  # type: ignore
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
    ) -> tuple[PRNGKeyArray, Float[Array, " ni dim"] | None, int, int | None, None]:
        return (
            self.key,
            self.initial,
            self.curr_initial_idx,
            self.initial_batch_size,
            None,
        )

    def initial_batch(
        self,
    ) -> tuple[CubicMeshPDENonStatio, Float[Array, " initial_batch_size dim"] | None]:
        if self.initial_batch_size is None or self.initial_batch_size == self.ni:
            # Avoid unnecessary reshuffling
            return self, self.initial

        bstart = self.curr_initial_idx
        bend = bstart + self.initial_batch_size

        n_eff = self.ni

        new_attributes = _reset_or_increment(
            bend,
            n_eff,
            self._get_initial_operands(),  # type: ignore
            # ignore since the case self.initial_batch_size is None has been
            # handled above
        )
        new = eqx.tree_at(
            lambda m: (m.key, m.initial, m.curr_initial_idx),  # type: ignore
            self,
            new_attributes,
        )
        return new, jax.lax.dynamic_slice(
            new.initial,
            start_indices=(new.curr_initial_idx, 0),
            slice_sizes=(new.initial_batch_size, new.dim),
        )

    def get_batch(self) -> tuple[CubicMeshPDENonStatio, PDENonStatioBatch]:
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
