"""
We define a new class AdaptativeCubicMesh
"""

import equinox as eqx
import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental import io_callback
from scipy.stats import qmc
from jaxtyping import Key, Array, Float
from typing import Literal
from jinns.data._Batchs import PDENonStatioBatch
from jinns.data._AbstractDataGenerator import AbstractDataGenerator
from jinns.data._utils import make_cartesian_product


class ACMPDENonStatio(AbstractDataGenerator):
    r"""
    class implementing data generation in adaptative way

    Parameters
    ----------

    key: Key Jax random key that we split from at each iteration to
         sample new data
    n : int The number of collocation points sampled at each
    nb: int | None
    ni: int
    dim : int
    min_pts :
    max_pts :
    tmin :
    tmax :
    method : Literal["uniform", "sobol", "halton"]
    """

    key: Key = eqx.field(kw_only=True)
    n: int = eqx.field(kw_only=True, static=True)
    nb: int | None = eqx.field(kw_only=True, static=True, default=None)
    ni: int = eqx.field(kw_only=True, static=True)
    dim: int = eqx.field(kw_only=True, static=True)
    min_pts: tuple[float, ...] = eqx.field(kw_only=True, static=True)
    max_pts: tuple[float, ...] = eqx.field(kw_only=True, static=True)
    method: Literal["uniform", "sobol", "halton"] = eqx.field(
        kw_only=True, static=True, default_factory=lambda: "uniform"
    )

    ## For Residual Adaptative Sampling
    rar_parameters: dict[str, int] = eqx.field(kw_only=True, default=None)
    rar_k: float = eqx.field(kw_only=True, default=None)
    rar_c: float = eqx.field(kw_only=True, default=None)
    residuals: Float[Array, "n 1+dim"] = eqx.field(kw_only=True, default=None)

    tmin: Float = eqx.field(kw_only=True, static=True)
    tmax: Float = eqx.field(kw_only=True, static=True)

    domain: Float[Array, "n 1+dim"] = eqx.field(init=False)
    border: (
        Float[Array, " (nb//2) 1+1 1+1"] | Float[Array, "(nb//2) 1+2 2+2"] | None
    ) = eqx.field(init=False)
    initial: Float[Array, " ni dim"] | None = eqx.field(init=False)

    def __post_init__(self):
        if len(self.min_pts) != self.dim or len(self.max_pts) != self.dim:
            raise ValueError(
                f"min_pts and max_pts must have length {self.dim}"
                f"for dimension {self.dim}"
            )
        self.key, domain_key, border_key, initial_key = jax.random.split(self.key, 4)

        # If RAR sampling

        # Domain sampling (same for uniform and QMC)
        if self.method == "uniform":
            self.domain = jax.random.uniform(
                domain_key,
                (self.n, self.dim + 1),
                minval=jnp.array((self.tmin,) + self.min_pts),
                maxval=jnp.array((self.tmax,) + self.max_pts),
            )
        else:  # "sobol" or "halton"
            self.domain = self.sample_in_domain(domain_key)

        # Border sampling
        if self.nb is not None:
            if self.method == "uniform":
                if self.dim == 1:
                    self.border = jnp.array([self.min_pts[0], self.max_pts[0]]).astype(
                        float
                    )
                elif self.dim == 2:
                    # Split border_key into 4 keys for uniform sampling in 2D
                    border_keys = jax.random.split(border_key, 4)
                    self.border = self.sample_in_domain_border(border_keys)
                else:
                    raise NotImplementedError(
                        "Uniform border sampling not implemented for dim > 2"
                    )
            else:  # "sobol" or "halton"
                self.border = self.qmc_sample_in_domain_border(border_key)
        else:
            self.border = None

        # Initial condition sampling
        if self.ni is not None:
            if self.method == "uniform":
                self.initial = jax.random.uniform(
                    initial_key,
                    (self.ni, self.dim),
                    minval=jnp.array(self.min_pts),
                    maxval=jnp.array(self.max_pts),
                )
            else:  # "sobol" or "halton"
                self.initial = self.sample_initial(initial_key)
        else:
            self.initial = None

            # Verify shapes
            assert self.domain.shape == (self.n, self.dim + 1)
            if self.border is not None:
                if self.dim == 1:
                    assert self.border.shape == (self.nb // 2, 2, 2)
                else:
                    assert self.border.shape == (
                        self.nb // (2 * self.dim),
                        self.dim + 1,
                        2 * self.dim,
                    )
            if self.initial is not None:
                assert self.initial.shape == (self.ni, self.dim)

    ## Sampling methods
    def sample_in_domain(self, key: Key) -> Float[Array, "n 1+dim"]:
        """ """
        key, subkey = jax.random.split(key, 2)
        qmc_generator = qmc.Sobol if self.method == "sobol" else qmc.Halton
        sampler = qmc_generator(
            d=self.dim + 1, scramble=True, rng=self._get_numpy_rng(subkey)
        )
        samples = sampler.random(n=self.n)
        samples[:, 1:] = qmc.scale(
            samples[:, 1:], l_bounds=self.min_pts, u_bounds=self.max_pts
        )  # We scale omega domain to be in (min_pts, max_pts)
        return jnp.array(samples, dtype=jnp.float32)

    def qmc_sample_in_domain_border(
        self, key: Key
    ) -> Float[Array, "nb 1+dim 2+2"] | Float[Array, "nb 1+dim 1+1"] | None:
        """ """
        qmc_generator = qmc.Sobol if self.method == "sobol" else qmc.Halton
        if self.nb is None:
            return None
        if self.dim == 1:
            omega_border = jnp.array([self.min_pts[0], self.max_pts[0]]).astype(float)
            qmc_seq = qmc_generator(d=1, scramble=True, rng=self._get_numpy_rng(key))
            boundary_times = jnp.array(
                qmc_seq.random(self.nb // (2 * self.dim))
            )  ## WE SHOULD TRIM TO THE POWER OF TWO IF IT IS NOT THE CASE
            boundary_times = boundary_times.reshape(-1, 1, 1)
            boundary_times = jnp.repeat(boundary_times, omega_border.shape[-1], axis=2)
            return make_cartesian_product(boundary_times, omega_border[None, None])
        if self.dim == 2:
            # currently hard-coded the 4 edges for d==2
            # TODO : find a general & efficient way to sample from the border
            # (facets) of the hypercube in general dim.
            key, *subkeys = jax.random.split(key, 5)
            facet_n = self.nb // (2 * self.dim)

            def generate_qmc_sample(key, min_val, max_val):
                qmc_seq = qmc_generator(
                    d=2,
                    scramble=True,
                    rng=self._get_numpy_rng(key),
                )
                u = qmc_seq.random(n=facet_n)
                u[:, 1:2] = qmc.scale(u[:, 1:2], l_bounds=min_val, u_bounds=max_val)
                return jnp.array(u, dtype=jnp.float32)

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
            return jnp.stack([xmin, xmax, ymin, ymax], axis=-1)
        raise NotImplementedError(
            "Generation of the border of a cube in dimension > 2 is not "
            + f"implemented yet. You are asking for generation in dimension d={self.dim}."
        )

    def sample_in_domain_border(
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
            key_0, subkey_0 = jax.random.split(keys[0], 2)
            xmin = jnp.hstack(
                [
                    jax.random.uniform(
                        key_0, (facet_n, 1), minval=self.tmin, maxval=self.tmax
                    ),
                    self.min_pts[0] * jnp.ones((facet_n, 1)),
                    jax.random.uniform(
                        subkey_0,
                        (facet_n, 1),
                        minval=self.min_pts[1],
                        maxval=self.max_pts[1],
                    ),
                ]
            )
            key_1, subkey_1 = jax.random.split(keys[1], 2)
            xmax = jnp.hstack(
                [
                    jax.random.uniform(
                        key_1, (facet_n, 1), minval=self.tmin, maxval=self.tmax
                    ),
                    self.max_pts[0] * jnp.ones((facet_n, 1)),
                    jax.random.uniform(
                        subkey_1,
                        (facet_n, 1),
                        minval=self.min_pts[1],
                        maxval=self.max_pts[1],
                    ),
                ]
            )
            key_2, subkey_2 = jax.random.split(keys[2], 2)
            ymin = jnp.hstack(
                [
                    jax.random.uniform(
                        key_2, (facet_n, 1), minval=self.tmin, maxval=self.tmax
                    ),
                    jax.random.uniform(
                        subkey_2,
                        (facet_n, 1),
                        minval=self.min_pts[0],
                        maxval=self.max_pts[0],
                    ),
                    self.min_pts[1] * jnp.ones((facet_n, 1)),
                ]
            )
            key_3, subkey_3 = jax.random.split(keys[3], 2)
            ymax = jnp.hstack(
                [
                    jax.random.uniform(
                        key_3, (facet_n, 1), minval=self.tmin, maxval=self.tmax
                    ),
                    jax.random.uniform(
                        subkey_3,
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

    def sample_initial(self, key: Key) -> Float[Array, "ni dim"]:
        """ """
        qmc_generator = qmc.Sobol if self.method == "sobol" else qmc.Halton
        sampler = qmc_generator(d=self.dim, scramble=True, rng=self._get_numpy_rng(key))
        samples = sampler.random(n=self.ni)
        samples = qmc.scale(
            samples, l_bounds=self.min_pts, u_bounds=self.max_pts
        )  # We scale omega domain to be in (min_pts, max_pts)
        return jnp.array(samples)

    def update_residuals(self, residuals: Array):
        """
        Update the current residuals of RAR sampling
        """
        self.residuals = jnp.abs(residuals)

    def sample_with_rar(self, key: Key, n_samples: int) -> Array:
        """ """
        # Compute weights based on residuals
        # Get current domain points (or generate new ones)
        # Sample according to weights
        pass

    ## Batch generation

    def get_batch(self):
        """
        Generate a random batch on points directly in the training
        """
        key, domain_key, border_key, initial_key = jax.random.split(self.key, 4)
        border_keys = jax.random.split(border_key, 4)

        def _sample_in_domain(key):
            return self.sample_in_domain(key)

        def _sample_in_domain_border(key):
            return self.qmc_sample_in_domain_border(key)

        def _sample_initial(key):
            return self.sample_initial(key)

        # Domain points
        domain_shape = (self.n, self.dim + 1)
        if self.method == "uniform":
            new_domain = jax.random.uniform(
                domain_key,
                domain_shape,
                minval=jnp.array((self.tmin,) + self.min_pts),
                maxval=jnp.array((self.tmax,) + self.max_pts),
            )
        else:
            new_domain = io_callback(
                _sample_in_domain,
                jax.ShapeDtypeStruct(domain_shape, jnp.float32),
                domain_key,
            )

        # Border points
        if self.border is not None:
            if self.method == "uniform":
                new_border = self.sample_in_domain_border(border_keys, self.nb)
            else:
                if self.dim == 1:
                    border_shape = (self.nb // 2, 1 + 1, 1 + 1)
                else:  # dim == 2
                    border_shape = (
                        self.nb // (2 * self.dim),
                        1 + self.dim,
                        2 * self.dim,
                    )
                new_border = io_callback(
                    _sample_in_domain_border,
                    jax.ShapeDtypeStruct(border_shape, jnp.float32),
                    border_key,
                )
        else:
            new_border = None

        # Initial points
        if self.initial is not None:
            initial_shape = (self.ni, self.dim)
            if self.method == "uniform":
                new_initial = jax.random.uniform(
                    initial_key,
                    initial_shape,
                    minval=jnp.array(self.min_pts),
                    maxval=jnp.array(self.max_pts),
                )
            else:
                new_initial = io_callback(
                    _sample_initial,
                    jax.ShapeDtypeStruct(initial_shape, jnp.float32),
                    initial_key,
                )
        else:
            new_initial = None

        # Update the state
        new = eqx.tree_at(
            lambda m: (m.key, m.domain, m.border, m.initial),
            self,
            (key, new_domain, new_border, new_initial),
        )

        return new, PDENonStatioBatch(new_domain, new_border, new_initial)

    def _get_numpy_rng(self, key: Key) -> np.random.Generator:
        """
        Convert JAX key to NumPy RNG.
        """

        return np.random.default_rng(np.uint32(key))
