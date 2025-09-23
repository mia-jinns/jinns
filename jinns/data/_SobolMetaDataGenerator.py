"""
Define the DataGenerators modules
"""

from __future__ import (
    annotations,
)  # https://docs.python.org/3/library/typing.html#constant
import equinox as eqx
import numpy as np
import jax
import jax.numpy as jnp
from scipy.stats import qmc
from jaxtyping import Key, Array, Float
from jinns.data._Batchs import PDENonStatioBatch
from jinns.data._utils import (
    make_cartesian_product,
    _reset_or_increment,
)

from jinns.data._AbstractDataGenerator import AbstractDataGenerator


class SobolMetaDataGenerator(AbstractDataGenerator):
    r"""
    A DataGenartor that generate a joint distribution of domain-time-parameters
    """

    key: Key = eqx.field(kw_only=True)
    keys: Key | dict[str, Key]
    n: int = eqx.field(static=True)
    nb: int | None = eqx.field(kw_only=True, static=True)
    ni: int = eqx.field(kw_only=True, static=True)
    np: int = eqx.field(kw_only=True, static=True)

    param_ranges: dict[str, tuple[Float, Float]] = eqx.field(
        static=True, default_factory=lambda: {}
    )
    user_data: dict[str, Float[Array, " n"]] | None = eqx.field(
        default_factory=lambda: {}
    )

    dim: int = eqx.field(kw_only=True, static=True)
    min_pts: tuple[float, ...] = eqx.field(kw_only=True)
    max_pts: tuple[float, ...] = eqx.field(kw_only=True)
    tmin: Float = eqx.field(kw_only=True)
    tmax: Float = eqx.field(kw_only=True)

    domain_batch_size: int | None = eqx.field(kw_only=True, static=True, default=None)
    initial_batch_size: int | None = eqx.field(kw_only=True, static=True, default=None)
    border_batch_size: int | None = eqx.field(kw_only=True, static=True, default=None)
    param_batch_size: int | None = eqx.field(static=True, default=None)

    curr_domain_idx: int = eqx.field(init=False)
    curr_initial_idx: int = eqx.field(init=False)
    curr_border_idx: int = eqx.field(init=False)
    curr_param_idx: dict[str, int] = eqx.field(init=False)

    domain: Float[Array, " n 1+dim"] = eqx.field(init=False)
    border: Float[Array, " (nb//2) 1+1 2"] | Float[Array, " (nb//4) 2+1 4"] | None = (
        eqx.field(init=False)
    )
    initial: Float[Array, " ni dim"] | None = eqx.field(init=False)
    param_n_samples: dict[str, Array] = eqx.field(init=False)

    def __post_init__(self):
        """
        Note that neither __init__ or __post_init__ are called when udating a
        Module with eqx.tree_at!
        """
        assert self.dim == len(self.min_pts) and isinstance(self.min_pts, tuple)
        assert self.dim == len(self.max_pts) and isinstance(self.max_pts, tuple)

        ## For the parameters
        if self.user_data is None:
            self.user_data = {}
        if self.param_ranges is None:
            self.param_ranges = {}
        if self.param_batch_size is not None and self.n < self.param_batch_size:
            raise ValueError(
                f"Number of data points ({self.n}) is smaller than the"
                f"number of batch points ({self.param_batch_size})."
            )
        if not isinstance(self.keys, dict):
            all_keys = set().union(self.param_ranges, self.user_data)
            self.keys = dict(zip(all_keys, jax.random.split(self.keys, len(all_keys))))

        if self.param_batch_size is None:
            self.curr_param_idx = None  # type: ignore
        else:
            self.curr_param_idx = {}
            for k in self.keys.keys():
                self.curr_param_idx[k] = self.n + self.param_batch_size
                # to be sure there is a shuffling at first get_batch()

        # The call to self.generate_data() creates
        # the dict self.param_n_samples and then we will only use this one
        # because it merges the scattered data between `user_data` and
        # `param_ranges`

        self.key, self.keys, self.domain, self.param_n_samples = (
            self.qmc_in_time_omega_domain_parameters(self.key, self.keys, self.n)
        )  # here add parameters

        if self.domain_batch_size is None:
            self.curr_domain_idx = 0
        else:
            self.curr_domain_idx = self.n + self.domain_batch_size
            # to be sure there is a shuffling at first get_batch()

        if self.nb is not None:
            if self.dim == 1:
                self.border_batch_size = None
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
                    self.border_batch_size is not None
                    and self.nb // (2 * self.dim) < self.border_batch_size
                ):
                    raise ValueError(
                        f"number of points per facets ({self.nb // (2 * self.dim)})"
                        f" cannot be lower than border batch size "
                        f" ({self.border_batch_size})."
                    )
                self.nb = int((2 * self.dim) * (self.nb // (2 * self.dim)))

            if self.border_batch_size is None:
                self.curr_border_idx = 0
            else:
                self.curr_border_idx = self.nb + self.border_batch_size
                # to be sure there is a shuffling at first get_batch()
        else:  # self.nb is None
            self.curr_border_idx = 0

        self.key, self.border = self.generate_omega_border_data(self.key)

        if self.nb is not None:
            self.key, boundary_times = self.generate_time_data(
                self.key, self.nb // (2 * self.dim)
            )
            boundary_times = boundary_times.reshape(-1, 1, 1)
            boundary_times = jnp.repeat(boundary_times, self.border.shape[-1], axis=2)
            if self.dim == 1:
                self.border = make_cartesian_product(
                    boundary_times, self.border[None, None]
                )
            else:
                self.border = jnp.concatenate([boundary_times, self.border], axis=1)

            if self.border_batch_size is None:
                self.curr_border_idx = 0
            else:
                self.curr_border_idx = self.nb + self.border_batch_size
                # to be sure there is a shuffling at first get_batch()

        else:
            self.border = None
            self.border_batch_size = None
            self.curr_border_idx = 0

        if self.ni is not None:
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

    def generate_time_data(
        self, key: Key, nt: int
    ) -> tuple[Key, Float[Array, " nt 1"]]:
        """
        Construct a complete set of `nt` time points according to the
        specified `self.method`
        """
        key, subkey = jax.random.split(key, 2)
        return key, self.sample_in_time_domain(subkey, nt)

    def sample_in_time_domain(self, key: Key, nt: int) -> Float[Array, " nt 1"]:
        return jax.random.uniform(key, (nt, 1), minval=self.tmin, maxval=self.tmax)

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
        if self.dim == 1:
            key, subkeys = jax.random.split(key, 2)
        else:
            key, *subkeys = jax.random.split(key, self.dim + 1)
        omega = self.sample_in_omega_domain(subkeys, sample_size=data_size)
        return key, omega

    def sample_in_omega_domain(
        self, keys: Key, sample_size: int
    ) -> Float[Array, " n dim"]:
        if self.dim == 1:
            xmin, xmax = self.min_pts[0], self.max_pts[0]
            return jax.random.uniform(
                keys, shape=(sample_size, 1), minval=xmin, maxval=xmax
            )

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

    def qmc_in_time_omega_domain_parameters(
        self,
        key: Key,
        keys: dict[str, Key],
        sample_size: int,
    ) -> tuple[
        Key, dict[str, Key], Float[Array, "n 1+dim"], dict[str, Float[Array, " n"]]
    ]:
        """
        Because in Quasi-Monte Carlo sampling we cannot concatenate two vectors generated independently
        We generate time and omega and paramters samples jointly
        """
        param_n_samples = {}

        all_keys = set().union(
            self.param_ranges,
            self.user_data,  # type: ignore this has been handled in post_init
        )
        idx = self.dim + 1
        k = list(keys.keys())[0]
        key, subkey = jax.random.split(key, 2)
        keys[k], subkey = jax.random.split(keys[k], 2)
        qmc_generator = qmc.Sobol
        sampler = qmc_generator(
            d=len(all_keys) + self.dim + 1,
            scramble=True,
            rng=self._get_numpy_rng(subkey),
        )
        samples = sampler.random(n=sample_size)

        for k in all_keys:
            if self.user_data and k in self.user_data.keys():
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
                param_n_samples[k] = jnp.array(
                    qmc.scale(
                        samples[:, idx].reshape(-1, 1),
                        l_bounds=self.param_ranges[k][0],
                        u_bounds=self.param_ranges[k][1],
                    )
                )  # We scale omega domain to be in (min_pts, max_pts)
                assert param_n_samples[k].shape == (self.n, 1)
                idx = idx + 1

        samples[:, 1 : (self.dim + 1)] = qmc.scale(
            samples[:, 1 : (self.dim + 1)], l_bounds=self.min_pts, u_bounds=self.max_pts
        )  # We scale omega domain to be in (min_pts, max_pts)

        return key, keys, jnp.array(samples[:, 0 : self.dim + 1]), param_n_samples

    def _get_domain_operands(
        self,
    ) -> tuple[Key, Float[Array, " n 1+dim"], int, int | None, Array | None]:
        return (
            self.key,
            self.domain,
            self.curr_domain_idx,
            self.domain_batch_size,
            None,
        )

    def domain_batch(
        self,
    ) -> tuple[SobolMetaDataGenerator, Float[Array, " domain_batch_size 1+dim"]]:
        if self.domain_batch_size is None or self.domain_batch_size == self.n:
            # Avoid unnecessary reshuffling
            return self, self.domain

        bstart = self.curr_domain_idx
        bend = bstart + self.domain_batch_size

        # Compute the effective number of used collocation points
        n_eff = self.n

        new_attributes = _reset_or_increment(
            bend,
            n_eff,
            self._get_domain_operands(),  # type: ignore
            # ignore since the case self.domain_batch_size is None has been
            # handled above
        )
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
        Key,
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
        SobolMetaDataGenerator,
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
    ) -> tuple[Key, Float[Array, " ni dim"] | None, int, int | None, None]:
        return (
            self.key,
            self.initial,
            self.curr_initial_idx,
            self.initial_batch_size,
            None,
        )

    def initial_batch(
        self,
    ) -> tuple[SobolMetaDataGenerator, Float[Array, " initial_batch_size dim"] | None]:
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
            lambda m: (m.key, m.initial, m.curr_initial_idx),
            self,
            new_attributes,
        )
        return new, jax.lax.dynamic_slice(
            new.initial,
            start_indices=(new.curr_initial_idx, 0),
            slice_sizes=(new.initial_batch_size, new.dim),
        )

    def _get_param_operands(
        self, k: str
    ) -> tuple[Key, Float[Array, " n"], int, int | None, None]:
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
                (key_k, param_k, idx_k, self.param_batch_size, None),  # type: ignore
                # ignore since the case self.param_batch_size is None has been
                # handled above
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

    def get_batch(self) -> tuple[SobolMetaDataGenerator, PDENonStatioBatch]:
        """
        Generic method to return a batch. Here we call `self.domain_batch()`,
        `self.border_batch()` and `self.initial_batch()`
        """
        new, domain = self.domain_batch()
        new, param = self.param_batch()
        if self.border is not None:
            new, border = new.border_batch()
        else:
            border = None
        if self.initial is not None:
            new, initial = new.initial_batch()
        else:
            initial = None

        return new, PDENonStatioBatch(
            domain_batch=domain,
            border_batch=border,
            initial_batch=initial,
            param_batch_dict=param,
        )

    def _get_numpy_rng(self, key: Key) -> np.random.Generator:
        """
        Convert JAX key to NumPy RNG.
        """

        return np.random.default_rng(np.uint32(key))
