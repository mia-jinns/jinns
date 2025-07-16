"""
Define the DataGenerators modules
"""

from __future__ import (
    annotations,
)  # https://docs.python.org/3/library/typing.html#constant
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Key, Int, Array, Float
from jinns.data._Batchs import ObsBatchDict
from jinns.data._utils import _reset_or_increment
from jinns.data._AbstractDataGenerator import AbstractDataGenerator


class DataGeneratorObservations(AbstractDataGenerator):
    r"""
    Despite the class name, it is rather a dataloader for user-provided
    observations which will are used in the observations loss.

    Parameters
    ----------
    key : tuple[Key, ...]
        A tuple of JAX random keys to shuffle batches
        We accept also Key type. The boradcasting to a tuple will then be done
        internally.
    obs_batch_size : tuple[int | None, ...]
        A tuple of the sizes of the batches of randomly selected points among
        the `n` points. If a single integer is passed, we use this same batch
        size for all observation batches. If None, no minibatch are used.
    observed_pinn_in : Float[Array, " n_obs nb_pinn_in"]
        Observed values corresponding to the input of the PINN
        (eg. the time at which we recorded the observations). The first
        dimension must corresponds to the number of observed_values.
        The second dimension depends on the input dimension of the PINN,
        that is `1` for ODE, `n_dim_x` for stationnary PDE and `n_dim_x + 1`
        for non-stationnary PDE.
    observed_values : Float[Array, " n_obs, nb_pinn_out"]
        Observed values that the PINN should learn to fit. The first
        dimension must be aligned with observed_pinn_in.
    observed_eq_params : dict[str, Float[Array, " n_obs 1"]], default={}
        XXX If some datasets do not have observed_eq_params, we expect {}
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

    key: tuple[Key, ...]
    obs_batch_size: tuple[int | None, ...] = eqx.field(
        static=True, converter=lambda x: (x,) if not isinstance(x, tuple) else x
    )
    observed_pinn_in: (
        tuple[Float[Array, " n_obs nb_pinn_in"], ...]
        | Float[Array, " n_obs nb_pinn_in"]
    ) = eqx.field(converter=lambda x: (x,) if not isinstance(x, tuple) else x)
    observed_values: (
        tuple[Float[Array, " n_obs nb_pinn_out"], ...]
        | Float[Array, " n_obs nb_pinn_out"]
    ) = eqx.field(converter=lambda x: (x,) if not isinstance(x, tuple) else x)
    observed_eq_params: (
        tuple[dict[str, Float[Array, " n_obs 1"]], ...]
        | dict[str, Float[Array, " n_obs 1"]]
    ) = eqx.field(static=True, default=None)
    # sharding_device: jax.sharding.Sharding = eqx.field(static=True, default=None)

    n: tuple[int, ...] = eqx.field(init=False, static=True)
    curr_idx: tuple[int, ...] = eqx.field(init=False)
    indices: tuple[Array, ...] = eqx.field(init=False)

    def __post_init__(self):
        # below is like the other converter but we need to do here because we
        # need the length of the other tuples
        if self.observed_eq_params is None:
            self.observed_eq_params = tuple(
                {} for i in range(len(self.observed_pinn_in))
            )

        def check_first_axis(a, b):
            if a.shape[0] != b.shape[0]:
                raise ValueError(
                    "Each matching elements of self.observed_pinn_in and self.observed_values must have same first axis"
                )

        jax.tree.map(check_first_axis, self.observed_pinn_in, self.observed_values)

        def check_first_axis2(a, b):
            for _, v in a.items():
                if v.shape[0] != b.shape[0]:
                    raise ValueError(
                        "Each matching elements of self.observed_pinn_in and self.observed_eq_params must have the same first axis"
                    )

        jax.tree.map(
            check_first_axis2,
            self.observed_eq_params,
            self.observed_pinn_in,
            is_leaf=lambda x: isinstance(x, dict),
        )

        self.observed_pinn_in = jax.tree.map(
            lambda x: x[:, None] if len(x.shape) == 1 else x, self.observed_pinn_in
        )

        self.observed_values = jax.tree.map(
            lambda x: x[:, None] if len(x.shape) == 1 else x, self.observed_values
        )

        self.observed_eq_params = jax.tree.map(
            lambda d: {k: v[:, None] if len(v.shape) == 1 else v for k, v in d.items()},
            self.observed_eq_params,
        )

        def check_ndim(a, b, c):
            if a.ndim > 2:
                raise ValueError(
                    "Each element of self.observed_pinn_in must have 2 dimensions"
                )
            if b.ndim > 2:
                raise ValueError(
                    "Each element of self.observed_values must have 2 dimensions"
                )
            for _, v in c.items():
                if v.ndim > 2:
                    raise ValueError(
                        "Each value of observed_eq_params must have 2 dimensions"
                    )

        jax.tree.map(
            check_ndim,
            self.observed_pinn_in,
            self.observed_values,
            self.observed_eq_params,
        )

        self.n = jax.tree.map(
            lambda o: o.shape[0],
            self.observed_pinn_in,
        )

        # if self.sharding_device is not None:
        #    self.observed_pinn_in = jax.lax.with_sharding_constraint(
        #        self.observed_pinn_in, self.sharding_device
        #    )
        #    self.observed_values = jax.lax.with_sharding_constraint(
        #        self.observed_values, self.sharding_device
        #    )
        #    self.observed_eq_params = jax.lax.with_sharding_constraint(
        #        self.observed_eq_params, self.sharding_device
        #    )

        if isinstance(self.obs_batch_size, int) or self.obs_batch_size is None:
            self.obs_batch_size = jax.tree.map(
                lambda _: self.obs_batch_size, self.observed_pinn_in
            )

        # When self.obs_batch_size leaf is None we will have self.curr_idx leaf
        # to None. (Previous behaviour would put an unused self.curr_idx to 0)
        self.curr_idx = jax.tree.map(
            lambda bs, n: bs + n if bs is not None else None,
            self.obs_batch_size,
            self.n,
            is_leaf=lambda x: x is None,
        )
        # For speed and to avoid duplicating data what is really
        # shuffled is a vector of indices
        # if self.sharding_device is not None:
        #    self.indices = jax.lax.with_sharding_constraint(
        #        jnp.arange(self.n), self.sharding_device
        #    )
        # else:
        self.indices = jax.tree.map(lambda x: jnp.arange(x), self.n)

        if not isinstance(self.key, tuple):
            # recall post_init is the only place with _init_ where we can set
            # self attribute in a in-place way
            self.key = jax.tree.unflatten(
                jax.tree.structure(self.n),
                jax.random.split(self.key, len(jax.tree.leaves(self.n))),
            )

    def _get_operands(
        self,
    ) -> tuple[
        tuple[Key, ...],
        tuple[Array, ...],
        tuple[int, ...],
        tuple[int | None, ...],
        None,
    ]:
        return (
            self.key,
            self.indices,
            self.curr_idx,
            self.obs_batch_size,
            None,
        )

    @staticmethod
    def obs_batch(
        n,
        obs_batch_size,
        observed_pinn_in,
        observed_values,
        observed_eq_params,
        curr_idx,
        key,
        indices,
    ) -> tuple[Key, Array, Int, ObsBatchDict]:
        """
        Return an update DataGeneratorObservations instance and an ObsBatchDict
        """
        if obs_batch_size is None or obs_batch_size == n:
            # Avoid unnecessary reshuffling
            return (
                key,
                indices,
                curr_idx,
                {
                    "pinn_in": observed_pinn_in,
                    "val": observed_values,
                    "eq_params": observed_eq_params,
                },
            )

        new_key, new_indices, new_curr_idx = _reset_or_increment(
            curr_idx + obs_batch_size,
            n,
            (key, indices, curr_idx, obs_batch_size, None),  # type: ignore
        )

        minib_indices = jax.lax.dynamic_slice(
            new_indices,
            start_indices=(new_curr_idx,),
            slice_sizes=(obs_batch_size,),
        )

        obs_batch: ObsBatchDict = {
            "pinn_in": jnp.take(
                observed_pinn_in, minib_indices, unique_indices=True, axis=0
            ),
            "val": jnp.take(
                observed_values, minib_indices, unique_indices=True, axis=0
            ),
            "eq_params": jax.tree_util.tree_map(
                lambda a: jnp.take(a, minib_indices, unique_indices=True, axis=0),
                observed_eq_params,
            ),
        }
        return new_key, new_indices, new_curr_idx, obs_batch

    def get_batch(
        self,
    ) -> tuple[DataGeneratorObservations, tuple[ObsBatchDict, ...]]:
        """
        Generic method to return a batch
        """

        ret = jax.tree.map(
            DataGeneratorObservations.obs_batch,
            self.n,
            self.obs_batch_size,
            self.observed_pinn_in,
            self.observed_values,
            self.observed_eq_params,
            self.curr_idx,
            self.key,
            self.indices,
        )
        new_key = jax.tree.map(
            lambda l: l[0], ret, is_leaf=lambda x: isinstance(x, tuple) and len(x) == 4
        )  # we must not traverse the second level
        new_indices = jax.tree.map(
            lambda l: l[1], ret, is_leaf=lambda x: isinstance(x, tuple) and len(x) == 4
        )
        new_curr_idx = jax.tree.map(
            lambda l: l[2], ret, is_leaf=lambda x: isinstance(x, tuple) and len(x) == 4
        )
        obs_batch_tuple = jax.tree.map(
            lambda l: l[3], ret, is_leaf=lambda x: isinstance(x, tuple) and len(x) == 4
        )

        new = eqx.tree_at(
            lambda m: (m.key, m.indices, m.curr_idx),
            self,
            (new_key, new_indices, new_curr_idx),
        )

        return new, obs_batch_tuple
