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
    key : Key
        Jax random key to shuffle batches
    obs_batch_size : int | None
        The size of the batch of randomly selected points among
        the `n` points. If None, no minibatch are used.
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
    obs_batch_size: int | None = eqx.field(static=True)
    observed_pinn_in: Float[Array, " n_obs nb_pinn_in"]
    observed_values: Float[Array, " n_obs nb_pinn_out"]
    observed_eq_params: dict[str, Float[Array, " n_obs 1"]] = eqx.field(
        static=True, default_factory=lambda: {}
    )
    sharding_device: jax.sharding.Sharding = eqx.field(static=True, default=None)

    n: int = eqx.field(init=False, static=True)
    curr_idx: int = eqx.field(init=False)
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
        if self.observed_pinn_in.ndim > 2:
            raise ValueError("self.observed_pinn_in must have 2 dimensions")
        if len(self.observed_values.shape) == 1:
            self.observed_values = self.observed_values[:, None]
        if self.observed_values.ndim > 2:
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

    def _get_operands(self) -> tuple[Key, Int[Array, " n"], int, int | None, None]:
        return (
            self.key,
            self.indices,
            self.curr_idx,
            self.obs_batch_size,
            None,
        )

    def obs_batch(
        self,
    ) -> tuple[DataGeneratorObservations, ObsBatchDict]:
        """
        Return an update DataGeneratorObservations instance and an ObsBatchDict
        """
        if self.obs_batch_size is None or self.obs_batch_size == self.n:
            # Avoid unnecessary reshuffling
            return self, {
                "pinn_in": self.observed_pinn_in,
                "val": self.observed_values,
                "eq_params": self.observed_eq_params,
            }

        new_attributes = _reset_or_increment(
            self.curr_idx + self.obs_batch_size,
            self.n,
            self._get_operands(),  # type: ignore
            # ignore since the case self.obs_batch_size is None has been
            # handled above
        )
        new = eqx.tree_at(
            lambda m: (m.key, m.indices, m.curr_idx), self, new_attributes
        )

        minib_indices = jax.lax.dynamic_slice(
            new.indices,
            start_indices=(new.curr_idx,),
            slice_sizes=(new.obs_batch_size,),
        )

        obs_batch: ObsBatchDict = {
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
    ) -> tuple[DataGeneratorObservations, ObsBatchDict]:
        """
        Generic method to return a batch
        """
        return self.obs_batch()
