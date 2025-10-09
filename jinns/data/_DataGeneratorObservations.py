"""
Define the DataGenerators modules
"""

from __future__ import (
    annotations,
)  # https://docs.python.org/3/library/typing.html#constant
import equinox as eqx
import jax
import jax.numpy as jnp
from typing import TYPE_CHECKING, Self
from jaxtyping import PRNGKeyArray, Int, Array, Float
from jinns.data._Batchs import ObsBatchDict
from jinns.data._utils import _reset_or_increment
from jinns.data._AbstractDataGenerator import AbstractDataGenerator
from jinns.utils._DictToModuleMeta import DictToModuleMeta


class DGObservedParams(metaclass=DictToModuleMeta):
    """
    However, static type checkers cannot know that DGObservedParams inherit from
    eqx.Module and explicit casting to the latter class will be needed
    """

    pass


if TYPE_CHECKING:
    # imports only used in type hints
    InputEqParams = (
        dict[str, Float[Array, "  n_obs"]] | dict[str, Float[Array, " n_obs 1"]]
    ) | None

    # Note that the lambda functions used below are with type: ignore just
    # because the lambda are not type annotated, but there is no proper way
    # to do this and we should assign the lambda to a type hinted variable
    # before hand: this is not practical, let us not get mad at this


class DataGeneratorObservations(AbstractDataGenerator):
    r"""
    Despite the class name, it is rather a dataloader for user-provided
    observations which will are used in the observations loss.

    Parameters
    ----------
    key : PRNGKeyArray
        Jax random key to shuffle batches
    obs_batch_size : tuple[int | None, ...]
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
        XXX If some datasets do not have observed_eq_params, we expect {}
        A dict with keys corresponding to
        the parameter name. The keys must match the keys in
        `params["eq_params"]`, ie., if only some parameters are observed, other
        keys **must still appear with None as value**. The values are jnp.array with 2 dimensions
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

    key: PRNGKeyArray
    obs_batch_size: tuple[int | None, ...] = eqx.field(static=True)
    observed_pinn_in: tuple[Float[Array, " n_obs nb_pinn_in"], ...]
    observed_values: tuple[Float[Array, " n_obs nb_pinn_out"], ...]
    observed_eq_params: tuple[eqx.Module | None, ...]
    sharding_device: jax.sharding.Sharding | None  # = eqx.field(static=True)

    n: tuple[int, ...] = eqx.field(init=False, static=True)
    curr_idx: tuple[int, ...] = eqx.field(init=False)
    indices: tuple[Array, ...] = eqx.field(init=False)

    def __init__(
        self,
        *,
        key: PRNGKeyArray,
        obs_batch_size: tuple[int | None, ...] | int | None = None,
        observed_pinn_in: tuple[Float[Array, " n_obs nb_pinn_in"], ...]
        | Float[Array, " n_obs nb_pinn_in"],
        observed_values: tuple[Float[Array, " n_obs nb_pinn_out"], ...]
        | Float[Array, " n_obs nb_pinn_out"],
        observed_eq_params: tuple[InputEqParams, ...] | InputEqParams | None = None,
        sharding_device: jax.sharding.Sharding | None = None,
    ) -> None:
        super().__init__()
        self.key = key

        if not isinstance(observed_pinn_in, tuple):
            observed_pinn_in = (observed_pinn_in,)
        if not isinstance(observed_values, tuple):
            observed_values = (observed_values,)

        def check_first_axis(a, b):
            if a.shape[0] != b.shape[0]:
                raise ValueError(
                    "Each matching elements of self.observed_pinn_in and self.observed_values must have same first axis"
                )

        jax.tree.map(check_first_axis, observed_pinn_in, observed_values)

        self.observed_pinn_in = observed_pinn_in
        self.observed_values = observed_values

        if observed_eq_params is not None:

            def check_first_axis2(a, b):
                for _, v in a.items():
                    if v.shape[0] != b.shape[0]:
                        raise ValueError(
                            "Each matching elements of self.observed_pinn_in and self.observed_eq_params must have the same first axis"
                        )

            jax.tree.map(
                check_first_axis2,
                observed_eq_params,
                observed_pinn_in,
                is_leaf=lambda x: isinstance(x, dict),
            )

            if not isinstance(observed_eq_params, tuple):
                observed_eq_params = (observed_eq_params,)
            observed_eq_params = jax.tree.map(
                lambda d: {
                    k: v[:, None] if len(v.shape) == 1 else v for k, v in d.items()
                },
                observed_eq_params,
            )

            # Convert the dict of observed parameters to the internal
            # `DGObservedParams`
            # class used by Jinns.
            self.observed_eq_params = tuple(
                DGObservedParams(o_, "DGObservedParams") for o_ in observed_eq_params
            )
        else:
            self.observed_eq_params = tuple(
                None for _ in range(len(self.observed_pinn_in))
            )

        self.observed_pinn_in = jax.tree.map(
            lambda x: x[:, None] if len(x.shape) == 1 else x, self.observed_pinn_in
        )

        self.observed_values = jax.tree.map(
            lambda x: x[:, None] if len(x.shape) == 1 else x, self.observed_values
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
            if c is not None:
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

        if isinstance(obs_batch_size, int) or obs_batch_size is None:
            self.obs_batch_size = jax.tree.map(
                lambda _: obs_batch_size,
                self.observed_pinn_in,
            )

        # NOTE currently disabled
        self.sharding_device = None
        # self.sharding_device = sharding_device
        # if self.sharding_device is not None:
        #     self.observed_pinn_in = jax.lax.with_sharding_constraint(
        #         self.observed_pinn_in, self.sharding_device
        #     )
        #     self.observed_values = jax.lax.with_sharding_constraint(
        #         self.observed_values, self.sharding_device
        #     )
        #     self.observed_eq_params = jax.lax.with_sharding_constraint(
        #         self.observed_eq_params, self.sharding_device
        #     )

        # When self.obs_batch_size leaf is None we will have self.curr_idx leaf
        # to None. (Previous behaviour would put an unused self.curr_idx to 0)
        print(self.obs_batch_size)
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
        # recall post_init is the only place with _init_ where we can set
        # self attribute in a in-place way
        ###self.key, _ = jax.random.split(self.key, 2)  # to make it equivalent to
        #### the call to _reset_batch_idx_and_permute in legacy DG

    def _get_operands(
        self,
    ) -> tuple[
        tuple[PRNGKeyArray, ...],
        tuple[Int[Array, " n"], ...],
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
    ) -> tuple[PRNGKeyArray, Array, Int, ObsBatchDict]:
        """
        Return an update DataGeneratorObservations instance and an ObsBatchDict
        """
        if obs_batch_size is None or obs_batch_size == n:
            # Avoid unnecessary reshuffling
            return (
                key,
                indices,
                curr_idx,
                ObsBatchDict(
                    {
                        "pinn_in": observed_pinn_in,
                        "val": observed_values,
                        "eq_params": observed_eq_params,
                    }
                ),
            )

        new_key, new_indices, new_curr_idx = _reset_or_increment(
            curr_idx + obs_batch_size,
            n,
            (key, indices, curr_idx, obs_batch_size, None),  # type: ignore
            # ignore since the case self.obs_batch_size is None has been
            # handled above
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
                lambda a: jnp.take(a, minib_indices, unique_indices=True, axis=0),  # type: ignore
                observed_eq_params,
            ),
        }
        return new_key, new_indices, new_curr_idx, obs_batch

    def get_batch(
        self,
    ) -> tuple[Self, tuple[ObsBatchDict, ...]]:
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
