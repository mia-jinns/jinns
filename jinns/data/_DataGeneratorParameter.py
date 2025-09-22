"""
Define the DataGenerators modules
"""

from __future__ import (
    annotations,
)  # https://docs.python.org/3/library/typing.html#constant
from typing import Self
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray, Array, Float
from jinns.data._utils import _reset_or_increment
from jinns.data._AbstractDataGenerator import AbstractDataGenerator
from jinns.utils._DictToModuleMeta import DictToModuleMeta


class DGParams(metaclass=DictToModuleMeta):
    """
    However, static type checkers cannot know that DGParams inherit from
    eqx.Module and explicit casting to the latter class will be needed
    """

    pass


class DataGeneratorParameter(AbstractDataGenerator):
    r"""
    A data generator for additional unidimensional equation parameter(s).
    Mostly useful for metamodeling where batch of `params.eq_params` are fed
    to the network.

    Parameters
    ----------
    key : PRNGKeyArray
        Jax random key to sample new time points and to shuffle batches.
    n : int
        The number of total points that will be divided in
        batches. Batches are made so that each data point is seen only
        once during 1 epoch.
    param_batch_size : int | None, default=None
        The size of the batch of randomly selected points among
        the `n` points.  **Important**: no check is performed but
        `param_batch_size` must be the same as other collocation points
        batch_size (time, space or timexspace depending on the context). This is because we vmap the network on all its axes at once to compute the MSE. Also, `param_batch_size` will be the same for all parameters. If None, no mini-batches are used.
    param_ranges : dict[str, tuple[Float, Float] | None, default={}
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
    user_data : dict[str, Float[Array, " n"]] | None, default={}
        A dictionary containing user-provided data for parameters.
        The keys corresponds to the parameter name,
        and must match the keys in `params["eq_params"]`. Only
        unidimensional `jnp.array` are supported. Therefore, the array at
        `user_data[k]` must have shape `(n, 1)` or `(n,)`.
        Note that if the same key appears in `param_ranges` and `user_data`
        priority goes for the content in `user_data`.
        Defaults to None.
    """

    key: PRNGKeyArray
    n: int = eqx.field(static=True)
    param_batch_size: int | None = eqx.field(static=True)
    param_ranges: dict[str, tuple[Float, Float]] = eqx.field(static=True)
    method: str = eqx.field(static=True)
    user_data: dict[str, Float[Array, " n"]]

    # --- Below fields are not passed as arguments to __init__
    _all_params_keys: set[str] = eqx.field(init=False, static=True)
    curr_param_idx: eqx.Module | None = eqx.field(init=False)
    param_n_samples: eqx.Module = eqx.field(init=False)

    def __init__(
        self,
        *,
        key: PRNGKeyArray,
        n: int,
        param_batch_size: int | None,
        param_ranges: dict[str, tuple[Float, Float]] = {},
        method: str = "uniform",
        user_data: dict[str, Float[Array, " n"]] = {},
    ):
        self.key = key
        self.n = n
        self.param_batch_size = param_batch_size
        self.param_ranges = param_ranges
        self.method = method
        self.user_data = user_data

        _all_keys = set().union(self.param_ranges, self.user_data)
        self._all_params_keys = _all_keys

        if self.param_batch_size is not None and self.n < self.param_batch_size:
            raise ValueError(
                f"Number of data points ({self.n}) is smaller than the"
                f"number of batch points ({self.param_batch_size})."
            )

        # NOTE from jinns > v1.5.1 we work with eqx.Module
        # because eq_params is not a dict anymore.
        # We have to use a different class from the publicly exposed EqParams
        # because fields(EqParams) are not necessarily all present in the
        # datagenerator, which would cause eqx.Module to error.

        # 1) Call self.generate_data() to generate a dictionnary that merges the scattered data between `user_data` and `param_ranges`
        self.key, _param_n_samples = self.generate_data(self.key)

        # 2) Use the dictionnary to populate the field of the eqx.Module.
        self.param_n_samples = DGParams(_param_n_samples, "DGParams")

        if self.param_batch_size is None:
            self.curr_param_idx = None
        else:
            curr_idx = self.n + self.param_batch_size
            param_keys_and_curr_idx = {k: curr_idx for k in self._all_params_keys}

            self.curr_param_idx = DGParams(param_keys_and_curr_idx)

    def generate_data(
        self, key: PRNGKeyArray
    ) -> tuple[PRNGKeyArray, dict[str, Float[Array, " n 1"]]]:
        """
        Generate parameter samples, either through generation
        or using user-provided data.
        """
        param_n_samples = {}

        # Some of the subkeys might not be used cause of user-provided data.
        # This is not a big deal and simpler like that.
        key, *subkeys = jax.random.split(key, len(self._all_params_keys) + 1)
        for i, k in enumerate(self._all_params_keys):
            if self.user_data and k in self.user_data.keys():
                try:
                    param_n_samples[k] = self.user_data[k].reshape((self.n, 1))
                except TypeError:
                    shape = self.user_data[k].shape
                    raise TypeError(
                        "Wrong shape for user provided parameters"
                        f" in user_data dictionary at key='{k}' got {shape} "
                        f"and expected {(self.n, 1)}."
                    )
            else:
                if self.method == "grid":
                    xmin, xmax = self.param_ranges[k][0], self.param_ranges[k][1]
                    partial = (xmax - xmin) / self.n
                    # shape (n, 1)
                    param_n_samples[k] = jnp.arange(xmin, xmax, partial)[:, None]
                elif self.method == "uniform":
                    xmin, xmax = self.param_ranges[k][0], self.param_ranges[k][1]
                    param_n_samples[k] = jax.random.uniform(
                        subkeys[i], shape=(self.n, 1), minval=xmin, maxval=xmax
                    )
                else:
                    raise ValueError("Method " + self.method + " is not implemented.")

        return key, param_n_samples

    def param_batch(self) -> tuple[Self, eqx.Module]:
        """
        Return an `eqx.Module` with batches of parameters at its leafs.
        If all the batches have been seen, we reshuffle them (or rather
        their indices), otherwise we just return the next unseen batch.
        """

        if self.param_batch_size is None or self.param_batch_size == self.n:
            # Full batch mode: nothing to do.
            return self, self.param_n_samples
        else:

            def _reset_or_increment_wrapper(
                param_k: Array, idx_k: int, key_k: PRNGKeyArray
            ):
                everything_but_key = _reset_or_increment(
                    idx_k + self.param_batch_size,  # type: ignore
                    self.n,
                    (key_k, param_k, idx_k, self.param_batch_size, None),  # type: ignore
                )[1:]
                return everything_but_key

            new_key, *subkeys = jax.random.split(
                self.key, len(self._all_params_keys) + 1
            )

            # From PRNGKeyArray to a pytree of keys with adequate structure
            subkeys = jax.tree.unflatten(
                jax.tree.structure(self.param_n_samples), subkeys
            )

            res = jax.tree.map(
                _reset_or_increment_wrapper,
                self.param_n_samples,
                self.curr_param_idx,
                subkeys,
            )
            # we must transpose the pytrees because both params and curr_idx # are merged in res
            # https://jax.readthedocs.io/en/latest/jax-101/05.1-pytrees.html#transposing-trees

            new_attributes = jax.tree.transpose(
                jax.tree.structure(self.param_n_samples),
                jax.tree.structure([0, 0]),
                res,
            )

            new = eqx.tree_at(
                lambda m: (m.param_n_samples, m.curr_param_idx),
                self,
                new_attributes,
            )

            new = eqx.tree_at(lambda m: m.key, new, new_key)

            return new, jax.tree_util.tree_map(
                lambda p, q: jax.lax.dynamic_slice(
                    p, start_indices=(q, 0), slice_sizes=(new.param_batch_size, 1)
                ),
                new.param_n_samples,
                new.curr_param_idx,
            )

    def get_batch(self) -> tuple[Self, eqx.Module]:
        """
        Generic method to return a batch
        """
        return self.param_batch()


if __name__ == "__main__":
    key = jax.random.PRNGKey(2)
    key, subkey = jax.random.split(key)

    n = 64
    param_batch_size = 32
    method = "uniform"
    param_ranges = {"theta": (10.0, 11.0)}
    user_data = {"nu": jnp.ones((n, 1))}

    x = DataGeneratorParameter(
        key=subkey,
        n=n,
        param_batch_size=param_batch_size,
        param_ranges=param_ranges,
        method=method,
        user_data=user_data,
    )
    print(x.key)
    x, batch = x.get_batch()
    print(x.key)
