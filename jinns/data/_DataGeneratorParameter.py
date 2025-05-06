"""
Define the DataGenerators modules
"""

from __future__ import (
    annotations,
)  # https://docs.python.org/3/library/typing.html#constant
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Key, Array, Float
from jinns.data._utils import _reset_or_increment
from jinns.data._AbstractDataGenerator import AbstractDataGenerator


class DataGeneratorParameter(AbstractDataGenerator):
    r"""
    A data generator for additional unidimensional equation parameter(s).
    Mostly useful for metamodeling where batch of `params.eq_params` are fed
    to the network.

    Parameters
    ----------
    keys : Key | dict[str, Key]
        Jax random key to sample new time points and to shuffle batches
        or a dict of Jax random keys with key entries from param_ranges
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

    keys: Key | dict[str, Key]
    n: int = eqx.field(static=True)
    param_batch_size: int | None = eqx.field(static=True, default=None)
    param_ranges: dict[str, tuple[Float, Float]] = eqx.field(
        static=True, default_factory=lambda: {}
    )
    method: str = eqx.field(static=True, default="uniform")
    user_data: dict[str, Float[Array, " n"]] | None = eqx.field(
        default_factory=lambda: {}
    )

    curr_param_idx: dict[str, int] = eqx.field(init=False)
    param_n_samples: dict[str, Array] = eqx.field(init=False)

    def __post_init__(self):
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
        self.keys, self.param_n_samples = self.generate_data(self.keys)

    def generate_data(
        self, keys: dict[str, Key]
    ) -> tuple[dict[str, Key], dict[str, Float[Array, " n"]]]:
        """
        Generate parameter samples, either through generation
        or using user-provided data.
        """
        param_n_samples = {}

        all_keys = set().union(
            self.param_ranges,
            self.user_data,  # type: ignore this has been handled in post_init
        )
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

    def get_batch(self):
        """
        Generic method to return a batch
        """
        return self.param_batch()
