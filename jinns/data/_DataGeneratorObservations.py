"""
Define the DataGenerators modules
"""

from __future__ import (
    annotations,
)  # https://docs.python.org/3/library/typing.html#constant
from functools import partial
from itertools import zip_longest
import equinox as eqx
import jax
import jax.numpy as jnp
from typing import TYPE_CHECKING, Self
from jaxtyping import PRNGKeyArray, Int, Array, Float
from jinns.data._Batchs import ObsBatchDict
from jinns.data._utils import _reset_or_increment
from jinns.data._AbstractDataGenerator import AbstractDataGenerator
from jinns.utils._DictToModuleMeta import DictToModuleMeta

if TYPE_CHECKING:
    # imports only used in type hints
    InputEqParams = (
        dict[str, Float[Array, "  n_obs"]] | dict[str, Float[Array, " n_obs 1"]]
    ) | None

    # Note that the lambda functions used below are with type: ignore just
    # because the lambda are not type annotated, but there is no proper way
    # to do this and we should assign the lambda to a type hinted variable
    # before hand: this is not practical, let us not get mad at this


def _merge_dict_arguments(fun, fixed_args):
    """
    a decorator function that transforms a tuple of 1-key-dict argument
    in a function call with a big merged unpacked dict. This is used for a
    dynamic construction of a tree map call, where an arbitrary number of arguments
    are fixed before the tree map call. The key to enable this, is that the
    function that needs to be called is kw only, but jax tree map does not
    support keyword only, so we pass through this decorator
    """

    def wrapper(tuple_of_dict):
        d = {}
        # the for loop below is needed because there is no unpack operator
        # authorized inside a comprehension for now: https://stackoverflow.com/a/37584733
        for d_ in tuple_of_dict:
            if len(d_.keys()) != 1:
                raise ValueError("Problem here, we expect 1-key-dict")
            if list(d_.keys())[0] not in fixed_args:
                d.update(d_)
        return fun(**d)

    return wrapper


class DGObservedParams(metaclass=DictToModuleMeta):
    """
    However, static type checkers cannot know that DGObservedParams inherit from
    eqx.Module and explicit casting to the latter class will be needed
    """

    pass


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
    observed_pinn_in : Float[Array, " n_obs nb_pinn_in"] | tuple[Float[Array, " n_obs nb_pinn_in"], ...]
        Observed values corresponding to the input of the PINN
        (eg. the time at which we recorded the observations). The first
        dimension must corresponds to the number of observed_values.
        The second dimension depends on the input dimension of the PINN,
        that is `1` for ODE, `n_dim_x` for stationnary PDE and `n_dim_x + 1`
        for non-stationnary PDE.
        Can be a tuple of such arguments to support multidatasets, see below.
    observed_values : Float[Array, " n_obs, nb_pinn_out"] | tuple[Float[Array, " n_obs, nb_pinn_out"], ...]
        Observed values that the PINN should learn to fit. The first
        dimension must be aligned with observed_pinn_in.
        Can be a tuple of such arguments to support multidatasets, see below.
    observed_eq_params : dict[str, Float[Array, " n_obs 1"]] | tuple[dict[str, Float[Array, " n_obs 1"], ...]], default=None
        A dict with keys corresponding to
        the parameter name. The keys must match the keys in
        `params["eq_params"]`, ie., if only some parameters are observed, other
        keys **must still appear with None as value**. The values are jnp.array with 2 dimensions
        with values corresponding to the parameter value for which we also
        have observed_pinn_in and observed_values. Hence the first
        dimension must be aligned with observed_pinn_in and observed_values.
        Can be a tuple of such arguments to support multidatasets, see below.
        Optional argument.
    sharding_device : jax.sharding.Sharding, default=None
        Default None. An optional sharding object to constraint the storage
        of observed inputs, values and parameters. Typically, a
        SingleDeviceSharding(cpu_device) to avoid loading on GPU huge
        datasets of observations. Note that computations for **batches**
        can still be performed on other devices (*e.g.* GPU, TPU or
        any pre-defined Sharding) thanks to the `obs_batch_sharding`
        arguments of `jinns.solve()`. Read `jinns.solve()` doc for more info.

    ** New in jinns vX.X.X:** We provide the possibility of specifying several
        datasets of observations, this can serve a variety of purposes, as for
        example, provided different observations for different channels of the
        solution (by using the `obs_slice` attribute of Loss objects, see the
        notebook on this topic in the documentation). To
        provide several datasets, it suffices to pass `observed_values` or
        `observed_pinn_in` or `observed_eq_params` as tuples. If you have
        several `observed_values` but the same `observed_eq_params` and or the
        same `observed_pinn_in`, the two latter should not be duplicated in
        tuples of the same length of `observed_values`. The last sentence
        remains true when interchanging the terms `observed_values`,
        `observed_pinn_in` and `observed_eq_params` in any position.
        This is not a syntaxic sugar.
        This is a real necessity to avoid duplicating data, to gain speed. This
        internally handled with dynamic freezing
        of non vectorized arguments (see code...)
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
        """ """
        super().__init__()
        self.key = key

        if not isinstance(observed_values, tuple):
            observed_values = (observed_values,)
        if not isinstance(observed_pinn_in, tuple):
            observed_pinn_in = (observed_pinn_in,)
        if observed_eq_params is not None:
            if not isinstance(observed_eq_params, tuple):
                observed_eq_params = (observed_eq_params,)
        else:
            observed_eq_params = (None,)

        # now if values, pinn_in, and eq_params does not have same length (as
        # tuples), we must find the longest one and the other either must be
        # length 1 or must be the same length as the longest
        len_longest_tuple = max(
            map(len, (observed_values, observed_pinn_in, observed_eq_params))
        )
        longest_tuple = max(
            (observed_values, observed_pinn_in, observed_eq_params), key=len
        )
        if len(observed_values) != 1 and len(observed_values) != len_longest_tuple:
            raise ValueError(
                "If observed_values is a tuple, it should"
                " be of length 1 (one array, the same for"
                " all the pinn_in and eq_params entries) or be of the same"
                " length as the longest tuple of entries (1 to 1 matching)"
            )
        if len(observed_pinn_in) != 1 and len(observed_pinn_in) != len_longest_tuple:
            raise ValueError(
                "If observed_pinn_in is a tuple, it should"
                " be of length 1 (one array, the same for"
                " all the values and eq_params entries) or be of the same"
                " length as the longest tuple of entries (1 to 1 matching)"
            )
        if (
            len(observed_eq_params) != 1
            and len(observed_eq_params) != len_longest_tuple
        ):
            raise ValueError(
                "If observed_eq_params is a tuple, it should"
                " be of length 1 (one array, the same for"
                " all the values and pinn_in entries) or be of the same"
                " length as the longest tuple of entries (1 to 1 matching)"
            )

        ### Start check first axis

        def check_first_axis(*, values, pinn_in_array):
            if values.shape[0] != pinn_in_array.shape[0]:
                raise ValueError(
                    "Each matching elements of self.observed_pinn_in and self.observed_values must have same first axis"
                )
            return values

        tree_map_args = tuple(
            ({"values": v}, {"pinn_in_array": p})
            for v, p in zip_longest(observed_values, observed_pinn_in)
        )
        fixed_args = ()
        if len(observed_values) != len(observed_pinn_in):
            if len(observed_pinn_in) == 1:
                check_first_axis = partial(
                    check_first_axis, pinn_in_array=observed_pinn_in[0]
                )
                fixed_args = fixed_args + ("pinn_in_array",)
            if len(observed_values) == 1:
                check_first_axis = partial(check_first_axis, values=observed_values[0])
                fixed_args = fixed_args + ("values",)
        # ... and then we do the tree map. Note that in the tree.map below,
        # self.observed_eq_params can have None leaves
        # tree_map_args is a tuple of tuple dicts: 1) outer tuples are those we
        # will vectorize over 2) inside tuples to be able to unpack
        # dynamically (i.e. varying nb of elements to pass to fun)
        # 3) then the dicts are merged to feed the kw only function
        # tree.map cannot directly feed a kw only
        # function such as check_first_axis (so we pass through
        # the decorator)
        jax.tree.map(
            _merge_dict_arguments(check_first_axis, fixed_args),
            tree_map_args,
            is_leaf=lambda x: (isinstance(x, tuple) and isinstance(x[0], dict)),
        )

        ### End check first axis

        self.observed_pinn_in = observed_pinn_in
        self.observed_values = observed_values
        if observed_eq_params == (None,):
            self.observed_eq_params = observed_eq_params  # pyright: ignore
            # (this is resolved later on one instanciating DGObservedParams)
        else:
            self.observed_eq_params = jax.tree.map(
                lambda d: {
                    k: v[:, None] if len(v.shape) == 1 else v for k, v in d.items()
                },
                observed_eq_params,
                is_leaf=lambda x: isinstance(x, dict),
            )

        self.observed_pinn_in = jax.tree.map(
            lambda x: x[:, None] if len(x.shape) == 1 else x, self.observed_pinn_in
        )

        self.observed_values = jax.tree.map(
            lambda x: x[:, None] if len(x.shape) == 1 else x, self.observed_values
        )

        ### Start check first axis 2
        def check_first_axis2(*, eq_params_dict, pinn_in_array):
            if eq_params_dict is not None:
                for _, v in eq_params_dict.items():
                    if v.shape[0] != pinn_in_array.shape[0]:
                        raise ValueError(
                            "Each matching elements of self.observed_pinn_in and self.observed_eq_params must have the same first axis"
                        )

        # the following tree_map_args will work if all lengths are equal either
        # 1 or more
        tree_map_args = tuple(
            ({"eq_params_dict": e}, {"pinn_in_array": p})
            for e, p in zip_longest(self.observed_eq_params, self.observed_pinn_in)
        )
        fixed_args = ()
        if len(self.observed_eq_params) != len(self.observed_pinn_in):
            if len(self.observed_pinn_in) == 1:
                check_first_axis2 = partial(
                    check_first_axis2, pinn_in_array=self.observed_pinn_in[0]
                )
                fixed_args = fixed_args + ("pinn_in_array",)

            if len(self.observed_eq_params) == 1:
                check_first_axis2 = partial(
                    check_first_axis2, eq_params_dict=self.observed_eq_params[0]
                )
                fixed_args = fixed_args + ("eq_params_dict",)
        jax.tree.map(
            _merge_dict_arguments(
                check_first_axis2, fixed_args
            ),  # https://stackoverflow.com/a/42421497
            tree_map_args,
            is_leaf=lambda x: (isinstance(x, tuple) and isinstance(x[0], dict)),
        )

        ### End check first axis 2

        ### Start check ndim

        def check_ndim(*, values, pinn_in_array, eq_params_dict):
            if values.ndim > 2:
                raise ValueError(
                    "Each element of self.observed_pinn_in must have 2 dimensions"
                )
            if pinn_in_array.ndim > 2:
                raise ValueError(
                    "Each element of self.observed_values must have 2 dimensions"
                )
            if eq_params_dict is not None:
                for _, v in eq_params_dict.items():
                    if v.ndim > 2:
                        raise ValueError(
                            "Each value of observed_eq_params must have 2 dimensions"
                        )

        # the following tree_map_args will work if all lengths are equal either
        # 1 or more
        tree_map_args = tuple(
            ({"eq_params_dict": e}, {"pinn_in_array": p}, {"values": v})
            for e, p, v in zip_longest(
                self.observed_eq_params, self.observed_pinn_in, self.observed_values
            )
        )
        # now, if some shape are different, it can only be because there are 1
        # while we expect a fixed n (thanks to the early tests above)
        # then we must fix the arguments that are single leaf pytree
        # and keep track of the arguments that are fixed to be able to remove
        # them in the wrapper
        fixed_args = ()
        if len(self.observed_eq_params) != len(self.observed_pinn_in) or len(
            self.observed_eq_params
        ) != len(self.observed_values):
            if len(self.observed_pinn_in) == 1:
                check_ndim = partial(check_ndim, pinn_in_array=self.observed_pinn_in[0])
                fixed_args = fixed_args + ("pinn_in_array",)
            if len(self.observed_eq_params) == 1:
                check_ndim = partial(
                    check_ndim, eq_params_dict=self.observed_eq_params[0]
                )
                fixed_args = fixed_args + ("eq_params_dict",)
            if len(self.observed_values) == 1:
                check_ndim = partial(check_ndim, values=self.observed_values[0])
                fixed_args = fixed_args + ("values",)

        jax.tree.map(
            _merge_dict_arguments(check_ndim, fixed_args),
            tree_map_args,
            is_leaf=lambda x: (isinstance(x, tuple) and isinstance(x[0], dict)),
        )
        ### End check ndim

        # longest_tuple will be used for correct jax tree map broadcast. Note
        # that even though self.observed_pinn_in and self.observed_values and
        # self.observed_eq_params does
        # not have the same len (as tuples), their components (jnp.arrays) do
        # have the same first axis. This is worked out by all the previous
        # checks
        self.n = jax.tree.map(
            lambda o: o.shape[0],
            tuple(_ for _ in jax.tree.leaves(longest_tuple)),  # jax.tree.leaves
            # because if longest_tuple is eq_params then it is a dict but we do
            # not want self.n to have the dict tree structure
        )

        if isinstance(obs_batch_size, int) or obs_batch_size is None:
            self.obs_batch_size = jax.tree.map(
                lambda _: obs_batch_size,
                tuple(_ for _ in jax.tree.leaves(longest_tuple)),  # jax tree leaves
                # because if longest_tuple is eq_params then it is a dict but we do
                # not want self.n to have the dict tree structure
            )
        elif isinstance(obs_batch_size, tuple):
            if len(obs_batch_size) != len_longest_tuple and len(obs_batch_size) != 1:
                raise ValueError(
                    "If obs_batch_size is a tuple, it must me"
                    " of length 1 or of length equal to the"
                    " maximum length between values, pinn_in and"
                    " eq_params."
                )
            self.obs_batch_size = obs_batch_size
        else:
            raise ValueError("obs_batch_size must be an int, a tuple or None")

        # After all the checks
        # Convert the dict of observed parameters to the internal
        # `DGObservedParams`
        # class used by Jinns.
        self.observed_eq_params = tuple(
            DGObservedParams(o_, "DGObservedParams")
            for o_ in self.observed_eq_params
            if o_ is not None
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
        self.indices = jax.tree.map(jnp.arange, self.n)

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
        *,
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
        # the following tree map over DataGeneratorObservations.obs_batch, must
        # be handled with pre-fixed arguments when, for memory reason,
        # observed_pinn_in or observed_values or observed_eq_params have not
        # does not have the same length. If all tuples are of size 1, this
        # should work totally transparently
        args = (
            self.observed_eq_params,
            self.observed_pinn_in,
            self.observed_values,
            self.n,
            self.obs_batch_size,
            self.curr_idx,
            self.key,
            self.indices,
        )

        tree_map_args = tuple(
            (
                {"observed_eq_params": e},
                {"observed_pinn_in": p},
                {"observed_values": v},
                {"n": n},
                {"obs_batch_size": b},
                {"curr_idx": c},
                {"key": k},
                {"indices": i},
            )
            for e, p, v, n, b, c, k, i in zip_longest(*args)
        )
        fixed_args = ()
        obs_batch_fun = DataGeneratorObservations.obs_batch
        if len(set(map(len, args))) > 1:  # at least 2 lengths differ
            # but since values, pinn_in and equations are the arguments that
            # generates all the others, it suffices to potentially fix the
            # former
            if len(self.observed_pinn_in) == 1:
                obs_batch_fun = partial(
                    obs_batch_fun, observed_pinn_in=self.observed_pinn_in[0]
                )
                fixed_args = fixed_args + ("observed_pinn_in",)
            if len(self.observed_eq_params) == 1:
                obs_batch_fun = partial(
                    obs_batch_fun, observed_eq_params=self.observed_eq_params[0]
                )
                fixed_args = fixed_args + ("observed_eq_params",)
            if len(self.observed_values) == 1:
                obs_batch_fun = partial(
                    obs_batch_fun, observed_values=self.observed_values[0]
                )
                fixed_args = fixed_args + ("observed_values",)

        ret = jax.tree.map(
            _merge_dict_arguments(obs_batch_fun, fixed_args),
            tree_map_args,
            is_leaf=lambda x: (isinstance(x, tuple) and isinstance(x[0], dict)),
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
