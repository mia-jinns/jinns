"""
Implements utility function to create HYPERPINNs
https://arxiv.org/pdf/2111.01008.pdf
"""

import warnings
from dataclasses import InitVar
from typing import Callable, Literal
import copy
from math import prod
import jax
import jax.numpy as jnp
from jax.tree_util import tree_leaves, tree_map
from jaxtyping import Array, Float, PyTree, Int, Key
import equinox as eqx
import numpy as onp

from jinns.utils._pinn import PINN, _MLP
from jinns.parameters._params import Params


def _get_param_nb(
    params: Params,
) -> tuple[Int[onp.ndarray, "1"], Int[onp.ndarray, "n_layers"]]:
    """Returns the number of parameters in a Params object and also
    the cumulative sum when parsing the object.


    Parameters
    ----------
    params :
        A Params object.
    """
    dim_prod_all_arrays = [
        prod(a.shape)
        for a in tree_leaves(params, is_leaf=lambda x: isinstance(x, jnp.ndarray))
    ]
    return onp.asarray(sum(dim_prod_all_arrays)), onp.cumsum(dim_prod_all_arrays)


class HYPERPINN(PINN):
    """
    A HYPERPINN object compatible with the rest of jinns.
    Composed of a PINN and an HYPER network. The HYPERPINN is typically
    instanciated using with `create_HYPERPINN`. However, a user could directly
    creates their HYPERPINN using this
    class by passing an eqx.Module for argument `mlp` (resp. for argument
    `hyper_mlp`) that plays the role of the NN (resp. hyper NN) and that is
    already instanciated.

    Parameters
    ----------
    hyperparams: list = eqx.field(static=True)
        A list of keys from Params.eq_params that will be considered as
        hyperparameters for metamodeling.
    hypernet_input_size: int
        An integer. The input size of the MLP used for the hypernetwork. Must
        be equal to the flattened concatenations for the array of parameters
        designated by the `hyperparams` argument.
    slice_solution : slice
        A jnp.s\_ object which indicates which axis of the PINN output is
        dedicated to the actual equation solution. Default None
        means that slice_solution = the whole PINN output. This argument is useful
        when the PINN is also used to output equation parameters for example
        Note that it must be a slice and not an integer (a preprocessing of the
        user provided argument takes care of it).
    eq_type : str
        A string with three possibilities.
        "ODE": the HYPERPINN is called with one input `t`.
        "statio_PDE": the HYPERPINN is called with one input `x`, `x`
        can be high dimensional.
        "nonstatio_PDE": the HYPERPINN is called with two inputs `t` and `x`, `x`
        can be high dimensional.
        **Note**: the input dimension as given in eqx_list has to match the sum
        of the dimension of `t` + the dimension of `x` or the output dimension
        after the `input_transform` function
    input_transform : Callable[[Float[Array, "input_dim"], Params], Float[Array, "output_dim"]]
        A function that will be called before entering the PINN. Its output(s)
        must match the PINN inputs (except for the parameters).
        Its inputs are the PINN inputs (`t` and/or `x` concatenated together)
        and the parameters. Default is no operation.
    output_transform : Callable[[Float[Array, "input_dim"], Float[Array, "output_dim"], Params], Float[Array, "output_dim"]]
        A function with arguments begin the same input as the PINN, the PINN
        output and the parameter. This function will be called after exiting the PINN.
        Default is no operation.
    output_slice : slice, default=None
        A jnp.s\_[] to determine the different dimension for the HYPERPINN.
        See `shared_pinn_outputs` argument of `create_HYPERPINN`.
    mlp : eqx.Module
        The actual neural network instanciated as an eqx.Module.
    hyper_mlp : eqx.Module
        The actual hyper neural network instanciated as an eqx.Module.
    """

    hyperparams: list[str] = eqx.field(static=True, kw_only=True)
    hypernet_input_size: int = eqx.field(kw_only=True)

    hyper_mlp: InitVar[eqx.Module] = eqx.field(kw_only=True)
    mlp: InitVar[eqx.Module] = eqx.field(kw_only=True)

    params_hyper: PyTree = eqx.field(init=False)
    static_hyper: PyTree = eqx.field(init=False, static=True)
    pinn_params_sum: Int[onp.ndarray, "1"] = eqx.field(init=False, static=True)
    pinn_params_cumsum: Int[onp.ndarray, "n_layers"] = eqx.field(
        init=False, static=True
    )

    def __post_init__(self, mlp, hyper_mlp):
        super().__post_init__(
            mlp,
        )
        self.params_hyper, self.static_hyper = eqx.partition(
            hyper_mlp, eqx.is_inexact_array
        )
        self.pinn_params_sum, self.pinn_params_cumsum = _get_param_nb(self.params)

    def init_params(self) -> Params:
        """
        Returns an initial set of parameters
        """
        return self.params_hyper

    def _hyper_to_pinn(self, hyper_output: Float[Array, "output_dim"]) -> PyTree:
        """
        From the output of the hypernetwork we set the well formed
        parameters of the pinn (`self.params`)
        """
        pinn_params_flat = eqx.tree_at(
            lambda p: tree_leaves(p, is_leaf=eqx.is_array),
            self.params,
            jnp.split(hyper_output, self.pinn_params_cumsum[:-1]),
        )

        return tree_map(
            lambda a, b: a.reshape(b.shape),
            pinn_params_flat,
            self.params,
            is_leaf=lambda x: isinstance(x, jnp.ndarray),
        )

    def eval_nn(
        self,
        inputs: Float[Array, "input_dim"],
        params: Params | PyTree,
    ) -> Float[Array, "output_dim"]:
        """
        Evaluate the HYPERPINN on some inputs with some params.
        """
        try:
            hyper = eqx.combine(params.nn_params, self.static_hyper)
        except (KeyError, AttributeError, TypeError) as e:  # give more flexibility
            hyper = eqx.combine(params, self.static_hyper)

        eq_params_batch = jnp.concatenate(
            [params.eq_params[k].flatten() for k in self.hyperparams], axis=0
        )

        hyper_output = hyper(eq_params_batch)

        pinn_params = self._hyper_to_pinn(hyper_output)

        pinn = eqx.combine(pinn_params, self.static)
        res = self.output_transform(
            inputs, pinn(self.input_transform(inputs, params)).squeeze(), params
        )

        if self.output_slice is not None:
            res = res[self.output_slice]

        ## force (1,) output for non vectorial solution (consistency)
        if not res.shape:
            return jnp.expand_dims(res, axis=-1)
        return res


def create_HYPERPINN(
    key: Key,
    eqx_list: tuple[tuple[Callable, int, int] | Callable, ...],
    eq_type: Literal["ODE", "statio_PDE", "nonstatio_PDE"],
    hyperparams: list[str],
    hypernet_input_size: int,
    dim_x: int = 0,
    input_transform: Callable[
        [Float[Array, "input_dim"], Params], Float[Array, "output_dim"]
    ] = None,
    output_transform: Callable[
        [Float[Array, "input_dim"], Float[Array, "output_dim"], Params],
        Float[Array, "output_dim"],
    ] = None,
    slice_solution: slice = None,
    shared_pinn_outputs: slice = None,
    eqx_list_hyper: tuple[tuple[Callable, int, int] | Callable, ...] = None,
) -> HYPERPINN | list[HYPERPINN]:
    r"""
    Utility function to create a standard PINN neural network with the equinox
    library.

    Parameters
    ----------
    key
        A JAX random key that will be used to initialize the network
        parameters.
    eqx_list
        A tuple of tuples of successive equinox modules and activation functions to
        describe the PINN architecture. The inner tuples must have the eqx module or
        activation function as first item, other items represent arguments
        that could be required (eg. the size of the layer).
        The `key` argument need not be given.
        Thus typical example is `eqx_list=
        ((eqx.nn.Linear, 2, 20),
            jax.nn.tanh,
            (eqx.nn.Linear, 20, 20),
            jax.nn.tanh,
            (eqx.nn.Linear, 20, 20),
            jax.nn.tanh,
            (eqx.nn.Linear, 20, 1)
        )`.
    eq_type
        A string with three possibilities.
        "ODE": the HYPERPINN is called with one input `t`.
        "statio_PDE": the HYPERPINN is called with one input `x`, `x`
        can be high dimensional.
        "nonstatio_PDE": the HYPERPINN is called with two inputs `t` and `x`, `x`
        can be high dimensional.
        **Note**: the input dimension as given in eqx_list has to match the sum
        of the dimension of `t` + the dimension of `x` or the output dimension
        after the `input_transform` function
    hyperparams
        A list of keys from Params.eq_params that will be considered as
        hyperparameters for metamodeling.
    hypernet_input_size
        An integer. The input size of the MLP used for the hypernetwork. Must
        be equal to the flattened concatenations for the array of parameters
        designated by the `hyperparams` argument.
    dim_x
        An integer. The dimension of `x`. Default `0`.
    input_transform
        A function that will be called before entering the PINN. Its output(s)
        must match the PINN inputs (except for the parameters).
        Its inputs are the PINN inputs (`t` and/or `x` concatenated together)
        and the parameters. Default is no operation.
    output_transform
        A function with arguments begin the same input as the PINN, the PINN
        output and the parameter. This function will be called after exiting the PINN.
        Default is no operation.
    slice_solution
        A jnp.s\_ object which indicates which axis of the PINN output is
        dedicated to the actual equation solution. Default None
        means that slice_solution = the whole PINN output. This argument is useful
        when the PINN is also used to output equation parameters for example
        Note that it must be a slice and not an integer (a preprocessing of the
        user provided argument takes care of it).
    shared_pinn_outputs
        Default is None, for a stantard PINN.
        A tuple of jnp.s\_[] (slices) to determine the different output for each
        network. In this case we return a list of PINNs, one for each output in
        shared_pinn_outputs. This is useful to create PINNs that share the
        same network and same parameters; **the user must then use the same
        parameter set in their manipulation**.
        See the notebook 2D Navier Stokes in pipeflow with metamodel for an
        example using this option.
    eqx_list_hyper
        Same as eqx_list but for the hypernetwork. Default is None, i.e., we
        use the same architecture as the PINN, up to the number of inputs and
        ouputs. Note that the number of inputs must be of the hypernetwork must
        be equal to the flattened concatenations for the array of parameters
        designated by the `hyperparams` argument;
        and the number of outputs must be equal to the number
        of parameters in the pinn network

    Returns
    -------
    hyperpinn
        A HYPERPINN instance or, when `shared_pinn_ouput` is not None,
        a list of HYPERPINN instances with the same structure is returned,
        only differing by there final slicing of the network output.


    Raises
    ------
    RuntimeError
        If the parameter value for eq_type is not in `["ODE", "statio_PDE",
        "nonstatio_PDE"]`
    RuntimeError
        If we have a `dim_x > 0` and `eq_type == "ODE"`
        or if we have a `dim_x = 0` and `eq_type != "ODE"`
    """
    if eq_type not in ["ODE", "statio_PDE", "nonstatio_PDE"]:
        raise RuntimeError("Wrong parameter value for eq_type")

    if eq_type == "ODE" and dim_x != 0:
        raise RuntimeError("Wrong parameter combination eq_type and dim_x")

    if eq_type != "ODE" and dim_x == 0:
        raise RuntimeError("Wrong parameter combination eq_type and dim_x")

    if eqx_list_hyper is None:
        eqx_list_hyper = copy.deepcopy(eqx_list)

    try:
        nb_outputs_declared = eqx_list[-1][2]  # normally we look for 3rd ele of
        # last layer
    except IndexError:
        nb_outputs_declared = eqx_list[-2][2]

    if slice_solution is None:
        slice_solution = jnp.s_[0:nb_outputs_declared]
    if isinstance(slice_solution, int):
        # rewrite it as a slice to ensure that axis does not disappear when
        # indexing
        slice_solution = jnp.s_[slice_solution : slice_solution + 1]

    if input_transform is None:

        def input_transform(_in, _params):
            return _in

    if output_transform is None:

        def output_transform(_in_pinn, _out_pinn, _params):
            return _out_pinn

    key, subkey = jax.random.split(key, 2)
    mlp = _MLP(key=subkey, eqx_list=eqx_list)
    # quick partitioning to get the params to get the correct number of neurons
    # for the last layer of hyper network
    params_mlp, _ = eqx.partition(mlp, eqx.is_inexact_array)
    pinn_params_sum, _ = _get_param_nb(params_mlp)
    # the number of parameters for the pinn will be the number of ouputs
    # for the hyper network
    if len(eqx_list_hyper[-1]) > 1:
        eqx_list_hyper = eqx_list_hyper[:-1] + (
            (eqx_list_hyper[-1][:2] + (pinn_params_sum,)),
        )
    else:
        eqx_list_hyper = (
            eqx_list_hyper[:-2]
            + ((eqx_list_hyper[-2][:2] + (pinn_params_sum,)),)
            + eqx_list_hyper[-1]
        )
    if len(eqx_list_hyper[0]) > 1:
        eqx_list_hyper = (
            (
                (eqx_list_hyper[0][0],)
                + (hypernet_input_size,)
                + (eqx_list_hyper[0][2],)
            ),
        ) + eqx_list_hyper[1:]
    else:
        eqx_list_hyper = (
            eqx_list_hyper[0]
            + (
                (
                    (eqx_list_hyper[1][0],)
                    + (hypernet_input_size,)
                    + (eqx_list_hyper[1][2],)
                ),
            )
            + eqx_list_hyper[2:]
        )
    key, subkey = jax.random.split(key, 2)

    with warnings.catch_warnings():
        # TODO check why this warning is raised here and not in the PINN
        # context ?
        warnings.filterwarnings("ignore", message="A JAX array is being set as static!")
        hyper_mlp = _MLP(key=subkey, eqx_list=eqx_list_hyper)

    if shared_pinn_outputs is not None:
        hyperpinns = []
        for output_slice in shared_pinn_outputs:
            with warnings.catch_warnings():
                # Catch the equinox warning because we put the number of
                # parameters as static while being jnp.Array. This this time
                # this is correct to do so, because they are used as indices
                # and will never be modified
                warnings.filterwarnings(
                    "ignore", message="A JAX array is being set as static!"
                )
                hyperpinn = HYPERPINN(
                    mlp=mlp,
                    hyper_mlp=hyper_mlp,
                    slice_solution=slice_solution,
                    eq_type=eq_type,
                    input_transform=input_transform,
                    output_transform=output_transform,
                    hyperparams=hyperparams,
                    hypernet_input_size=hypernet_input_size,
                    output_slice=output_slice,
                )
            hyperpinns.append(hyperpinn)
        return hyperpinns
    with warnings.catch_warnings():
        # Catch the equinox warning because we put the number of
        # parameters as static while being jnp.Array. This this time
        # this is correct to do so, because they are used as indices
        # and will never be modified
        warnings.filterwarnings("ignore", message="A JAX array is being set as static!")
        hyperpinn = HYPERPINN(
            mlp=mlp,
            hyper_mlp=hyper_mlp,
            slice_solution=slice_solution,
            eq_type=eq_type,
            input_transform=input_transform,
            output_transform=output_transform,
            hyperparams=hyperparams,
            hypernet_input_size=hypernet_input_size,
            output_slice=None,
        )
    return hyperpinn
