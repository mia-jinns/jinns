"""
Implements utility function to create PINNs
"""

from typing import Callable, Literal
from dataclasses import InitVar
import jax
import jax.numpy as jnp
import equinox as eqx

from jaxtyping import Array, Key, PyTree, Float

from jinns.parameters._params import Params


class _MLP(eqx.Module):
    """
    Class to construct an equinox module from a key and a eqx_list. To be used
    in pair with the function `create_PINN`.

    Parameters
    ----------
    key : InitVar[Key]
        A jax random key for the layer initializations.
    eqx_list : InitVar[tuple[tuple[Callable, int, int] | Callable, ...]]
        A tuple of tuples of successive equinox modules and activation functions to
        describe the PINN architecture. The inner tuples must have the eqx module or
        activation function as first item, other items represents arguments
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
    """

    key: InitVar[Key] = eqx.field(kw_only=True)
    eqx_list: InitVar[tuple[tuple[Callable, int, int] | Callable, ...]] = eqx.field(
        kw_only=True
    )

    # NOTE that the following should NOT be declared as static otherwise the
    # eqx.partition that we use in the PINN module will misbehave
    layers: list[eqx.Module] = eqx.field(init=False)

    def __post_init__(self, key, eqx_list):
        self.layers = []
        for l in eqx_list:
            if len(l) == 1:
                self.layers.append(l[0])
            else:
                key, subkey = jax.random.split(key, 2)
                self.layers.append(l[0](*l[1:], key=subkey))

    def __call__(self, t: Float[Array, "input_dim"]) -> Float[Array, "output_dim"]:
        for layer in self.layers:
            t = layer(t)
        return t


class PINN(eqx.Module):
    r"""
    A PINN object, i.e., a neural network compatible with the rest of jinns.
    This is typically created with `create_PINN` which creates iternally a
    `_MLP` object. However, a user could directly creates their PINN using this
    class by passing a eqx.Module (for argument `mlp`)
    that plays the role of the NN and that is
    already instanciated.

    Parameters
    ----------
    slice_solution : slice
        A jnp.s\_ object which indicates which axis of the PINN output is
        dedicated to the actual equation solution. Default None
        means that slice_solution = the whole PINN output. This argument is useful
        when the PINN is also used to output equation parameters for example
        Note that it must be a slice and not an integer (a preprocessing of the
        user provided argument takes care of it).
    eq_type : Literal["ODE", "statio_PDE", "nonstatio_PDE"]
        A string with three possibilities.
        "ODE": the PINN is called with one input `t`.
        "statio_PDE": the PINN is called with one input `x`, `x`
        can be high dimensional.
        "nonstatio_PDE": the PINN is called with two inputs `t` and `x`, `x`
        can be high dimensional.
        **Note**: the input dimension as given in eqx_list has to match the sum
        of the dimension of `t` + the dimension of `x` or the output dimension
        after the `input_transform` function.
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
        A jnp.s\_[] to determine the different dimension for the PINN.
        See `shared_pinn_outputs` argument of `create_PINN`.
    mlp : eqx.Module
        The actual neural network instanciated as an eqx.Module.
    """

    slice_solution: slice = eqx.field(static=True, kw_only=True)
    eq_type: Literal["ODE", "statio_PDE", "nonstatio_PDE"] = eqx.field(
        static=True, kw_only=True
    )
    input_transform: Callable[
        [Float[Array, "input_dim"], Params], Float[Array, "output_dim"]
    ] = eqx.field(static=True, kw_only=True)
    output_transform: Callable[
        [Float[Array, "input_dim"], Float[Array, "output_dim"], Params],
        Float[Array, "output_dim"],
    ] = eqx.field(static=True, kw_only=True)
    output_slice: slice = eqx.field(static=True, kw_only=True, default=None)

    mlp: InitVar[eqx.Module] = eqx.field(kw_only=True)

    params: PyTree = eqx.field(init=False)
    static: PyTree = eqx.field(init=False, static=True)

    def __post_init__(self, mlp):
        self.params, self.static = eqx.partition(mlp, eqx.is_inexact_array)

    def init_params(self) -> PyTree:
        """
        Returns an initial set of parameters
        """
        return self.params

    def __call__(self, *args) -> Float[Array, "output_dim"]:
        """
        Calls `eval_nn` with rearranged arguments
        """
        if self.eq_type == "ODE":
            (t, params) = args
            if len(t.shape) == 0:
                t = t[..., None]  #  Add mandatory dimension which can be lacking
                # (eg. for the ODE batches) but this dimension can already
                # exists (eg. for user provided observation times)
            return self.eval_nn(t, params)
        if self.eq_type == "statio_PDE":
            (x, params) = args
            return self.eval_nn(x, params)
        if self.eq_type == "nonstatio_PDE":
            (t, x, params) = args
            t_x = jnp.concatenate([t, x], axis=-1)
            return self.eval_nn(t_x, params)
        raise ValueError("Wrong value for self.eq_type")

    def eval_nn(
        self,
        inputs: Float[Array, "input_dim"],
        params: Params | PyTree,
    ) -> Float[Array, "output_dim"]:
        """
        Evaluate the PINN on some inputs with some params.
        """
        try:
            model = eqx.combine(params.nn_params, self.static)
        except (KeyError, AttributeError, TypeError) as e:  # give more flexibility
            model = eqx.combine(params, self.static)
        res = self.output_transform(
            inputs, model(self.input_transform(inputs, params)).squeeze(), params
        )

        if self.output_slice is not None:
            res = res[self.output_slice]

        ## force (1,) output for non vectorial solution (consistency)
        if not res.shape:
            return jnp.expand_dims(res, axis=-1)
        return res


def create_PINN(
    key: Key,
    eqx_list: tuple[tuple[Callable, int, int] | Callable, ...],
    eq_type: Literal["ODE", "statio_PDE", "nonstatio_PDE"],
    dim_x: int = 0,
    input_transform: Callable[
        [Float[Array, "input_dim"], Params], Float[Array, "output_dim"]
    ] = None,
    output_transform: Callable[
        [Float[Array, "input_dim"], Float[Array, "output_dim"], Params],
        Float[Array, "output_dim"],
    ] = None,
    shared_pinn_outputs: tuple[slice] = None,
    slice_solution: slice = None,
) -> PINN | list[PINN]:
    r"""
    Utility function to create a standard PINN neural network with the equinox
    library.

    Parameters
    ----------
    key
        A JAX random key that will be used to initialize the network
        parameters.
    eqx_list
        A tuple of tuples of successive equinox modules and activation
        functions to describe the PINN architecture. The inner tuples must have
        the eqx module or activation function as first item, other items
        represent arguments that could be required (eg. the size of the layer).

        The `key` argument do not need to be given.

        A typical example is `eqx_list = (
            (eqx.nn.Linear, input_dim, 20),
            (jax.nn.tanh,),
            (eqx.nn.Linear, 20, 20),
            (jax.nn.tanh,),
            (eqx.nn.Linear, 20, 20),
            (jax.nn.tanh,),
            (eqx.nn.Linear, 20, output_dim)
        )`.
    eq_type
        A string with three possibilities.
        "ODE": the PINN is called with one input `t`.
        "statio_PDE": the PINN is called with one input `x`, `x`
        can be high dimensional.
        "nonstatio_PDE": the PINN is called with two inputs `t` and `x`, `x`
        can be high dimensional.
        **Note**: the input dimension as given in eqx_list has to match the sum
        of the dimension of `t` + the dimension of `x` or the output dimension
        after the `input_transform` function.
    dim_x
        An integer. The dimension of `x`. Default `0`.
    input_transform
        A function that will be called before entering the PINN. Its output(s)
        must match the PINN inputs (except for the parameters).
        Its inputs are the PINN inputs (`t` and/or `x` concatenated together)
        and the parameters. Default is no operation.
    output_transform
        A function with arguments begin the same input as the PINN, the PINN
        output and the parameter. This function will be called after exiting
        the PINN.
        Default is no operation.
    shared_pinn_outputs
        Default is None, for a stantard PINN.
        A tuple of jnp.s\_[] (slices) to determine the different output for each
        network. In this case we return a list of PINNs, one for each output in
        shared_pinn_outputs. This is useful to create PINNs that share the
        same network and same parameters; **the user must then use the same
        parameter set in their manipulation**.
        See the notebook 2D Navier Stokes in pipeflow with metamodel for an
        example using this option.
    slice_solution
        A jnp.s\_ object which indicates which axis of the PINN output is
        dedicated to the actual equation solution. Default None
        means that slice_solution = the whole PINN output. This argument is
        useful when the PINN is also used to output equation parameters for
        example Note that it must be a slice and not an integer (a
        preprocessing of the user provided argument takes care of it).


    Returns
    -------
    pinn
        A PINN instance or, when `shared_pinn_ouput` is not None,
        a list of PINN instances with the same structure is returned,
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

    mlp = _MLP(key=key, eqx_list=eqx_list)

    if shared_pinn_outputs is not None:
        pinns = []
        for output_slice in shared_pinn_outputs:
            pinn = PINN(
                mlp=mlp,
                slice_solution=slice_solution,
                eq_type=eq_type,
                input_transform=input_transform,
                output_transform=output_transform,
                output_slice=output_slice,
            )
            pinns.append(pinn)
        return pinns
    pinn = PINN(
        mlp=mlp,
        slice_solution=slice_solution,
        eq_type=eq_type,
        input_transform=input_transform,
        output_transform=output_transform,
        output_slice=None,
    )
    return pinn
