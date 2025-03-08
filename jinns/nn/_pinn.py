"""
Implement abstract class for PINN architectures
"""

from typing import Literal, Callable, Union, Any
from dataclasses import InitVar
import equinox as eqx
from jaxtyping import Float, Array, PyTree
import jax.numpy as jnp
from jinns.parameters._params import Params, ParamsDict


class PINN(eqx.Module):
    r"""
    Base class for PINN objects. It can be seen as a wrapper on
    an `eqx.Module` which actually implement the NN architectures, with extra
    arguments handling the "physics-informed" aspect.

    !!! Note
        We use the `eqx.partition` and `eqx.combine` strategy of Equinox: a
        `filter_spec` is applied on the PyTree and splits it into two PyTree with
        the same structure: a static one (invisible to JAX transform such as JIT,
        grad, etc.) and dynamic one. By convention, anything not static is
        considered a parameter in Jinns.

    For compatibility with jinns, we require that a `PINN` architecture:

        1) has an eqx.Module (`eqx_network`) InitVar passed to __post_init__
        representing the network architecture.
        2) calls `eqx.partition` in __post_init__ in order to store the
        static part of the model and the initial parameters.
        3) has a `eq_type` argument, used for handling internal operations in
        jinns.
        4) has a `slice_solution` argument. It is a `jnp.s\_` object which
    indicates which axis of the PINN output is dedicated to the actual equation
    solution. Default None means that slice_solution = the whole PINN output.
    For example, this argument is useful when the PINN is also used to output
    equation parameters. Note that it must be a slice and not an integer (a
    preprocessing of the user provided argument takes care of it).

    Parameters
    ----------
    slice_solution : slice
        Default is jnp.s\_[...]. A jnp.s\_ object which indicates which axis of the PINN output is
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
    eqx_network : eqx.Module
        The actual neural network instanciated as an eqx.Module.
    filter_spec : PyTree[Union[bool, Callable[[Any], bool]]]
        Default is `eqx.is_inexact_array`. This tells Jinns what to consider as
        a trainable parameter. Quoting from equinox documentation:
        a PyTree whose structure should be a prefix of the structure of pytree.
        Each of its leaves should either be 1) True, in which case the leaf or
        subtree is kept; 2) False, in which case the leaf or subtree is
        replaced with replace; 3) a callable Leaf -> bool, in which case this is evaluated on the leaf or mapped over the subtree, and the leaf kept or replaced as appropriate.


    Raises
    ------
    RuntimeError
        If the parameter value for eq_type is not in `["ODE", "statio_PDE",
        "nonstatio_PDE"]`
    """

    slice_solution: slice = eqx.field(static=True, kw_only=True, default=None)
    eq_type: Literal["ODE", "statio_PDE", "nonstatio_PDE"] = eqx.field(
        static=True, kw_only=True
    )
    input_transform: Callable[
        [Float[Array, "input_dim"], Params], Float[Array, "output_dim"]
    ] = eqx.field(static=True, kw_only=True, default=None)
    output_transform: Callable[
        [Float[Array, "input_dim"], Float[Array, "output_dim"], Params],
        Float[Array, "output_dim"],
    ] = eqx.field(static=True, kw_only=True, default=None)

    eqx_network: InitVar[eqx.Module] = eqx.field(kw_only=True)
    filter_spec: PyTree[Union[bool, Callable[[Any], bool]]] = eqx.field(
        static=True, kw_only=True, default=eqx.is_inexact_array
    )

    init_params: PyTree = eqx.field(init=False)
    static: PyTree = eqx.field(init=False, static=True)

    def __post_init__(self, eqx_network):

        if self.eq_type not in ["ODE", "statio_PDE", "nonstatio_PDE"]:
            raise RuntimeError("Wrong parameter value for eq_type")
        # saving the static part of the model and initial parameters

        if self.filter_spec is None:
            self.filter_spec = eqx.is_inexact_array

        self.init_params, self.static = eqx.partition(eqx_network, self.filter_spec)

        if self.input_transform is None:
            self.input_transform = lambda _in, _params: _in

        if self.output_transform is None:
            self.output_transform = lambda _in_pinn, _out_pinn, _params: _out_pinn

        if self.slice_solution is None:
            self.slice_solution = jnp.s_[:]

        if isinstance(self.slice_solution, int):
            # rewrite it as a slice to ensure that axis does not disappear when
            # indexing
            self.slice_solution = jnp.s_[self.slice_solution : self.slice_solution + 1]

    def eval(self, network, inputs, *args, **kwargs):
        """How to call your Equinox module `network`. The purpose of this method
        is to give more flexibility : user should re-implement `eval`
        when inheriting from `PINN` if they desire more flexibility on how to
        evaluate the network.

        Defaults to using `network.__call__(inputs)` but it could be more refined *e.g.* `network.anymethod(inputs)`.

        Parameters
        ----------
        network : eqx.Module
            Your neural network with the parameters set, usually returned by
            `eqx.combine(self.static, current_params)`.
        inputs : Array
            The inputs, evetually transformed by `self.input_transformed` if
            specified by the user.

        Returns
        -------
        Array
            The output
        """

        return network(inputs)

    def __call__(
        self,
        inputs: Float[Array, "input_dim"],
        params: Params | ParamsDict | PyTree,
        *args,
        **kwargs,
    ) -> Float[Array, "output_dim"]:
        """
        A proper __call__ implementation performs an eqx.combine here with
        `params` and `self.static` to recreate the callable eqx.Module
        architecture. The rest of the content of this function is dependent on
        the network.
        """

        if len(inputs.shape) == 0:
            # This can happen often when the user directly provides some
            # collocation points (eg for plotting, whithout using
            # DataGenerators)
            inputs = inputs[None]

        try:
            model = eqx.combine(params.nn_params, self.static)
        except (KeyError, AttributeError, TypeError) as e:  # give more flexibility
            model = eqx.combine(params, self.static)

        # evaluate the model
        res = self.eval(model, self.input_transform(inputs, params), *args, **kwargs)

        res = self.output_transform(inputs, res.squeeze(), params)

        # force (1,) output for non vectorial solution (consistency)
        if not res.shape:
            return jnp.expand_dims(res, axis=-1)
        return res
