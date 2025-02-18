"""
Implements utility function to create PINNs
"""

from typing import Callable, Literal, Self
from dataclasses import InitVar
import jax
import jax.numpy as jnp
import equinox as eqx

from jaxtyping import Array, Key, PyTree, Float

from jinns.parameters._params import Params, ParamsDict
from jinns.networks._pinn import PINN


class JinnsMLP(eqx.Module):
    """
    Custom MLP equinox module from a key and a eqx_list

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
        nb_keys_required = sum(1 if len(l) > 1 else 0 for l in eqx_list)
        keys = jax.random.split(key, nb_keys_required)
        # we need a global split
        # before the loop to maintain strict equivalency with eqx.nn.MLP
        # for debugging purpose
        k = 0
        for l in eqx_list:
            if len(l) == 1:
                self.layers.append(l[0])
            else:
                self.layers.append(l[0](*l[1:], key=keys[k]))
                k += 1

    def __call__(self, t: Float[Array, "input_dim"]) -> Float[Array, "output_dim"]:
        for layer in self.layers:
            t = layer(t)
        return t


class MLP(PINN):
    r"""
    A MLP PINN object, i.e., a neural network compatible with the rest of jinns.
    This is typically created with `create`.

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
    output_slice : slice, default=None
        A jnp.s\_[] to determine the different dimension for the PINN.
        See `shared_pinn_outputs` argument of `create_PINN`.
    mlp : eqx.Module
        The actual neural network instanciated as an eqx.Module.
    """

    input_transform: Callable[
        [Float[Array, "input_dim"], Params], Float[Array, "output_dim"]
    ] = eqx.field(static=True, kw_only=True)
    output_transform: Callable[
        [Float[Array, "input_dim"], Float[Array, "output_dim"], Params],
        Float[Array, "output_dim"],
    ] = eqx.field(static=True, kw_only=True)
    output_slice: slice = eqx.field(static=True, kw_only=True, default=None)

    def __call__(
        self,
        inputs: Float[Array, "1"] | Float[Array, "dim"] | Float[Array, "1+dim"],
        params: Params | ParamsDict | PyTree,
    ) -> Float[Array, "output_dim"]:
        """
        Evaluate the PINN on some inputs with some params.
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
        res = self.output_transform(
            inputs, model(self.input_transform(inputs, params)).squeeze(), params
        )

        if self.output_slice is not None:
            res = res[self.output_slice]

        ## force (1,) output for non vectorial solution (consistency)
        if not res.shape:
            return jnp.expand_dims(res, axis=-1)
        return res

    @classmethod
    def create(
        cls,
        eq_type: Literal["ODE", "statio_PDE", "nonstatio_PDE"],
        eqx_network: eqx.nn.MLP = None,
        key: Key = None,
        eqx_list: tuple[tuple[Callable, int, int] | Callable, ...] = None,
        input_transform: Callable[
            [Float[Array, "input_dim"], Params], Float[Array, "output_dim"]
        ] = None,
        output_transform: Callable[
            [Float[Array, "input_dim"], Float[Array, "output_dim"], Params],
            Float[Array, "output_dim"],
        ] = None,
        shared_pinn_outputs: tuple[slice] = None,
        slice_solution: slice = None,
    ) -> tuple[Self | list[Self], PyTree | list[PyTree]]:
        r"""
        Instanciate standard MLP PINN object. The actual NN is either passed as
        a eqx.nn.MLP (`eqx_network` argument) or constructed as a custom
        JinnsMLP when `key` and `eqx_list` is provided.

        Parameters
        ----------
        eq_type
            A string with three possibilities.
            "ODE": the MLP is called with one input `t`.
            "statio_PDE": the MLP is called with one input `x`, `x`
            can be high dimensional.
            "nonstatio_PDE": the MLP is called with two inputs `t` and `x`, `x`
            can be high dimensional.
            **Note**: the input dimension as given in eqx_list has to match the sum
            of the dimension of `t` + the dimension of `x` or the output dimension
            after the `input_transform` function.
        eqx_network
            Default is None. A eqx.nn.MLP object that will be wrapped inside
            our MLP object in order to make it easily jinns compatible.
        key
            Default is None. Must be provided with `eqx_list` if `eqx_network`
            is not provided. A JAX random key that will be used to initialize the network
            parameters.
        eqx_list
            Default is None. Must be provided with `eqx_list` if `eqx_network`
            is not provided.A tuple of tuples of successive equinox modules and activation
            functions to describe the MLP architecture. The inner tuples must have
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
        input_transform
            A function that will be called before entering the MLP. Its output(s)
            must match the MLP inputs (except for the parameters).
            Its inputs are the MLP inputs (`t` and/or `x` concatenated together)
            and the parameters. Default is no operation.
        output_transform
            A function with arguments begin the same input as the MLP, the MLP
            output and the parameter. This function will be called after exiting
            the MLP.
            Default is no operation.
        shared_pinn_outputs
            Default is None, for a stantard MLP.
            A tuple of jnp.s\_[] (slices) to determine the different output for each
            network. In this case we return a list of MLPs, one for each output in
            shared_pinn_outputs. This is useful to create MLPs that share the
            same network and same parameters; **the user must then use the same
            parameter set in their manipulation**.
            See the notebook 2D Navier Stokes in pipeflow with metamodel for an
            example using this option.
        slice_solution
            A jnp.s\_ object which indicates which axis of the MLP output is
            dedicated to the actual equation solution. Default None
            means that slice_solution = the whole MLP output. This argument is
            useful when the MLP is also used to output equation parameters for
            example Note that it must be a slice and not an integer (a
            preprocessing of the user provided argument takes care of it).


        Returns
        -------
        mlp
            A MLP instance or, when `shared_pinn_ouput` is not None,
            a list of MLP instances with the same structure is returned,
            only differing by there final slicing of the network output.
        mlp.init_params
            An initial set of parameters for the MLP or a list of the latter
            when `shared_pinn_ouput` is not None.

        """
        if eqx_network is None:
            if eqx_list is None or key is None:
                raise ValueError(
                    "If eqx_network is None, then key and eqx_list" " must be provided"
                )
            eqx_network = JinnsMLP(key=key, eqx_list=eqx_list)

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

        if shared_pinn_outputs is not None:
            mlps = []
            for output_slice in shared_pinn_outputs:
                mlp = cls(
                    eqx_network=eqx_network,
                    slice_solution=slice_solution,
                    eq_type=eq_type,
                    input_transform=input_transform,
                    output_transform=output_transform,
                    output_slice=output_slice,
                )
                mlps.append(mlp)
            return mlps, [p.init_params for p in mlps]
        mlp = cls(
            eqx_network=eqx_network,
            slice_solution=slice_solution,
            eq_type=eq_type,
            input_transform=input_transform,
            output_transform=output_transform,
            output_slice=None,
        )
        return mlp, mlp.init_params
