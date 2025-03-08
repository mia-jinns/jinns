"""
Implements utility function to create PINNs
"""

from typing import Callable, Literal, Self, Union, Any
from dataclasses import InitVar
import jax
import equinox as eqx

from jaxtyping import Array, Key, PyTree, Float

from jinns.parameters._params import Params
from jinns.nn._pinn import PINN


class MLP(eqx.Module):
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
        # nb_keys_required = sum(1 if len(l) > 1 else 0 for l in eqx_list)
        # keys = jax.random.split(key, nb_keys_required)
        # we need a global split
        # before the loop to maintain strict equivalency with eqx.nn.MLP
        # for debugging purpose
        k = 0
        for l in eqx_list:
            if len(l) == 1:
                self.layers.append(l[0])
            else:
                key, subkey = jax.random.split(key, 2)  # nb_keys_required)
                self.layers.append(l[0](*l[1:], key=subkey))
                k += 1

    def __call__(self, t: Float[Array, "input_dim"]) -> Float[Array, "output_dim"]:
        for layer in self.layers:
            t = layer(t)
        return t


class PINN_MLP(PINN):
    """
    An implementable PINN based on a MLP architecture
    """

    # Here we could have a more complex __call__ method that redefined the
    # parent's __call__. But there is no need for the simple PINN_MLP

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
        slice_solution: slice = None,
        filter_spec: PyTree[Union[bool, Callable[[Any], bool]]] = None,
    ) -> tuple[Self, PyTree]:
        r"""
        Instanciate standard PINN MLP object. The actual NN is either passed as
        a eqx.nn.MLP (`eqx_network` argument) or constructed as a custom
        jinns.nn.MLP when `key` and `eqx_list` is provided.

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
            our PINN_MLP object in order to make it easily jinns compatible.
        key
            Default is None. Must be provided with `eqx_list` if `eqx_network`
            is not provided. A JAX random key that will be used to initialize the network
            parameters.
        eqx_list
            Default is None. Must be provided  if `eqx_network`
            is not provided. A tuple of tuples of successive equinox modules and activation
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
        slice_solution
            A jnp.s\_ object which indicates which axis of the MLP output is
            dedicated to the actual equation solution. Default None
            means that slice_solution = the whole MLP output. This argument is
            useful when the MLP is also used to output equation parameters for
            example Note that it must be a slice and not an integer (a
            preprocessing of the user provided argument takes care of it).

        filter_spec : PyTree[Union[bool, Callable[[Any], bool]]]
            Default is None which leads to `eqx.is_inexact_array` in the class
            instanciation. This tells Jinns what to consider as
            a trainable parameter. Quoting from equinox documentation:
            a PyTree whose structure should be a prefix of the structure of pytree.
            Each of its leaves should either be 1) True, in which case the leaf or
            subtree is kept; 2) False, in which case the leaf or subtree is
            replaced with replace; 3) a callable Leaf -> bool, in which case this is evaluated on the leaf or mapped over the subtree, and the leaf kept or replaced as appropriate.

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
                    "If eqx_network is None, then key and eqx_list must be provided"
                )
            eqx_network = MLP(key=key, eqx_list=eqx_list)

        mlp = cls(
            eqx_network=eqx_network,
            slice_solution=slice_solution,
            eq_type=eq_type,
            input_transform=input_transform,
            output_transform=output_transform,
            filter_spec=filter_spec,
        )
        return mlp, mlp.init_params
