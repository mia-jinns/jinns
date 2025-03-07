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
from jinns.nn._pinn import PINN
from jinns.nn._mlp import MLP


class PPINN_MLP(PINN):
    r"""
    A PPINN MLP (Parallel PINN with MLPs) object which mimicks the PFNN architecture from
    DeepXDE. This is in fact a PINN MLP that encompasses several PINN MLPs internally.

    Parameters
    ----------
    slice_solution : slice
        A jnp.s\_ object which indicates which axis of the PPINN output is
        dedicated to the actual equation solution. Default None
        means that slice_solution = the whole PPINN output. This argument is useful
        when the PINN is also used to output equation parameters for example
        Note that it must be a slice and not an integer (a preprocessing of the
        user provided argument takes care of it).
    eq_type : Literal["ODE", "statio_PDE", "nonstatio_PDE"]
        A string with three possibilities.
        "ODE": the PPINN is called with one input `t`.
        "statio_PDE": the PPINN is called with one input `x`, `x`
        can be high dimensional.
        "nonstatio_PDE": the PPINN is called with two inputs `t` and `x`, `x`
        can be high dimensional.
        **Note**: the input dimension as given in eqx_list has to match the sum
        of the dimension of `t` + the dimension of `x` or the output dimension
        after the `input_transform` function.
    input_transform : Callable[[Float[Array, "input_dim"], Params], Float[Array, "output_dim"]]
        A function that will be called before entering the PPINN. Its output(s)
        must match the PPINN inputs (except for the parameters).
        Its inputs are the PPINN inputs (`t` and/or `x` concatenated together)
        and the parameters. Default is no operation.
    output_transform : Callable[[Float[Array, "input_dim"], Float[Array, "output_dim"], Params], Float[Array, "output_dim"]]
        A function with arguments begin the same input as the PPINN, the PPINN
        output and the parameter. This function will be called after exiting
        the PPINN.
        Default is no operation.
    filter_spec : PyTree[Union[bool, Callable[[Any], bool]]]
        Default is `eqx.is_inexact_array`. This tells Jinns what to consider as
        a trainable parameter. Quoting from equinox documentation:
        a PyTree whose structure should be a prefix of the structure of pytree.
        Each of its leaves should either be 1) True, in which case the leaf or
        subtree is kept; 2) False, in which case the leaf or subtree is
        replaced with replace; 3) a callable Leaf -> bool, in which case this is evaluated on the leaf or mapped over the subtree, and the leaf kept or replaced as appropriate.
    eqx_network_list
            A list of eqx.nn.MLP objects with same input
            dimensions. They represent the parallel subnetworks of the PPIN MLP.
            Their respective outputs are concatenated.
    """

    eqx_network_list: InitVar[list[eqx.Module]] = eqx.field(kw_only=True)

    def __post_init__(self, eqx_network, eqx_network_list):
        super().__post_init__(
            eqx_network=eqx_network_list[0],  # this is not used since it is
            # overwritten just below
        )
        self.init_params, self.static = (), ()
        for eqx_network_ in eqx_network_list:
            params, static = eqx.partition(eqx_network_, self.filter_spec)
            self.init_params = self.init_params + (params,)
            self.static = self.static + (static,)

    def __call__(
        self,
        inputs: Float[Array, "1"] | Float[Array, "dim"] | Float[Array, "1+dim"],
        params: PyTree,
    ) -> Float[Array, "output_dim"]:
        """
        Evaluate the PPINN on some inputs with some params.
        """
        if len(inputs.shape) == 0:
            # This can happen often when the user directly provides some
            # collocation points (eg for plotting, whithout using
            # DataGenerators)
            inputs = inputs[None]
        transformed_inputs = self.input_transform(inputs, params)

        outs = []

        try:
            for params_, static in zip(params.nn_params, self.static):
                model = eqx.combine(params_, static)
                outs += [model(transformed_inputs)]
        except (KeyError, AttributeError, TypeError) as e:
            for params_, static in zip(params, self.static):
                model = eqx.combine(params_, static)
                outs += [model(transformed_inputs)]
        # Note that below is then a global output transform
        res = self.output_transform(inputs, jnp.concatenate(outs, axis=0), params)

        ## force (1,) output for non vectorial solution (consistency)
        if not res.shape:
            return jnp.expand_dims(res, axis=-1)
        return res

    @classmethod
    def create(
        cls,
        eq_type: Literal["ODE", "statio_PDE", "nonstatio_PDE"],
        eqx_network_list: list[eqx.nn.MLP] = None,
        key: Key = None,
        eqx_list_list: list[tuple[tuple[Callable, int, int] | Callable, ...]] = None,
        input_transform: Callable[
            [Float[Array, "input_dim"], Params], Float[Array, "output_dim"]
        ] = None,
        output_transform: Callable[
            [Float[Array, "input_dim"], Float[Array, "output_dim"], Params],
            Float[Array, "output_dim"],
        ] = None,
        slice_solution: slice = None,
    ) -> tuple[Self, PyTree]:
        r"""
        Utility function to create a Parrallel PINN neural network for Jinns.

        Parameters
        ----------
        eq_type
            A string with three possibilities.
            "ODE": the PPINN MLP is called with one input `t`.
            "statio_PDE": the PPINN MLP is called with one input `x`, `x`
            can be high dimensional.
            "nonstatio_PDE": the PPINN MLP is called with two inputs `t` and `x`, `x`
            can be high dimensional.
            **Note**: the input dimension as given in eqx_list has to match the sum
            of the dimension of `t` + the dimension of `x` or the output dimension
            after the `input_transform` function.
        eqx_network_list
                Default is None. A list of eqx.nn.MLP objects with same input
                dimensions. They represent the parallel subnetworks of the PPIN MLP.
                Their respective outputs are concatenated.
        key
            Default is None. Must be provided with `eqx_list_list` if
            `eqx_network_list` is not provided. A JAX random key that will be used
            to initialize the networks parameters.
        eqx_list_list
            Default is None. Must be provided if `eqx_network_list` is not
            provided. A list of `eqx_list` (see `PINN_MLP.create()`). The input dimension must be the
            same for each sub-`eqx_list`. Then the parallel subnetworks can be
            different. Their respective outputs are concatenated.
        input_transform
            A function that will be called before entering the PPINN MLP. Its output(s)
            must match the PPINN MLP inputs (except for the parameters).
            Its inputs are the PPINN MLP inputs (`t` and/or `x` concatenated together)
            and the parameters. Default is no operation.
        output_transform
            This function will be called after exiting
            the PPINN MLP, i.e., on the concatenated outputs of all parallel networks
            Default is no operation.
        slice_solution
            A jnp.s\_ object which indicates which axis of the PPINN MLP output is
            dedicated to the actual equation solution. Default None
            means that slice_solution = the whole PPINN MLP output. This argument is
            useful when the PPINN MLP is also used to output equation parameters for
            example Note that it must be a slice and not an integer (a
            preprocessing of the user provided argument takes care of it).


        Returns
        -------
        ppinn
            A PPINN MLP instance
        ppinn.init_params
            An initial set of parameters for the PPINN MLP

        """

        if eqx_network_list is None:
            if eqx_list_list is None or key is None:
                raise ValueError(
                    "If eqx_network_list is None, then key and eqx_list_list"
                    " must be provided"
                )

            eqx_network_list = []
            for eqx_list in eqx_list_list:
                key, subkey = jax.random.split(key, 2)
                print(subkey)
                eqx_network_list.append(MLP(key=subkey, eqx_list=eqx_list))

        ppinn = cls(
            eqx_network=None,
            eqx_network_list=eqx_network_list,
            slice_solution=slice_solution,
            eq_type=eq_type,
            input_transform=input_transform,
            output_transform=output_transform,
        )
        return ppinn, ppinn.init_params
