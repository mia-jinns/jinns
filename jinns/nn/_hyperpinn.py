"""
Implements utility function to create HyperPINNs
https://arxiv.org/pdf/2111.01008.pdf
"""

import warnings
from dataclasses import InitVar
from typing import Callable, Literal, Self, Union, Any
from math import prod
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree, Key
import equinox as eqx
import numpy as onp

from jinns.nn._pinn import PINN
from jinns.nn._mlp import MLP
from jinns.parameters._params import Params, ParamsDict


def _get_param_nb(
    params: Params,
) -> tuple[int, list]:
    """Returns the number of parameters in a Params object and also
    the cumulative sum when parsing the object.


    Parameters
    ----------
    params :
        A Params object.
    """
    dim_prod_all_arrays = [
        prod(a.shape)
        for a in jax.tree.leaves(params, is_leaf=lambda x: isinstance(x, jnp.ndarray))
    ]
    return (
        sum(dim_prod_all_arrays),
        onp.cumsum(dim_prod_all_arrays).tolist(),
    )


class HyperPINN(PINN):
    r"""
    An HyperPINN object compatible with the rest of jinns.
    Composed of a PINN and an HYPER network. The HyperPINN is typically
    instanciated using with `create`.

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
        "ODE": the HyperPINN is called with one input `t`.
        "statio_PDE": the HyperPINN is called with one input `x`, `x`
        can be high dimensional.
        "nonstatio_PDE": the HyperPINN is called with two inputs `t` and `x`, `x`
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
    mlp : eqx.Module
        The actual neural network instanciated as an eqx.Module.
    hyper_mlp : eqx.Module
        The actual hyper neural network instanciated as an eqx.Module.
    filter_spec : PyTree[Union[bool, Callable[[Any], bool]]]
        Default is `eqx.is_inexact_array`. This tells Jinns what to consider as
        a trainable parameter. Quoting from equinox documentation:
        a PyTree whose structure should be a prefix of the structure of pytree.
        Each of its leaves should either be 1) True, in which case the leaf or
        subtree is kept; 2) False, in which case the leaf or subtree is
        replaced with replace; 3) a callable Leaf -> bool, in which case this is evaluated on the leaf or mapped over the subtree, and the leaf kept or replaced as appropriate.
    """

    hyperparams: list[str] = eqx.field(static=True, kw_only=True)
    hypernet_input_size: int = eqx.field(kw_only=True)

    eqx_hyper_network: InitVar[eqx.Module] = eqx.field(kw_only=True)

    pinn_params_sum: int = eqx.field(init=False, static=True)
    pinn_params_cumsum: list = eqx.field(init=False, static=True)

    init_params_hyper: PyTree = eqx.field(init=False)
    static_hyper: PyTree = eqx.field(init=False, static=True)

    def __post_init__(self, eqx_network, eqx_hyper_network):
        super().__post_init__(
            eqx_network,
        )
        # In addition, we store the PyTree structure of the hypernetwork as well
        self.init_params_hyper, self.static_hyper = eqx.partition(
            eqx_hyper_network, self.filter_spec
        )
        self.pinn_params_sum, self.pinn_params_cumsum = _get_param_nb(self.init_params)

    def _hyper_to_pinn(self, hyper_output: Float[Array, "output_dim"]) -> PyTree:
        """
        From the output of the hypernetwork, transform to a well formed
        parameters for the pinn network (i.e. with the same PyTree structure as
        `self.init_params`)
        """

        pinn_params_flat = eqx.tree_at(
            jax.tree.leaves,  # is_leaf=eqx.is_array argument for jax.tree.leaves
            # is not needed in general when working
            # with eqx.nn.Linear for examples: jax.tree.leaves
            # already returns the array of weights and biases only, since the
            # other stuff (that we do not want to be returned) is marked as
            # static (in eqx.nn.Linear), hence is not part of the leaves.
            # Note, that custom layers should then be properly designed to pass
            # this jax.tree.leaves.
            self.init_params,
            jnp.split(hyper_output, self.pinn_params_cumsum[:-1]),
        )

        return jax.tree.map(
            lambda a, b: a.reshape(b.shape),
            pinn_params_flat,
            self.init_params,
            is_leaf=lambda x: isinstance(x, jnp.ndarray),
        )

    def __call__(
        self,
        inputs: Float[Array, "input_dim"],
        params: Params | ParamsDict | PyTree,
        *args,
        **kwargs,
    ) -> Float[Array, "output_dim"]:
        """
        Evaluate the HyperPINN on some inputs with some params.
        """
        if len(inputs.shape) == 0:
            # This can happen often when the user directly provides some
            # collocation points (eg for plotting, whithout using
            # DataGenerators)
            inputs = inputs[None]

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
        res = self.eval(pinn, self.input_transform(inputs, params), *args, **kwargs)

        res = self.output_transform(inputs, res.squeeze(), params)

        # force (1,) output for non vectorial solution (consistency)
        if not res.shape:
            return jnp.expand_dims(res, axis=-1)
        return res

    @classmethod
    def create(
        cls,
        eq_type: Literal["ODE", "statio_PDE", "nonstatio_PDE"],
        hyperparams: list[str],
        hypernet_input_size: int,
        eqx_network: eqx.nn.MLP = None,
        eqx_hyper_network: eqx.nn.MLP = None,
        key: Key = None,
        eqx_list: tuple[tuple[Callable, int, int] | Callable, ...] = None,
        eqx_list_hyper: tuple[tuple[Callable, int, int] | Callable, ...] = None,
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
        Utility function to create a standard PINN neural network with the equinox
        library.

        Parameters
        ----------
        key
            A JAX random key that will be used to initialize the network
            parameters.
        eq_type
            A string with three possibilities.
            "ODE": the HyperPINN is called with one input `t`.
            "statio_PDE": the HyperPINN is called with one input `x`, `x`
            can be high dimensional.
            "nonstatio_PDE": the HyperPINN is called with two inputs `t` and `x`, `x`
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
        eqx_network
            Default is None. A eqx.nn.MLP for the base network that will be wrapped inside
            our PINN_MLP object in order to make it easily jinns compatible.
        eqx_hyper_network
            Default is None. A eqx.nn.MLP for the hyper network that will be wrapped inside
            our PINN_MLP object in order to make it easily jinns compatible.
        key
            Default is None. Must be provided with `eqx_list` and
            `eqx_list_hyper` if `eqx_network` or `eqx_hyper_network`
            is not provided. A JAX random key that will be used to initialize the network
            parameters.
        eqx_list
            Default is None. Must be provided  if `eqx_network` or
            `eqx_hyper_network`
            is not provided.
            A tuple of tuples of successive equinox modules and activation functions to
            describe the base network architecture. The inner tuples must have the eqx module or
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
        eqx_list_hyper
            Default is None. Must be provided  if `eqx_network` or
            `eqx_hyper_network`
            is not provided.
            A tuple of tuples of successive equinox modules and activation functions to
            describe the hyper network architecture. The inner tuples must have the eqx module or
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
        eqx_list_hyper
            Same as eqx_list but for the hypernetwork. Default is None, i.e., we
            use the same architecture as the PINN, up to the number of inputs and
            ouputs. Note that the number of inputs must be of the hypernetwork must
            be equal to the flattened concatenations for the array of parameters
            designated by the `hyperparams` argument;
            and the number of outputs must be equal to the number
            of parameters in the pinn network
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
        hyperpinn
            A HyperPINN instance or, when `shared_pinn_ouput` is not None,
            a list of HyperPINN instances with the same structure is returned,
            only differing by there final slicing of the network output.
        hyperpinn.init_params
            The initial set of parameters for the HyperPINN or a list of the latter
            when `shared_pinn_ouput` is not None.

        """
        if eqx_network is None or eqx_hyper_network is None:
            if eqx_list is None or key is None or eqx_list_hyper is None:
                raise ValueError(
                    "If eqx_network is None or eqx_hyper_network is None, then"
                    " key and eqx_list and eqx_hyper_network must be provided"
                )

            ### Now we finetune the hypernetwork architecture

            key, subkey = jax.random.split(key, 2)
            # with warnings.catch_warnings():
            #    warnings.filterwarnings("ignore", message="A JAX array is being set as static!")
            eqx_network = MLP(key=subkey, eqx_list=eqx_list)
            # quick partitioning to get the params to get the correct number of neurons
            # for the last layer of hyper network
            params_mlp, _ = eqx.partition(eqx_network, eqx.is_inexact_array)
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
            # with warnings.catch_warnings():
            #    warnings.filterwarnings("ignore", message="A JAX array is being set as static!")
            eqx_hyper_network = MLP(key=subkey, eqx_list=eqx_list_hyper)

            ### End of finetuning the hypernetwork architecture

        with warnings.catch_warnings():
            # Catch the equinox warning because we put the number of
            # parameters as static while being jnp.Array. This this time
            # this is correct to do so, because they are used as indices
            # and will never be modified
            warnings.filterwarnings(
                "ignore", message="A JAX array is being set as static!"
            )
            hyperpinn = cls(
                eqx_network=eqx_network,
                eqx_hyper_network=eqx_hyper_network,
                slice_solution=slice_solution,
                eq_type=eq_type,
                input_transform=input_transform,
                output_transform=output_transform,
                hyperparams=hyperparams,
                hypernet_input_size=hypernet_input_size,
                filter_spec=filter_spec,
            )
        return hyperpinn, hyperpinn.init_params_hyper
