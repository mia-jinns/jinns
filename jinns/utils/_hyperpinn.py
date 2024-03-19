"""
Implements utility function to create HYPERPINNs
https://arxiv.org/pdf/2111.01008.pdf
"""

from functools import partial
import copy
from math import prod
import numpy as onp
import jax
import jax.numpy as jnp
from jax.tree_util import tree_leaves, tree_map
import equinox as eqx

from jinns.utils._pinn import PINN, _MLP


def _get_param_nb(params):
    """
    Returns the number of parameters in a equinox module whose parameters
    are stored in the pytree of parameters params but also the cumulative
    sum when parsing the pytree
    In reality, multiply the dimensions of the Arrays in this  pytree and
    sum everything, using pytree utility functions
    """
    dim_prod_all_arrays = [
        prod(a.shape)
        for a in tree_leaves(params, is_leaf=lambda x: isinstance(x, jnp.ndarray))
    ]
    return sum(dim_prod_all_arrays), onp.cumsum(dim_prod_all_arrays)


class HYPERPINN(PINN):
    """
    Composed of a PINN and an hypernetwork
    """

    def __init__(
        self,
        key,
        eqx_list,
        eqx_list_hyper,
        slice_solution,
        eq_type,
        input_transform,
        output_transform,
        hyperparams,
        hypernet_input_size,
        output_slice=None,
    ):
        key, subkey = jax.random.split(key, 2)
        super().__init__(
            subkey,
            eqx_list,
            slice_solution,
            eq_type,
            input_transform,
            output_transform,
            output_slice,
        )
        self.pinn_params_sum, self.pinn_params_cumsum = _get_param_nb(self.params)
        # the number of parameters for the pinn will be the number of ouputs
        # for the hypetnetwork
        self.hyperparams = hyperparams
        self.hypernet_input_size = hypernet_input_size
        key, subkey = jax.random.split(key, 2)
        try:
            eqx_list_hyper[-1][2] = self.pinn_params_sum
        except IndexError:
            eqx_list_hyper[-2][2] = self.pinn_params_sum
        try:
            eqx_list_hyper[0][1] = self.hypernet_input_size
        except IndexError:
            eqx_list_hyper[0][1] = self.hypernet_input_size
        _hyper = _MLP(subkey, eqx_list_hyper)
        self.params_hyper, self.static_hyper = eqx.partition(
            _hyper, eqx.is_inexact_array
        )

    def init_params(self):
        return self.params_hyper

    def hyper_to_pinn(self, hyper_output):
        """
        From the output of the hypernetwork we set the well formed
        parameters of the pinn (`self.params`)
        """
        pinn_params_flat = eqx.tree_at(
            lambda p: tree_leaves(p, is_leaf=lambda x: isinstance(x, jnp.ndarray)),
            self.params,
            [hyper_output[0 : self.pinn_params_cumsum[0]]]
            + [
                hyper_output[
                    self.pinn_params_cumsum[i] : self.pinn_params_cumsum[i + 1]
                ]
                for i in range(len(self.pinn_params_cumsum) - 1)
            ],
        )

        return tree_map(
            lambda a, b: a.reshape(b.shape),
            pinn_params_flat,
            self.params,
            is_leaf=lambda x: isinstance(x, jnp.ndarray),
        )

    def _eval_nn(self, inputs, params, input_transform, output_transform):
        """
        inner function to factorize code. apply_fn (which takes varying forms)
        call _eval_nn which always have the same content.
        """
        try:
            hyper = eqx.combine(params["nn_params"], self.static_hyper)
        except (KeyError, TypeError) as e:  # give more flexibility
            hyper = eqx.combine(params, self.static_hyper)

        eq_params_batch = jnp.concatenate(
            [params["eq_params"][k].flatten() for k in self.hyperparams], axis=0
        )

        hyper_output = hyper(eq_params_batch)

        pinn_params = self.hyper_to_pinn(hyper_output)

        pinn = eqx.combine(pinn_params, self.static)
        res = output_transform(inputs, pinn(input_transform(inputs, params)).squeeze())

        if self.output_slice is not None:
            res = res[self.output_slice]

        ## force (1,) output for non vectorial solution (consistency)
        if not res.shape:
            return jnp.expand_dims(res, axis=-1)
        return res


def create_HYPERPINN(
    key,
    eqx_list,
    eq_type,
    hyperparams,
    hypernet_input_size,
    dim_x=0,
    input_transform=None,
    output_transform=None,
    slice_solution=None,
    shared_pinn_outputs=None,
    eqx_list_hyper=None,
):
    r"""
    Utility function to create a standard PINN neural network with the equinox
    library.

    Parameters
    ----------
    key
        A jax random key that will be used to initialize the network parameters
    eqx_list
        A list of list of successive equinox modules and activation functions to
        describe the PINN architecture. The inner lists have the eqx module or
        axtivation function as first item, other items represents arguments
        that could be required (eg. the size of the layer).
        __Note:__ the `key` argument need not be given.
        Thus typical example is `eqx_list=
        [[eqx.nn.Linear, 2, 20],
        [jax.nn.tanh],
        [eqx.nn.Linear, 20, 20],
        [jax.nn.tanh],
        [eqx.nn.Linear, 20, 20],
        [jax.nn.tanh],
        [eqx.nn.Linear, 20, 1]
        ]`
    eq_type
        A string with three possibilities.
        "ODE": the PINN is called with one input `t`.
        "statio_PDE": the PINN is called with one input `x`, `x`
        can be high dimensional.
        "nonstatio_PDE": the PINN is called with two inputs `t` and `x`, `x`
        can be high dimensional.
        **Note**: the input dimension as given in eqx_list has to match the sum
        of the dimension of `t` + the dimension of `x` or the output dimension
        after the `input_transform` function
    hyperparams
        A list of keys from params["eq_params"] that will be considered as
        hyperparameters for metamodeling
    hypernet_input_size
        An integer. The input size of the MLP used for the hypernetwork. Must
        be equal to the flattened concatenations for the array of parameters
        designated by the `hyperparams` argument
    dim_x
        An integer. The dimension of `x`. Default `0`
    input_transform
        A function that will be called before entering the PINN. Its output(s)
        must match the PINN inputs. Its inputs are the PINN inputs (`t` and/or
        `x` concatenated together and the parameters). Default is the No operation
    output_transform
        A function with arguments the same input(s) as the PINN AND the PINN
        output that will be called after exiting the PINN. Default is the No
        operation
    slice_solution
        A jnp.s\_ object which indicates which axis of the PINN output is
        dedicated to the actual equation solution. Default None
        means that slice_solution = the whole PINN output. This argument is useful
        when the PINN is also used to output equation parameters for example
        Note that it must be a slice and not an integer (a preprocessing of the
        user provided argument takes care of it)
    shared_pinn_outputs
        Default is None, for a stantard PINN.
        A tuple of jnp.s_[] (slices) to determine the different output for each
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
    init_fn
        A function which (re-)initializes the PINN parameters with the provided
        jax random key
    apply_fn
        A function to apply the neural network on given inputs for given
        parameters. A typical call will be of the form `u(t, params)` for
        ODE or `u(t, x, params)` for nD PDEs (`x` being multidimensional)

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

        def output_transform(_in_pinn, _out_pinn):
            return _out_pinn

    if shared_pinn_outputs is not None:
        hyperpinns = []
        static = None
        for output_slice in shared_pinn_outputs:
            hyperpinn = HYPERPINN(
                key,
                eqx_list,
                eqx_list_hyper,
                slice_solution,
                eq_type,
                input_transform,
                output_transform,
                hyperparams,
                hypernet_input_size,
                output_slice,
            )
            # all the pinns are in fact the same so we share the same static
            if static is None:
                static = hyperpinn.static
            else:
                hyperpinn.static = static
            hyperpinns.append(hyperpinn)
        return hyperpinns
    hyperpinn = HYPERPINN(
        key,
        eqx_list,
        eqx_list_hyper,
        slice_solution,
        eq_type,
        input_transform,
        output_transform,
        hyperparams,
        hypernet_input_size,
    )
    return hyperpinn
