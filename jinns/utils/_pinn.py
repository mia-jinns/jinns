import jax
import jax.numpy as jnp
import equinox as eqx


class _MLP(eqx.Module):
    """
    Class to construct an equinox module from a key and a eqx_list. To be used
    in pair with the function `create_PINN`
    """

    layers: list

    def __init__(self, key, eqx_list):
        """
        Parameters
        ----------
        key
            A jax random key
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
        """

        self.layers = []
        # TODO we are limited currently in the number of layer type we can
        # parse and we lack some safety checks
        for l in eqx_list:
            if len(l) == 1:
                self.layers.append(l[0])
            else:
                # By default we append a random key at the end of the
                # arguments fed into a layer module call
                key, subkey = jax.random.split(key, 2)
                # the argument key is keyword only
                self.layers.append(l[0](*l[1:], key=subkey))

    def __call__(self, t):
        for layer in self.layers:
            t = layer(t)
        return t


class PINN:
    """
    Basically a wrapper around the `__call__` function to be able to give a type to
    our former `self.u`
    The function create_PINN has the role to population the `__call__` function
    """

    def __init__(self, key, eqx_list, output_slice=None):
        _pinn = _MLP(key, eqx_list)
        self.params, self.static = eqx.partition(_pinn, eqx.is_inexact_array)
        self.output_slice = output_slice

    def init_params(self):
        return self.params

    def __call__(self, *args, **kwargs):
        return self.apply_fn(self, *args, **kwargs)

    def _eval_nn(self, inputs, u_params, eq_params, input_transform, output_transform):
        """
        inner function to factorize code. apply_fn (which takes varying forms)
        call _eval_nn which always have the same content.
        """
        model = eqx.combine(u_params, self.static)
        res = output_transform(inputs, model(input_transform(inputs)).squeeze())

        if self.output_slice is not None:
            res = res[self.output_slice]

        ## force (1,) output for non vectorial solution (consistency)
        if not res.shape:
            return jnp.expand_dims(res, axis=-1)
        else:
            return res


def create_PINN(
    key,
    eqx_list,
    eq_type,
    dim_x=0,
    with_eq_params=None,
    input_transform=None,
    output_transform=None,
    shared_pinn_outputs=None,
):
    """
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
        **Note: the input dimension as given in eqx_list has to match the sum
        of the dimension of `t` + the dimension of `x` + the number of
        parameters in `eq_params` if with_eq_params is `True` (see below)**
    dim_x
        An integer. The dimension of `x`. Default `0`
    with_eq_params
        Default is None. Otherwise a list of keys from the dict `eq_params`
        that  the network also takes as inputs.
        the equation parameters (`eq_params`).
        **If some keys are provided, the input dimension
        as given in eqx_list must take into account the number of such provided
        keys (i.e., the input dimension is the addition of the dimension of ``t``
        + the dimension of ``x`` + the number of ``eq_params``)**
    input_transform
        A function that will be called before entering the PINN. Its output(s)
        must match the PINN inputs. Default is the No operation
    output_transform
        A function with arguments the same input(s) as the PINN AND the PINN
        output that will be called after exiting the PINN. Default is the No
        operation
    shared_pinn_outputs
        A tuple of jnp.s_[] (slices) to determine the different output for each
        network. In this case we return a list of PINNs, one for each output in
        shared_pinn_outputs. This is useful to create PINNs that share the
        same network and same parameters. Default is None, we only return one PINN.


    Returns
    -------
    init_fn
        A function which (re-)initializes the PINN parameters with the provided
        jax random key
    apply_fn
        A function to apply the neural network on given inputs for given
        parameters. A typical call will be of the form `u(t, nn_params)` for
        ODE or `u(t, x, nn_params)` for nD PDEs (`x` being multidimensional)
        or even `u(t, x, nn_params, eq_params)` if with_eq_params is `True`

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

    dim_t = 0 if eq_type == "statio_PDE" else 1
    dim_in_params = len(with_eq_params) if with_eq_params is not None else 0
    try:
        nb_inputs_declared = eqx_list[0][1]  # normally we look for 2nd ele of 1st layer
    except IndexError:
        nb_inputs_declared = eqx_list[1][1]
        # but we can have, eg, a flatten first layer

    try:
        nb_outputs_declared = eqx_list[-1][2]  # normally we look for 3rd ele of
        # last layer
    except IndexError:
        nb_outputs_declared = eqx_list[-2][2]
        # but we can have, eg, a `jnp.exp` last layer

    # NOTE Currently the check below is disabled because we added
    # input_transform
    # if dim_t + dim_x + dim_in_params != nb_inputs_declared:
    #    raise RuntimeError("Error in the declarations of the number of parameters")

    if input_transform is None:

        def input_transform(_in):
            return _in

    if output_transform is None:

        def output_transform(_in_pinn, _out_pinn):
            return _out_pinn

    if eq_type == "ODE":
        if with_eq_params is None:

            def apply_fn(self, t, u_params, eq_params=None):
                t = t[
                    None
                ]  # Note that we added a dimension to t which is lacking for the ODE batches
                return self._eval_nn(
                    t, u_params, eq_params, input_transform, output_transform
                ).squeeze()

        else:

            def apply_fn(self, t, u_params, eq_params):
                t = t[
                    None
                ]  # We added a dimension to t which is lacking for the ODE batches
                eq_params_flatten = jnp.concatenate(
                    [e.ravel() for k, e in eq_params.items() if k in with_eq_params]
                )
                t_eq_params = jnp.concatenate([t, eq_params_flatten], axis=-1)
                return self._eval_nn(
                    t_eq_params, u_params, eq_params, input_transform, output_transform
                )

    elif eq_type == "statio_PDE":
        # Here we add an argument `x` which can be high dimensional
        if with_eq_params is None:

            def apply_fn(self, x, u_params, eq_params=None):
                return self._eval_nn(
                    x, u_params, eq_params, input_transform, output_transform
                )

        else:

            def apply_fn(self, x, u_params, eq_params):
                eq_params_flatten = jnp.concatenate(
                    [e.ravel() for k, e in eq_params.items() if k in with_eq_params]
                )
                x_eq_params = jnp.concatenate([x, eq_params_flatten], axis=-1)
                return self._eval_nn(
                    x_eq_params, u_params, eq_params, input_transform, output_transform
                )

    elif eq_type == "nonstatio_PDE":
        # Here we add an argument `x` which can be high dimensional
        if with_eq_params is None:

            def apply_fn(self, t, x, u_params, eq_params=None):
                t_x = jnp.concatenate([t, x], axis=-1)
                return self._eval_nn(
                    t_x, u_params, eq_params, input_transform, output_transform
                )

        else:

            def apply_fn(self, t, x, u_params, eq_params):
                t_x = jnp.concatenate([t, x], axis=-1)
                eq_params_flatten = jnp.concatenate(
                    [e.ravel() for k, e in eq_params.items() if k in with_eq_params]
                )
                t_x_eq_params = jnp.concatenate([t_x, eq_params_flatten], axis=-1)
                return self._eval_nn(
                    t_x_eq_params,
                    u_params,
                    eq_params,
                    input_transform,
                    output_transform,
                )

    else:
        raise RuntimeError("Wrong parameter value for eq_type")

    if shared_pinn_outputs is not None:
        pinns = []
        static = None
        for output_slice in shared_pinn_outputs:
            pinn = PINN(key, eqx_list, output_slice)
            pinn.apply_fn = apply_fn
            # all the pinns are in fact the same so we share the same static
            if static is None:
                static = pinn.static
            else:
                pinn.static = static
            pinns.append(pinn)
        return pinns
    else:
        pinn = PINN(key, eqx_list)
        pinn.apply_fn = apply_fn
        return pinn
