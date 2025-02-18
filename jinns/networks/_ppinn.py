"""
Implements utility function to create PINNs
"""

from typing import Callable, Literal
from dataclasses import InitVar
import jax
import jax.numpy as jnp
import equinox as eqx

from jaxtyping import Array, Key, PyTree, Float

from jinns.parameters._params import Params, ParamsDict

from jinns.utils._pinn import PINN, _MLP


class PPINN(PINN):
    r"""
    A PPINN (Parallel PINN) object which mimicks the PFNN architecture from
    DeepXDE. This is in fact a PINN that encompasses several PINNs internally.

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
    mlp_list : list[eqx.Module]
        The actual neural networks instanciated as eqx.Modules
    """

    slice_solution: slice = eqx.field(static=True, kw_only=True)
    output_slice: slice = eqx.field(static=True, kw_only=True, default=None)

    mlp_list: InitVar[list[eqx.Module]] = eqx.field(kw_only=True)

    params: PyTree = eqx.field(init=False)
    static: PyTree = eqx.field(init=False, static=True)

    def __post_init__(self, mlp, mlp_list):
        super().__post_init__(
            mlp=mlp_list[0],
        )
        self.params, self.static = (), ()
        for mlp in mlp_list:
            params, static = eqx.partition(mlp, eqx.is_inexact_array)
            self.params = self.params + (params,)
            self.static = self.static + (static,)

    @property
    def init_params(self) -> PyTree:
        """
        Returns an initial set of parameters
        """
        return self.params

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
        for params_, static in zip(params.nn_params, self.static):
            model = eqx.combine(params_, static)
            outs += [model(transformed_inputs)]
        # Note that below is then a global output transform
        res = self.output_transform(inputs, jnp.concatenate(outs, axis=0), params)

        ## force (1,) output for non vectorial solution (consistency)
        if not res.shape:
            return jnp.expand_dims(res, axis=-1)
        return res


def create_PPINN(
    key: Key,
    eqx_list_list: list[tuple[tuple[Callable, int, int] | Callable, ...]],
    eq_type: Literal["ODE", "statio_PDE", "nonstatio_PDE"],
    dim_x: int = 0,
    input_transform: Callable[
        [Float[Array, "input_dim"], Params], Float[Array, "output_dim"]
    ] = None,
    output_transform: Callable[
        [Float[Array, "input_dim"], Float[Array, "output_dim"], Params],
        Float[Array, "output_dim"],
    ] = None,
    slice_solution: slice = None,
) -> tuple[PINN | list[PINN], PyTree | list[PyTree]]:
    r"""
    Utility function to create a standard PINN neural network with the equinox
    library.

    Parameters
    ----------
    key
        A JAX random key that will be used to initialize the network
        parameters.
    eqx_list_list
        A list of `eqx_list` (see `create_PINN`). The input dimension must be the
        same for each sub-`eqx_list`. Then the parallel subnetworks can be
        different. Their respective outputs are concatenated.
    eq_type
        A string with three possibilities.
        "ODE": the PPINN is called with one input `t`.
        "statio_PDE": the PPINN is called with one input `x`, `x`
        can be high dimensional.
        "nonstatio_PDE": the PPINN is called with two inputs `t` and `x`, `x`
        can be high dimensional.
        **Note**: the input dimension as given in eqx_list has to match the sum
        of the dimension of `t` + the dimension of `x` or the output dimension
        after the `input_transform` function.
    dim_x
        An integer. The dimension of `x`. Default `0`.
    input_transform
        A function that will be called before entering the PPINN. Its output(s)
        must match the PPINN inputs (except for the parameters).
        Its inputs are the PPINN inputs (`t` and/or `x` concatenated together)
        and the parameters. Default is no operation.
    output_transform
        This function will be called after exiting
        the PPINN, i.e., on the concatenated outputs of all parallel networks
        Default is no operation.
    slice_solution
        A jnp.s\_ object which indicates which axis of the PPINN output is
        dedicated to the actual equation solution. Default None
        means that slice_solution = the whole PPINN output. This argument is
        useful when the PPINN is also used to output equation parameters for
        example Note that it must be a slice and not an integer (a
        preprocessing of the user provided argument takes care of it).


    Returns
    -------
    ppinn
        A PPINN instance
    ppinn.init_params
        An initial set of parameters for the PPINN

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

    nb_outputs_declared = 0
    for eqx_list in eqx_list_list:
        try:
            nb_outputs_declared += eqx_list[-1][2]  # normally we look for 3rd ele of
            # last layer
        except IndexError:
            nb_outputs_declared += eqx_list[-2][2]

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

    mlp_list = []
    for eqx_list in eqx_list_list:
        mlp_list.append(_MLP(key=key, eqx_list=eqx_list))

    ppinn = PPINN(
        mlp=None,
        mlp_list=mlp_list,
        slice_solution=slice_solution,
        eq_type=eq_type,
        input_transform=input_transform,
        output_transform=output_transform,
    )
    return ppinn, ppinn.init_params
