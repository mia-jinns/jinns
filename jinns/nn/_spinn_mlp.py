"""
Implements utility function to create Separable PINNs
https://arxiv.org/abs/2211.08761
"""

from dataclasses import InitVar
from typing import Callable, Literal, Self, Union, Any
import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Key, Array, Float, PyTree

from jinns.parameters._params import Params, ParamsDict
from jinns.nn._mlp import MLP
from jinns.nn._spinn import SPINN


class SMLP(eqx.Module):
    """
    Construct a Separable MLP

    Parameters
    ----------
    key : InitVar[Key]
        A jax random key for the layer initializations.
    d : int
        The number of dimensions to treat separately, including time `t` if
        used for non-stationnary equations.
    eqx_list : InitVar[tuple[tuple[Callable, int, int] | Callable, ...]]
        A tuple of tuples of successive equinox modules and activation functions to
        describe the PINN architecture. The inner tuples must have the eqx module or
        activation function as first item, other items represents arguments
        that could be required (eg. the size of the layer).
        The `key` argument need not be given.
        Thus typical example is `eqx_list=
        ((eqx.nn.Linear, 1, 20),
            jax.nn.tanh,
            (eqx.nn.Linear, 20, 20),
            jax.nn.tanh,
            (eqx.nn.Linear, 20, 20),
            jax.nn.tanh,
            (eqx.nn.Linear, 20, r * m)
        )`.
    """

    key: InitVar[Key] = eqx.field(kw_only=True)
    eqx_list: InitVar[tuple[tuple[Callable, int, int] | Callable, ...]] = eqx.field(
        kw_only=True
    )
    d: int = eqx.field(static=True, kw_only=True)

    separated_mlp: list[MLP] = eqx.field(init=False)

    def __post_init__(self, key, eqx_list):
        keys = jax.random.split(key, self.d)
        self.separated_mlp = [
            MLP(key=keys[d_], eqx_list=eqx_list) for d_ in range(self.d)
        ]

    def __call__(
        self, inputs: Float[Array, "dim"] | Float[Array, "dim+1"]
    ) -> Float[Array, "d embed_dim*output_dim"]:
        outputs = []
        for d in range(self.d):
            x_i = inputs[d : d + 1]
            outputs += [self.separated_mlp[d](x_i)]
        return jnp.asarray(outputs)


class SPINN_MLP(SPINN):
    """
    An implementable SPINN based on a MLP architecture
    """

    @classmethod
    def create(
        cls,
        key: Key,
        d: int,
        r: int,
        eqx_list: tuple[tuple[Callable, int, int] | Callable, ...],
        eq_type: Literal["ODE", "statio_PDE", "nonstatio_PDE"],
        m: int = 1,
        filter_spec: PyTree[Union[bool, Callable[[Any], bool]]] = None,
    ) -> tuple[Self, PyTree]:
        """
        Utility function to create a SPINN neural network with the equinox
        library.

        *Note* that a SPINN is not vmapped and expects the
        same batch size for each of its input axis. It directly outputs a
        solution of shape `(batchsize,) * d`. See the paper for more
        details.

        Parameters
        ----------
        key : Key
            A JAX random key that will be used to initialize the network parameters
        d : int
            The number of dimensions to treat separately.
        r : int
            An integer. The dimension of the embedding.
        eqx_list : tuple[tuple[Callable, int, int] | Callable, ...],
            A tuple of tuples of successive equinox modules and activation functions to
            describe the PINN architecture. The inner tuples must have the eqx module or
            activation function as first item, other items represents arguments
            that could be required (eg. the size of the layer).
            The `key` argument need not be given.
            Thus typical example is
            `eqx_list=((eqx.nn.Linear, 1, 20),
                jax.nn.tanh,
                (eqx.nn.Linear, 20, 20),
                jax.nn.tanh,
                (eqx.nn.Linear, 20, 20),
                jax.nn.tanh,
                (eqx.nn.Linear, 20, r * m)
            )`.
        eq_type : Literal["ODE", "statio_PDE", "nonstatio_PDE"]
            A string with three possibilities.
            "ODE": the PINN is called with one input `t`.
            "statio_PDE": the PINN is called with one input `x`, `x`
            can be high dimensional.
            "nonstatio_PDE": the PINN is called with two inputs `t` and `x`, `x`
            can be high dimensional.
            **Note**: the input dimension as given in eqx_list has to match the sum
            of the dimension of `t` + the dimension of `x`.
        m : int
            The output dimension of the neural network. According to
            the SPINN article, a total embedding dimension of `r*m` is defined. We
            then sum groups of `r` embedding dimensions to compute each output.
            Default is 1.
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
        spinn
            An instanciated SPINN
        spinn.init_params
            The initial set of parameters of the model

        Raises
        ------
        RuntimeError
            If the parameter value for eq_type is not in `["ODE", "statio_PDE",
            "nonstatio_PDE"]` and for various failing checks
        """

        if eq_type not in ["ODE", "statio_PDE", "nonstatio_PDE"]:
            raise RuntimeError("Wrong parameter value for eq_type")

        try:
            nb_inputs_declared = eqx_list[0][
                1
            ]  # normally we look for 2nd ele of 1st layer
        except IndexError:
            nb_inputs_declared = eqx_list[1][
                1
            ]  # but we can have, eg, a flatten first layer
        if nb_inputs_declared != 1:
            raise ValueError("Input dim must be set to 1 in SPINN!")

        try:
            nb_outputs_declared = eqx_list[-1][2]  # normally we look for 3rd ele of
            # last layer
        except IndexError:
            nb_outputs_declared = eqx_list[-2][2]
            # but we can have, eg, a `jnp.exp` last layer
        if nb_outputs_declared != r * m:
            raise ValueError("Output dim must be set to r * m in SPINN!")

        if d > 24:
            raise ValueError(
                "Too many dimensions, not enough letters available in jnp.einsum"
            )

        smlp = SMLP(key=key, d=d, eqx_list=eqx_list)
        spinn = cls(
            eqx_spinn_network=smlp,
            d=d,
            r=r,
            eq_type=eq_type,
            m=m,
            filter_spec=filter_spec,
        )

        return spinn, spinn.init_params
