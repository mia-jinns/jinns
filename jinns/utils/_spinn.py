"""
Implements utility function to create Separable PINNs
https://arxiv.org/abs/2211.08761
"""

from dataclasses import InitVar
from typing import Callable, Literal
import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Key, Array, Float, PyTree


class _SPINN(eqx.Module):
    """
    Construct a Separable PINN as proposed in
    Cho et al., _Separable Physics-Informed Neural Networks_, NeurIPS, 2023

    Parameters
    ----------
    key : InitVar[Key]
        A jax random key for the layer initializations.
    d : int
        The number of dimensions to treat separately.
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

    d: int = eqx.field(static=True, kw_only=True)

    key: InitVar[Key] = eqx.field(kw_only=True)
    eqx_list: InitVar[tuple[tuple[Callable, int, int] | Callable, ...]] = eqx.field(
        kw_only=True
    )

    layers: list = eqx.field(init=False)
    separated_mlp: list = eqx.field(init=False)

    def __post_init__(self, key, eqx_list):
        self.separated_mlp = []
        for _ in range(self.d):
            self.layers = []
            for l in eqx_list:
                if len(l) == 1:
                    self.layers.append(l[0])
                else:
                    key, subkey = jax.random.split(key, 2)
                    self.layers.append(l[0](*l[1:], key=subkey))
            self.separated_mlp.append(self.layers)

    def __call__(
        self, t: Float[Array, "1"], x: Float[Array, "omega_dim"]
    ) -> Float[Array, "d embed_dim*output_dim"]:
        if t is not None:
            dimensions = jnp.concatenate([t, x.flatten()], axis=0)
        else:
            dimensions = jnp.concatenate([x.flatten()], axis=0)
        outputs = []
        for d in range(self.d):
            t_ = dimensions[d][None]
            for layer in self.separated_mlp[d]:
                t_ = layer(t_)
            outputs += [t_]
        return jnp.asarray(outputs)


class SPINN(eqx.Module):
    """
    A SPINN object compatible with the rest of jinns.
    This is typically created with `create_SPINN`.

    **NOTE**: SPINNs with `t` and `x` as inputs are best used with a
    DataGenerator with `self.cartesian_product=False` for memory consideration

    Parameters
    ----------
    d : int
        The number of dimensions to treat separately.

    """

    d: int = eqx.field(static=True, kw_only=True)
    r: int = eqx.field(static=True, kw_only=True)
    eq_type: str = eqx.field(static=True, kw_only=True)
    m: int = eqx.field(static=True, kw_only=True)

    spinn_mlp: InitVar[eqx.Module] = eqx.field(kw_only=True)

    params: PyTree = eqx.field(init=False)
    static: PyTree = eqx.field(init=False, static=True)

    def __post_init__(self, spinn_mlp):
        self.params, self.static = eqx.partition(spinn_mlp, eqx.is_inexact_array)

    def init_params(self) -> PyTree:
        """
        Returns an initial set of parameters
        """
        return self.params

    def __call__(self, *args) -> Float[Array, "output_dim"]:
        """
        Calls `eval_nn` with rearranged arguments
        """
        if self.eq_type == "statio_PDE":
            (x, params) = args
            try:
                spinn = eqx.combine(params.nn_params, self.static)
            except (KeyError, AttributeError, TypeError) as e:
                spinn = eqx.combine(params, self.static)
            v_model = jax.vmap(spinn, (0))
            res = v_model(t=None, x=x)
            return self.eval_nn(res)
        if self.eq_type == "nonstatio_PDE":
            (t, x, params) = args
            try:
                spinn = eqx.combine(params.nn_params, self.static)
            except (KeyError, AttributeError, TypeError) as e:
                spinn = eqx.combine(params, self.static)
            v_model = jax.vmap(spinn, ((0, 0)))
            res = v_model(t, x)
            return self.eval_nn(res)
        raise RuntimeError("Wrong parameter value for eq_type")

    def eval_nn(
        self, res: Float[Array, "d embed_dim*output_dim"]
    ) -> Float[Array, "output_dim"]:
        """
        Evaluate the SPINN on some inputs with some params.
        """
        a = ", ".join([f"{chr(97 + d)}z" for d in range(res.shape[1])])
        b = "".join([f"{chr(97 + d)}" for d in range(res.shape[1])])
        res = jnp.stack(
            [
                jnp.einsum(
                    f"{a} -> {b}",
                    *(
                        res[:, d, m * self.r : (m + 1) * self.r]
                        for d in range(res.shape[1])
                    ),
                )
                for m in range(self.m)
            ],
            axis=-1,
        )  # compute each output dimension

        # force (1,) output for non vectorial solution (consistency)
        if len(res.shape) == self.d:
            return jnp.expand_dims(res, axis=-1)
        return res


def create_SPINN(
    key: Key,
    d: int,
    r: int,
    eqx_list: tuple[tuple[Callable, int, int] | Callable, ...],
    eq_type: Literal["ODE", "statio_PDE", "nonstatio_PDE"],
    m: int = 1,
) -> SPINN:
    """
    Utility function to create a SPINN neural network with the equinox
    library.

    *Note* that a SPINN is not vmapped and expects the
    same batch size for each of its input axis. It directly outputs a solution
    of shape `(batchsize, batchsize)`. See the paper for more details.

    Parameters
    ----------
    key
        A JAX random key that will be used to initialize the network parameters
    d
        The number of dimensions to treat separately.
    r
        An integer. The dimension of the embedding.
    eqx_list
        A tuple of tuples of successive equinox modules and activation functions to
        describe the PINN architecture. The inner tuples must have the eqx module or
        activation function as first item, other items represents arguments
        that could be required (eg. the size of the layer).
        The `key` argument need not be given.
        Thus typical example is
        `eqx_list=((eqx.nn.Linear, 2, 20),
            jax.nn.tanh,
            (eqx.nn.Linear, 20, 20),
            jax.nn.tanh,
            (eqx.nn.Linear, 20, 20),
            jax.nn.tanh,
            (eqx.nn.Linear, 20, 1)
        )`.
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
    m
        The output dimension of the neural network. According to
        the SPINN article, a total embedding dimension of `r*m` is defined. We
        then sum groups of `r` embedding dimensions to compute each output.
        Default is 1.

    !!! note
        SPINNs with `t` and `x` as inputs are best used with a
        DataGenerator with `self.cartesian_product=False` for memory
        consideration


    Returns
    -------
    spinn
        An instanciated SPINN

    Raises
    ------
    RuntimeError
        If the parameter value for eq_type is not in `["ODE", "statio_PDE",
        "nonstatio_PDE"]` and for various failing checks
    """

    if eq_type not in ["ODE", "statio_PDE", "nonstatio_PDE"]:
        raise RuntimeError("Wrong parameter value for eq_type")

    try:
        nb_inputs_declared = eqx_list[0][1]  # normally we look for 2nd ele of 1st layer
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

    spinn_mlp = _SPINN(key=key, d=d, eqx_list=eqx_list)
    spinn = SPINN(spinn_mlp=spinn_mlp, d=d, r=r, eq_type=eq_type, m=m)

    return spinn
