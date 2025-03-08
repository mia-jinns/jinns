from typing import Union, Callable, Any
from dataclasses import InitVar
from jaxtyping import PyTree, Float, Array
import jax
import jax.numpy as jnp
import equinox as eqx

from jinns.parameters._params import Params, ParamsDict


class SPINN(eqx.Module):
    """
    A Separable PINN object compatible with the rest of jinns.

    Parameters
    ----------
    d : int
        The number of dimensions to treat separately, including time `t` if
        used for non-stationnary equations.
    r : int
        An integer. The dimension of the embedding.
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
        Default is `eqx.is_inexact_array`. This tells Jinns what to consider as
        a trainable parameter. Quoting from equinox documentation:
        a PyTree whose structure should be a prefix of the structure of pytree.
        Each of its leaves should either be 1) True, in which case the leaf or
        subtree is kept; 2) False, in which case the leaf or subtree is
        replaced with replace; 3) a callable Leaf -> bool, in which case this is evaluated on the leaf or mapped over the subtree, and the leaf kept or replaced as appropriate.
    eqx_spinn_network : eqx.Module
        The actual neural network instanciated as an eqx.Module. It should be
        an architecture taking `d` inputs and returning `d` times an embedding
        of dimension `r`*`m`. See the Separable PINN paper for more details.

    """

    d: int = eqx.field(static=True, kw_only=True)
    r: int = eqx.field(static=True, kw_only=True)
    eq_type: str = eqx.field(static=True, kw_only=True)
    m: int = eqx.field(static=True, kw_only=True, default=1)

    filter_spec: PyTree[Union[bool, Callable[[Any], bool]]] = eqx.field(
        static=True, kw_only=True, default=None
    )
    eqx_spinn_network: InitVar[eqx.Module] = eqx.field(kw_only=True)

    init_params: PyTree = eqx.field(init=False)
    static: PyTree = eqx.field(init=False, static=True)

    def __post_init__(self, eqx_spinn_network):

        if self.filter_spec is None:
            self.filter_spec = eqx.is_inexact_array

        self.init_params, self.static = eqx.partition(
            eqx_spinn_network, self.filter_spec
        )

    def __call__(
        self,
        t_x: Float[Array, "batch_size 1+dim"],
        params: Params | ParamsDict | PyTree,
    ) -> Float[Array, "output_dim"]:
        """
        Evaluate the SPINN on some inputs with some params.
        """
        try:
            spinn = eqx.combine(params.nn_params, self.static)
        except (KeyError, AttributeError, TypeError) as e:
            spinn = eqx.combine(params, self.static)
        v_model = jax.vmap(spinn)
        res = v_model(t_x)

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
