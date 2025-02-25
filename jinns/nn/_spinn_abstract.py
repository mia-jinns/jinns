from typing import Union, Callable, Any
from dataclasses import InitVar
from jaxtyping import PyTree, Float, Array
import jax
import jax.numpy as jnp
import equinox as eqx

from jinns.parameters._params import Params, ParamsDict


class SPINNAbstract(eqx.Module):
    """
    A SPINN object compatible with the rest of jinns.
    This is typically created with `create_SPINN`.

    Parameters
    ----------
    d : int
        The number of dimensions to treat separately, including time `t` if
        used for non-stationnary equations.

    """

    d: int = eqx.field(static=True, kw_only=True)
    r: int = eqx.field(static=True, kw_only=True)
    eq_type: str = eqx.field(static=True, kw_only=True)
    m: int = eqx.field(static=True, kw_only=True, default=1)

    filter_spec: PyTree[Union[bool, Callable[[Any], bool]]] = eqx.field(
        static=True, kw_only=True, default=eqx.is_inexact_array
    )
    eqx_spinn_network: InitVar[eqx.Module] = eqx.field(kw_only=True)

    init_params: PyTree = eqx.field(init=False)
    static: PyTree = eqx.field(init=False, static=True)

    def __post_init__(self, eqx_spinn_network):
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
