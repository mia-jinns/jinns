from functools import partial
import jax

from jinns.parameters import Params


def test_success():
    """
    This would fail with ValueError with jinns prior to 1.5.1 and
    equinox>=0.13.0 and JAX>=0.7.0 because of dict with string
    which is unhashable while they should be since we require it as static
    """

    params = Params(nn_params=(0, 0), eq_params={"a": 1, "b": 1})

    @partial(jax.jit, static_argnames=["params"])
    def f(a, params):
        return jax.tree.map(lambda t: t * a, params)

    f(10, params)
