from functools import partial
import jax

import jinns
from jinns.parameters import Params, _update_eq_params


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

    print(f(10, params))
    assert f(10, params).eq_params.a == 10


def test_update_eq_params():
    key = jax.random.PRNGKey(0)
    params = jinns.parameters.Params(
        nn_params=None,
        eq_params={"nu": None},
    )
    key, subkey = jax.random.split(key)
    np = 100
    param_batch_size = 10  # must be equal to batch size of the main DataGenerator
    param_train_data = jinns.data.DataGeneratorParameter(
        subkey,
        np,
        param_batch_size,
        {"nu": (2e-4, 1.9e-3)},
        method="grid",
    )

    param_train_data, param_batch = param_train_data.get_batch()

    params = _update_eq_params(params, param_batch)

    assert params.eq_params.nu.shape == (10, 1)
