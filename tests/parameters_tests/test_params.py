import pytest
from functools import partial
import jax

import jinns
from jinns.parameters import Params, update_eq_params


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
        key=subkey,
        n=np,
        param_batch_size=param_batch_size,
        param_ranges={"nu": (2e-4, 1.9e-3)},
        method="grid",
    )

    param_train_data, param_batch = param_train_data.get_batch()

    params = update_eq_params(params, param_batch)

    assert params.eq_params.nu.shape == (10, 1)


def test_metaclass1():
    """
    Here we test the metaclass in its main role of providing a single EqParams
    class template so that different instance of params really are of the same
    class!
    """
    d_float = {"theta": 0.0, "beta": 1.0}
    d_bool = {"theta": True, "beta": False}

    tree1 = Params(nn_params=None, eq_params=d_float)
    tree2 = Params(nn_params=None, eq_params=d_bool)

    res = jax.tree.map(lambda p1, p2: p2 if p1 is True else None, tree2, tree1)
    assert jax.tree.leaves(res) == [0.0]


def test_metaclass2():
    """
    Here we test the metaclass in its main role of providing a single EqParams
    class template so that different instance of params really are of the same
    class!
    """
    d_float = {"theta": 0.0, "beta": 1.0}
    d_bool = {"theta": True, "beta": False, "gamma": False}

    _ = Params(nn_params=None, eq_params=d_float)
    with pytest.raises(ValueError):
        # this fails because it does not comply the EqParams of the problem!
        _ = Params(nn_params=None, eq_params=d_bool)
