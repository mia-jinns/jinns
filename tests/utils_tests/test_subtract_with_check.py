import pytest
import jax
import jax.numpy as jnp
from jinns.utils._utils import _subtract_with_check


def test_corrected_broadcast1():

    with pytest.warns(UserWarning):
        res = _subtract_with_check(
            jnp.ones((2,)),
            jnp.ones((2, 1)),
        )
    assert res.shape == (2, 1)


def test_corrected_broadcast2():
    with pytest.warns(UserWarning):
        res = _subtract_with_check(
            jnp.ones((2, 1)),
            jnp.ones((2,)),
        )
    assert res.shape == (2,)
