import pytest
import os

"""
Here we set up the env variable to run on GPU if available or on CPU. Such set
up is done in conftest.py since it needs to be done before import JAX
"""


def pytest_addoption(parser):
    parser.addoption(
        "--gpu",
        action="store_true",
        default=False,
        help="whether to run test scripts on GPU",
    )


def pytest_configure(config):
    gpu = config.getoption("--gpu")
    if not gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""


# Note that the we cannot get the option using fixtures as suggested by the
# pytest doc since the option would then be accessed after jax import, which is
# too late. See:

# @pytest.fixture
# def cmdopt(request):
#    return request.config.getoption("--gpu")
#
# @pytest.fixture(autouse=True)
# def set_device(cmdopt):
#    gpu = cmdopt
#    if not gpu:
#        os.environ["CUDA_VISIBLE_DEVICES"]=""
#    return 0
