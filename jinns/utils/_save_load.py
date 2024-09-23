"""
Implements save and load functions
"""

from typing import Callable, Literal
import pickle
import jax
import equinox as eqx

from jinns.utils._pinn import create_PINN, PINN
from jinns.utils._spinn import create_SPINN, SPINN
from jinns.utils._hyperpinn import create_HYPERPINN, HYPERPINN
from jinns.parameters._params import Params


def function_to_string(
    eqx_list: tuple[tuple[Callable, int, int] | Callable, ...]
) -> tuple[tuple[str, int, int] | str, ...]:
    """
    We need this transformation for eqx_list to be pickled

    From `((eqx.nn.Linear, 2, 20),
            (jax.nn.tanh),
            (eqx.nn.Linear, 20, 20),
            (jax.nn.tanh),
            (eqx.nn.Linear, 20, 20),
            (jax.nn.tanh),
            (eqx.nn.Linear, 20, 1))` to
    `(("Linear", 2, 20),
                ("tanh"),
                ("Linear", 20, 20),
                ("tanh"),
                ("Linear", 20, 20),
                ("tanh"),
                ("Linear", 20, 1))`
    """
    return jax.tree_util.tree_map(
        lambda x: x.__name__ if hasattr(x, "__call__") else x, eqx_list
    )


def string_to_function(
    eqx_list_with_string: tuple[tuple[str, int, int] | str, ...]
) -> tuple[tuple[Callable, int, int] | Callable, ...]:
    """
    We need this transformation for eqx_list at the loading ("unpickling")
    operation.

    From `(("Linear", 2, 20),
                ("tanh"),
                ("Linear", 20, 20),
                ("tanh"),
                ("Linear", 20, 20),
                ("tanh"),
                ("Linear", 20, 1))`
    to  `((eqx.nn.Linear, 2, 20),
            (jax.nn.tanh),
            (eqx.nn.Linear, 20, 20),
            (jax.nn.tanh),
            (eqx.nn.Linear, 20, 20),
            (jax.nn.tanh),
            (eqx.nn.Linear, 20, 1))`
    """

    def _str_to_fun(l):
        try:
            try:
                try:
                    return getattr(jax.nn, l)
                except AttributeError:
                    return getattr(jax.numpy, l)
            except AttributeError:
                return getattr(eqx.nn, l)
        except AttributeError as exc:
            raise ValueError(
                "Activation functions must be from jax.nn or jax.numpy,"
                + "or layers must be eqx.nn layers"
            ) from exc

    return jax.tree_util.tree_map(
        lambda x: _str_to_fun(x) if isinstance(x, str) else x, eqx_list_with_string
    )


def save_pinn(
    filename: str, u: PINN | HYPERPINN | SPINN, params: Params, kwargs_creation
):
    """
    Save a PINN / HyperPINN / SPINN model
    This function creates 3 files, beggining by `filename`

     1. an eqx file to save the eqx.Module (the PINN, HyperPINN, ...)
     2. a pickle file for the parameters of the equation
     3. a pickle file for the arguments that have been used at PINN creation and that we need to reconstruct the eqx.module later on.

    Note that the equation parameters (typically `Params.eq_params`) go in the
    pickle file while the neural network parameters (typically
    `Params.nn_params`) go in the `"*-module.eqx"` file (normal behaviour with
    `eqx.tree_serialise_leaves`. Currently, equation parameters are saved apart
    because the initial type of attribute `params` in PINN / HYPERPINN / SPINN
    is not `Params` but `PyTree` as inherited from `eqx.partition`.
    Therefore, if we want to ensure a proper serialization/deserialization we
    cannot save a `Params` object at this attribute field ; the `Params` object
    must be split into `Params.nn_params` (type `PyTree`)
    and `Params.eq_params` (type `dict`).

    Parameters
    ----------
    filename
        Filename (prefix) without extension
    u
        The PINN
    params
        Params
    kwargs_creation
        The dictionary of arguments that were used to create the PINN, e.g.
        the layers list, O/PDE type, etc.
    """
    if isinstance(u, HYPERPINN):
        u = eqx.tree_at(lambda m: m.params_hyper, u, params)
    elif isinstance(u, (PINN, SPINN)):
        u = eqx.tree_at(lambda m: m.params, u, params)
    eqx.tree_serialise_leaves(filename + "-module.eqx", u)

    if isinstance(params, Params):
        with open(filename + "-eq_params.pkl", "wb") as f:
            pickle.dump(params.eq_params, f)

    kwargs_creation = kwargs_creation.copy()  # avoid side-effect that would be
    # very probably harmless anyway

    # we now need to transform the functions in eqx_list into strings otherwise
    # it could not be pickled
    kwargs_creation["eqx_list"] = function_to_string(kwargs_creation["eqx_list"])

    # same thing if there is an hypernetwork:
    try:
        kwargs_creation["eqx_list_hyper"] = function_to_string(
            kwargs_creation["eqx_list_hyper"]
        )
    except KeyError:
        pass

    with open(filename + "-arguments.pkl", "wb") as f:
        pickle.dump(kwargs_creation, f)


def load_pinn(filename: str, type_: Literal["pinn", "hyperpinn", "spinn"]):
    """
    Load a PINN model. This function needs to access 3 files :
    `{filename}-module.eqx`, `{filename}-parameters.pkl` and
    `{filename}-arguments.pkl`.

    These files are created by `jinns.utils.save_pinn`.

    Note that this requires equinox>v0.11.3 for the
    `eqx.filter_eval_shape` to work.

    See note in `save_pinn` for more details about the saving process

    Parameters
    ----------
    filename
        Filename (prefix) without extension.
    type_
        Type of model to load. Must be in ["pinn", "hyperpinn", "spinn"].

    Returns
    -------
    u_reloaded
        The reloaded PINN
    params
        The reloaded parameters
    """
    with open(filename + "-arguments.pkl", "rb") as f:
        kwargs_reloaded = pickle.load(f)
    try:
        with open(filename + "-eq_params.pkl", "rb") as f:
            eq_params_reloaded = pickle.load(f)
    except FileNotFoundError:
        eq_params_reloaded = None
        print("No pickle file for equation parameters found!")
    kwargs_reloaded["eqx_list"] = string_to_function(kwargs_reloaded["eqx_list"])
    if type_ == "pinn":
        # next line creates a shallow model, the jax arrays are just shapes and
        # not populated, this just recreates the correct pytree structure
        u_reloaded_shallow = eqx.filter_eval_shape(create_PINN, **kwargs_reloaded)
    elif type_ == "spinn":
        u_reloaded_shallow = eqx.filter_eval_shape(create_SPINN, **kwargs_reloaded)
    elif type_ == "hyperpinn":
        kwargs_reloaded["eqx_list_hyper"] = string_to_function(
            kwargs_reloaded["eqx_list_hyper"]
        )
        u_reloaded_shallow = eqx.filter_eval_shape(create_HYPERPINN, **kwargs_reloaded)
    else:
        raise ValueError(f"{type_} is not valid")
    # now the empty structure is populated with the actual saved array values
    # stored in the eqx file
    u_reloaded = eqx.tree_deserialise_leaves(
        filename + "-module.eqx", u_reloaded_shallow
    )
    if eq_params_reloaded is None:
        params = u_reloaded.init_params()
    else:
        params = Params(
            nn_params=u_reloaded.init_params(), eq_params=eq_params_reloaded
        )
    return u_reloaded, params
