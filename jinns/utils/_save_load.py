"""
Implements save and load functions
"""

import pickle
import jax
import equinox as eqx

from jinns.utils._pinn import create_PINN
from jinns.utils._spinn import create_SPINN
from jinns.utils._hyperpinn import create_HYPERPINN


def function_to_string(eqx_list):
    """
    We need this transformation for eqx_list to be pickled

    From `[[eqx.nn.Linear, 2, 20],
            [jax.nn.tanh],
            [eqx.nn.Linear, 20, 20],
            [jax.nn.tanh],
            [eqx.nn.Linear, 20, 20],
            [jax.nn.tanh],
            [eqx.nn.Linear, 20, 1]` to
    `[["Linear", 2, 20],
                ["tanh"],
                ["Linear", 20, 20],
                ["tanh"],
                ["Linear", 20, 20],
                ["tanh"],
                ["Linear", 20, 1]`
    """
    return jax.tree_util.tree_map(
        lambda x: x.__name__ if hasattr(x, "__call__") else x, eqx_list
    )


def string_to_function(eqx_list_with_string):
    """
    We need this transformation for eqx_list at the loading ("unpickling")
    operation.

    From `[["Linear", 2, 20],
                ["tanh"],
                ["Linear", 20, 20],
                ["tanh"],
                ["Linear", 20, 20],
                ["tanh"],
                ["Linear", 20, 1]`
    to  `[[eqx.nn.Linear, 2, 20],
            [jax.nn.tanh],
            [eqx.nn.Linear, 20, 20],
            [jax.nn.tanh],
            [eqx.nn.Linear, 20, 20],
            [jax.nn.tanh],
            [eqx.nn.Linear, 20, 1]` to
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


def save_pinn(filename, u, params, kwargs_creation):
    """
    Save a PINN / HyperPINN / SPINN model
    This function creates 3 files, beggining by `filename`

     1. an eqx file to save the eqx.Module (the PINN, HyperPINN, ...)
     2. a pickle file for the parameters
     3. a pickle file for the arguments that have been used at PINN

     creation and that we need to reconstruct the eqx.module later on.

    Parameters
    ----------
    filename
        Filename (prefix) without extension
    u
        The PINN
    params
        The dictionary of parameters of the model.
        Typically, it is a dictionary of
        dictionaries: `eq_params` and `nn_params`, respectively the
        differential equation parameters and the neural network parameter
    kwargs_creation
        The dictionary of arguments that were used to create the PINN, e.g.
        the layers list, O/PDE type, etc.
    """
    eqx.tree_serialise_leaves(filename + "-module.eqx", u)
    with open(filename + "-parameters.pkl", "wb") as f:
        pickle.dump(params, f)
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


def load_pinn(filename, type_):
    """
    Load a PINN model. This function needs to access 3 files :
    `{filename}-module.eqx`, `{filename}-parameters.pkl` and
    `{filename}-arguments.pkl`.

    These files are created by `jinns.utils.save_pinn`.

    Note that this requires equinox v0.11.3 (currently latest version) for the
    `eqx.filter_eval_shape` to work.

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
    params_reloaded
        The reloaded parameters
    """
    with open(filename + "-arguments.pkl", "rb") as f:
        kwargs_reloaded = pickle.load(f)
    with open(filename + "-parameters.pkl", "rb") as f:
        params_reloaded = pickle.load(f)
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
    return u_reloaded, params_reloaded
