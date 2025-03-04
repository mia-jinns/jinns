"""
Implements save and load functions
"""

from typing import Callable, Literal
import pickle
import jax
import equinox as eqx

from jinns.nn._pinn import PINN
from jinns.nn._spinn import SPINN
from jinns.nn._mlp import PINN_MLP
from jinns.nn._spinn_mlp import SPINN_MLP
from jinns.nn._hyperpinn import HyperPINN
from jinns.parameters._params import Params, ParamsDict


def function_to_string(
    eqx_list: tuple[tuple[Callable, int, int] | Callable, ...],
) -> tuple[tuple[str, int, int] | str, ...]:
    """
    We need this transformation for eqx_list to be pickled

    From `((eqx.nn.Linear, 2, 20),
            (jax.nn.tanh,),
            (eqx.nn.Linear, 20, 20),
            (jax.nn.tanh,),
            (eqx.nn.Linear, 20, 20),
            (jax.nn.tanh,),
            (eqx.nn.Linear, 20, 1))` to
    `(("Linear", 2, 20),
                ("tanh",),
                ("Linear", 20, 20),
                ("tanh",),
                ("Linear", 20, 20),
                ("tanh",),
                ("Linear", 20, 1))`
    """
    return jax.tree_util.tree_map(
        lambda x: x.__name__ if hasattr(x, "__call__") else x, eqx_list
    )


def string_to_function(
    eqx_list_with_string: tuple[tuple[str, int, int] | str, ...],
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
    filename: str,
    u: PINN | HyperPINN | SPINN,
    params: Params | ParamsDict,
    kwargs_creation,
):
    """
    Save a PINN / HyperPINN / SPINN model
    This function creates 3 files, beggining by `filename`

     1. an eqx file to save the eqx.Module (the PINN, HyperPINN, ...)
     2. a pickle file for the parameters of the equation
     3. a pickle file for the arguments that have been used at PINN creation
     and that we need to reconstruct the eqx.module later on.

    Note that the equation parameters `Params.eq_params` go in the
    pickle file while the neural network parameters `Params.nn_params` go in
    the `"*-module.eqx"` file (normal behaviour with `eqx.
    tree_serialise_leaves`).

    Equation parameters are saved apart because the initial type of attribute
    `params` in PINN / HyperPINN / SPINN is not `Params` nor `ParamsDict`
    but `PyTree` as inherited from `eqx.partition`.
    Therefore, if we want to ensure a proper serialization/deserialization:
    - we cannot save a `Params` object at this
      attribute field ; the `Params` object must be split into `Params.nn_params`
      (type `PyTree`) and `Params.eq_params` (type `dict`).
    - in the case of a `ParamsDict` we cannot save `ParamsDict.nn_params` at
      the attribute field `params` because it is not a `PyTree` (as expected in
      the PINN / HyperPINN / SPINN signature) but it is still a dictionary.

    Parameters
    ----------
    filename
        Filename (prefix) without extension
    u
        The PINN
    params
        Params or ParamsDict to be save
    kwargs_creation
        The dictionary of arguments that were used to create the PINN, e.g.
        the layers list, O/PDE type, etc.
    """
    if isinstance(params, Params):
        if isinstance(u, HyperPINN):
            u = eqx.tree_at(lambda m: m.init_params_hyper, u, params)
        elif isinstance(u, (PINN, SPINN)):
            u = eqx.tree_at(lambda m: m.init_params, u, params)
        eqx.tree_serialise_leaves(filename + "-module.eqx", u)

    elif isinstance(params, ParamsDict):
        for key, params_ in params.nn_params.items():
            if isinstance(u, HyperPINN):
                u = eqx.tree_at(lambda m: m.init_params_hyper, u, params_)
            elif isinstance(u, (PINN, SPINN)):
                u = eqx.tree_at(lambda m: m.init_params, u, params_)
            eqx.tree_serialise_leaves(filename + f"-module_{key}.eqx", u)

    else:
        raise ValueError("The parameters to be saved must be a Params or a ParamsDict")

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


def load_pinn(
    filename: str,
    type_: Literal["pinn_mlp", "hyperpinn", "spinn_mlp"],
    key_list_for_paramsdict: list[str] = None,
) -> tuple[eqx.Module, Params | ParamsDict]:
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
        Type of model to load. Must be in ["pinn_mlp", "hyperpinn", "spinn"].
    key_list_for_paramsdict
        Pass the name of the keys of the dictionnary `ParamsDict.nn_params`. Default is None. In this case, we expect to retrieve a ParamsDict.

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
        eq_params_reloaded = {}
        print("No pickle file for equation parameters found!")
    kwargs_reloaded["eqx_list"] = string_to_function(kwargs_reloaded["eqx_list"])
    if type_ == "pinn_mlp":
        # next line creates a shallow model, the jax arrays are just shapes and
        # not populated, this just recreates the correct pytree structure
        u_reloaded_shallow, _ = eqx.filter_eval_shape(
            PINN_MLP.create, **kwargs_reloaded
        )
    elif type_ == "spinn_mlp":
        u_reloaded_shallow, _ = eqx.filter_eval_shape(
            SPINN_MLP.create, **kwargs_reloaded
        )
    elif type_ == "hyperpinn":
        kwargs_reloaded["eqx_list_hyper"] = string_to_function(
            kwargs_reloaded["eqx_list_hyper"]
        )
        u_reloaded_shallow, _ = eqx.filter_eval_shape(
            HyperPINN.create, **kwargs_reloaded
        )
    else:
        raise ValueError(f"{type_} is not valid")
    if key_list_for_paramsdict is None:
        # now the empty structure is populated with the actual saved array values
        # stored in the eqx file
        u_reloaded = eqx.tree_deserialise_leaves(
            filename + "-module.eqx", u_reloaded_shallow
        )
        if isinstance(u_reloaded, HyperPINN):
            params = Params(
                nn_params=u_reloaded.init_params_hyper, eq_params=eq_params_reloaded
            )
        elif isinstance(u_reloaded, (PINN, SPINN)):
            params = Params(
                nn_params=u_reloaded.init_params, eq_params=eq_params_reloaded
            )
    else:
        nn_params_dict = {}
        for key in key_list_for_paramsdict:
            u_reloaded = eqx.tree_deserialise_leaves(
                filename + f"-module_{key}.eqx", u_reloaded_shallow
            )
            if isinstance(u_reloaded, HyperPINN):
                nn_params_dict[key] = u_reloaded.init_params_hyper
            elif isinstance(u_reloaded, (PINN, SPINN)):
                nn_params_dict[key] = u_reloaded.init_params
        params = ParamsDict(nn_params=nn_params_dict, eq_params=eq_params_reloaded)
    return u_reloaded, params
