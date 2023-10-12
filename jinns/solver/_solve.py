from functools import partial
from jaxopt import OptaxSolver
from jax_tqdm import scan_tqdm
import jax
from jax import jit
import jax.numpy as jnp
from jinns.solver._seq2seq import (
    _initialize_seq2seq,
    _seq2seq_triggerer,
    _update_seq2seq_false,
)
from jinns.solver._rar import _rar_step_init, _rar_step_triggerer
from jinns.utils._utils import _check_nan_in_pytree
from jinns.data._DataGenerators import (
    DataGeneratorODE,
    CubicMeshPDEStatio,
    CubicMeshPDENonStatio,
    append_param_batch,
)


def solve(
    n_iter,
    init_params,
    data,
    loss,
    optax_solver,
    opt_state=None,
    seq2seq=None,
    accu_vars=[],
    param_data=None,
):
    """
    Performs the optimization process via stochastic gradient descent
    algorithm. We minimize the function defined `loss.evaluate()` with
    respect to the learnable parameters of the problem whose initial values
    are given in `init_params`.


    Parameters
    ----------
    n_iter
        The number of iterations in the optimization
    init_params
        The initial dictionary of parameters. Typically, it is a dictionary of
        dictionaries: `eq_params` and `nn_params``, respectively the
        differential equation parameters and the neural network parameter
    data
        A DataGenerator object which implements a `get_batch()`
        method which returns a 3-tuple with (omega_grid, omega_border, time grid).
        It must be jittable (e.g. implements via a pytree
        registration)
    loss
        A loss object (e.g. a LossODE, SystemLossODE, LossPDEStatio [...]
        object). It must be jittable (e.g. implements via a pytree
        registration)
    optax_solver
        An optax solver (e.g. adam with a given step-size)
    opt_state
        Default None. Provide an optional initial optional state to the
        optimizer
    seq2seq
        Default None. A dictionary with keys 'times_steps'
        and 'iter_steps' which mush have same length. The first represents
        the time steps which represents the different time interval upon
        which we perform the incremental learning. The second represents
        the number of iteration we perform in each time interval.
        The seq2seq approach we reimplements is defined in
        "Characterizing possible failure modes in physics-informed neural
        networks", A. S. Krishnapriyan, NeurIPS 2021
    accu_vars
        Default []. Otherwise it is a list of list of (integers, strings)
        to access a leaf in init_params (and later on params). Each
        selected leaf will be tracked and stored at each iteration and
        returned by the solve function
    param_data
        Default None. A DataGeneratorParameter object which can be used to
        sample equation parameters.


    Returns
    -------
    params
        The last non NaN value of the dictionaries of parameters at then end of the
        optimization process
    accu[0, :]
        An array of the total loss term along the gradient steps
    res["stored_loss_terms"]
        A dictionary. At each key an array of the values of a given loss
        term is stored
    data
        The input data object
    loss
        The input loss object
    opt_state
        The final optimized state
    res["stored_values"]
        A dictionary. At each key an array of the values of the parameters
        given in accu_vars is stored
    """
    params = init_params

    # Wrap the optax solver with jaxopt
    optax_solver = OptaxSolver(
        opt=optax_solver,
        fun=loss,
        has_aux=True,  # because the objective has aux output
        maxiter=n_iter,
    )

    if param_data is not None:
        if (
            (
                isinstance(data, DataGeneratorODE)
                and param_data.param_batch_size != data.temporal_batch_size
            )
            or (
                isinstance(data, CubicMeshPDEStatio)
                and param_data.param_batch_size != data.omega_batch_size
            )
            or (
                isinstance(data, CubicMeshPDENonStatio)
                and param_data.param_batch_size
                != data.omega_batch_size * data.temporal_batch_size
            )
        ):
            raise ValueError(
                "Optional param_data.param_batch_size must be"
                " equal to data.temporal_batch_size or data.omega_batch_size or"
                " the product of both dependeing on the type of the main"
                " datagenerator"
            )

    if opt_state is None:
        batch = data.get_batch()
        if param_data is not None:
            batch = append_param_batch(batch, param_data.get_batch())
        opt_state = optax_solver.init_state(params, batch=batch)

    curr_seq = 0
    if seq2seq is not None:
        assert data.method == "uniform", "data.method must be uniform if"
        " using seq2seq learning !"

        _update_seq2seq_true = _initialize_seq2seq(loss, data, seq2seq, opt_state)

    else:
        _update_seq2seq_true = None

    if data.rar_parameters is not None:
        _rar_step_true, _rar_step_false = _rar_step_init(
            data.rar_parameters["sample_size"],
            data.rar_parameters["selected_sample_size"],
        )
        data.rar_parameters["iter_from_last_sampling"] = 0

    # initialize the dict for stored parameter values
    stored_values = {}
    for params_leaf_path in accu_vars:
        stored_values["-".join(map(str, params_leaf_path))] = jnp.zeros((n_iter,))

    # initialize the dict for stored loss values
    stored_loss_terms = {}
    batch = data.get_batch()
    if param_data is not None:
        batch = append_param_batch(batch, param_data.get_batch())

    _, loss_terms = loss(params, batch)
    for loss_name, _ in loss_terms.items():
        stored_loss_terms[loss_name] = jnp.zeros((n_iter,))

    def nested_get(dic, keys):
        for k in keys:
            dic = dic[k]
        return dic

    @scan_tqdm(n_iter)
    def scan_func_solve_one_iter(carry, i):
        """
        Main optimization loop
        """
        batch = carry["data"].get_batch()
        if carry["param_data"] is not None:
            batch = append_param_batch(batch, carry["param_data"].get_batch())
        carry["params"], carry["opt_state"] = optax_solver.update(
            params=carry["params"], state=carry["state"], batch=batch
        )

        # check if any of the parameters is NaN
        carry["last_non_nan_params"] = jax.lax.cond(
            _check_nan_in_pytree(carry["params"]),
            lambda _: carry["last_non_nan_params"],
            lambda _: carry["params"],
            None,
        )

        total_loss_val, loss_terms = loss(carry["params"], batch)

        if seq2seq is not None:
            carry = _seq2seq_triggerer(
                carry, i, _update_seq2seq_true, _update_seq2seq_false
            )
        else:
            carry["curr_seq"] = -1

        if carry["data"].rar_parameters is not None:
            carry = _rar_step_triggerer(carry, i, _rar_step_true, _rar_step_false)

        # saving selected parameters values with accumulator
        accu = [total_loss_val]
        for params_leaf_path in accu_vars:
            carry["stored_values"]["-".join(map(str, params_leaf_path))] = (
                carry["stored_values"]["-".join(map(str, params_leaf_path))]
                .at[i]
                .set(nested_get(carry["params"], params_leaf_path).squeeze())
            )

        # saving values of each loss term
        for loss_name, loss_value in loss_terms.items():
            carry["stored_loss_terms"][loss_name] = (
                carry["stored_loss_terms"][loss_name].at[i].set(loss_value)
            )

        return carry, accu

    res, accu = jax.lax.scan(
        scan_func_solve_one_iter,
        {
            "params": init_params,
            "last_non_nan_params": init_params.copy(),
            "state": opt_state,
            "data": data,
            "curr_seq": curr_seq,
            "seq2seq": seq2seq,
            "stored_values": stored_values,
            "stored_loss_terms": stored_loss_terms,
            "loss": loss,
            "param_data": param_data,
            "opt_state": opt_state,
        },
        jnp.arange(n_iter),
    )

    params = res["params"]
    last_non_nan_params = res["last_non_nan_params"]
    opt_state = res["state"]
    data = res["data"]
    loss = res["loss"]

    accu = jnp.array(accu)

    return (
        last_non_nan_params,
        accu[0, :],
        res["stored_loss_terms"],
        data,
        loss,
        opt_state,
        res["stored_values"],
    )
