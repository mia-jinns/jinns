from functools import partial
from jaxopt import OptaxSolver, LBFGS
from optax import GradientTransformation
from jax_tqdm import scan_tqdm
import jax
import jax.numpy as jnp
from jinns.solver._seq2seq import (
    _initialize_seq2seq,
    _seq2seq_triggerer,
    _update_seq2seq_false,
)
from jinns.solver._rar import _rar_step_init, _rar_step_triggerer
from jinns.utils._utils import _check_nan_in_pytree, _tracked_parameters
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
    optimizer,
    print_loss_every=1000,
    opt_state=None,
    seq2seq=None,
    tracked_params_key_list=None,
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
    optimizer
        Can be an `optax` optimizer (e.g. `optax.adam`).
        In such case, it is wrapped in the `jaxopt.OptaxSolver` wrapper.
        Can be a `jaxopt` optimizer (e.g. `jaxopt.BFGS`) which supports the
        methods `init_state` and `update`.
        Can be a string (currently only `bfgs`), in such case a `jaxopt`
        optimizer is created with default parameters.
    print_loss_every
        Integer. Default 100. The rate at which we print the loss value in the
        gradient step loop.
    opt_state
        Default None. Provide an optional initial optional state to the
        optimizer. Not valid for all optimizers.
    seq2seq
        Default None. A dictionary with keys 'times_steps'
        and 'iter_steps' which mush have same length. The first represents
        the time steps which represents the different time interval upon
        which we perform the incremental learning. The second represents
        the number of iteration we perform in each time interval.
        The seq2seq approach we reimplements is defined in
        "Characterizing possible failure modes in physics-informed neural
        networks", A. S. Krishnapriyan, NeurIPS 2021
    tracked_params_key_list
        Default None. Otherwise it is a list of list of strings
        to access a leaf in params. Each selected leaf will be tracked
        and stored at each iteration and returned by the solve function
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
    res["stored_params"]
        A dictionary. At each key an array of the values of the parameters
        given in tracked_params_key_list is stored
    """
    params = init_params

    if isinstance(optimizer, GradientTransformation):
        optimizer = OptaxSolver(
            opt=optimizer,
            fun=loss,
            has_aux=True,
            maxiter=n_iter,
        )
    elif optimizer == "lbfgs":
        optimizer = LBFGS(fun=loss, has_aux=True, maxiter=n_iter)
    # else, we trust that the user has given a valid jaxopt optimizer

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
        opt_state = optimizer.init_state(params, batch=batch)

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

    batch = data.get_batch()
    if param_data is not None:
        batch = append_param_batch(batch, param_data.get_batch())
    _, loss_terms = loss(params, batch)

    # initialize the dict for stored parameter values
    if tracked_params_key_list is None:
        tracked_params_key_list = []
    tracked_params = _tracked_parameters(params, tracked_params_key_list)
    stored_params = jax.tree_util.tree_map(
        lambda tracked_param, param: jnp.zeros((n_iter,) + param.shape)
        if tracked_param
        else None,
        tracked_params,
        params,
    )

    # initialize the dict for stored loss values
    stored_loss_terms = jax.tree_util.tree_map(
        lambda x: jnp.zeros((n_iter)), loss_terms
    )

    @scan_tqdm(n_iter)
    def scan_func_solve_one_iter(carry, i):
        """
        Main optimization loop
        """
        batch = carry["data"].get_batch()
        if carry["param_data"] is not None:
            batch = append_param_batch(batch, carry["param_data"].get_batch())
        carry["params"], carry["opt_state"] = optimizer.update(
            params=carry["params"], state=carry["opt_state"], batch=batch
        )

        # check if any of the parameters is NaN
        carry["last_non_nan_params"] = jax.lax.cond(
            _check_nan_in_pytree(carry["params"]),
            lambda _: carry["last_non_nan_params"],
            lambda _: carry["params"],
            None,
        )

        total_loss_val, loss_terms = loss(carry["params"], batch)

        # Print loss during optimization
        _ = jax.lax.cond(
            i % print_loss_every == 0,
            lambda _: jax.debug.print(
                "Iteration {i}: loss value = " "{total_loss_val}",
                i=i,
                total_loss_val=total_loss_val,
            ),
            lambda _: None,
            (None,),
        )

        # optionnal seq2seq
        if seq2seq is not None:
            carry = _seq2seq_triggerer(
                carry, i, _update_seq2seq_true, _update_seq2seq_false
            )
        else:
            carry["curr_seq"] = -1

        # optional residual adaptative refinement
        if carry["data"].rar_parameters is not None:
            carry = _rar_step_triggerer(carry, i, _rar_step_true, _rar_step_false)

        # saving selected parameters values
        carry["stored_params"] = jax.tree_util.tree_map(
            lambda stored_value, param, tracked_param: jax.lax.cond(
                tracked_param,
                lambda ope: ope[0].at[i].set(ope[1]),
                lambda ope: ope[0],
                (stored_value, param),
            ),
            carry["stored_params"],
            carry["params"],
            tracked_params,
        )

        # saving values of each loss term
        carry["stored_loss_terms"] = jax.tree_util.tree_map(
            lambda stored_term, loss_term: stored_term.at[i].set(loss_term),
            carry["stored_loss_terms"],
            loss_terms,
        )

        return carry, [total_loss_val]

    res, accu = jax.lax.scan(
        scan_func_solve_one_iter,
        {
            "params": init_params,
            "last_non_nan_params": init_params.copy(),
            "data": data,
            "curr_seq": curr_seq,
            "seq2seq": seq2seq,
            "stored_params": stored_params,
            "stored_loss_terms": stored_loss_terms,
            "loss": loss,
            "param_data": param_data,
            "opt_state": opt_state,
        },
        jnp.arange(n_iter),
    )

    jax.debug.print(
        "Iteration {i}: loss value = " "{total_loss_val}",
        i=n_iter,
        total_loss_val=accu[-1][-1],
    )

    params = res["params"]
    last_non_nan_params = res["last_non_nan_params"]
    opt_state = res["opt_state"]
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
        res["stored_params"],
    )
