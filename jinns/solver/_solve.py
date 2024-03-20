"""
This modules implements the main `solve()` function of jinns which
handles the optimization process
"""

import optax
from tqdm import tqdm
from jax_tqdm import scan_tqdm
import jax
from jax import jit
import jax.numpy as jnp
from jinns.solver._seq2seq import trigger_seq2seq, initialize_seq2seq
from jinns.solver._rar import init_rar, trigger_rar
from jinns.utils._utils import _check_nan_in_pytree, _tracked_parameters
from jinns.data._DataGenerators import (
    DataGeneratorODE,
    CubicMeshPDEStatio,
    CubicMeshPDENonStatio,
    append_param_batch,
    append_obs_batch,
)

from functools import partial


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
    obs_data=None,
    obs_batch_sharding=None,
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
        An `optax` optimizer (e.g. `optax.adam`).
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
    obs_data
        Default None. A DataGeneratorObservations object which can be used to
        sample minibatches of observations
    obs_batch_sharding
        Default None. An optional sharding object to constraint the obs_batch.
        Typically, a SingleDeviceSharding(gpu_device) when obs_data has been
        created with sharding_device=SingleDeviceSharding(cpu_device) to avoid
        loading on GPU huge datasets of observations

    Returns
    -------
    params
        The last non NaN value of the dictionaries of parameters at then end of the
        optimization process
    total_loss_values
        An array of the total loss term along the gradient steps
    stored_loss_terms
        A dictionary. At each key an array of the values of a given loss
        term is stored
    data
        The input data object
    loss
        The input loss object
    opt_state
        The final optimized state
    stored_params
        A dictionary. At each key an array of the values of the parameters
        given in tracked_params_key_list is stored
    """
    params = init_params
    last_non_nan_params = init_params.copy()

    if param_data is not None:
        if (
            (
                isinstance(data, DataGeneratorODE)
                and param_data.param_batch_size != data.temporal_batch_size
                and obs_data.obs_batch_size != data.temporal_batch_size
            )
            or (
                isinstance(data, CubicMeshPDEStatio)
                and not isinstance(data, CubicMeshPDENonStatio)
                and param_data.param_batch_size != data.omega_batch_size
                and obs_data.obs_batch_size != data.omega_batch_size
            )
            or (
                isinstance(data, CubicMeshPDENonStatio)
                and param_data.param_batch_size
                != data.omega_batch_size * data.temporal_batch_size
                and obs_data.obs_batch_size
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
        opt_state = optimizer.init(params)

    # RAR sampling init (ouside scanned function to avoid dynamic slice error)
    # If RAR is not used the _rar_step_*() are juste None and data is unchanged
    data, _rar_step_true, _rar_step_false = init_rar(data)

    # Seq2seq
    curr_seq = 0
    if seq2seq is not None:
        assert (
            data.method == "uniform"
        ), "data.method must be uniform if using seq2seq learning !"
        data, opt_state = initialize_seq2seq(loss, data, seq2seq, opt_state)

    total_loss_values = jnp.zeros((n_iter))
    # depending on obs_batch_sharding we will get the simple get_batch or the
    # get_batch with device_put, the latter is not jittable
    get_batch = get_get_batch(obs_batch_sharding)

    # initialize the dict for stored parameter values
    # we need to get a loss_term to init stuff
    batch_ini, data, param_data, obs_data = get_batch(data, param_data, obs_data)
    _, loss_terms = loss(params, batch_ini)
    if tracked_params_key_list is None:
        tracked_params_key_list = []
    tracked_params = _tracked_parameters(params, tracked_params_key_list)
    stored_params = jax.tree_util.tree_map(
        lambda tracked_param, param: (
            jnp.zeros((n_iter,) + param.shape) if tracked_param else None
        ),
        tracked_params,
        params,
    )

    # initialize the dict for stored loss values
    stored_loss_terms = jax.tree_util.tree_map(
        lambda x: jnp.zeros((n_iter)), loss_terms
    )

    carry = (
        init_params,
        init_params.copy(),
        data,
        curr_seq,
        seq2seq,
        stored_params,
        stored_loss_terms,
        loss,
        param_data,
        obs_data,
        opt_state,
        total_loss_values,
    )

    def one_iteration(carry, i):
        (
            params,
            last_non_nan_params,
            data,
            curr_seq,
            seq2seq,
            stored_params,
            stored_loss_terms,
            loss,
            param_data,
            obs_data,
            opt_state,
            total_loss_values,
        ) = carry
        batch, data, param_data, obs_data = get_batch(data, param_data, obs_data)

        (
            loss_val,
            loss_terms,
            params,
            opt_state,
            last_non_nan_params,
        ) = gradient_step(
            loss, optimizer, batch, params, opt_state, last_non_nan_params
        )

        # Print loss during optimization
        print_fn(i, loss_val, print_loss_every)

        # Trigger RAR
        loss, params, data = trigger_rar(
            i, loss, params, data, _rar_step_true, _rar_step_false
        )

        # Trigger seq2seq
        loss, params, data, opt_state, curr_seq, seq2seq = trigger_seq2seq(
            i, loss, params, data, opt_state, curr_seq, seq2seq
        )

        # save loss value and selected parameters
        stored_params, stored_loss_terms, total_loss_values = store_loss_and_params(
            i,
            params,
            stored_params,
            stored_loss_terms,
            total_loss_values,
            loss_val,
            loss_terms,
            tracked_params,
        )
        return (
            params,
            last_non_nan_params,
            data,
            curr_seq,
            seq2seq,
            stored_params,
            stored_loss_terms,
            loss,
            param_data,
            obs_data,
            opt_state,
            total_loss_values,
        ), None

    # Main optimization loop. We use the fully scanned (fully jitted) version
    # if no mixing devices. Otherwise we use the for loop. Here devices only
    # concern obs_batch, but it could lead to more complex scheme in the future
    if obs_batch_sharding is not None:
        for i in tqdm(range(n_iter)):
            carry, _ = one_iteration(carry, i)
    else:
        carry, _ = jax.lax.scan(
            scan_tqdm(n_iter)(one_iteration),
            carry,
            jnp.arange(n_iter),
        )

    (
        init_params,
        last_non_nan_params,
        data,
        curr_seq,
        seq2seq,
        stored_params,
        stored_loss_terms,
        loss,
        param_data,
        obs_data,
        opt_state,
        total_loss_values,
    ) = carry

    jax.debug.print(
        "Iteration {i}: loss value = {total_loss_val}",
        i=n_iter,
        total_loss_val=total_loss_values[-1],
    )

    return (
        last_non_nan_params,
        total_loss_values,
        stored_loss_terms,
        data,
        loss,
        opt_state,
        stored_params,
    )


@partial(jit, static_argnames=["loss", "optimizer"])
def gradient_step(loss, optimizer, batch, params, opt_state, last_non_nan_params):
    """
    loss and optimizer cannot be jit-ted.
    """
    value_grad_loss = jax.value_and_grad(loss, has_aux=True)
    (loss_val, loss_terms), grads = value_grad_loss(params, batch)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    # check if any of the parameters is NaN
    last_non_nan_params = jax.lax.cond(
        _check_nan_in_pytree(params),
        lambda _: last_non_nan_params,
        lambda _: params,
        None,
    )

    return (
        loss_val,
        loss_terms,
        params,
        opt_state,
        last_non_nan_params,
    )


@jit
def print_fn(i, loss_val, print_loss_every):
    # note that if the following is not jitted in the main lor loop, it is
    # super slow
    _ = jax.lax.cond(
        i % print_loss_every == 0,
        lambda _: jax.debug.print(
            "Iteration {i}: loss value = {loss_val}",
            i=i,
            loss_val=loss_val,
        ),
        lambda _: None,
        (None,),
    )


@jit
def store_loss_and_params(
    i,
    params,
    stored_params,
    stored_loss_terms,
    total_loss_values,
    loss_val,
    loss_terms,
    tracked_params,
):
    stored_params = jax.tree_util.tree_map(
        lambda stored_value, param, tracked_param: jax.lax.cond(
            tracked_param,
            lambda ope: ope[0].at[i].set(ope[1]),
            lambda ope: ope[0],
            (stored_value, param),
        ),
        stored_params,
        params,
        tracked_params,
    )
    stored_loss_terms = jax.tree_util.tree_map(
        lambda stored_term, loss_term: stored_term.at[i].set(loss_term),
        stored_loss_terms,
        loss_terms,
    )

    total_loss_values = total_loss_values.at[i].set(loss_val)
    return stored_params, stored_loss_terms, total_loss_values


def get_get_batch(obs_batch_sharding):
    """
    Return the get_batch function that will be used either the jittable one or
    the non-jittable one with sharding
    """

    def get_batch_sharding(data, param_data, obs_data):
        """
        This function is used at each loop but it cannot be jitted because of
        device_put

        Note: return all that's modified or unwanted dirty undefined behaviour
        """
        batch = data.get_batch()
        if param_data is not None:
            batch = append_param_batch(batch, param_data.get_batch())
        if obs_data is not None:
            # This is the part that motivated the transition from scan to for loop
            # Indeed we need to be transit obs_batch from CPU to GPU when we have
            # huge observations that cannot fit on GPU. Such transfer wasn't meant
            # to be jitted, i.e. in a scan loop
            obs_batch = jax.device_put(obs_data.get_batch(), obs_batch_sharding)
            batch = append_obs_batch(batch, obs_batch)
        return batch, data, param_data, obs_data

    @jit
    def get_batch(data, param_data, obs_data):
        """
        Original get_batch with not sharding

        Note: return all that's modified or unwanted dirty undefined behaviour
        """
        batch = data.get_batch()
        if param_data is not None:
            batch = append_param_batch(batch, param_data.get_batch())
        if obs_data is not None:
            batch = append_obs_batch(batch, obs_data.get_batch())
        return batch, data, param_data, obs_data

    if obs_batch_sharding is not None:
        return get_batch_sharding
    return get_batch
