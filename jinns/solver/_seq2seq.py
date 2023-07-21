import jax
from jinns.data._DataGenerators import (
    DataGeneratorODE,
    _reset_batch_idx_and_permute,
)
from jinns.loss._LossODE import SystemLossODE
from jinns.loss._LossPDE import LossPDENonStatio

import jax.numpy as jnp


def initialize_seq2seq(loss, data, seq2seq):
    """
    Initialize the seq2seq parameters as described in “Characterizing possible
    failure modes in physics-informed neural networks”, A. S. Krishnapriyan,
    NeurIPS 2021.

    Parameters
    ----------
    loss
        A loss object (e.g. a LossODE, SystemLossODE, LossPDEStatio [...]
        object). It must be jittable (e.g. implements via a pytree
        registration)
    data
        A DataGenerator object which implements a `get_batch()`
        method which returns a 3-tuple with (omega_grid, omega_border, time grid).
        It must be jittable (e.g. implements via a pytree
        registration)
    seq2seq
        A dictionary with keys 'times_steps'
        and 'iter_steps' which mush have same length. The first represents
        the time steps which represents the different time interval upon
        which we perform the incremental learning. The second represents
        the number of iteration we perform in each time interval.

    Returns
    -------
    update_seq2seq
        A function which performs the update of the seq2seq method
    """
    curr_seq = 0
    if isinstance(loss, SystemLossODE) and isinstance(data, DataGeneratorODE):
        update_seq2seq = update_seq2seq_SystemLossODE
        # Note that boundaries for the first PINN are OK
        # set new boundaries for the batch generator
        data.tmin = seq2seq["time_steps"][curr_seq]
        data.tmax = seq2seq["time_steps"][curr_seq + 1]
        # and do not forget to regenerate the data
        data.curr_omega_idx = 0
        data.generate_time_data()
        data._key, data.times, _ = _reset_batch_idx_and_permute(
            (data._key, data.times, data.curr_omega_idx, None)
        )
    elif isinstance(loss, LossPDENonStatio):
        raise RuntimeError("Untrusted function, do not use")
        # update_seq2seq = update_seq2seq_LossPDENonStatio
        ## Note that boundaries for the first PINN are OK
        ## set new boundaries for the batch generator
        # data.tmin = seq2seq["time_steps"][curr_seq]
        # data.tmax = seq2seq["time_steps"][curr_seq + 1]
        ## and do not forget to regenerate the data
        # data.curr_omega_idx = 0
        # data.generate_data()
        # data._key, data.omega, _ = _reset_batch_idx_and_permute(
        #    (data._key, data.omega, data.curr_omega_idx, None)
        # )
        # data.curr_time_idx = 0
        # data.generate_data_nonstatio()
        # data._key, data.times, _ = _reset_batch_idx_and_permute(
        #    (data._key, data.times, data.curr_time_idx, None)
        # )
        # if data.omega_border is not None:
        #    data.curr_omega_border_idx = 0
        #    data._key, data.omega_border, _ = _reset_batch_idx_and_permute(
        #        (
        #            data._key,
        #            data.omega_border,
        #            data.curr_omega_border_idx,
        #            None,
        #        )
        #    )

    return update_seq2seq


def update_seq2seq_SystemLossODE(operands):
    """
    __TODO__ check the u0 update in this function as it is not sure it is well
    programed as we have temporal_boundary_fun in aux_data.

    Make all the necessary updates for a SystemLossODE in seq2seq learning mode

    Parameters
    ----------
    operands
        A tuple which comprises.

        loss
            A loss object (e.g. a LossODE, SystemLossODE, LossPDEStatio [...]
            object). It must be jittable (e.g. implements via a pytree
            registration)
        seq2seq
            A dictionary with keys 'times_steps'
            and 'iter_steps' which mush have same length. The first represents
            the time steps which represents the different time interval upon
            which we perform the incremental learning. The second represents
            the number of iteration we perform in each time interval.
        data
            A DataGenerator object which implements a `get_batch()`
            method which returns a 3-tuple with (omega_grid, omega_border, time grid).
            It must be jittable (e.g. implements via a pytree
            registration)
        params
            The dictionary of parameters of the model.
            Typically, it is a dictionary of
            dictionaries: `eq_params` and `nn_params``, respectively the
            differential equation parameters and the neural network parameter
        curr_seq
            A integer which represents which sequence we currently are in
    """
    loss, seq2seq, data, params, curr_seq = operands
    curr_seq += 1
    # NOTE this can cause a UnexpectedTracerException when reusing the loss
    # whose attribute has been set in the following (jitted) line
    loss.initial_condition_dict = {
        i: (
            seq2seq["time_steps"][curr_seq],
            loss.u_dict[i](
                jnp.array(seq2seq["time_steps"][curr_seq]),
                params["nn_params"][i],
                params["eq_params"][i],
            ),
        )
        for i in loss.u_dict.keys()
    }
    # set new boundaries for the batch generator
    data.tmin = seq2seq["time_steps"][curr_seq]
    data.tmax = seq2seq["time_steps"][curr_seq + 1]
    # and do not forget to regenerate the data
    data.curr_omega_idx = 0
    data.generate_time_data()
    data._key, data.times, _ = _reset_batch_idx_and_permute(
        (data._key, data.times, data.curr_omega_idx, None)
    )
    return curr_seq


def update_seq2seq_LossPDENonStatio(operands):
    """
    Make all the necessary updates for a LossPDENonStatio in seq2seq learning mode

    Parameters
    ----------
    operands
        A tuple which comprises.

        loss
            A loss object (e.g. a LossODE, SystemLossODE, LossPDEStatio [...]
            object). It must be jittable (e.g. implements via a pytree
            registration)
        seq2seq
            A dictionary with keys 'times_steps'
            and 'iter_steps' which mush have same length. The first represents
            the time steps which represents the different time interval upon
            which we perform the incremental learning. The second represents
            the number of iteration we perform in each time interval.
        data
            A DataGenerator object which implements a `get_batch()`
            method which returns a 3-tuple with (omega_grid, omega_border, time grid).
            It must be jittable (e.g. implements via a pytree
            registration)
        params
            The dictionary of parameters of the model.
            Typically, it is a dictionary of
            dictionaries: `eq_params` and `nn_params``, respectively the
            differential equation parameters and the neural network parameter
        curr_seq
            A integer which represents which sequence we currently are in
    """
    loss, seq2seq, data, params, curr_seq = operands
    curr_seq += 1
    # set new boundaries for the PINNs
    # NOTE this can cause a UnexpectedTracerException when reusing the loss
    # whose attribute has been set in the following (jitted) line
    loss.u0 = lambda t, x: loss.u(t, x, params["nn_params"], params["eq_params"])
    # set new boundaries for the batch generator
    data.tmin = seq2seq["time_steps"][curr_seq]
    data.tmax = seq2seq["time_steps"][curr_seq + 1]
    # and do not forget to regenerate the data
    data.curr_omega_idx = 0
    data.generate_data()
    data._key, data.omega, _ = _reset_batch_idx_and_permute(
        (data._key, data.omega, data.curr_omega_idx, None)
    )
    data.curr_time_idx = 0
    data.generate_data_nonstatio()
    data._key, data.times, _ = _reset_batch_idx_and_permute(
        (data._key, data.times, data.curr_time_idx, None)
    )
    if data.omega_border is not None:
        data.curr_omega_border_idx = 0
        data._key, data.omega_border, _ = _reset_batch_idx_and_permute(
            (
                data._key,
                data.omega_border,
                data.curr_omega_border_idx,
                None,
            )
        )
    return curr_seq
