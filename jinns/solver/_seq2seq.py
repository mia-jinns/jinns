import jax
from jinns.data._DataGenerators import (
    DataGeneratorODE,
    _reset_batch_idx_and_permute,
)
from jinns.loss._LossODE import SystemLossODE, LossODE
from jinns.loss._LossPDE import LossPDENonStatio, LossPDEStatio, SystemLossPDE

import jax.numpy as jnp


def _seq2seq_triggerer(carry, i, _update_seq2seq_true, _update_seq2seq_false):
    carry["curr_seq"], carry["loss"], carry["data"], carry["state"] = jax.lax.cond(
        carry["curr_seq"] + 1
        < jnp.sum(
            seq2seq["iter_steps"] < i
        ),  # check if we fall in another time interval
        _update_seq2seq_true,
        _update_seq2seq_false,
        (
            carry["loss"],
            carry["seq2seq"],
            carry["data"],
            carry["params"],
            carry["curr_seq"],
            carry["state"],
        ),
    )
    return carry


def _initialize_seq2seq(loss, data, seq2seq, opt_state):
    """
    Initialize the seq2seq parameters as described in “Characterizing possible
    failure modes in physics-informed neural networks”, A. S. Krishnapriyan,
    NeurIPS 2021.

    **Note:** we do not change tmin, we only let the interval grow longer.
    Indeed we noticed some unlearning happening.

    **Note:** using seq2seq might create some instability in training when
    interval changes. Some of this instability comes from the fact that Tmax in
    the dynamic loss rescaling must be the true and final (and potentially large and
    unstable) one from the beginning if we want to be able to catch the real dynamic.
    However it does offer some better results for learning on long time intervals.

    **Note:** As this is experimental some changes in the future might be:
        - to dig deeper and try to attenuate the instability
        - to try to attenuate the discrepancy with the real dynamic when we
          also change Tmax in dynamic loss (this requires to treat the dynamic
          loss as a dynamic attribute of a Loss class...).
        - to investigate Tmax as input of the PINN

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
        update_seq2seq = _update_seq2seq_SystemLossODE
        # Note that boundaries for the first PINN are OK
        # set new boundaries for the batch generator
        data.tmax = seq2seq["time_steps"][curr_seq + 1]
        # and do not forget to regenerate the data
        data.curr_omega_idx = 0
        data.generate_time_data()
        data._key, data.times, _ = _reset_batch_idx_and_permute(
            (data._key, data.times, data.curr_omega_idx, None, data.p)
        )
        opt_state.internal_state.hyperparams["learning_rate"] = seq2seq[
            "learning_rate"
        ][curr_seq]

    elif (
        isinstance(loss, LossPDENonStatio)
        or isinstance(loss, LossPDE)
        or isinstance(loss, SystemLossPDE)
    ):
        raise RuntimeError("Not implemented")

    # No need to return data here since this function will not be jitted and
    # side effects are allowed
    return update_seq2seq


def _update_seq2seq_SystemLossODE(operands):
    """
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
    loss, seq2seq, data, params, curr_seq, opt_state = operands
    curr_seq += 1

    # set new boundaries for the batch generator
    data.tmax = seq2seq["time_steps"][curr_seq + 1]
    # and do not forget to regenerate the data
    data.curr_omega_idx = 0
    data.generate_time_data()
    data._key, data.times, _ = _reset_batch_idx_and_permute(
        (data._key, data.times, data.curr_omega_idx, None, data.p)
    )
    opt_state.internal_state.hyperparams["learning_rate"] = seq2seq["learning_rate"][
        curr_seq
    ]
    return curr_seq, loss, data, opt_state


def _update_seq2seq_false(operands):
    return (operands[-2], operands[0], operands[2], operands[-1])
