from jax_tqdm import scan_tqdm
import jax
from jax import jit
import jax.numpy as jnp
from jinns.solver._seq2seq import initialize_seq2seq


class PinnSolver:
    """A class for optimizing the loss functions
    Main method : <object>.solve()
    """

    def __init__(self, optax_solver, loss, n_iter):
        """
        Parameters
        ----------
        optax_solver
            A JaxOpt OptaxSolver object: an optax solver with predefined
            algorithm (e.g. adam with a given step-size)
        loss
            A loss object (e.g. a LossODE, SystemLossODE, LossPDEStatio [...]
            object). It must be jittable (e.g. implements via a pytree
            registration)
        n_iter
            The number of iterations in the optimization
        """

        self.optax_solver = optax_solver
        self.loss = loss
        self.n_iter = n_iter

    def solve(
        self,
        init_params,
        data,
        opt_state=None,
        seq2seq=None,
        accu_vars=[],
    ):
        """
        Performs the optimization process via stochastic gradient descent
        algorithm. We minimize the function defined `loss.evaluate()` with
        respect to the learnable parameters of the problem whose initial values
        are given in `init_params`.


        Parameters
        ----------
        init_params
            The initial dictionary of parameters. Typically, it is a dictionary of
            dictionaries: `eq_params` and `nn_params``, respectively the
            differential equation parameters and the neural network parameter
        data
            A DataGenerator object which implements a `get_batch()`
            method which returns a 3-tuple with (omega_grid, omega_border, time grid).
            It must be jittable (e.g. implements via a pytree
            registration)
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


        Returns
        -------
        params
            The values of the dictionaries of parameters at then end of the
            optimization process
        accu[0, :]
            An array of the total loss term along the gradient steps
        res["stored_loss_terms"]
            A dictionary. At each key an array of the values of a given loss
            term is stored
        opt_state
            The final optimized state
        res["stored_values"]
            A dictionary. At each key an array of the values of the parameters
            given in accu_vars is stored
        """
        params = init_params
        if opt_state is None:
            batch = data.get_batch()
            opt_state = self.optax_solver.init_state(params, batch=batch)

        curr_seq = 0
        if seq2seq is not None:
            assert data.method == "uniform", "data.method must be uniform if"
            " using seq2seq learning !"
            # NOTE this was the cause to a very hard to debug error probably
            # due to the omnistaging of the jnp.arange which then do not
            # accept that shape changes after compilation. Solution to use
            # np.arange not available since although we declare tmin tmax
            # or other as static argnums as the whole dataset class is jitted
            # https://jax.readthedocs.io/en/latest/jep/4410-omnistaging.html
            # proabably because data.omega is not static ?!

            update_seq2seq = initialize_seq2seq(self.loss, data, seq2seq)

        else:
            update_seq2seq = None

        # initialize the dict for stored parameter values
        stored_values = {}
        for params_leaf_path in accu_vars:
            stored_values["-".join(map(str, params_leaf_path))] = jnp.zeros(
                (self.n_iter,)
            )

        # initialize the dict for stored loss values
        stored_loss_terms = {}
        batch = data.get_batch()
        _, loss_terms = self.loss(params, batch)
        for loss_name, _ in loss_terms.items():
            stored_loss_terms[loss_name] = jnp.zeros((self.n_iter,))

        def nested_get(dic, keys):
            for k in keys:
                dic = dic[k]
            return dic

        @scan_tqdm(self.n_iter)
        def scan_func_solve_one_iter(carry, i):
            """
            Main optimization loop
            """
            batch = carry["data"].get_batch()
            params, opt_state = self.optax_solver.update(
                params=carry["params"], state=carry["state"], batch=batch
            )
            total_loss_val, loss_terms = self.loss(carry["params"], batch)

            # seq2seq learning updates
            if seq2seq is not None:
                curr_seq = jax.lax.cond(
                    carry["curr_seq"] + 1
                    < jnp.sum(
                        seq2seq["iter_steps"] < i
                    ),  # check if we fall in another time interval
                    update_seq2seq,
                    lambda operands: operands[-1],
                    (
                        carry["loss"],
                        seq2seq,
                        carry["data"],
                        carry["params"],
                        carry["curr_seq"],
                    ),
                )
            else:
                curr_seq = -1

            # saving selected parameters values with accumulator
            accu = [total_loss_val]
            for params_leaf_path in accu_vars:
                carry["stored_values"]["-".join(map(str, params_leaf_path))] = (
                    carry["stored_values"]["-".join(map(str, params_leaf_path))]
                    .at[i]
                    .set(nested_get(params, params_leaf_path).squeeze())
                )

            # saving values of each loss term
            for loss_name, loss_value in loss_terms.items():
                carry["stored_loss_terms"][loss_name] = (
                    carry["stored_loss_terms"][loss_name].at[i].set(loss_value)
                )

            return {
                "params": params,
                "state": opt_state,
                "data": carry["data"],
                "curr_seq": curr_seq,
                "seq2seq": seq2seq,
                "stored_values": carry["stored_values"],
                "stored_loss_terms": carry["stored_loss_terms"],
                "loss": carry["loss"],
            }, accu

        res, accu = jax.lax.scan(
            scan_func_solve_one_iter,
            {
                "params": init_params,
                "state": opt_state,
                "data": data,
                "curr_seq": curr_seq,
                "seq2seq": seq2seq,
                "stored_values": stored_values,
                "stored_loss_terms": stored_loss_terms,
                "loss": self.loss,
            },
            jnp.arange(self.n_iter),
        )

        params = res["params"]
        opt_state = res["state"]
        data = res["data"]
        loss = res["loss"]

        accu = jnp.array(accu)

        return (
            params,
            accu[0, :],
            res["stored_loss_terms"],
            opt_state,
            res["stored_values"],
        )
