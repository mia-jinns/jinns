import jax
import jax.numpy as jnp
from jax import vmap
from jax.tree_util import register_pytree_node_class
from jinns.data._DataGenerators import ODEBatch
from jinns.utils._utils import _get_vmap_in_axes_params


@register_pytree_node_class
class LossODE:
    r"""Loss object for an ordinary differential equation

    .. math::
        \mathcal{N}[u](t) = 0, \forall t \in I

    where :math:`\mathcal{N}[\cdot]` is a differential operator and the
    initial condition is :math:`u(t_0)=u_0`.


    **Note:** LossODE is jittable. Hence it implements the tree_flatten() and
    tree_unflatten methods.
    """

    def __init__(
        self,
        u,
        loss_weights,
        dynamic_loss,
        initial_condition,
        obs_batch=None,
    ):
        r"""
        Parameters
        ----------
        u :
            the PINN
        loss_weights :
            a dictionary with values used to ponderate each term in the loss
            function. Valid keys are `dyn_loss`, `initial_condition` and `observations`
        dynamic_loss :
            the ODE dynamic part of the loss, basically the differential
            operator :math:`\mathcal{N}[u](t)`. Should implement a method
            `dynamic_loss.evaluate(t, u, params)`.
            Can be None in order to
            access only some part of the evaluate call results.
        initial_condition :
            tuple of length 2 with initial condition :math:`(t0, u0)`.
            Can be None in order to
            access only some part of the evaluate call results.
        obs_batch :
            Default is None.
            A fixed batch of the observations that we have from the
            process and their corresponding timesteps. It is implemented as a
            list containing 2 jnp.array of the same size (on their first
            axis) encoding for observations [:math:`x_i, u(x_i)`] for
            computing a MSE. Default is None.

        Raises
        ------
        ValueError
            if initial condition is not a tuple.
        """
        self.dynamic_loss = dynamic_loss
        self.u = u
        if initial_condition is not None:
            if not isinstance(initial_condition, tuple) or len(initial_condition) != 2:
                raise ValueError(
                    f"Initial condition should be a tuple of len 2 with (t0, u0), {initial_condition} was passed."
                )
        self.initial_condition = initial_condition
        self.loss_weights = loss_weights
        self.obs_batch = obs_batch
        if self.obs_batch is None:
            self.loss_weights["observations"] = 0

    def __call__(self, *args, **kwargs):
        return self.evaluate(*args, **kwargs)

    def evaluate(self, params, batch):
        """
        Evaluate the loss function at a batch of points for given parameters.


        Parameters
        ---------
        params
            The dictionary of parameters of the model.
            Typically, it is a dictionary of
            dictionaries: `eq_params` and `nn_params``, respectively the
            differential equation parameters and the neural network parameter
        batch
            A batch of time points at which to evaluate the loss
        """
        if isinstance(params, tuple):
            params_ = params[0]
        else:
            params_ = params

        temporal_batch = batch.temporal_batch

        vmap_in_axes_t = (0,)

        # Retrieve the optional eq_params_batch
        # and update eq_params with the latter
        # and update vmap_in_axes
        if batch.param_batch_dict is not None:
            eq_params_batch_dict = batch.param_batch_dict

            # feed the eq_params with the batch
            for k in eq_params_batch_dict.keys():
                params["eq_params"][k] = eq_params_batch_dict[k]

        vmap_in_axes_params = _get_vmap_in_axes_params(batch.param_batch_dict, params)

        # dynamic part
        if self.dynamic_loss is not None:
            v_dyn_loss = vmap(
                lambda t, params: self.dynamic_loss.evaluate(t, self.u, params),
                vmap_in_axes_t + vmap_in_axes_params,
                0,
            )
            mse_dyn_loss = jnp.mean(v_dyn_loss(temporal_batch, params) ** 2)
        else:
            mse_dyn_loss = 0

        # initial condition
        if self.initial_condition is not None:
            t0, u0 = self.initial_condition
            t0 = jnp.array(t0)
            u0 = jnp.array(u0)
            mse_initial_condition = (
                self.u(
                    t0,
                    params["nn_params"],
                    jax.lax.stop_gradient(params["eq_params"]),
                )
                - u0
            ) ** 2
        else:
            mse_initial_condition = 0

        # MSE loss wrt to an observed batch
        if self.obs_batch is not None:
            v_u = vmap(
                lambda t: self.u(t, params["nn_params"], params["eq_params"]),
                0,
                0,
            )
            mse_observation_loss = jnp.mean(
                (v_u(self.obs_batch[0]) - self.obs_batch[1]) ** 2
            )
        else:
            mse_observation_loss = 0

        # total loss
        total_loss = (
            self.loss_weights["dyn_loss"] * mse_dyn_loss
            + self.loss_weights["initial_condition"] * mse_initial_condition
            + self.loss_weights["observations"] * mse_observation_loss
        )
        return total_loss, (
            {
                "dyn_loss": mse_dyn_loss,
                "initial_condition": mse_initial_condition,
                "observations": mse_observation_loss,
            }
        )

    def tree_flatten(self):
        children = (self.initial_condition, self.obs_batch)
        aux_data = {
            "loss_weights": self.loss_weights,
            "u": self.u,
            "dynamic_loss": self.dynamic_loss,
        }
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (initial_condition, obs_batch) = children
        loss_ode = cls(
            initial_condition=initial_condition,
            obs_batch=obs_batch,
            **aux_data,
        )
        return loss_ode


@register_pytree_node_class
class SystemLossODE:
    """
    Class to implement a system of ODEs.
    The goal is to give maximum freedom to the user. The class is created with
    a dict of dynamic loss and a dict of initial conditions. When then iterate
    over the dynamic losses that compose the system. All the PINNs with all the
    parameter dictionaries are passed as arguments to each dynamic loss
    evaluate functions; it is inside the dynamic loss that specification are
    performed.

    **Note:** All the dictionaries (except `dynamic_loss_dict`) must have the same keys.
    Indeed, these dictionaries (except `dynamic_loss_dict`) are tied to one
    solution.

    **Note:** SystemLossODE is jittable. Hence it implements the tree_flatten() and
    tree_unflatten methods.
    """

    def __init__(
        self,
        u_dict,
        loss_weights,
        dynamic_loss_dict,
        initial_condition_dict,
        obs_batch_dict=None,
    ):
        r"""
        Parameters
        ----------
        u_dict
            dict of PINNs
        loss_weights
            a dictionary with values used to ponderate each term in the loss
            function. Valid keys are `dyn_loss`, `initial_condition` and `observations`
        initial_condition_dict
            dict of tuple of length 2 with initial condition :math:`(t_0, u_0)`
            Must share the keys of `u_dict`
        dynamic_loss_dict
            dict of dynamic part of the loss, basically the differential
            operator :math:`\mathcal{N}[u](t)`. Should implement a method
            `dynamic_loss.evaluate(t, u, params)`
        obs_batch_dict
            Default is None.
            A dictionary, a key per element of `u_dict`, with value being a tuple
            containing 2 jnp.array of the same size (on their first
            axis) encoding the timestep and the observations [:math:`x_i, u(x_i)`].
            A particular key can be None.

        Raises
        ------
        ValueError
            if initial condition is not a dict of tuple
        ValueError
            if the dictionaries that should share the keys of u_dict do not
        """
        self.loss_weights = loss_weights
        if obs_batch_dict is None:
            # if the user did not provide at all this optional argument,
            # we make sure there is a null ponderating loss_weight and we
            # create a dummy dict with the required keys and all the values to
            # None
            self.loss_weights["observations"] = 0
            self.obs_batch_dict = {k: None for k in u_dict.keys()}
        else:
            self.obs_batch_dict = obs_batch_dict
            if u_dict.keys() != obs_batch_dict.keys():
                raise ValueError(
                    "All the dicts (except dynamic_loss_dict) should have same keys"
                )
        if u_dict.keys() != initial_condition_dict.keys():
            raise ValueError(
                "All the dicts (except dynamic_loss_dict) should have same keys"
            )
        self.dynamic_loss_dict = dynamic_loss_dict
        self.u_dict = u_dict
        self.initial_condition_dict = initial_condition_dict

        # The constaints on the solutions will be implemented by reusing a
        # LossODE class without dynamic loss term
        self.u_constraints_dict = {}
        for i in self.u_dict.keys():
            self.u_constraints_dict[i] = LossODE(
                u=u_dict[i],
                loss_weights={
                    "dyn_loss": 0.0,
                    "initial_condition": self.loss_weights["initial_condition"],
                    "observations": self.loss_weights["observations"],
                },
                dynamic_loss=None,
                initial_condition=self.initial_condition_dict[i],
                obs_batch=self.obs_batch_dict[i],
            )

    def __call__(self, *args, **kwargs):
        return self.evaluate(*args, **kwargs)

    def evaluate(self, params_dict, batch):
        """
        Evaluate the loss function at a batch of points for given parameters.


        Parameters
        ---------
        params
            A dictionary of dictionaries of parameters of the model.
            Typically, it is a dictionary of dictionaries of
            dictionaries: `eq_params` and `nn_params``, respectively the
            differential equation parameters and the neural network parameter
        batch
            A batch of time points at which to evaluate the loss
        """
        if self.u_dict.keys() != params_dict["nn_params"].keys():
            raise ValueError("u_dict and params_dict[nn_params] should have same keys ")

        temporal_batch = batch.temporal_batch

        vmap_in_axes_t = (0,)

        # Retrieve the optional eq_params_batch
        # and update eq_params with the latter
        # and update vmap_in_axes
        if batch.param_batch_dict is not None:
            eq_params_batch_dict = batch.param_batch_dict

            # feed the eq_params with the batch
            for k in eq_params_batch_dict.keys():
                params_dict["eq_params"][k] = eq_params_batch_dict[k]

        vmap_in_axes_params = _get_vmap_in_axes_params(
            batch.param_batch_dict, params_dict
        )

        mse_dyn_loss = 0
        mse_initial_condition = 0
        mse_observation_loss = 0

        for i in self.dynamic_loss_dict.keys():
            # dynamic part
            v_dyn_loss = vmap(
                lambda t, params_dict: self.dynamic_loss_dict[i].evaluate(
                    t, self.u_dict, params_dict
                ),
                vmap_in_axes_t + vmap_in_axes_params,
                0,
            )
            mse_dyn_loss += jnp.mean(v_dyn_loss(temporal_batch, params_dict) ** 2)

        # initial conditions and observation_loss via the internal LossODE
        for i in self.u_dict.keys():
            _, res_dict = self.u_constraints_dict[i].evaluate(
                {
                    "nn_params": params_dict["nn_params"][i],
                    "eq_params": params_dict["eq_params"],
                },
                batch,
            )
            # note that the results have already been scaled internally in the
            # call to evaluate
            mse_initial_condition += res_dict["initial_condition"]
            mse_observation_loss += res_dict["observations"]

        # total loss
        total_loss = (
            self.loss_weights["dyn_loss"] * mse_dyn_loss
            + self.loss_weights["initial_condition"] * mse_initial_condition
            + self.loss_weights["observations"] * mse_observation_loss
        )
        return total_loss, (
            {
                "dyn_loss": mse_dyn_loss,
                "initial_condition": mse_initial_condition,
                "observations": mse_observation_loss,
            }
        )

    def tree_flatten(self):
        children = (
            self.initial_condition_dict,
            self.obs_batch_dict,
        )
        aux_data = {
            "loss_weights": self.loss_weights,
            "u_dict": self.u_dict,
            "dynamic_loss_dict": self.dynamic_loss_dict,
        }
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (initial_condition_dict, obs_batch_dict) = children
        loss_ode = cls(
            initial_condition_dict=initial_condition_dict,
            obs_batch_dict=obs_batch_dict,
            **aux_data,
        )

        return loss_ode
