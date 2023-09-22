import jax
from jax import jit, grad
import jax.numpy as jnp


class DynamicLoss:
    r"""
    Abstract base class for dynamic losses whose aim is to implement the term:

    .. math::
        \mathcal{N}[u](t, x) = 0
    """

    def __init__(self, Tmax=None, derivatives="nn_params"):
        """
        Parameters
        ----------
        Tmax
            Tmax needs to be given when the PINN time input is normalized in
            [0, 1], ie. we have performed renormalization of the differential
            equation
        derivatives
            A string. Either ``nn_params``, ``eq_params``, ``both``. Determines
            with respect to which set of parameters gradients of the dynamic
            loss are computed. Default "nn_params", this is what is typically
            done in solving forward problems, when we only estimate the
            equation solution with as PINN.
        """
        self.Tmax = Tmax
        if derivatives not in ["nn_params", "eq_params", "both"]:
            raise RuntimeError("derivative argument has an invalid value")
        self.derivatives = derivatives

    def _eval_heterogeneous_array_parameter(self, p, t=None, x=None):
        """
        For time and/or space heterogeneous params defined by an n-dimensional
        array `p` we return the value `p[t, x]` with discretization of the
        collocation point

        **Note** t is assumed to be normalized in [0, 1] as well as x!
        """
        if x is None and t is None:
            raise ValueError("Either x or t must be provided")
        elif t is None:
            coords = (x * jnp.array(p.shape)).astype(int)
            return jnp.take(p, jnp.ravel_multi_index(coords, p.shape, mode="clip"))
        elif x is None:
            return p[(t * len(p)).astype(int)]
        else:
            coords = jnp.concatenate(
                [(t * len(p))[:, None], x * jnp.array(p.shape)], axis=1
            ).astype(int)
            return jnp.take(p, jnp.ravel_multi_index(coords, p.shape, mode="clip"))

    def set_stop_gradient(self, params_dict):
        """
        Set the stop gradient operators in the dynamic loss `evaluate`
        function according to the rule defined by the attribute
        `self.derivatives`

        Parameters
        ----------
        params_dict
            The dictionary of parameters of the model.
            Typically, it is a dictionary of
            dictionaries: `eq_params` and `nn_params``, respectively the
            differential equation parameters and the neural network parameter
        """
        nn_params = params_dict["nn_params"]
        eq_params = params_dict["eq_params"]

        if self.derivatives == "nn_params":
            return (nn_params, jax.lax.stop_gradient(eq_params))
        elif self.derivatives == "eq_params":
            return (jax.lax.stop_gradient(nn_params), eq_params)
        elif self.derivatives == "both":
            return (nn_params, eq_params)
        else:
            return (
                jax.lax.stop_gradient(nn_params),
                jax.lax.stop_gradient(eq_params),
            )


class ODE(DynamicLoss):
    r"""
    Abstract base class for ODE dynamic losses
    """

    def __init__(self, Tmax=None, derivatives="nn_params"):
        """
        Parameters
        ----------
        Tmax
            Tmax needs to be given when the PINN time input is normalized in
            [0, 1], ie. we have performed renormalization of the differential
            equation
        derivatives
            A string. Either ``nn_params``, ``eq_params``, ``both``. Determines
            with respect to which set of parameters gradients of the dynamic
            loss are computed. Default "nn_params", this is what is typically
            done in solving forward problems, when we only estimate the
            equation solution with as PINN.
        """
        super().__init__(Tmax, derivatives)


class PDEStatio(DynamicLoss):
    r"""
    Abstract base class for PDE statio dynamic losses
    """

    def __init__(self, Tmax=None, derivatives="nn_params"):
        """
        Parameters
        ----------
        Tmax
            Tmax needs to be given when the PINN time input is normalized in
            [0, 1], ie. we have performed renormalization of the differential
            equation
        derivatives
            A string. Either ``nn_params``, ``eq_params``, ``both``. Determines
            with respect to which set of parameters gradients of the dynamic
            loss are computed. Default "nn_params", this is what is typically
            done in solving forward problems, when we only estimate the
            equation solution with as PINN.
        """
        super().__init__(Tmax, derivatives)


class PDENonStatio(DynamicLoss):
    r"""
    Abstract base class for PDE Non statio dynamic losses
    """

    def __init__(self, Tmax=None, derivatives="nn_params"):
        """
        Parameters
        ----------
        Tmax
            Tmax needs to be given when the PINN time input is normalized in
            [0, 1], ie. we have performed renormalization of the differential
            equation
        derivatives
            A string. Either ``nn_params``, ``eq_params``, ``both``. Determines
            with respect to which set of parameters gradients of the dynamic
            loss are computed. Default "nn_params", this is what is typically
            done in solving forward problems, when we only estimate the
            equation solution with as PINN.
        """
        super().__init__(Tmax, derivatives)
