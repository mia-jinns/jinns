import jax
from jax import jit, grad
import jax.numpy as jnp


class DynamicLoss:
    r"""
    Abstract base class for dynamic losses whose aim is to implement the term:

    .. math::
        \mathcal{N}[u](t, x) = 0
    """

    def __init__(
        self, Tmax=None, derivatives="nn_params", eq_params_heterogeneity=None
    ):
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
        eq_params_heterogeneity
            Default None. A dict with the keys being the same as in eq_params
            and the value being `time`, `space`, `both` or None which corresponds to
            the heterogeneity of a given parameter. A value can be missing, in
            this case there is no heterogeneity (=None). If
            eq_params_heterogeneity is None this means there is no
            heterogeneity for no parameters.
        """
        self.Tmax = Tmax
        if derivatives not in ["nn_params", "eq_params", "both"]:
            raise RuntimeError("derivative argument has an invalid value")
        self.derivatives = derivatives
        self.eq_params_heterogeneity = eq_params_heterogeneity

    def _eval_heterogeneous_parameters(
        self, eq_params, t, x, eq_params_heterogeneity=None
    ):
        eq_params_ = {}
        if eq_params_heterogeneity is None:
            return eq_params
        for k, p in eq_params.items():
            try:
                eq_params_[k] = self._eval_heterogeneous_array_parameter(
                    p, t, x, heterogeneity=eq_params_heterogeneity[k]
                )
            except KeyError:
                # we authorize missing eq_params_heterogeneity key
                # is its heterogeneity is None anyway
                eq_params_[k] = p
        return eq_params_

    def _eval_heterogeneous_array_parameter(self, p, t, x, heterogeneity=None):
        """
        For time and/or space heterogeneous params defined by an n-dimensional
        array `p` we return the value `p[t, x]` with discretization of the
        collocation point

        Parameters
        ----------
        p
            The parameter
        heterogeneity
            A string. Either `time`, `space`, `both` or None to specify which
            kind of heterogeneity we have. Default is None, is this case we do
            not have heterogeneity.


        **Note** t is assumed to be normalized in [0, 1] as well as x!
        """
        if heterogeneity is None:
            return p
        elif heterogeneity == "time":
            return p[(t * len(p)).astype(int)]
        elif heterogeneity == "space":
            coords = (x * jnp.array(p.shape)).astype(int)
            return jnp.take(p, jnp.ravel_multi_index(coords, p.shape, mode="clip"))
        elif heterogeneity == "both":
            coords = jnp.concatenate(
                [(t * len(p))[:, None], x * jnp.array(p.shape)], axis=1
            ).astype(int)
            return jnp.take(p, jnp.ravel_multi_index(coords, p.shape, mode="clip"))
        else:
            raise ValueError("Wrong paramater value for parameter `heterogeneity`")

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

    def __init__(
        self, Tmax=None, derivatives="nn_params", eq_params_heterogeneity=None
    ):
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
        eq_params_heterogeneity
            Default None. A dict with the keys being the same as in eq_params
            and the value being `time`, `space`, `both` or None which corresponds to
            the heterogeneity of a given parameter. A value can be missing, in
            this case there is no heterogeneity (=None). If
            eq_params_heterogeneity is None this means there is no
            heterogeneity for no parameters.
        """
        super().__init__(Tmax, derivatives, eq_params_heterogeneity)


class PDEStatio(DynamicLoss):
    r"""
    Abstract base class for PDE statio dynamic losses
    """

    def __init__(self, derivatives="nn_params", eq_params_heterogeneity=None):
        """
        Parameters
        ----------
        derivatives
            A string. Either ``nn_params``, ``eq_params``, ``both``. Determines
            with respect to which set of parameters gradients of the dynamic
            loss are computed. Default "nn_params", this is what is typically
            done in solving forward problems, when we only estimate the
            equation solution with as PINN.
        eq_params_heterogeneity
            Default None. A dict with the keys being the same as in eq_params
            and the value being `time`, `space`, `both` or None which corresponds to
            the heterogeneity of a given parameter. A value can be missing, in
            this case there is no heterogeneity (=None). If
            eq_params_heterogeneity is None this means there is no
            heterogeneity for no parameters.
        """
        super().__init__(
            derivatives=derivatives, eq_params_heterogeneity=eq_params_heterogeneity
        )


class PDENonStatio(DynamicLoss):
    r"""
    Abstract base class for PDE Non statio dynamic losses
    """

    def __init__(
        self, Tmax=None, derivatives="nn_params", eq_params_heterogeneity=None
    ):
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
        eq_params_heterogeneity
            Default None. A dict with the keys being the same as in eq_params
            and the value being `time`, `space`, `both` or None which corresponds to
            the heterogeneity of a given parameter. A value can be missing, in
            this case there is no heterogeneity (=None). If
            eq_params_heterogeneity is None this means there is no
            heterogeneity for no parameters.
        """
        super().__init__(Tmax, derivatives, eq_params_heterogeneity)
