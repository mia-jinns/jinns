"""
Implements several dynamic losses
"""

from typing import Union, Callable, Dict
from jaxtyping import Float
from jax import grad
import jax.numpy as jnp
import equinox as eqx

from jinns.utils._utils import _extract_nn_params
from jinns.loss._DynamicLossAbstract_eqx import ODE


class GeneralizedLotkaVolterra_eqx(ODE):
    r"""
    Return a dynamic loss from an equation of a Generalized Lotka Volterra
    system. Say we implement the equation for population :math:`i`:

    .. math::
        \frac{\partial}{\partial t}u_i(t) = r_iu_i(t) - \sum_{j\neq i}\alpha_{ij}u_j(t)
        -\alpha_{i,i}u_i(t) + c_iu_i(t) + \sum_{j \neq i} c_ju_j(t)

    with :math:`r_i` the growth rate parameter, :math:`c_i` the carrying
    capacities and :math:`\alpha_{ij}` the interaction terms.

    Parameters
    ----------
    key_main
        The dictionary key (in the dictionaries ``u`` and ``params`` that
        are arguments of the ``evaluate`` function) of the main population
        :math:`i` of the particular equation of the system implemented
        by this dynamic loss
    keys_other
        The list of dictionary keys (in the dictionaries ``u`` and ``params`` that
        are arguments of the ``evaluate`` function) of the other
        populations that appear in the equation of the system implemented
        by this dynamic loss
    Tmax
        Tmax needs to be given when the PINN time input is normalized in
        [0, 1], ie. we have performed renormalization of the differential
        equation
    eq_params_heterogeneity
        Default None. A dict with the keys being the same as in eq_params
        and the value being `time`, `space`, `both` or None which corresponds to
        the heterogeneity of a given parameter. A value can be missing, in
        this case there is no heterogeneity (=None). If
        eq_params_heterogeneity is None this means there is no
        heterogeneity for no parameters.
    """

    key_main: list[str]
    keys_other: list[str]

    def evaluate(self, t, u_dict, params_dict):
        """
        Evaluate the dynamic loss at `t`.
        For stability we implement the dynamic loss in log space.

        Parameters
        ---------
        t
            A time point
        u_dict
            A dictionary of PINNS. Must have the same keys as `params_dict`
        params_dict
            The dictionary of dictionaries of parameters of the model. Keys at
            top level are "nn_params" and "eq_params"
        """
        params_main = _extract_nn_params(params_dict, self.key_main)

        u = u_dict[self.key_main]
        # need to index with [0] since u output is nec (1,)
        du_dt = grad(lambda t: jnp.log(u(t, params_main)[0]), 0)(t)
        carrying_term = params_main["eq_params"]["carrying_capacity"] * u(
            t, params_main
        )
        # NOTE the following assumes interaction term with oneself is at idx 0
        interaction_terms = params_main["eq_params"]["interactions"][0] * u(
            t, params_main
        )

        # TODO write this for loop with tree_util functions?
        for i, k in enumerate(self.keys_other):
            params_k = _extract_nn_params(params_dict, k)
            carrying_term += params_main["eq_params"]["carrying_capacity"] * u_dict[k](
                t, params_k
            )
            interaction_terms += params_main["eq_params"]["interactions"][
                i + 1
            ] * u_dict[k](t, params_k)

        return du_dt + self.Tmax * (
            -params_main["eq_params"]["growth_rate"] - interaction_terms + carrying_term
        )
