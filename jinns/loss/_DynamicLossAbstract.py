"""
Implements abstract classes for dynamic losses
"""

from __future__ import (
    annotations,
)  # https://docs.python.org/3/library/typing.html#constant

import equinox as eqx
from typing import Callable, Dict, TYPE_CHECKING, ClassVar
from jaxtyping import Float, Array
from functools import partial
import abc


# See : https://docs.kidger.site/equinox/api/module/advanced_fields/#equinox.AbstractClassVar--known-issues
if TYPE_CHECKING:
    from typing import ClassVar as AbstractClassVar
    from jinns.parameters import Params, ParamsDict
else:
    from equinox import AbstractClassVar


def _decorator_heteregeneous_params(evaluate):

    def wrapper(*args):
        self, inputs, u, params = args
        _params = eqx.tree_at(
            lambda p: p.eq_params,
            params,
            self._eval_heterogeneous_parameters(
                inputs, u, params, self.eq_params_heterogeneity
            ),
        )
        new_args = args[:-1] + (_params,)
        res = evaluate(*new_args)
        return res

    return wrapper


class DynamicLoss(eqx.Module):
    r"""
    Abstract base class for dynamic losses. Implements the physical term:

    $$
        \mathcal{N}[u](t, x) = 0
    $$

    for **one** point $t$, $x$ or $(t, x)$, depending on the context.

    Parameters
    ----------
    Tmax : Float, default=1
        Tmax needs to be given when the PINN time input is normalized in
        [0, 1], ie. we have performed renormalization of the differential
        equation
    eq_params_heterogeneity : Dict[str, Callable | None], default=None
        A dict with the same keys as eq_params and the value being either None
        (no heterogeneity) or a function which encodes for the spatio-temporal
        heterogeneity of the parameter.
        Such a function must be jittable and take four arguments `t`, `x`,
        `u` and `params` even if some are not used. Therefore,
        one can introduce spatio-temporal covariates upon which a particular
        parameter can depend, e.g. in a Generalized Linear Model fashion. The
        effect of these covariates can themselves be estimated by being in
        `eq_params` too.
        A value can be missing, in this case there is no heterogeneity (=None).
        Default None, meaning there is no heterogeneity in the equation
        parameters.
    """

    _eq_type = AbstractClassVar[str]  # class variable denoting the type of
    # differential equation
    Tmax: Float = eqx.field(kw_only=True, default=1)
    eq_params_heterogeneity: Dict[str, Callable | None] = eqx.field(
        kw_only=True, default=None, static=True
    )

    def _eval_heterogeneous_parameters(
        self,
        inputs: Float[Array, "1"] | Float[Array, "dim"] | Float[Array, "1+dim"],
        u: eqx.Module,
        params: Params | ParamsDict,
        eq_params_heterogeneity: Dict[str, Callable | None] = None,
    ) -> Dict[str, float | Float[Array, "parameter_dimension"]]:
        eq_params_ = {}
        if eq_params_heterogeneity is None:
            return params.eq_params
        for k, p in params.eq_params.items():
            try:
                if eq_params_heterogeneity[k] is None:
                    eq_params_[k] = p
                else:
                    eq_params_[k] = eq_params_heterogeneity[k](inputs, u, params)
            except KeyError:
                # we authorize missing eq_params_heterogeneity key
                # if its heterogeneity is None anyway
                eq_params_[k] = p
        return eq_params_

    def _evaluate(
        self,
        inputs: Float[Array, "1"] | Float[Array, "dim"] | Float[Array, "1+dim"],
        u: eqx.Module,
        params: Params | ParamsDict,
    ) -> float:
        evaluation = self.equation(inputs, u, params)
        if len(evaluation.shape) == 0:
            raise ValueError(
                "The output of dynamic loss must be vectorial, "
                "i.e. of shape (d,) with d >= 1"
            )
        return evaluation

    @abc.abstractmethod
    def equation(self, *args, **kwargs):
        # TO IMPLEMENT
        # Point-wise evaluation of the differential equation N[u](.)
        raise NotImplementedError("You should implement your equation.")


class ODE(DynamicLoss):
    r"""
    Abstract base class for ODE dynamic losses. All dynamic loss must subclass
    this class and override the abstract method `equation`.

    Attributes
    ----------
    Tmax : float, default=1
        Tmax needs to be given when the PINN time input is normalized in
        [0, 1], ie. we have performed renormalization of the differential
        equation
    eq_params_heterogeneity : Dict[str, Callable | None], default=None
        Default None. A dict with the keys being the same as in eq_params
        and the value being either None (no heterogeneity) or a function
        which encodes for the spatio-temporal heterogeneity of the parameter.
        Such a function must be jittable and take four arguments `t`, `x`,
        `u` and `params` even if one is not used. Therefore,
        one can introduce spatio-temporal covariates upon which a particular
        parameter can depend, e.g. in a GLM fashion. The effect of these
        covariables can themselves be estimated by being in `eq_params` too.
        Some key can be missing, in this case there is no heterogeneity (=None).
        If eq_params_heterogeneity is None this means there is no
        heterogeneity for no parameters.
    """

    _eq_type: ClassVar[str] = "ODE"

    @partial(_decorator_heteregeneous_params)
    def evaluate(
        self,
        t: Float[Array, "1"],
        u: eqx.Module | Dict[str, eqx.Module],
        params: Params | ParamsDict,
    ) -> float:
        """Here we call DynamicLoss._evaluate with x=None"""
        return self._evaluate(t, u, params)

    @abc.abstractmethod
    def equation(
        self, t: Float[Array, "1"], u: eqx.Module, params: Params | ParamsDict
    ) -> float:
        r"""
        The differential operator defining the ODE.

        !!! warning

            This is an abstract method to be implemented by users.

        Parameters
        ----------
        t : Float[Array, "1"]
            A 1-dimensional jnp.array representing the time point.
        u : eqx.Module
            The network with a call signature `u(t, params)`.
        params : Params | ParamsDict
            The equation and neural network parameters $\theta$ and $\nu$.

        Returns
        -------
        float
            The residual, *i.e.* the differential operator $\mathcal{N}_\theta[u_\nu](t)$ evaluated at point `t`.

        Raises
        ------
        NotImplementedError
            This is an abstract method to be implemented.
        """
        raise NotImplementedError


class PDEStatio(DynamicLoss):
    r"""
    Abstract base class for stationnary PDE dynamic losses. All dynamic loss must subclass this class and override the abstract method `equation`.

    Attributes
    ----------
    Tmax : float, default=1
        Tmax needs to be given when the PINN time input is normalized in
        [0, 1], ie. we have performed renormalization of the differential
        equation
    eq_params_heterogeneity : Dict[str, Callable | None], default=None
        Default None. A dict with the keys being the same as in eq_params
        and the value being either None (no heterogeneity) or a function
        which encodes for the spatio-temporal heterogeneity of the parameter.
        Such a function must be jittable and take four arguments `t`, `x`,
        `u` and `params` even if one is not used. Therefore,
        one can introduce spatio-temporal covariates upon which a particular
        parameter can depend, e.g. in a GLM fashion. The effect of these
        covariables can themselves be estimated by being in `eq_params` too.
        Some key can be missing, in this case there is no heterogeneity (=None).
        If eq_params_heterogeneity is None this means there is no
        heterogeneity for no parameters.
    """

    _eq_type: ClassVar[str] = "Statio PDE"

    @partial(_decorator_heteregeneous_params)
    def evaluate(
        self, x: Float[Array, "dimension"], u: eqx.Module, params: Params | ParamsDict
    ) -> float:
        """Here we call the DynamicLoss._evaluate with t=None"""
        return self._evaluate(x, u, params)

    @abc.abstractmethod
    def equation(
        self, x: Float[Array, "d"], u: eqx.Module, params: Params | ParamsDict
    ) -> float:
        r"""The differential operator defining the stationnary PDE.

        !!! warning

            This is an abstract method to be implemented by users.

        Parameters
        ----------
        x : Float[Array, "d"]
            A `d` dimensional jnp.array representing a point in the spatial domain $\Omega$.
        u : eqx.Module
            The neural network.
        params : Params | ParamsDict
            The parameters of the equation and the networks, $\theta$ and $\nu$ respectively.

        Returns
        -------
        float
            The residual, *i.e.* the differential operator $\mathcal{N}_\theta[u_\nu](x)$ evaluated at point `x`.

        Raises
        ------
        NotImplementedError
            This is an abstract method to be implemented.
        """
        raise NotImplementedError


class PDENonStatio(DynamicLoss):
    """
    Abstract base class for non-stationnary PDE dynamic losses. All dynamic loss must subclass this class and override the abstract method `equation`.

    Attributes
    ----------
    Tmax : float, default=1
        Tmax needs to be given when the PINN time input is normalized in
        [0, 1], ie. we have performed renormalization of the differential
        equation
    eq_params_heterogeneity : Dict[str, Callable | None], default=None
        Default None. A dict with the keys being the same as in eq_params
        and the value being either None (no heterogeneity) or a function
        which encodes for the spatio-temporal heterogeneity of the parameter.
        Such a function must be jittable and take four arguments `t`, `x`,
        `u` and `params` even if one is not used. Therefore,
        one can introduce spatio-temporal covariates upon which a particular
        parameter can depend, e.g. in a GLM fashion. The effect of these
        covariables can themselves be estimated by being in `eq_params` too.
        Some key can be missing, in this case there is no heterogeneity (=None).
        If eq_params_heterogeneity is None this means there is no
        heterogeneity for no parameters.
    """

    _eq_type: ClassVar[str] = "Non-statio PDE"

    @partial(_decorator_heteregeneous_params)
    def evaluate(
        self,
        t_x: Float[Array, "1 + dim"],
        u: eqx.Module,
        params: Params | ParamsDict,
    ) -> float:
        """Here we call the DynamicLoss._evaluate with full arguments"""
        ans = self._evaluate(t_x, u, params)
        return ans

    @abc.abstractmethod
    def equation(
        self,
        t_x: Float[Array, "1 + dim"],
        u: eqx.Module,
        params: Params | ParamsDict,
    ) -> float:
        r"""The differential operator defining the non-stationnary PDE.

        !!! warning

            This is an abstract method to be implemented by users.

        Parameters
        ----------
        t_x : Float[Array, "1 + dim"]
            A jnp array containing the concatenation of a time point and a point in $\Omega$
        u : eqx.Module
            The neural network.
        params : Params | ParamsDict
            The parameters of the equation and the networks, $\theta$ and $\nu$ respectively.
        Returns
        -------
        float
            The residual, *i.e.* the differential operator $\mathcal{N}_\theta[u_\nu](t, x)$ evaluated at point `(t, x)`.


        Raises
        ------
        NotImplementedError
            This is an abstract method to be implemented.
        """
        raise NotImplementedError
