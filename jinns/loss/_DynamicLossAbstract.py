"""
Implements abstract classes for dynamic losses
"""

from __future__ import (
    annotations,
)  # https://docs.python.org/3/library/typing.html#constant

import warnings
import abc
from functools import partial
from typing import Callable, TYPE_CHECKING, ClassVar, Generic, TypeVar
import equinox as eqx
from jaxtyping import Float, Array
import jax.numpy as jnp


# See : https://docs.kidger.site/equinox/api/module/advanced_fields/#equinox.AbstractClassVar--known-issues
if TYPE_CHECKING:
    from typing import ClassVar as AbstractClassVar
    from jinns.parameters import Params
    from jinns.nn._abstract_pinn import AbstractPINN
else:
    from equinox import AbstractClassVar

InputDim = TypeVar("InputDim")


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


class DynamicLoss(eqx.Module, Generic[InputDim]):
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
    eq_params_heterogeneity : dict[str, Callable | None], default=None
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
    vectorial_dyn_loss_ponderation : Float[Array, " dim"], default=None
        Add a different ponderation weight to each of the dimension to the
        dynamic loss. This array must have the same dimension as the output of
        the dynamic loss equation or an error is raised. Default is None which
        means that a ponderation of 1 is applied on each dimension.
        `vectorial_dyn_loss_ponderation`
        is different from loss weights, which are attributes of Loss
        classes and which implement scalar (and possibly dynamic)
        ponderations for each term of the total loss.
        `vectorial_dyn_loss_ponderation` can be used with loss weights.
    """

    _eq_type = AbstractClassVar[str]  # class variable denoting the type of
    # differential equation
    Tmax: Float = eqx.field(kw_only=True, default=1)
    eq_params_heterogeneity: dict[str, Callable | None] | None = eqx.field(
        kw_only=True, default=None, static=True
    )
    vectorial_dyn_loss_ponderation: Float[Array, " dim"] | None = eqx.field(
        kw_only=True, default=None
    )

    def __post_init__(self):
        if self.vectorial_dyn_loss_ponderation is None:
            self.vectorial_dyn_loss_ponderation = jnp.array(1.0)

    def _eval_heterogeneous_parameters(
        self,
        inputs: InputDim,
        u: AbstractPINN,
        params: Params[Array],
        eq_params_heterogeneity: dict[str, Callable | None] | None = None,
    ) -> dict[str, Array]:
        eq_params_ = {}
        if eq_params_heterogeneity is None:
            return params.eq_params

        for k, p in params.eq_params.items():
            try:
                if eq_params_heterogeneity[k] is not None:
                    eq_params_[k] = eq_params_heterogeneity[k](inputs, u, params)  # type: ignore don't know why pyright says
                    # eq_params_heterogeneity[k] can be None here
                else:
                    eq_params_[k] = p
            except KeyError:
                # we authorize missing eq_params_heterogeneity key
                # if its heterogeneity is None anyway
                eq_params_[k] = p
        return eq_params_

    @partial(_decorator_heteregeneous_params)
    def evaluate(
        self,
        inputs: InputDim,
        u: AbstractPINN,
        params: Params[Array],
    ) -> float:
        evaluation = self.vectorial_dyn_loss_ponderation * self.equation(
            inputs, u, params
        )
        if len(evaluation.shape) == 0:
            raise ValueError(
                "The output of dynamic loss must be vectorial, "
                "i.e. of shape (d,) with d >= 1"
            )
        if len(evaluation.shape) > 1:
            warnings.warn(
                "Return value from DynamicLoss' equation has more "
                "than one dimension. This is in general a mistake (probably from "
                "an unfortunate broadcast in jnp.array computations) resulting in "
                "bad reduction operations in losses."
            )
        return evaluation

    @abc.abstractmethod
    def equation(self, *args, **kwargs):
        # TO IMPLEMENT
        # Point-wise evaluation of the differential equation N[u](.)
        raise NotImplementedError("You should implement your equation.")


class ODE(DynamicLoss[Float[Array, " 1"]]):
    r"""
    Abstract base class for ODE dynamic losses. All dynamic loss must subclass
    this class and override the abstract method `equation`.

    Attributes
    ----------
    Tmax : float, default=1
        Tmax needs to be given when the PINN time input is normalized in
        [0, 1], ie. we have performed renormalization of the differential
        equation
    eq_params_heterogeneity : dict[str, Callable | None], default=None
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

    @abc.abstractmethod
    def equation(
        self, t: Float[Array, " 1"], u: AbstractPINN, params: Params[Array]
    ) -> float:
        r"""
        The differential operator defining the ODE.

        !!! warning

            This is an abstract method to be implemented by users.

        Parameters
        ----------
        t : Float[Array, " 1"]
            A 1-dimensional jnp.array representing the time point.
        u : AbstractPINN
            The network with a call signature `u(t, params)`.
        params : Params[Array]
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


class PDEStatio(DynamicLoss[Float[Array, " dim"]]):
    r"""
    Abstract base class for stationnary PDE dynamic losses. All dynamic loss must subclass this class and override the abstract method `equation`.

    Attributes
    ----------
    Tmax : float, default=1
        Tmax needs to be given when the PINN time input is normalized in
        [0, 1], ie. we have performed renormalization of the differential
        equation
    eq_params_heterogeneity : dict[str, Callable | None], default=None
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

    @abc.abstractmethod
    def equation(
        self, x: Float[Array, " dim"], u: AbstractPINN, params: Params[Array]
    ) -> float:
        r"""The differential operator defining the stationnary PDE.

        !!! warning

            This is an abstract method to be implemented by users.

        Parameters
        ----------
        x : Float[Array, " dim"]
            A `d` dimensional jnp.array representing a point in the spatial domain $\Omega$.
        u : AbstractPINN
            The neural network.
        params : Params[Array]
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


class PDENonStatio(DynamicLoss[Float[Array, " 1 + dim"]]):
    """
    Abstract base class for non-stationnary PDE dynamic losses. All dynamic loss must subclass this class and override the abstract method `equation`.

    Attributes
    ----------
    Tmax : float, default=1
        Tmax needs to be given when the PINN time input is normalized in
        [0, 1], ie. we have performed renormalization of the differential
        equation
    eq_params_heterogeneity : dict[str, Callable | None], default=None
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

    @abc.abstractmethod
    def equation(
        self,
        t_x: Float[Array, " 1 + dim"],
        u: AbstractPINN,
        params: Params[Array],
    ) -> float:
        r"""The differential operator defining the non-stationnary PDE.

        !!! warning

            This is an abstract method to be implemented by users.

        Parameters
        ----------
        t_x : Float[Array, " 1 + dim"]
            A jnp array containing the concatenation of a time point and a point in $\Omega$
        u : AbstractPINN
            The neural network.
        params : Params[Array]
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
