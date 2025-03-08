"""
Implements several dynamic losses
"""

from __future__ import (
    annotations,
)  # https://docs.python.org/3/library/typing.html#constant

from typing import TYPE_CHECKING, Dict
from jaxtyping import Float
import jax
from jax import grad
import jax.numpy as jnp
import equinox as eqx

from jinns.nn._pinn import PINN
from jinns.nn._spinn_mlp import SPINN

from jinns.utils._utils import get_grid
from jinns.loss._DynamicLossAbstract import ODE, PDEStatio, PDENonStatio
from jinns.loss._operators import (
    laplacian_rev,
    laplacian_fwd,
    divergence_rev,
    divergence_fwd,
    vectorial_laplacian_rev,
    vectorial_laplacian_fwd,
    _u_dot_nabla_times_u_rev,
    _u_dot_nabla_times_u_fwd,
)

from jaxtyping import Array, Float

if TYPE_CHECKING:
    from jinns.parameters import Params, ParamsDict


class FisherKPP(PDENonStatio):
    r"""
    Return the Fisher KPP dynamic loss term. Dimension of $x$ can be
    arbitrary

    $$
    \frac{\partial}{\partial t} u(t,x)=D\Delta u(t,x) + u(t,x)(r(x) - \gamma(x)u(t,x))
    $$

    Parameters
    ----------
    dim_x : int, default=1
        The dimension of x, the space domain. Default is 1.
    """

    dim_x: int = eqx.field(default=1, static=True)

    def equation(
        self,
        t_x: Float[Array, "1+dim"],
        u: eqx.Module,
        params: Params,
    ) -> Float[Array, "1"]:
        r"""
        Evaluate the dynamic loss at $(t, x)$.

        Parameters
        ---------
        t_x
            A jnp array containing the concatenation of a time point
            and a point in $\Omega$
        u
            The PINN
        params
            The dictionary of parameters of the model.
            Typically, it is a dictionary of
            dictionaries: `eq_params` and `nn_params`, respectively the
            differential equation parameters and the neural network parameter
        """
        if isinstance(u, PINN):
            # Note that the last dim of u is nec. 1
            u_ = lambda t_x: u(t_x, params)[0]

            du_dt = grad(u_)(t_x)[0]

            lap = laplacian_rev(t_x, u, params, eq_type=u.eq_type)[..., None]

            return du_dt + self.Tmax * (
                -params.eq_params["D"] * lap
                - u(t_x, params)
                * (params.eq_params["r"] - params.eq_params["g"] * u(t_x, params))
            )
        if isinstance(u, SPINN):
            s = jnp.zeros((1, self.dim_x + 1))
            s = s.at[0].set(1.0)
            v0 = jnp.repeat(s, t_x.shape[0], axis=0)
            u_tx, du_dt = jax.jvp(
                lambda t_x: u(t_x, params),
                (t_x,),
                (v0,),
            )
            lap = laplacian_fwd(t_x, u, params, eq_type=u.eq_type)

            return du_dt + self.Tmax * (
                -params.eq_params["D"] * lap
                - u_tx * (params.eq_params["r"] - params.eq_params["g"] * u_tx)
            )
        raise ValueError("u is not among the recognized types (PINN or SPINN)")


class GeneralizedLotkaVolterra(ODE):
    r"""
    Return a dynamic loss from an equation of a Generalized Lotka Volterra
    system. Say we implement the equation for population $i$

    $$
        \frac{\partial}{\partial t}u_i(t) = r_iu_i(t) - \sum_{j\neq i}\alpha_{ij}u_j(t)
        -\alpha_{i,i}u_i(t) + c_iu_i(t) + \sum_{j \neq i} c_ju_j(t)
    $$
    with $r_i$ the growth rate parameter, $c_i$ the carrying
    capacities and $\alpha_{ij}$ the interaction terms.

    Parameters
    ----------
    key_main
        The dictionary key (in the dictionaries `u` and `params` that
        are arguments of the `evaluate` function) of the main population
        $i$ of the particular equation of the system implemented
        by this dynamic loss
    keys_other
        The list of dictionary keys (in the dictionaries `u` and `params` that
        are arguments of the `evaluate` function) of the other
        populations that appear in the equation of the system implemented
        by this dynamic loss
    Tmax
        Tmax needs to be given when the PINN time input is normalized in
        $[0, 1]$, ie. we have performed renormalization of the differential
        equation.
    eq_params_heterogeneity
        Default None. A dict with the keys being the same as in eq_params
        and the value being `time`, `space`, `both` or None which corresponds to
        the heterogeneity of a given parameter. A value can be missing, in
        this case there is no heterogeneity (=None). If
        eq_params_heterogeneity is None this means there is no
        heterogeneity for no parameters.
    """

    # they should be static because they are list of strings
    key_main: list[str] = eqx.field(static=True)
    keys_other: list[str] = eqx.field(static=True)

    def equation(
        self,
        t: Float[Array, "1"],
        u_dict: Dict[str, eqx.Module],
        params_dict: ParamsDict,
    ) -> Float[Array, "1"]:
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
        params_main = params_dict.extract_params(self.key_main)

        u = u_dict[self.key_main]
        # need to index with [0] since u output is nec (1,)
        du_dt = grad(lambda t: jnp.log(u(t, params_main)[0]), 0)(t)
        carrying_term = params_main.eq_params["carrying_capacity"] * u(t, params_main)
        # NOTE the following assumes interaction term with oneself is at idx 0
        interaction_terms = params_main.eq_params["interactions"][0] * u(t, params_main)

        # TODO write this for loop with tree_util functions?
        for i, k in enumerate(self.keys_other):
            params_k = params_dict.extract_params(k)
            carrying_term += params_main.eq_params["carrying_capacity"] * u_dict[k](
                t, params_k
            )
            interaction_terms += params_main.eq_params["interactions"][i + 1] * u_dict[
                k
            ](t, params_k)

        return du_dt + self.Tmax * (
            -params_main.eq_params["growth_rate"] - interaction_terms + carrying_term
        )


class BurgersEquation(PDENonStatio):
    r"""
    Return the Burgers dynamic loss term (in 1 space dimension):

    $$
        \frac{\partial}{\partial t} u(t,x) + u(t,x)\frac{\partial}{\partial x}
        u(t,x) - \theta \frac{\partial^2}{\partial x^2} u(t,x) = 0
    $$

    Parameters
    ----------
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

    def equation(
        self,
        t_x: Float[Array, "1+dim"],
        u: eqx.Module,
        params: Params,
    ) -> Float[Array, "1"]:
        r"""
        Evaluate the dynamic loss at :math:`(t,x)`.

        Parameters
        ----------
        t_x
            A jnp array containing the concatenation of a time point
            and a point in $\Omega$
        u
            The PINN
        params
            The dictionary of parameters of the model.
        """
        if isinstance(u, PINN):
            u_ = lambda t_x: jnp.squeeze(u(t_x, params)[u.slice_solution])
            du_dtx = grad(u_)
            d2u_dx_dtx = grad(lambda t_x: du_dtx(t_x)[1])
            du_dtx_values = du_dtx(t_x)

            return du_dtx_values[0:1] + self.Tmax * (
                u_(t_x) * du_dtx_values[1:2]
                - params.eq_params["nu"] * d2u_dx_dtx(t_x)[1:2]
            )

        if isinstance(u, SPINN):
            # d=2 JVP calls are expected since we have time and x
            # then with a batch of size B, we then have Bd JVP calls
            v0 = jnp.repeat(jnp.array([[1.0, 0.0]]), t_x.shape[0], axis=0)
            v1 = jnp.repeat(jnp.array([[0.0, 1.0]]), t_x.shape[0], axis=0)
            u_tx, du_dt = jax.jvp(
                lambda t_x: u(t_x, params),
                (t_x,),
                (v0,),
            )
            _, du_dx = jax.jvp(
                lambda t_x: u(t_x, params),
                (t_x,),
                (v1,),
            )
            # both calls above could be condensed into the one jacfwd below
            # u_ = lambda t_x: u(t_x, params)
            # J = jax.jacfwd(u_)(t_x)

            du_dx_fun = lambda t_x: jax.jvp(
                lambda t_x: u(t_x, params),
                (t_x,),
                (v1,),
            )[1]
            _, d2u_dx2 = jax.jvp(du_dx_fun, (t_x,), (v1,))
            # Note that ones_like(x) works because x is Bx1 !
            return du_dt + self.Tmax * (u_tx * du_dx - params.eq_params["nu"] * d2u_dx2)
        raise ValueError("u is not among the recognized types (PINN or SPINN)")


class FPENonStatioLoss2D(PDENonStatio):
    r"""
    Return the dynamic loss for a non-stationary Fokker Planck Equation in two
    dimensions:

    $$
        -\sum_{i=1}^2\frac{\partial}{\partial \mathbf{x}}
        \left[\mu(t, \mathbf{x})u(t, \mathbf{x})\right] +
        \sum_{i=1}^2\sum_{j=1}^2\frac{\partial^2}{\partial x_i \partial x_j}
        \left[D(t, \mathbf{x})u(t, \mathbf{x})\right]= \frac{\partial}
        {\partial t}u(t,\mathbf{x})
    $$
    where $\mu(t, \mathbf{x})$ is the drift term and $D(t, \mathbf{x})$ is the
    diffusion term.

    The drift and diffusion terms are not specified here, hence this class
    is `abstract`.
    Other classes inherit from FPENonStatioLoss2D and define the drift and
    diffusion terms, which then defines several other dynamic losses
    (Ornstein-Uhlenbeck, Cox-Ingersoll-Ross, ...)

    Parameters
    ----------
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

    def equation(
        self,
        t_x: Float[Array, "1+dim"],
        u: eqx.Module,
        params: Params,
    ) -> Float[Array, "1"]:
        r"""
        Evaluate the dynamic loss at $(t,\mathbf{x})$.

        Parameters
        ---------
        t_x
            A collocation point in  $I\times\Omega$
        u
            The PINN
        params
            The dictionary of parameters of the model.
            Typically, it is a dictionary of
            dictionaries: `eq_params` and `nn_params`, respectively the
            differential equation parameters and the neural network parameter
        """
        if isinstance(u, PINN):
            # Note that the last dim of u is nec. 1
            u_ = lambda t_x: u(t_x, params)[0]

            order_1_fun = lambda t_x: self.drift(t_x[1:], params.eq_params) * u_(t_x)
            grad_order_1 = jnp.trace(jax.jacrev(order_1_fun)(t_x)[..., 1:])[None]

            order_2_fun = lambda t_x: self.diffusion(t_x[1:], params.eq_params) * u_(
                t_x
            )
            grad_order_2_fun = lambda t_x: jax.jacrev(order_2_fun)(t_x)[..., 1:]
            grad_grad_order_2 = (
                jnp.trace(
                    jax.jacrev(lambda t_x: grad_order_2_fun(t_x)[0, :, 0])(t_x)[..., 1:]
                )[None]
                + jnp.trace(
                    jax.jacrev(lambda t_x: grad_order_2_fun(t_x)[1, :, 1])(t_x)[..., 1:]
                )[None]
            )
            # This is be a condensed form of the explicit which is less efficient
            # since 4 jacrev are called (as compared to 2)
            # grad_order_2_fun = lambda t_x, i, j: jax.jacrev(order_2_fun)(t_x)[i, j, 1:]
            # grad_grad_order_2 = (
            #    jax.jacrev(lambda t_x: grad_order_2_fun(t_x, 0, 0))(t_x)[0, 1] +
            #    jax.jacrev(lambda t_x: grad_order_2_fun(t_x, 1, 0))(t_x)[1, 1] +
            #    jax.jacrev(lambda t_x: grad_order_2_fun(t_x, 0, 1))(t_x)[0, 2] +
            #    jax.jacrev(lambda t_x: grad_order_2_fun(t_x, 1, 1))(t_x)[1, 2]
            # )[None]

            du_dt = grad(u_)(t_x)[0:1]

            return -du_dt + self.Tmax * (-grad_order_1 + grad_grad_order_2)

        if isinstance(u, SPINN):
            v0 = jnp.repeat(jnp.array([[1.0, 0.0, 0.0]]), t_x.shape[0], axis=0)
            _, du_dt = jax.jvp(
                lambda t_x: u(t_x, params),
                (t_x,),
                (v0,),
            )

            # in forward AD we do not have the results for all the input
            # dimension at once (as it is the case with grad), we then write
            # two jvp calls
            v1 = jnp.repeat(jnp.array([[0.0, 1.0, 0.0]]), t_x.shape[0], axis=0)
            v2 = jnp.repeat(jnp.array([[0.0, 0.0, 1.0]]), t_x.shape[0], axis=0)
            _, dau_dx1 = jax.jvp(
                lambda t_x: self.drift(get_grid(t_x[:, 1:]), params.eq_params)[
                    None, ..., 0:1
                ]
                * u(t_x, params),
                (t_x,),
                (v1,),
            )
            _, dau_dx2 = jax.jvp(
                lambda t_x: self.drift(get_grid(t_x[:, 1:]), params.eq_params)[
                    None, ..., 1:2
                ]
                * u(t_x, params),
                (t_x,),
                (v2,),
            )

            dsu_dx1_fun = lambda t_x, i, j: jax.jvp(
                lambda t_x: self.diffusion(
                    get_grid(t_x[:, 1:]), params.eq_params, i, j
                )[None, None, None, None]
                * u(t_x, params),
                (t_x,),
                (v1,),
            )[1]
            dsu_dx2_fun = lambda t_x, i, j: jax.jvp(
                lambda t_x: self.diffusion(
                    get_grid(t_x[:, 1:]), params.eq_params, i, j
                )[None, None, None, None]
                * u(t_x, params),
                (t_x,),
                (v2,),
            )[1]
            _, d2su_dx12 = jax.jvp(lambda t_x: dsu_dx1_fun(t_x, 0, 0), (t_x,), (v1,))
            _, d2su_dx1dx2 = jax.jvp(lambda t_x: dsu_dx1_fun(t_x, 0, 1), (t_x,), (v2,))
            _, d2su_dx22 = jax.jvp(lambda t_x: dsu_dx2_fun(t_x, 1, 1), (t_x,), (v2,))
            _, d2su_dx2dx1 = jax.jvp(lambda t_x: dsu_dx2_fun(t_x, 1, 0), (t_x,), (v1,))

            return -du_dt + self.Tmax * (
                -(dau_dx1 + dau_dx2)
                + (d2su_dx12 + d2su_dx22 + d2su_dx1dx2 + d2su_dx2dx1)
            )
        raise ValueError("u is not among the recognized types (PINN or SPINN)")

    def drift(self, *args, **kwargs):
        # To be implemented in child classes
        raise NotImplementedError("Drift function should be implemented")

    def diffusion(self, *args, **kwargs):
        # To be implemented in child classes
        raise NotImplementedError("Diffusion function should be implemented")


class OU_FPENonStatioLoss2D(FPENonStatioLoss2D):
    r"""
    Return the dynamic loss for a stationary Fokker Planck Equation in two
    dimensions:

    $$
        -\sum_{i=1}^2\frac{\partial}{\partial \mathbf{x}}
        \left[(\alpha(\mu - \mathbf{x}))u(t,\mathbf{x})\right] +
        \sum_{i=1}^2\sum_{j=1}^2\frac{\partial^2}{\partial x_i \partial x_j}
        \left[\frac{\sigma^2}{2}u(t,\mathbf{x})\right]=
        \frac{\partial}
        {\partial t}u(t,\mathbf{x})
    $$

    Parameters
    ----------
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

    def drift(self, x, eq_params):
        r"""
        Return the drift term

        Parameters
        ----------
        x
            A point in $\Omega$
        eq_params
            A dictionary containing the equation parameters
        """
        return eq_params["alpha"] * (eq_params["mu"] - x)

    def sigma_mat(self, x, eq_params):
        r"""
        Return the square root of the diffusion tensor in the sense of the outer
        product used to create the diffusion tensor

        Parameters
        ----------
        x
            A point in $\Omega$
        eq_params
            A dictionary containing the equation parameters
        """

        return jnp.diag(eq_params["sigma"])

    def diffusion(self, x, eq_params, i=None, j=None):
        r"""
        Return the computation of the diffusion tensor term in 2D (or
        higher)

        Parameters
        ----------
        x
            A point in $\Omega$
        eq_params
            A dictionary containing the equation parameters
        """
        if i is None or j is None:
            return 0.5 * (
                jnp.matmul(
                    self.sigma_mat(x, eq_params),
                    jnp.transpose(self.sigma_mat(x, eq_params)),
                )
            )
        return 0.5 * (
            jnp.matmul(
                self.sigma_mat(x, eq_params),
                jnp.transpose(self.sigma_mat(x, eq_params)),
            )[i, j]
        )


class MassConservation2DStatio(PDEStatio):
    r"""
    Returns the so-called mass conservation equation.

    $$
        \nabla \cdot \mathbf{u} = \frac{\partial}{\partial x}u(x,y) +
        \frac{\partial}{\partial y}u(x,y) = 0,
    $$
    where $u$ is a stationary function, i.e., it does not depend on
    $t$.

    Parameters
    ----------
    nn_key
        A dictionary key which identifies, in `u_dict` the PINN that
        appears in the mass conservation equation.
    eq_params_heterogeneity
        Default None. A dict with the keys being the same as in eq_params
        and the value being `time`, `space`, `both` or None which corresponds to
        the heterogeneity of a given parameter. A value can be missing, in
        this case there is no heterogeneity (=None). If
        eq_params_heterogeneity is None this means there is no
        heterogeneity for no parameters.
    """

    # an str field should be static (not a valid JAX type)
    nn_key: str = eqx.field(static=True)

    def equation(
        self,
        x: Float[Array, "dim"],
        u_dict: Dict[str, eqx.Module],
        params_dict: ParamsDict,
    ) -> Float[Array, "1"]:
        r"""
        Evaluate the dynamic loss at `\mathbf{x}`.
        For stability we implement the dynamic loss in log space.

        Parameters
        ---------
        x
            A point in $\Omega\subset\mathbb{R}^2$
        u_dict
            A dictionary of PINNs. Must have the same keys as `params_dict`
        params_dict
            The dictionary of dictionaries of parameters of the model.
            Typically, each sub-dictionary is a dictionary
            with keys: `eq_params` and `nn_params`, respectively the
            differential equation parameters and the neural network parameter.
            Must have the same keys as `u_dict`
        """
        params = params_dict.extract_params(self.nn_key)

        if isinstance(u_dict[self.nn_key], PINN):
            u = u_dict[self.nn_key]

            return divergence_rev(x, u, params)[..., None]

        if isinstance(u_dict[self.nn_key], SPINN):
            u = u_dict[self.nn_key]

            return divergence_fwd(x, u, params)[..., None]
        raise ValueError("u is not among the recognized types (PINN or SPINN)")


class NavierStokes2DStatio(PDEStatio):
    r"""
    Return the dynamic loss for all the components of the stationary Navier Stokes
    equation which is a 2D vectorial PDE.

    $$
       (\mathbf{u}\cdot\nabla)\mathbf{u} + \frac{1}{\rho}\nabla p - \theta
       \nabla^2\mathbf{u}=0,
    $$

    or, in 2D,


    $$
        \begin{pmatrix}u_x\frac{\partial}{\partial x} u_x +
        u_y\frac{\partial}{\partial y} u_x, \\
        u_x\frac{\partial}{\partial x} u_y + u_y\frac{\partial}{\partial y} u_y  \end{pmatrix} +
        \frac{1}{\rho} \begin{pmatrix} \frac{\partial}{\partial x} p, \\ \frac{\partial}{\partial y} p \end{pmatrix}
        - \theta
        \begin{pmatrix}
        \frac{\partial^2}{\partial x^2} u_x + \frac{\partial^2}{\partial y^2}
        u_x, \\
        \frac{\partial^2}{\partial x^2} u_y + \frac{\partial^2}{\partial y^2} u_y
        \end{pmatrix} = 0,
    $$
    with $\theta$ the viscosity coefficient and $\rho$ the density coefficient.

    Parameters
    ----------
    u_key
        A dictionary key which indices the NN u in `u_dict`
        the PINN with the role of the velocity in the equation.
        Its input is bimensional (points in $\Omega\subset\mathbb{R}^2$).
        Its output is bimensional as it represents a velocity vector
        field
    p_key
        A dictionary key which indices the NN p in `u_dict`
        the PINN with the role of the pressure in the equation.
        Its input is bimensional (points in $\Omega\subset\mathbb{R}^2).
        Its output is unidimensional as it represents a pressure scalar
        field
    eq_params_heterogeneity
        Default None. A dict with the keys being the same as in eq_params
        and the value being `time`, `space`, `both` or None which corresponds to
        the heterogeneity of a given parameter. A value can be missing, in
        this case there is no heterogeneity (=None). If
        eq_params_heterogeneity is None this means there is no
        heterogeneity for no parameters.
    """

    u_key: str = eqx.field(static=True)
    p_key: str = eqx.field(static=True)

    def equation(
        self,
        x: Float[Array, "dim"],
        u_dict: Dict[str, eqx.Module],
        params_dict: ParamsDict,
    ) -> Float[Array, "1"]:
        r"""
        Evaluate the dynamic loss at `x`.
        For stability we implement the dynamic loss in log space.

        Parameters
        ---------
        x
            A point in $\Omega\subset\mathbb{R}^2$
        u_dict
            A dictionary of PINNs. Must have the same keys as `params_dict`
        params_dict
            The dictionary of dictionaries of parameters of the model.
            Typically, each sub-dictionary is a dictionary
            with keys: `eq_params` and `nn_params`, respectively the
            differential equation parameters and the neural network parameter.
            Must have the same keys as `u_dict`
        """
        u_params = params_dict.extract_params(self.u_key)
        p_params = params_dict.extract_params(self.p_key)

        if isinstance(u_dict[self.u_key], PINN):
            u = u_dict[self.u_key]

            u_dot_nabla_x_u = _u_dot_nabla_times_u_rev(x, u, u_params)

            p = lambda x: u_dict[self.p_key](x, p_params)
            jac_p = jax.jacrev(p, 0)(x)  # compute the gradient

            vec_laplacian_u = vectorial_laplacian_rev(x, u, u_params, dim_out=2)

            # dynamic loss on x axis
            result_x = (
                u_dot_nabla_x_u[0]
                + 1 / params_dict.eq_params["rho"] * jac_p[0, 0]
                - params_dict.eq_params["nu"] * vec_laplacian_u[0]
            )

            # dynamic loss on y axis
            result_y = (
                u_dot_nabla_x_u[1]
                + 1 / params_dict.eq_params["rho"] * jac_p[0, 1]
                - params_dict.eq_params["nu"] * vec_laplacian_u[1]
            )

            # output is 2D
            return jnp.stack([result_x, result_y], axis=-1)

        if isinstance(u_dict[self.u_key], SPINN):
            u = u_dict[self.u_key]

            u_dot_nabla_x_u = _u_dot_nabla_times_u_fwd(x, u, u_params)

            p = lambda x: u_dict[self.p_key](x, p_params)

            tangent_vec_0 = jnp.repeat(jnp.array([1.0, 0.0])[None], x.shape[0], axis=0)
            _, dp_dx = jax.jvp(p, (x,), (tangent_vec_0,))
            tangent_vec_1 = jnp.repeat(jnp.array([0.0, 1.0])[None], x.shape[0], axis=0)
            _, dp_dy = jax.jvp(p, (x,), (tangent_vec_1,))

            vec_laplacian_u = vectorial_laplacian_fwd(x, u, u_params, dim_out=2)

            # dynamic loss on x axis
            result_x = (
                u_dot_nabla_x_u[..., 0]
                + 1 / params_dict.eq_params["rho"] * dp_dx.squeeze()
                - params_dict.eq_params["nu"] * vec_laplacian_u[..., 0]
            )
            # dynamic loss on y axis
            result_y = (
                u_dot_nabla_x_u[..., 1]
                + 1 / params_dict.eq_params["rho"] * dp_dy.squeeze()
                - params_dict.eq_params["nu"] * vec_laplacian_u[..., 1]
            )

            # output is 2D
            return jnp.stack([result_x, result_y], axis=-1)
        raise ValueError("u is not among the recognized types (PINN or SPINN)")
