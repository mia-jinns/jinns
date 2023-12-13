import jax
from jax import jit, grad, jacrev, jacfwd
import jax.numpy as jnp
from jinns.utils._utils import _get_grid
from jinns.utils._pinn import PINN
from jinns.utils._spinn import SPINN
from jinns.loss._DynamicLossAbstract import ODE, PDEStatio, PDENonStatio
from jinns.loss._operators import (
    _laplacian_rev,
    _laplacian_fwd,
    _div_rev,
    _div_fwd,
    _vectorial_laplacian,
    _u_dot_nabla_times_u_rev,
    _u_dot_nabla_times_u_fwd,
)


class FisherKPP(PDENonStatio):
    r"""
    Return the Fisher KPP dynamic loss term. Dimension of :math:`x` can be
    arbitrary

    .. math::
        \frac{\partial}{\partial t} u(t,x)=D\Delta u(t,x) + u(t,x)(r(x) - \gamma(x)u(t,x))

    """

    def __init__(self, Tmax=1, derivatives="nn_params", eq_params_heterogeneity=None):
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

    def evaluate(self, t, x, u, params):
        """
        Evaluate the dynamic loss at :math:`(t,x)`.

        Parameters
        ---------
        t
            A time point
        x
            A point in :math:`\Omega`
        u
            The PINN
        params
            The dictionary of parameters of the model.
            Typically, it is a dictionary of
            dictionaries: `eq_params` and `nn_params``, respectively the
            differential equation parameters and the neural network parameter
        """
        if isinstance(u, PINN):
            nn_params, eq_params = self.set_stop_gradient(params)
            eq_params = self._eval_heterogeneous_parameters(
                eq_params, t, x, self.eq_params_heterogeneity
            )

            # Note that the last dim of u is nec. 1
            u_ = lambda t, x: u(t, x, nn_params, eq_params)[0]

            du_dt = grad(u_, 0)(t, x)

            lap = _laplacian_rev(u, nn_params, eq_params, x, t)[..., None]

            return du_dt + self.Tmax * (
                -eq_params["D"] * lap
                - u(t, x, nn_params, eq_params)
                * (eq_params["r"] - eq_params["g"] * u(t, x, nn_params, eq_params))
            )
        elif isinstance(u, SPINN):
            nn_params, eq_params = self.set_stop_gradient(params)
            x_grid = _get_grid(x)
            eq_params = self._eval_heterogeneous_parameters(
                eq_params, t, x_grid, self.eq_params_heterogeneity
            )

            u_tx, du_dt = jax.jvp(
                lambda t: u(t, x, nn_params, eq_params),
                (t,),
                (jnp.ones_like(t),),
            )
            lap = _laplacian_fwd(u, nn_params, eq_params, x, t)[..., None]
            return du_dt + self.Tmax * (
                -eq_params["D"] * lap
                - u_tx * (eq_params["r"][..., None] - eq_params["g"] * u_tx)
            )


class Malthus(ODE):
    r"""
    Return a Malthus dynamic loss term following the PINN logic:

    .. math::
        \frac{\partial}{\partial t} u(t)=ru(t)

    """

    def __init__(self, Tmax=1, derivatives="nn_params", eq_params_heterogeneity=None):
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

    def evaluate(self, t, u, params):
        """
        Evaluate the dynamic loss at `t`.
        For stability we implement the dynamic loss in log space.

        Parameters
        ---------
        t
            A time point
        u
            The PINN
        params
            The dictionary of parameters of the model.
            Typically, it is a dictionary of
            dictionaries: `eq_params` and `nn_params``, respectively the
            differential equation parameters and the neural network parameter
        """
        nn_params, eq_params = self.set_stop_gradient(params)

        eq_params = self._eval_heterogeneous_parameters(
            eq_params, t, x, self.eq_params_heterogeneity
        )

        # NOTE the log formulation of the loss for stability
        du_dt = grad(lambda t: jnp.log(u(t, nn_params, eq_params)), 0)(t)
        return du_dt - eq_params["growth_rate"]


class BurgerEquation(PDENonStatio):
    r"""
    Return the Burger dynamic loss term (in 1 space dimension):

    .. math::
        \frac{\partial}{\partial t} u(t,x) + u(t,x)\frac{\partial}{\partial x}
        u(t,x) - \theta \frac{\partial^2}{\partial x^2} u(t,x) = 0

    """

    def __init__(
        self,
        Tmax=1,
        derivatives="nn_params",
        eq_params_heterogeneity=None,
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

    def evaluate(self, t, x, u, params):
        """
        Evaluate the dynamic loss at :math:`(t,x)`.

        Parameters
        ---------
        t
            A time point
        x
            A point in :math:`\Omega`
        u
            The PINN
        params
            The dictionary of parameters of the model.
            Typically, it is a dictionary of
            dictionaries: `eq_params` and `nn_params``, respectively the
            differential equation parameters and the neural network parameter
        """
        if isinstance(u, PINN):
            nn_params, eq_params = self.set_stop_gradient(params)
            eq_params = self._eval_heterogeneous_parameters(
                eq_params, t, x, self.eq_params_heterogeneity
            )

            # Note that the last dim of u is nec. 1
            u_ = lambda t, x: u(t, x, nn_params, eq_params)[0]
            du_dt = grad(u_, 0)
            du_dx = grad(u_, 1)
            d2u_dx2 = grad(
                lambda t, x: du_dx(t, x)[0],
                1,
            )

            return du_dt(t, x) + self.Tmax * (
                u(t, x, nn_params, eq_params) * du_dx(t, x)
                - eq_params["nu"] * d2u_dx2(t, x)
            )

        elif isinstance(u, SPINN):
            nn_params, eq_params = self.set_stop_gradient(params)
            x_grid = _get_grid(x)
            eq_params = self._eval_heterogeneous_parameters(
                eq_params, t, x_grid, self.eq_params_heterogeneity
            )
            # d=2 JVP calls are expected since we have time and x
            # then with a batch of size B, we then have Bd JVP calls
            u_tx, du_dt = jax.jvp(
                lambda t: u(t, x, nn_params, eq_params),
                (t,),
                (jnp.ones_like(t),),
            )
            du_dx_fun = lambda x: jax.jvp(
                lambda x: u(t, x, nn_params, eq_params),
                (x,),
                (jnp.ones_like(x),),
            )[1]
            du_dx, d2u_dx2 = jax.jvp(du_dx_fun, (x,), (jnp.ones_like(x),))
            # Note that ones_like(x) works because x is Bx1 !
            return du_dt + self.Tmax * (u_tx * du_dx - eq_params["nu"] * d2u_dx2)


class GeneralizedLotkaVolterra(ODE):
    r"""
    Return a dynamic loss from an equation of a Generalized Lotka Volterra
    system. Say we implement the equation for population :math:`i`:

    .. math::
        \frac{\partial}{\partial t}u_i(t) = r_iu_i(t) - \sum_{j\neq i}\alpha_{ij}u_j(t)
        -\alpha_{i,i}u_i(t) + c_iu_i(t) + \sum_{j \neq i} c_ju_j(t)

    with :math:`r_i` the growth rate parameter, :math:`c_i` the carrying
    capacities and :math:`\alpha_{ij}` the interaction terms.

    """

    def __init__(
        self,
        key_main,
        keys_other,
        Tmax=1,
        derivatives="nn_params",
        eq_params_heterogeneity=None,
    ):
        """
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
        self.key_main = key_main
        self.keys_other = keys_other

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
            The dictionary of dictionaries of parameters of the model.
            Typically, each sub-dictionary is a dictionary
            with keys: `eq_params` and `nn_params``, respectively the
            differential equation parameters and the neural network parameter.
            Must have the same keys as `u_dict`
        """
        nn_params, eq_params = self.set_stop_gradient(params_dict)

        u_nn_params = nn_params[self.key_main]
        u_eq_params = eq_params[self.key_main]

        u = u_dict[self.key_main]
        du_dt = grad(lambda t: jnp.log(u(t, u_nn_params, u_eq_params)), 0)(t)
        carrying_term = u_eq_params["carrying_capacity"] * u(
            t, u_nn_params, u_eq_params
        )
        for k in self.keys_other:
            carrying_term += u_eq_params["carrying_capacity"] * u_dict[k](
                t, nn_params[k], eq_params[k]
            )
        # NOTE the following assumes interaction term with oneself is at idx 0
        interaction_terms = u_eq_params["interactions"][0] * u(
            t, u_nn_params, u_eq_params
        )
        for i, k in enumerate(self.keys_other):
            interaction_terms += u_eq_params["interactions"][i + 1] * u_dict[k](
                t, nn_params[k], eq_params[k]
            )

        return du_dt + self.Tmax * (
            -u_eq_params["growth_rate"] - interaction_terms + carrying_term
        )


class FPENonStatioLoss2D(PDENonStatio):
    r"""
    Return the dynamic loss for a non-stationary Fokker Planck Equation in two
    dimensions:

    .. math::
        -\sum_{i=1}^2\frac{\partial}{\partial \mathbf{x}}
        \left[\mu(t, \mathbf{x})u(t, \mathbf{x})\right] +
        \sum_{i=1}^2\sum_{j=1}^2\frac{\partial^2}{\partial x_i \partial x_j}
        \left[D(t, \mathbf{x})u(t, \mathbf{x})\right]= \frac{\partial}
        {\partial t}u(t,\mathbf{x})

    where :math:`\mu(t, \mathbf{x})` is the drift term and :math:`D(t, \mathbf{x})` is the diffusion
    term.

    The drift and diffusion terms are not specified here, hence this class
    is `abstract`.
    Other classes inherit from FPENonStatioLoss2D and define the drift and
    diffusion terms, which then defines several other dynamic losses
    (Ornstein-Uhlenbeck, Cox-Ingersoll-Ross, ...)
    """

    def __init__(self, Tmax, derivatives="nn_params", eq_params_heterogeneity=None):
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

    def evaluate(self, t, x, u, params):
        """
        Evaluate the dynamic loss at :math:`(t,\mathbf{x})`.

        Parameters
        ---------
        t
            A time point
        x
            A point in :math:`\Omega`
        u
            The PINN
        params
            The dictionary of parameters of the model.
            Typically, it is a dictionary of
            dictionaries: `eq_params` and `nn_params``, respectively the
            differential equation parameters and the neural network parameter
        """
        if isinstance(u, PINN):
            nn_params, eq_params = self.set_stop_gradient(params)
            eq_params = self._eval_heterogeneous_parameters(
                eq_params, t, x, self.eq_params_heterogeneity
            )

            # Note that the last dim of u is nec. 1
            u_ = lambda t, x: u(t, x, nn_params, eq_params)[0]

            order_1 = (
                grad(
                    lambda t, x: self.drift(t, x, eq_params)[0] * u_(t, x),
                    1,
                )(
                    t, x
                )[0:1]
                + grad(
                    lambda t, x: self.drift(t, x, eq_params)[1] * u_(t, x),
                    1,
                )(
                    t, x
                )[1:2]
            )

            order_2 = (
                grad(
                    lambda t, x: grad(
                        lambda t, x: u_(t, x) * self.diffusion(t, x, eq_params)[0, 0],
                        1,
                    )(t, x)[0],
                    1,
                )(t, x)[0:1]
                + grad(
                    lambda t, x: grad(
                        lambda t, x: u_(t, x) * self.diffusion(t, x, eq_params)[1, 0],
                        1,
                    )(t, x)[1],
                    1,
                )(t, x)[0:1]
                + grad(
                    lambda t, x: grad(
                        lambda t, x: u_(t, x) * self.diffusion(t, x, eq_params)[0, 1],
                        1,
                    )(t, x)[0],
                    1,
                )(t, x)[1:2]
                + grad(
                    lambda t, x: grad(
                        lambda t, x: u_(t, x) * self.diffusion(t, x, eq_params)[1, 1],
                        1,
                    )(t, x)[1],
                    1,
                )(t, x)[1:2]
            )

            du_dt = grad(u_, 0)(t, x)

            return -du_dt + self.Tmax * (-order_1 + order_2)

        elif isinstance(u, SPINN):
            nn_params, eq_params = self.set_stop_gradient(params)
            x_grid = _get_grid(x)
            eq_params = self._eval_heterogeneous_parameters(
                eq_params, t, x_grid, self.eq_params_heterogeneity
            )

            _, du_dt = jax.jvp(
                lambda t: u(t, x, nn_params, eq_params),
                (t,),
                (jnp.ones_like(t),),
            )

            # in forward AD we do not have the results for all the input
            # dimension at once (as it is the case with grad), we then write
            # two jvp calls
            tangent_vec_0 = jnp.repeat(jnp.array([1.0, 0.0])[None], x.shape[0], axis=0)
            tangent_vec_1 = jnp.repeat(jnp.array([0.0, 1.0])[None], x.shape[0], axis=0)
            _, dau_dx1 = jax.jvp(
                lambda x: self.drift(t, _get_grid(x), eq_params)[None, ..., 0:1]
                * u(t, x, nn_params, eq_params)[..., 0:1],
                (x,),
                (tangent_vec_0,),
            )
            _, dau_dx2 = jax.jvp(
                lambda x: self.drift(t, _get_grid(x), eq_params)[None, ..., 1:2]
                * u(t, x, nn_params, eq_params)[..., 0:1],
                (x,),
                (tangent_vec_1,),
            )

            dsu_dx1_fun = lambda x, i, j: jax.jvp(
                lambda x: self.diffusion(t, _get_grid(x), eq_params, i, j)[
                    None, None, None, None
                ]
                * u(t, x, nn_params, eq_params)[..., 0:1],
                (x,),
                (tangent_vec_0,),
            )[1]
            dsu_dx2_fun = lambda x, i, j: jax.jvp(
                lambda x: self.diffusion(t, _get_grid(x), eq_params, i, j)[
                    None, None, None, None
                ]
                * u(t, x, nn_params, eq_params)[..., 0:1],
                (x,),
                (tangent_vec_1,),
            )[1]
            _, d2su_dx12 = jax.jvp(
                lambda x: dsu_dx1_fun(x, 0, 0), (x,), (tangent_vec_0,)
            )
            _, d2su_dx1dx2 = jax.jvp(
                lambda x: dsu_dx1_fun(x, 0, 1), (x,), (tangent_vec_1,)
            )
            _, d2su_dx22 = jax.jvp(
                lambda x: dsu_dx2_fun(x, 1, 1), (x,), (tangent_vec_1,)
            )
            _, d2su_dx2dx1 = jax.jvp(
                lambda x: dsu_dx2_fun(x, 1, 0), (x,), (tangent_vec_0,)
            )

            return -du_dt + self.Tmax * (
                -(dau_dx1 + dau_dx2)
                + (d2su_dx12 + d2su_dx22 + d2su_dx1dx2 + d2su_dx2dx1)
            )


class OU_FPENonStatioLoss2D(FPENonStatioLoss2D):
    r"""
    Return the dynamic loss for a stationary Fokker Planck Equation in two
    dimensions:

    .. math::
        -\sum_{i=1}^2\frac{\partial}{\partial \mathbf{x}}
        \left[(\alpha(\mu - \mathbf{x}))u(t,\mathbf{x})\right] +
        \sum_{i=1}^2\sum_{j=1}^2\frac{\partial^2}{\partial x_i \partial x_j}
        \left[\frac{\sigma^2}{2}u(t,\mathbf{x})\right]=
        \frac{\partial}
        {\partial t}u(t,\mathbf{x})

    """

    def __init__(self, Tmax=1, derivatives="nn_params", eq_params_heterogeneity=None):
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

    def drift(self, t, x, eq_params):
        r"""
        Return the drift term

        Parameters
        ----------
        t
            A time point
        x
            A point in :math:`\Omega`
        eq_params
            A dictionary containing the equation parameters
        """
        return eq_params["alpha"] * (eq_params["mu"] - x)

    def sigma_mat(self, t, x, eq_params):
        r"""
        Return the square root of the diffusion tensor in the sense of the outer
        product used to create the diffusion tensor

        Parameters
        ----------
        t
            A time point
        x
            A point in :math:`\Omega`
        eq_params
            A dictionary containing the equation parameters
        """

        return jnp.diag(eq_params["sigma"])

    def diffusion(self, t, x, eq_params, i=None, j=None):
        r"""
        Return the computation of the diffusion tensor term in 2D (or
        higher)

        Parameters
        ----------
        t
            A time point
        x
            A point in :math:`\Omega`
        eq_params
            A dictionary containing the equation parameters
        """
        if i is None or j is None:
            return 0.5 * (
                jnp.matmul(
                    self.sigma_mat(t, x, eq_params),
                    jnp.transpose(self.sigma_mat(t, x, eq_params)),
                )
            )
        else:
            return 0.5 * (
                jnp.matmul(
                    self.sigma_mat(t, x, eq_params),
                    jnp.transpose(self.sigma_mat(t, x, eq_params)),
                )[i, j]
            )


class ConvectionDiffusionNonStatio(FPENonStatioLoss2D):
    r"""
    Return the dynamic loss for a non-stationary convection diffusion process

    .. math::
        -\mathbf{v}\sum_{i=1}^2\frac{\partial}{\partial \mathbf{x}}
        \left[u(t, \mathbf{x})\right] +
        \sum_{i=1}^2\sum_{j=1}^2\frac{\partial^2}{\partial x_i \partial x_j}
        \left[D(t, \mathbf{x})u(t, \mathbf{x})\right]= \frac{\partial}
        {\partial t}u(t,\mathbf{x})

    where :math:`\mathbf{v}` is the velocity field and :math:`D` is a constant
    diffusion coefficent
    term.

    **Note:** That we inherit from a Fokker Planck Equation class. Indeed,
    the differential operators that are found in Convection Diffusion equations
    and FPE equations are the same.
    """

    def __init__(self, Tmax=1, derivatives="nn_params", eq_params_heterogeneity=None):
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

    def drift(self, t, x, eq_params):
        r"""
        Return the `equivalent` of the drift term in a Convection Diffusion
        problem

        Parameters
        ----------
        t
            A time point
        x
            A point in :math:`\Omega`
        eq_params
            A dictionary containing the equation parameters
        """
        return eq_params["v"]

    def diffusion(self, t, x, eq_params):
        r"""
        Return the computation of the diffusion tensor term in 2D (or
        higher) (its equivalent in a Convection Diffusion problem)

        Parameters
        ----------
        t
            A time point
        x
            A point in :math:`\Omega`
        eq_params
            A dictionary containing the equation parameters
        """
        return jnp.diag(eq_params["D"])


class MassConservation2DStatio(PDEStatio):
    r"""
    Returns the so-called mass conservation equation.

    .. math::
        \nabla \cdot \mathbf{u} = \frac{\partial}{\partial x}u(x,y) +
        \frac{\partial}{\partial y}u(x,y) = 0,

    where :math:`u` is a stationary function, i.e., it does not depend on
    :math:`t`.
    """

    def __init__(self, nn_key, derivatives="nn_params", eq_params_heterogeneity=None):
        """
        Parameters
        ----------
        nn_key
            A dictionary key which identifies, in `u_dict` the PINN that
            appears in the mass conservation equation.
        derivatives
            A string. Either ``nn_params``, ``eq_params``, ``both``. Determines
            with respect to which set of parameters gradients of the dynamic
            loss are computed. Default `"nn_params"`, this is what is typically
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
        self.nn_key = nn_key
        super().__init__(derivatives, eq_params_heterogeneity)

    def evaluate(self, x, u_dict, params_dict):
        """
        Evaluate the dynamic loss at `\mathbf{x}`.
        For stability we implement the dynamic loss in log space.

        Parameters
        ---------
        x
            A point in :math:`\Omega\subset\mathbb{R}^2`
        u_dict
            A dictionary of PINNs. Must have the same keys as `params_dict`
        params_dict
            The dictionary of dictionaries of parameters of the model.
            Typically, each sub-dictionary is a dictionary
            with keys: `eq_params` and `nn_params``, respectively the
            differential equation parameters and the neural network parameter.
            Must have the same keys as `u_dict`
        """
        if isinstance(u_dict[self.nn_key], PINN):
            nn_params, eq_params = self.set_stop_gradient(params_dict)

            nn_params = nn_params[self.nn_key]
            eq_params = eq_params

            u = u_dict[self.nn_key]

            return _div_rev(u, nn_params, eq_params, x)[..., None]

        elif isinstance(u_dict[self.nn_key], SPINN):
            nn_params, eq_params = self.set_stop_gradient(params_dict)

            nn_params = nn_params[self.nn_key]
            eq_params = eq_params

            u = u_dict[self.nn_key]

            return _div_fwd(u, nn_params, eq_params, x)[..., None]


class NavierStokes2DStatio(PDEStatio):
    r"""
    Return the dynamic loss for all the components of the stationary Navier Stokes
    equation which is a 2D vectorial PDE.

    .. math::
       (\mathbf{u}\cdot\nabla)\mathbf{u} + \frac{1}{\rho}\nabla p - \theta
       \nabla^2\mathbf{u}=0,


    or, in 2D,


    .. math::
        \begin{pmatrix}u_x\frac{\partial}{\partial x} u_x + u_y\frac{\partial}{\partial y} u_x \\
        u_x\frac{\partial}{\partial x} u_y + u_y\frac{\partial}{\partial y} u_y  \end{pmatrix} +
        \frac{1}{\rho} \begin{pmatrix} \frac{\partial}{\partial x} p \\ \frac{\partial}{\partial y} p \end{pmatrix}
        - \theta
        \begin{pmatrix}
        \frac{\partial^2}{\partial x^2} u_x + \frac{\partial^2}{\partial y^2} u_x \\
        \frac{\partial^2}{\partial x^2} u_y + \frac{\partial^2}{\partial y^2} u_y
        \end{pmatrix} = 0,

    with $\theta$ the viscosity coefficient and $\rho$ the density coefficient.

    **Note:** Note that the solution to the Navier Stokes equation is a vector
    field. Hence the MSE must concern all the axes.
    """

    def __init__(
        self, u_key, p_key, derivatives="nn_params", eq_params_heterogeneity=None
    ):
        """
        Parameters
        ----------
        u_key
            A dictionary key which indices in `u_dict`
            the PINN with the role of the velocity in the equation.
            Its input is bimensional (points in :math:`\Omega\subset\mathbb{R}^2`).
            Its output is bimensional as it represents a velocity vector
            field
        p_key
            A dictionary key which indices in `u_dict`
            the PINN with the role of the pressure in the equation.
            Its input is bimensional (points in :math:`\Omega\subset\mathbb{R}^2`).
            Its output is unidimensional as it represents a pressure scalar
            field
        derivatives
            A string. Either ``nn_params``, ``eq_params``, ``both``. Determines
            with respect to which set of parameters gradients of the dynamic
            loss are computed. Default `"nn_params"`, this is what is typically
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
        self.u_key = u_key
        self.p_key = p_key
        super().__init__(derivatives, eq_params_heterogeneity)

    def evaluate(self, x, u_dict, params_dict):
        """
        Evaluate the dynamic loss at `\mathbf{x}`.
        For stability we implement the dynamic loss in log space.

        Parameters
        ---------
        x
            A point in :math:`\Omega\subset\mathbb{R}^2`
        u_dict
            A dictionary of PINNs. Must have the same keys as `params_dict`
        params_dict
            The dictionary of dictionaries of parameters of the model.
            Typically, each sub-dictionary is a dictionary
            with keys: `eq_params` and `nn_params``, respectively the
            differential equation parameters and the neural network parameter.
            Must have the same keys as `u_dict`
        """
        if isinstance(u_dict[self.u_key], PINN):
            nn_params, eq_params = self.set_stop_gradient(params_dict)

            u_nn_params = nn_params[self.u_key]
            p_nn_params = nn_params[self.p_key]
            eq_params = eq_params

            u = u_dict[self.u_key]

            u_dot_nabla_x_u = _u_dot_nabla_times_u_rev(u, u_nn_params, eq_params, x)

            p = lambda x: u_dict[self.p_key](x, p_nn_params, eq_params)
            jac_p = jacrev(p, 0)(x)  # compute the gradient

            vec_laplacian_u = _vectorial_laplacian(
                u, u_nn_params, eq_params, x, u_vec_ndim=2
            )

            # dynamic loss on x axis
            result_x = (
                u_dot_nabla_x_u[0]
                + 1 / eq_params["rho"] * jac_p[0, 0]
                - eq_params["nu"] * vec_laplacian_u[0]
            )

            # dynamic loss on y axis
            result_y = (
                u_dot_nabla_x_u[1]
                + 1 / eq_params["rho"] * jac_p[0, 1]
                - eq_params["nu"] * vec_laplacian_u[1]
            )

            # output is 2D
            return jnp.stack([result_x, result_y], axis=-1)

        elif isinstance(u_dict[self.u_key], SPINN):
            nn_params, eq_params = self.set_stop_gradient(params_dict)

            u_nn_params = nn_params[self.u_key]
            p_nn_params = nn_params[self.p_key]
            eq_params = eq_params

            u = u_dict[self.u_key]

            u_dot_nabla_x_u = _u_dot_nabla_times_u_fwd(u, u_nn_params, eq_params, x)

            p = lambda x: u_dict[self.p_key](x, p_nn_params, eq_params)

            tangent_vec_0 = jnp.repeat(jnp.array([1.0, 0.0])[None], x.shape[0], axis=0)
            _, dp_dx = jax.jvp(p, (x,), (tangent_vec_0,))
            tangent_vec_1 = jnp.repeat(jnp.array([0.0, 1.0])[None], x.shape[0], axis=0)
            _, dp_dy = jax.jvp(p, (x,), (tangent_vec_1,))

            vec_laplacian_u = jnp.moveaxis(
                _vectorial_laplacian(u, u_nn_params, eq_params, x, u_vec_ndim=2),
                source=0,
                destination=-1,
            )

            # dynamic loss on x axis
            result_x = (
                u_dot_nabla_x_u[..., 0]
                + 1 / eq_params["rho"] * dp_dx.squeeze()
                - eq_params["nu"] * vec_laplacian_u[..., 0]
            )
            # dynamic loss on y axis
            result_y = (
                u_dot_nabla_x_u[..., 1]
                + 1 / eq_params["rho"] * dp_dy.squeeze()
                - eq_params["nu"] * vec_laplacian_u[..., 1]
            )

            # output is 2D
            return jnp.stack([result_x, result_y], axis=-1)
