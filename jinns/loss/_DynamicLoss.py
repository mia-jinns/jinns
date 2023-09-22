import jax
from jax import jit, grad
import jax.numpy as jnp
from jinns.loss._DynamicLossAbstract import ODE, PDEStatio, PDENonStatio


class FisherKPP(PDENonStatio):
    r"""
    Return the Fisher KPP dynamic loss term (in 1 space dimension):

    .. math::
        \frac{\partial}{\partial t} u(t,x)=D\frac{\partial^2}
        {\partial x^2} u(t,x) + u(t,x)(r(x) - \gamma(x)u(t,x))

    """

    def __init__(self, Tmax=1, derivatives="nn_params"):
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

    def evaluate(self, t, x, u, params):
        """
        Evaluate the dynamic loss at :math:`(t,x)`.

        **Note:** In practice this `u` is vectorized and `t` and `x` have a
        batch dimension.

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
        nn_params, eq_params = self.set_stop_gradient(params)

        du_dt = grad(u, 0)(t, x, nn_params)[0]

        d2u_dx2 = grad(
            lambda t, x, nn_params, eq_params: grad(u, 1)(t, x, nn_params, eq_params)[
                0
            ],
            1,
        )(t, x, nn_params, eq_params)[0]

        return du_dt + self.Tmax * (
            -eq_params["D"] * d2u_dx2
            - u(t, x, nn_params, eq_params)
            * (eq_params["r"] - eq_params["g"] * u(t, x, nn_params, eq_params))
        )


class Malthus(ODE):
    r"""
    Return a Malthus dynamic loss term following the PINN logic:

    .. math::
        \frac{\partial}{\partial t} u(t)=ru(t)

    """

    def __init__(self, Tmax=1, derivatives="nn_params"):
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

    def evaluate(self, t, u, params):
        """
        Evaluate the dynamic loss at `t`.
        For stability we implement the dynamic loss in log space.

        **Note:** In practice this `u` is vectorized and `t` has a
        batch dimension.

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

    def __init__(self, Tmax=1, derivatives="nn_params"):
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

    def evaluate(self, t, x, u, params):
        """
        Evaluate the dynamic loss at :math:`(t,x)`.

        **Note:** In practice this `u` is vectorized and `t` and `x` have a
        batch dimension.

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
        nn_params, eq_params = self.set_stop_gradient(params)

        du_dt = grad(u, 0)
        du_dx = grad(u, 1)
        du2_dx2 = grad(
            lambda t, x, nn_params, eq_params: du_dx(t, x, nn_params, eq_params)[0],
            1,
        )

        return du_dt(t, x, nn_params, eq_params)[0] + self.Tmax * (
            u(t, x, nn_params, eq_params) * du_dx(t, x, nn_params, eq_params)[0]
            - eq_params["nu"] * du2_dx2(t, x, nn_params, eq_params)[0]
        )


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

    def __init__(self, key_main, keys_other, Tmax=1, derivatives="nn_params"):
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
        """
        super().__init__(Tmax, derivatives)
        self.key_main = key_main
        self.keys_other = keys_other

    def evaluate(self, t, u_dict, params_dict):
        """
        Evaluate the dynamic loss at `t`.
        For stability we implement the dynamic loss in log space.

        **Note:** In practice each `u` from `u_dict` is vectorized and `t` has a
        batch dimension.

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


class FPEStatioLoss1D(PDEStatio):
    r"""
    Return the dynamic loss for a stationary Fokker Planck Equation in one
    dimension:

    .. math::
        -\frac{\partial}{\partial x}\left[\mu(x)u(x)\right] +
        \frac{\partial^2}{\partial x^2}\left[D(x)u(x)\right]=0

    where :math:`\mu(x)` is the drift term and :math:`D(x)` is the diffusion
    term.

    The drift and diffusion terms are not specified here, hence this class
    is `abstract`.
    Other classes inherit from FPEStatioLoss1D and define the drift and
    diffusion terms, which then defines several other dynamic losses
    (Ornstein-Uhlenbeck, Cox-Ingersoll-Ross, ...)
    """

    def __init__(self, derivatives="nn_params"):
        """
        Parameters
        ----------
        derivatives
            A string. Either ``nn_params``, ``eq_params``, ``both``. Determines
            with respect to which set of parameters gradients of the dynamic
            loss are computed. Default "nn_params", this is what is typically
            done in solving forward problems, when we only estimate the
            equation solution with as PINN.
        """
        super().__init__(derivatives)

    def evaluate(self, x, u, params):
        """
        Evaluate the dynamic loss at `x`.

        **Note:** In practice this `u` is vectorized and `x` has a
        batch dimension.

        Parameters
        ---------
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
        nn_params, eq_params = self.set_stop_gradient(params)

        # (drift * u)'
        order_1 = grad(
            lambda x: (self.drift(x, eq_params) * u(x, nn_params, eq_params))[0],
            0,
        )(x)

        # (diffusion * u)''
        order_2 = grad(
            lambda x: grad(
                lambda x: (self.diffusion(x, eq_params) * u(x, nn_params, eq_params))[
                    0
                ],
                0,
            )(x)[0],
            0,
        )(x)

        return -order_1 + order_2


class OU_FPEStatioLoss1D(FPEStatioLoss1D):
    r"""
    Return the dynamic loss whose solution is the probability density
    function of a stationary Ornstein-Uhlenbeck process in one dimension:

    .. math::
        -\frac{\partial}{\partial x}\left[(\alpha(\mu - x))u(x)\right] +
        \frac{\partial^2}{\partial x^2}\left[\frac{\sigma^2}{2}u(x)\right]=0

    """

    def __init__(self, derivatives="nn_params"):
        """
        Parameters
        ----------
        derivatives
            A string. Either ``nn_params``, ``eq_params``, ``both``. Determines
            with respect to which set of parameters gradients of the dynamic
            loss are computed. Default "nn_params", this is what is typically
            done in solving forward problems, when we only estimate the
            equation solution with as PINN.
        """
        super().__init__(derivatives)

    def drift(self, x, eq_params):
        r"""
        Return the drift term

        Parameters
        ----------
        x
            A point in :math:`\Omega`
        eq_params
            A dictionary containing the equation parameters
        """
        return eq_params["alpha"] * (eq_params["mu"] - x)

    def diffusion(self, x, eq_params):
        r"""
        Return the computation of the diffusion tensor term in 1D

        Parameters
        ----------
        x
            A point in :math:`\Omega`
        eq_params
            A dictionary containing the equation parameters
        """
        return 0.5 * eq_params["sigma"] ** 2


class CIR_FPEStatioLoss1D(FPEStatioLoss1D):
    r"""
    Return the dynamic loss whose solution is the probability density
    function of a stationary Cox-Ingersoll-Ross process in one dimension:

    .. math::
        -\frac{\partial}{\partial x}\left[(\mu - \alpha x)u(x)\right] +
        \frac{\partial^2}{\partial x^2}\left[\frac{\sigma^2}{2}xu(x)\right]=0

    """

    def __init__(self, derivatives="nn_params"):
        """
        Parameters
        ----------
        derivatives
            A string. Either ``nn_params``, ``eq_params``, ``both``. Determines
            with respect to which set of parameters gradients of the dynamic
            loss are computed. Default "nn_params", this is what is typically
            done in solving forward problems, when we only estimate the
            equation solution with as PINN.
        """
        super().__init__(derivatives)

    def drift(self, x, eq_params):
        r"""
        Return the drift term

        Parameters
        ----------
        x
            A point in :math:`\Omega`
        eq_params
            A dictionary containing the equation parameters
        """
        return eq_params["mu"] - eq_params["alpha"] * x

    def diffusion(self, x, eq_params):
        r"""
        Return the computation of the diffusion tensor term in 1D

        Parameters
        ----------
        x
            A point in :math:`\Omega`
        eq_params
            A dictionary containing the equation parameters
        """
        return 0.5 * (eq_params["sigma"] ** 2) * x


class FPENonStatioLoss1D(PDENonStatio):
    r"""
    Return the dynamic loss for a non stationary Fokker Planck Equation in one
    dimension:

    .. math::
        -\frac{\partial}{\partial x}\left[\mu(t, x)u(t, x)\right] +
        \frac{\partial^2}{\partial x^2}\left[D(t, x)u(t, x)\right] =
        \frac{\partial}{\partial t}u(t,x)

    where :math:`\mu(t, x)` is the drift term and :math:`D(t, x)` is the diffusion
    term.

    The drift and diffusion terms are not specified here, hence this class
    is `abstract`.
    Other classes inherit from FPENonStatioLoss1D and define the drift and
    diffusion terms, which then defines several other dynamic losses
    (Ornstein-Uhlenbeck, Cox-Ingersoll-Ross, ...)
    """

    def __init__(self, Tmax=1, derivatives="nn_params"):
        """
        Parameters
        ----------
        Tmax
            Tmax needs to be given when the PINN time input is normalized in
            [0, 1], ie. we have performed renormalization of the differential
            equation
        """
        super().__init__(Tmax, derivatives)

    def evaluate(self, t, x, u, params):
        """
        Evaluate the dynamic loss at :math:`(t,x)`.

        **Note:** In practice this `u` is vectorized and `t` and `x` have a
        batch dimension.

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
        nn_params, eq_params = self.set_stop_gradient(params)
        # (drift * u)'

        order_1 = grad(
            lambda t, x: (self.drift(t, x, eq_params) * u(t, x, nn_params, eq_params))[
                0
            ],
            1,
        )(t, x)

        # (diffusion * u)''
        order_2 = grad(
            lambda t, x: grad(
                lambda t, x: (
                    self.diffusion(t, x, eq_params) * u(t, x, nn_params, eq_params)
                )[0],
                1,
            )(t, x)[0],
            1,
        )(t, x)

        du_dt = grad(u, 0)(t, x, nn_params, eq_params)

        return -du_dt + self.Tmax * (-order_1 + order_2)


class OU_FPENonStatioLoss1D(FPENonStatioLoss1D):
    r"""
    Return the dynamic loss whose solution is the probability density
    function of a non-stationary Ornstein-Uhlenbeck process in one dimension:

    .. math::
        -\frac{\partial}{\partial x}\left[(\alpha(\mu - x))u(t,x)\right] +
        \frac{\partial^2}{\partial x^2}\left[\frac{\sigma^2}{2}u(t,x)\right] =
        \frac{\partial}{\partial t}u(t,x)

    """

    def __init__(self, Tmax=1, derivatives="nn_params"):
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

    def diffusion(self, t, x, eq_params):
        r"""
        Return the computation of the diffusion tensor term in 1D

        Parameters
        ----------
        t
            A time point
        x
            A point in :math:`\Omega`
        eq_params
            A dictionary containing the equation parameters
        """
        return 0.5 * eq_params["sigma"] ** 2


class CIR_FPENonStatioLoss1D(FPENonStatioLoss1D):
    r"""
    Return the dynamic loss whose solution is the probability density
    function of a stationary Cox-Ingersoll-Ross process in one dimension:

    .. math::
        -\frac{\partial}{\partial x}\left[(\mu - \alpha x)u(x)\right] +
        \frac{\partial^2}{\partial x^2}\left[\frac{\sigma^2}{2}xu(x)\right] =
        \frac{\partial}{\partial t}u(t,x)

    """

    def __init__(self, Tmax=1, derivatives="nn_params"):
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
        return eq_params["mu"] - eq_params["alpha"] * x

    def diffusion(self, t, x, eq_params):
        r"""
        Return the computation of the diffusion tensor term in 1D

        Parameters
        ----------
        t
            A time point
        x
            A point in :math:`\Omega`
        eq_params
            A dictionary containing the equation parameters
        """
        return 0.5 * (eq_params["sigma"] ** 2) * x


class Sinus_FPENonStatioLoss1D(FPENonStatioLoss1D):
    r"""
    Return the dynamic loss whose solution is the probability density
    function of a non-stationary Ornstein-Uhlenbeck process in one dimension:

    .. math::
        -\frac{\partial}{\partial x}\left[\sin(x)u(x)\right] +
        \frac{\partial^2}{\partial x^2}\left[\frac{1}{2}u(x)\right] =
        \frac{\partial}{\partial t}u(t,x)

    """

    def __init__(self, Tmax=1, derivatives="nn_params"):
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
        return jnp.sin(x)

    def diffusion(self, t, x, eq_params):
        r"""
        Return the computation of the diffusion tensor term in 1D

        Parameters
        ----------
        t
            A time point
        x
            A point in :math:`\Omega`
        eq_params
            A dictionary containing the equation parameters
        """
        return 0.5 * jnp.ones((1))


class FPEStatioLoss2D(PDEStatio):
    r"""
    Return the dynamic loss for a stationary Fokker Planck Equation in two
    dimensions:

    .. math::
        -\sum_{i=1}^2\frac{\partial}{\partial \mathbf{x}}
        \left[\mu(\mathbf{x})u(\mathbf{x})\right] +
        \sum_{i=1}^2\sum_{j=1}^2\frac{\partial^2}{\partial x_i \partial x_j}
        \left[D(\mathbf{x})u(\mathbf{x})\right]=0

    where :math:`\mu(\mathbf{x})` is the drift term and :math:`D(\mathbf{x})` is the diffusion
    term.

    The drift and diffusion terms are not specified here, hence this class
    is `abstract`.
    Other classes inherit from FPEStatioLoss2D and define the drift and
    diffusion terms, which then defines several other dynamic losses
    (Ornstein-Uhlenbeck, Cox-Ingersoll-Ross, ...)
    """

    def __init__(self, derivatives="nn_params"):
        """
        Parameters
        ----------
        derivatives
            A string. Either ``nn_params``, ``eq_params``, ``both``. Determines
            with respect to which set of parameters gradients of the dynamic
            loss are computed. Default "nn_params", this is what is typically
            done in solving forward problems, when we only estimate the
            equation solution with as PINN.
        """
        super().__init__(derivatives)

    def evaluate(self, x, u, params):
        """
        Evaluate the dynamic loss at :math:`\mathbf{x}`.

        **Note:** For computational purpose we use compositions of calls to
        `jax.grad` instead of a call to `jax.hessian`

        **Note:** In practice this `u` is vectorized and :math:`\mathbf{x}` has a
        batch dimension.

        Parameters
        ---------
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
        nn_params, eq_params = self.set_stop_gradient(params)

        order_1 = (
            grad(
                lambda x: self.drift(x, eq_params)[0] * u(x, nn_params, eq_params),
                0,
            )(x)[0]
            + grad(
                lambda x: self.drift(x, eq_params)[1] * u(x, nn_params, eq_params),
                0,
            )(x)[1]
        )

        order_2 = (
            grad(
                lambda x: grad(
                    lambda x: u(x, nn_params, eq_params)
                    * self.diffusion(x, eq_params)[0, 0],
                    0,
                )(x)[0],
                0,
            )(x)[0]
            + grad(
                lambda x: grad(
                    lambda x: u(x, nn_params, eq_params)
                    * self.diffusion(x, eq_params)[1, 0],
                    0,
                )(x)[1],
                0,
            )(x)[0]
            + grad(
                lambda x: grad(
                    lambda x: u(x, nn_params, eq_params)
                    * self.diffusion(x, eq_params)[0, 1],
                    0,
                )(x)[0],
                0,
            )(x)[1]
            + grad(
                lambda x: grad(
                    lambda x: u(x, nn_params, eq_params)
                    * self.diffusion(x, eq_params)[1, 1],
                    0,
                )(x)[1],
                0,
            )(x)[1]
        )
        return -order_1 + order_2


class OU_FPEStatioLoss2D(FPEStatioLoss2D):
    r"""
    Return the dynamic loss for a stationary Fokker Planck Equation in two
    dimensions:

    .. math::
        -\sum_{i=1}^2\frac{\partial}{\partial \mathbf{x}}
        \left[(\alpha(\mu - \mathbf{x}))u(\mathbf{x})\right] +
        \sum_{i=1}^2\sum_{j=1}^2\frac{\partial^2}{\partial x_i \partial x_j}
        \left[\frac{\sigma^2}{2}u(\mathbf{x})\right]=0

    """

    def __init__(self, derivatives="nn_params"):
        """
        Parameters
        ----------
        derivatives
            A string. Either ``nn_params``, ``eq_params``, ``both``. Determines
            with respect to which set of parameters gradients of the dynamic
            loss are computed. Default "nn_params", this is what is typically
            done in solving forward problems, when we only estimate the
            equation solution with as PINN.
        """
        super().__init__(derivatives)

    def drift(self, x, eq_params):
        r"""
        Return the drift term

        Parameters
        ----------
        x
            A point in :math:`\Omega`
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
            A point in :math:`\Omega`
        eq_params
            A dictionary containing the equation parameters
        """
        return jnp.diag(eq_params["sigma"])

    def diffusion(self, x, eq_params):
        r"""
        Return the computation of the diffusion tensor term in 2D (or
        higher)

        Parameters
        ----------
        x
            A point in :math:`\Omega`
        eq_params
            A dictionary containing the equation parameters
        """
        return 0.5 * (
            jnp.matmul(
                self.sigma_mat(x, eq_params),
                jnp.transpose(self.sigma_mat(x, eq_params)),
            )
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

    def __init__(self, Tmax, derivatives="nn_params"):
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

    def evaluate(self, t, x, u, params):
        """
        Evaluate the dynamic loss at :math:`(t,\mathbf{x})`.

        **Note:** In practice this `u` is vectorized and `t` and
        :math:`\mathbf{x}` have a batch dimension.

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
        nn_params, eq_params = self.set_stop_gradient(params)

        order_1 = (
            grad(
                lambda t, x: self.drift(t, x, eq_params)[0]
                * u(t, x, nn_params, eq_params),
                1,
            )(t, x)[0]
            + grad(
                lambda t, x: self.drift(t, x, eq_params)[1]
                * u(t, x, nn_params, eq_params),
                1,
            )(t, x)[1]
        )

        order_2 = (
            grad(
                lambda t, x: grad(
                    lambda t, x: u(t, x, nn_params, eq_params)
                    * self.diffusion(t, x, eq_params)[0, 0],
                    1,
                )(t, x)[0],
                1,
            )(t, x)[0]
            + grad(
                lambda t, x: grad(
                    lambda t, x: u(t, x, nn_params, eq_params)
                    * self.diffusion(t, x, eq_params)[1, 0],
                    1,
                )(t, x)[1],
                1,
            )(t, x)[0]
            + grad(
                lambda t, x: grad(
                    lambda t, x: u(t, x, nn_params, eq_params)
                    * self.diffusion(t, x, eq_params)[0, 1],
                    1,
                )(t, x)[0],
                1,
            )(t, x)[1]
            + grad(
                lambda t, x: grad(
                    lambda t, x: u(t, x, nn_params, eq_params)
                    * self.diffusion(t, x, eq_params)[1, 1],
                    1,
                )(t, x)[1],
                1,
            )(t, x)[1]
        )

        du_dt = grad(u, 0)(t, x, nn_params, eq_params)

        return -du_dt + self.Tmax * (-order_1 + order_2)


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

    def __init__(self, Tmax=1, derivatives="nn_params"):
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

    def diffusion(self, t, x, eq_params):
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
        return 0.5 * (
            jnp.matmul(
                self.sigma_mat(t, x, eq_params),
                jnp.transpose(self.sigma_mat(t, x, eq_params)),
            )
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

    def __init__(self, Tmax=1, derivatives="nn_params"):
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

    def __init__(self, nn_key, derivatives="nn_params"):
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
        """
        self.nn_key = nn_key
        super().__init__(derivatives)

    def evaluate(self, x, u_dict, params_dict):
        """
        Evaluate the dynamic loss at `\mathbf{x}`.
        For stability we implement the dynamic loss in log space.

        **Note:** In practice each `u` from `u_dict` is vectorized and
        `\mathbf{x}` has a batch dimension.

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
        nn_params, eq_params = self.set_stop_gradient(params_dict)

        nn_params = nn_params[self.nn_key]
        eq_params = eq_params

        u = u_dict[self.nn_key]
        # as u is a vector with two components here we create functions for
        # each of the components
        # We also fix the parameters for clarity
        ux = lambda x: u(x, nn_params, eq_params)[0]
        uy = lambda x: u(x, nn_params, eq_params)[1]

        return grad(ux, 0)(x)[0] + grad(uy, 0)(x)[1]


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

    def __init__(self, u_key, p_key, derivatives="nn_params"):
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
        """
        self.u_key = u_key
        self.p_key = p_key
        super().__init__(derivatives)

    def evaluate(self, x, u_dict, params_dict):
        """
        Evaluate the dynamic loss at `\mathbf{x}`.
        For stability we implement the dynamic loss in log space.

        **Note:** In practice each `u` from `u_dict` is vectorized and
        `\mathbf{x}` has a batch dimension.

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
        nn_params, eq_params = self.set_stop_gradient(params_dict)

        u_nn_params = nn_params[self.u_key]
        p_nn_params = nn_params[self.p_key]
        eq_params = eq_params

        u = u_dict[self.u_key]
        # as u is a vector with two components here we create functions for
        # each of the components
        # We also fix the parameters for clarity
        ux = lambda x: u(x, u_nn_params, eq_params)[0]
        uy = lambda x: u(x, u_nn_params, eq_params)[1]

        dux_dx = lambda x: grad(ux, 0)(x)[0]
        d2ux_dx2 = lambda x: grad(dux_dx, 0)(x)[0]
        dux_dy = lambda x: grad(ux, 0)(x)[1]
        d2ux_dy2 = lambda x: grad(dux_dy, 0)(x)[1]

        duy_dx = lambda x: grad(uy, 0)(x)[0]
        d2uy_dx2 = lambda x: grad(duy_dx, 0)(x)[0]
        duy_dy = lambda x: grad(uy, 0)(x)[1]
        d2uy_dy2 = lambda x: grad(duy_dy, 0)(x)[1]

        p = lambda x: u_dict[self.p_key](x, p_nn_params, eq_params)
        dp_dx = lambda x: grad(p, 0)(x)[0]
        dp_dy = lambda x: grad(p, 0)(x)[1]

        # dynamic loss on x axis
        result_x = (
            ux(x) * dux_dx(x)
            + uy(x) * dux_dy(x)
            + 1 / eq_params["rho"] * dp_dx(x)
            - eq_params["nu"] * (d2ux_dx2(x) + d2ux_dy2(x))
        )
        # dynamic loss on y axis
        result_y = (
            ux(x) * duy_dx(x)
            + uy(x) * duy_dy(x)
            + 1 / eq_params["rho"] * dp_dy(x)
            - eq_params["nu"] * (d2uy_dx2(x) + d2uy_dy2(x))
        )

        # output is 2D
        return jnp.stack([result_x, result_y], axis=-1)
