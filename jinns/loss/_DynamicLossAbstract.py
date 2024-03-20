"""
Implements abstract classes for dynamic losses
"""


class DynamicLoss:
    r"""
    Abstract base class for dynamic losses whose aim is to implement the term:

    .. math::
        \mathcal{N}[u](t, x) = 0
    """

    def __init__(self, Tmax=None, eq_params_heterogeneity=None):
        """
        Parameters
        ----------
        Tmax
            Tmax needs to be given when the PINN time input is normalized in
            [0, 1], ie. we have performed renormalization of the differential
            equation
        eq_params_heterogeneity
            Default None. A dict with the keys being the same as in eq_params
            and the value being either None (no heterogeneity) or a function
            which encodes for the spatio-temporal heterogeneity of the parameter.
            Such a function must be jittable and take four arguments `t`, `x`,
            `u` and `params` even if one is not used. Therefore,
            one can introduce spatio-temporal covariates upon which a particular
            parameter can depend, e.g. in a GLM fashion. The effect of these
            covariables can themselves be estimated by being in `eq_params` too.
            A value can be missing, in this case there is no heterogeneity (=None).
            If eq_params_heterogeneity is None this means there is no
            heterogeneity for no parameters.
        """
        self.Tmax = Tmax
        self.eq_params_heterogeneity = eq_params_heterogeneity

    @staticmethod
    def _eval_heterogeneous_parameters(t, x, u, params, eq_params_heterogeneity=None):
        eq_params_ = {}
        if eq_params_heterogeneity is None:
            return params["eq_params"]
        for k, p in params["eq_params"].items():
            try:
                if eq_params_heterogeneity[k] is None:
                    eq_params_[k] = p
                else:
                    if t is None:
                        eq_params_[k] = eq_params_heterogeneity[k](
                            x, u, params  # heterogeneity encoded through a function
                        )
                    else:
                        eq_params_[k] = eq_params_heterogeneity[k](
                            t, x, u, params  # heterogeneity encoded through a function
                        )
            except KeyError:
                # we authorize missing eq_params_heterogeneity key
                # is its heterogeneity is None anyway
                eq_params_[k] = p
        return eq_params_


class ODE(DynamicLoss):
    r"""
    Abstract base class for ODE dynamic losses
    """

    def __init__(self, Tmax=None, eq_params_heterogeneity=None):
        """
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
        super().__init__(Tmax, eq_params_heterogeneity)

    def eval_heterogeneous_parameters(self, t, u, params, eq_params_heterogeneity=None):
        return super()._eval_heterogeneous_parameters(
            t, None, u, params, eq_params_heterogeneity
        )

    @staticmethod
    def evaluate_heterogeneous_parameters(evaluate):
        """
        Decorator which aims to decorate the evaluate methods of Dynamic losses
        in order. It calls _eval_heterogeneous_parameters which applies the
        user defined rules to obtain spatially / temporally heterogeneous
        parameters
        """

        def wrapper(*args):
            self, t, u, params = args
            # avoid side effect with in-place modif of param["eq_params"]
            # TODO NamedTuple for params and use _replace() see Issue 1
            _params = {
                "nn_params": params["nn_params"],
                "eq_params": self.eval_heterogeneous_parameters(
                    t, u, params, self.eq_params_heterogeneity
                ),
            }
            new_args = args[:-1] + (_params,)
            res = evaluate(*new_args)
            return res

        return wrapper


class PDEStatio(DynamicLoss):
    r"""
    Abstract base class for PDE statio dynamic losses
    """

    def __init__(self, eq_params_heterogeneity=None):
        """
        Parameters
        ----------
        eq_params_heterogeneity
            Default None. A dict with the keys being the same as in eq_params
            and the value being `time`, `space`, `both` or None which corresponds to
            the heterogeneity of a given parameter. A value can be missing, in
            this case there is no heterogeneity (=None). If
            eq_params_heterogeneity is None this means there is no
            heterogeneity for no parameters.
        """
        super().__init__(eq_params_heterogeneity=eq_params_heterogeneity)

    def eval_heterogeneous_parameters(self, x, u, params, eq_params_heterogeneity=None):
        return super()._eval_heterogeneous_parameters(
            None, x, u, params, eq_params_heterogeneity
        )

    @staticmethod
    def evaluate_heterogeneous_parameters(evaluate):
        """
        Decorator which aims to decorate the evaluate methods of Dynamic losses
        in order. It calls _eval_heterogeneous_parameters which applies the
        user defined rules to obtain spatially / temporally heterogeneous
        parameters
        """

        def wrapper(*args):
            self, x, u, params = args
            # avoid side effect with in-place modif of param["eq_params"]
            # TODO NamedTuple for params and use _replace() see Issue 1
            _params = {
                "nn_params": params["nn_params"],
                "eq_params": self.eval_heterogeneous_parameters(
                    t, u, params, self.eq_params_heterogeneity
                ),
            }
            new_args = args[:-1] + (_params,)
            res = evaluate(*new_args)
            return res

        return wrapper


class PDENonStatio(DynamicLoss):
    r"""
    Abstract base class for PDE Non statio dynamic losses
    """

    def __init__(self, Tmax=None, eq_params_heterogeneity=None):
        """
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
        super().__init__(Tmax, eq_params_heterogeneity)

    def eval_heterogeneous_parameters(
        self, t, x, u, params, eq_params_heterogeneity=None
    ):
        return super()._eval_heterogeneous_parameters(
            t, x, u, params, eq_params_heterogeneity
        )

    @staticmethod
    def evaluate_heterogeneous_parameters(evaluate):
        """
        Decorator which aims to decorate the evaluate methods of Dynamic losses
        in order. It calls _eval_heterogeneous_parameters which applies the
        user defined rules to obtain spatially / temporally heterogeneous
        parameters
        """

        def wrapper(*args):
            self, t, x, u, params = args
            # avoid side effect with in-place modif of param["eq_params"]
            # TODO NamedTuple for params and use _replace() see Issue 1
            _params = {
                "nn_params": params["nn_params"],
                "eq_params": self.eval_heterogeneous_parameters(
                    t, x, u, params, self.eq_params_heterogeneity
                ),
            }
            new_args = args[:-1] + (_params,)
            res = evaluate(*new_args)
            return res

        return wrapper
