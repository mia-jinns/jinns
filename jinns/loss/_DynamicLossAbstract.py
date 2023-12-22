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
            Such a function must be jittable and take three arguments `t`,
            `x` and `params["eq_params"]` even if one is not used. Therefore,
            one can introduce spatio-temporal covariates upon which a particular
            parameter can depend, e.g. in a GLM fashion. The effect of these
            covariables can themselves be estimated by being in `eq_params` too.
            A value can be missing, in this case there is no heterogeneity (=None).
            If eq_params_heterogeneity is None this means there is no
            heterogeneity for no parameters.
        """
        self.Tmax = Tmax
        self.eq_params_heterogeneity = eq_params_heterogeneity

    def _eval_heterogeneous_parameters(
        self, eq_params, t, x, eq_params_heterogeneity=None
    ):
        eq_params_ = {}
        if eq_params_heterogeneity is None:
            return eq_params
        for k, p in eq_params.items():
            try:
                if eq_params_heterogeneity[k] is None:
                    eq_params_[k] = p
                else:
                    eq_params_[k] = eq_params_heterogeneity[k](
                        t, x, eq_params  # heterogeneity encoded through a function
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
