from ._natural_gradient import vanilla_ngd as vanilla_ngd
from ._ssbfgs import self_scaled_bfgs_or_broyden as self_scaled_bfgs_or_broyden


__all__ = ["vanilla_ngd", "self_scaled_bfgs_or_broyden"]
