from jinns import data as data
from jinns import loss as loss
from jinns import solver as solver
from jinns import utils as utils
from jinns import experimental as experimental
from jinns import parameters as parameters
from jinns import plot as plot
from jinns import nn as nn
from jinns.solver._solve import solve
from jinns.solver._solve_alternate import solve_alternate

__all__ = ["nn", "solve", "solve_alternate"]

import warnings

warnings.filterwarnings(
    action="ignore",
    message=r"Using `field\(init=False\)`",
)
