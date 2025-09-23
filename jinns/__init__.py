from jinns import data as data
from jinns import loss as loss
from jinns import solver as solver
from jinns import utils as utils
from jinns import experimental as experimental
from jinns import parameters as parameters
from jinns import plot as plot
from jinns import nn as nn
from jinns.solver._solve import solve
from jinns.solver._solve_rar import solve_rar
from jinns.solver._solve_sobol import solve_sobol

__all__ = ["nn", "solve", "solve_rar", "solve_sobol"]

import warnings

warnings.filterwarnings(
    action="ignore",
    message=r"Using `field\(init=False\)`",
)
