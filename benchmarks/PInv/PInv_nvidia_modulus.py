import os

# os.environ['HYDRA_FULL_ERROR']='1'
import numpy as np

seed = 1
np.random.seed(seed)
noisy_obs = np.random.normal(loc=0, scale=0.1, size=(2500, 1))

import matplotlib.pyplot as plt

# from PINNACLE
DEFAULT_NUM_DOMAIN_POINTS = 8192 // 10
DEFAULT_NUM_BOUNDARY_POINTS = 2048 // 10


bc_x = np.linspace(0, 1, 50)
bc_y = np.linspace(0, 1, 50)
bc_x, bc_y = np.meshgrid(bc_x, bc_y)
bc_xy = np.stack((bc_x.reshape(-1), bc_y.reshape(-1))).T

from sympy import Symbol, Function, Number, sin, Eq, exp, Lambda, cos

from modulus.sym.eq.pde import PDE


def u_ref(xy):
    x, y = xy[:, 0:1], xy[:, 1:2]
    return np.sin(np.pi * x) * np.sin(np.pi * y)


def a_ref_(xy):
    x, y = xy[:, 0:1], xy[:, 1:2]
    return 1 / (1 + x**2 + y**2 + (x - 1) ** 2 + (y - 1) ** 2)


ref_sol = lambda xy: np.concatenate((u_ref(xy), a_ref_(xy)), axis=1)


class Poisson(PDE):
    name = "Poisson"

    def __init__(self, a, f):

        # coordinates
        x = Symbol("x")
        y = Symbol("y")

        # make input variables
        input_variables = {"x": x, "y": y}

        u = Function("u")(*input_variables)

        # set equations
        self.equations = {}
        self.equations["poisson"] = (a * u.diff(x, 1)).diff(x, 1) + f


import modulus.sym
from modulus.sym import Node
from modulus.sym.hydra import to_yaml
from modulus.distributed import DistributedManager
from modulus.sym.hydra import instantiate_arch, ModulusConfig
from modulus.sym.eq.phy_informer import PhysicsInformer
from modulus.models.mlp.fully_connected import FullyConnected
from modulus.sym.models.fully_connected import (
    FullyConnectedArchCore,
    FullyConnectedArch,
)
from modulus.sym.eq.pdes.navier_stokes import NavierStokes
from modulus.sym.solver import Solver
from modulus.sym.domain import Domain
from modulus.sym.geometry.primitives_2d import Rectangle
from modulus.sym.domain.constraint import (
    PointwiseConstraint,
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
)
from modulus.sym.geometry.parameterization import Parameterization, Parameter
from modulus.sym.domain.validator import PointwiseValidator
from modulus.sym.domain.monitor import PointwiseMonitor
from modulus.sym.key import Key

from modulus.sym.utils.benchmark import timeit

import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import time
import sys

sys.path.append("../")
from PINNacle_utils import trans_time_data_to_dataset, compute_relative_errors


@modulus.sym.main(config_path="./", config_name="config_modulus")
def run(cfg: ModulusConfig) -> None:
    # print(to_yaml(cfg))
    DistributedManager.initialize()  # Only call this once in the entire script!
    dist = DistributedManager()  # call if required elsewhere

    # add constraints to solver
    # make geometry
    a = Symbol("a")
    x, y = Symbol("x"), Symbol("y")
    geo = Rectangle((0, 0), (1, 1))
    domain = Domain()

    a_ref = 1 / (1 + x**2 + y**2 + (x - 1) ** 2 + (y - 1) ** 2)

    f = (
        2 * np.pi**2 * sin(np.pi * x) * sin(np.pi * y) * a_ref
        + 2
        * np.pi
        * (
            (2 * x + 1) * cos(np.pi * x) * sin(np.pi * y)
            + (2 * y + 1) * sin(np.pi * x) * cos(np.pi * y)
        )
        * a_ref**2
    )

    poiss = Poisson(a, f)

    model = instantiate_arch(
        input_keys=[Key("x"), Key("y")],
        output_keys=[Key("u")],
        cfg=cfg.arch.fully_connected,
        nr_layers=5,
        layer_size=50,
        activation_fn=torch.nn.Tanh(),
    )
    # print(model)

    a_model = instantiate_arch(
        input_keys=[Key("x"), Key("y")],
        output_keys=[Key("a")],
        cfg=cfg.arch.fully_connected,
        nr_layers=5,
        layer_size=50,
        activation_fn=torch.nn.Tanh(),
    )

    # NOTE about detaching names, read carefully:https://docs.nvidia.com/deeplearning/modulus/modulus-sym/user_guide/foundational/inverse_problem.html
    model_node = model.make_node(name="poiss_network", optimize=True)
    a_node = a_model.make_node(name="a_network", optimize=True)
    poiss_nodes = poiss.make_nodes()
    all_nodes = poiss.make_nodes() + [model_node] + [a_node]

    # boundary condition
    BC_1 = PointwiseBoundaryConstraint(
        nodes=all_nodes,
        geometry=geo,
        outvar={"a": a_ref},
        batch_size=DEFAULT_NUM_BOUNDARY_POINTS,
        batch_per_epoch=1,  # total nb of points=batch_size*batch_per_epoch
        criteria=Eq(x, 0) | Eq(x, 1) | Eq(y, 0) | Eq(y, 1),
        fixed_dataset=True,
    )
    domain.add_constraint(BC_1, "BC_1")

    # observations constraint
    outvar = {"u": u_ref(bc_xy) + noisy_obs}
    obs_constraint = PointwiseConstraint.from_numpy(
        nodes=all_nodes,
        invar={"x": bc_xy[:, 0:1], "y": bc_xy[:, 1:2]},
        outvar=outvar,
        batch_size=bc_xy.shape[0],
    )
    domain.add_constraint(obs_constraint, "obs_constraint")

    # interior
    interior = PointwiseInteriorConstraint(
        nodes=all_nodes,
        geometry=geo,
        outvar={"poisson": 0},
        batch_size=DEFAULT_NUM_DOMAIN_POINTS,
        batch_per_epoch=1,  # total nb of points=batch_size*batch_per_epoch
        fixed_dataset=True,
    )
    domain.add_constraint(interior, "interior")

    test_x = np.concatenate(
        (np.random.uniform(size=(2500, 1)), np.random.uniform(size=(2500, 1))), axis=1
    )
    invar_numpy = {"x": test_x[:, 0:1], "y": test_x[:, 1:2]}
    true_outvar = {"u": ref_sol(test_x)[:, 0:1], "a": ref_sol(test_x)[:, 1:2]}
    validator = PointwiseValidator(
        nodes=all_nodes,
        invar=invar_numpy,
        true_outvar=true_outvar,
        batch_size=invar_numpy["x"].shape[0],
    )
    domain.add_validator(validator)

    solver = Solver(cfg, domain)

    start = time.time()
    solver.solve()
    end = time.time()
    print("Training time=", end - start)

    with np.load("validators/validator.npz", allow_pickle=True) as data:
        compute_relative_errors(
            data["arr_0"][()]["true_u"], data["arr_0"][()]["pred_u"]
        )  # about indexing https://stackoverflow.com/a/37949466


run()
