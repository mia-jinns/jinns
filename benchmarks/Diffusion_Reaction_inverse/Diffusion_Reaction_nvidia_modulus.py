import os

# os.environ['HYDRA_FULL_ERROR']='1'
seed = 1

import numpy as np
import matplotlib.pyplot as plt

# from PINNACLE
DEFAULT_NUM_DOMAIN_POINTS = 2000
DEFAULT_NUM_BOUNDARY_POINTS = 100
DEFAULT_NUM_INITIAL_POINTS = 100


def gen_traindata():
    data = np.load("../../reaction.npz")
    t, x, ca, cb = data["t"], data["x"], data["Ca"], data["Cb"]
    X, T = np.meshgrid(x, t)
    X = np.reshape(X, (-1, 1))
    T = np.reshape(T, (-1, 1))
    Ca = np.reshape(ca, (-1, 1))
    Cb = np.reshape(cb, (-1, 1))
    return X, T, Ca, Cb


kf_init = 0.05
D_init = 1.0

from sympy import Symbol, Function, Number, sin, Eq, exp, Lambda

from modulus.sym.eq.pde import PDE


class DiffusionReaction(PDE):
    name = "DiffusionReaction"

    def __init__(self, D, kf):

        # coordinates
        x = Symbol("x")
        # time
        t = Symbol("t")

        # make input variables
        input_variables = {"x": x, "t": t}

        ca = Function("ca")(*input_variables)
        cb = Function("cb")(*input_variables)

        if isinstance(D, str):
            D = Function(D)(*input_variables)
        elif isinstance(D, (float, int)):
            D = Number(D)
        # toherwise D is a symbol and it works ?!?!
        if isinstance(kf, str):
            kf = Function(kf)(*input_variables)
        elif isinstance(kf, (float, int)):
            kf = Number(kf)
        # toherwise kf is a symbol
        # D = 1
        # kf = 1

        # set equations
        self.equations = {}
        self.equations["eq1"] = ca.diff(t, 1) - D * ca.diff(x, 2) + kf * ca * cb**2
        self.equations["eq2"] = cb.diff(t, 1) - D * cb.diff(x, 2) + 2 * kf * ca * cb**2


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
from modulus.sym.geometry.primitives_1d import Point1D, Line1D
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


class OneParameterNN(FullyConnectedArchCore):
    def __init__(self, init_value):
        super().__init__()

        self.param = nn.Parameter(
            torch.Tensor([init_value]),
            requires_grad=True,
        )

    def forward(self, x):
        # overwrite on FullyConnectedArchCore forward function
        return self.param


@modulus.sym.main(config_path="./", config_name="config_modulus")
def run(cfg: ModulusConfig) -> None:
    # print(to_yaml(cfg))
    DistributedManager.initialize()  # Only call this once in the entire script!
    dist = DistributedManager()  # call if required elsewhere

    # add constraints to solver
    # make geometry
    D, kf = Symbol("D"), Symbol("kf")
    x, t_symbol = Symbol("x"), Symbol("t")
    geo = Line1D(0, 1)
    domain = Domain()

    dr = DiffusionReaction(D, kf)

    model = instantiate_arch(
        input_keys=[Key("x"), Key("t")],
        output_keys=[Key("ca"), Key("cb")],
        cfg=cfg.arch.fully_connected,
        nr_layers=3,
        layer_size=20,
        activation_fn=torch.nn.Tanh(),
    )
    # print(model)

    # D_node = Node(
    #    inputs=[Key("x"), Key("t")], # inputs not used
    #    outputs=[Key('D')],
    #    evaluate=OneParameterNN(D_init, "D_node"),
    #    name="D_node",
    #    optimize=True
    # )
    D_model = FullyConnectedArch(
        input_keys=[Key("x"), Key("t")], output_keys=[Key("D")]
    )
    D_model._impl = OneParameterNN(D_init)

    kf_model = FullyConnectedArch(
        input_keys=[Key("x"), Key("t")], output_keys=[Key("kf")]
    )
    kf_model._impl = OneParameterNN(kf_init)

    # NOTE about detaching names, read carefully:https://docs.nvidia.com/deeplearning/modulus/modulus-sym/user_guide/foundational/inverse_problem.html
    all_nodes = (
        dr.make_nodes()
        + [model.make_node(name="dr_network", optimize=True)]
        + [D_model.make_node(name="d_network", optimize=True)]
        + [kf_model.make_node(name="kf_network", optimize=True)]
    )
    nodes = [model.make_node(name="dr_network")]
    param_nodes = [D_model.make_node(name="d_network", optimize=True)] + [
        kf_model.make_node(name="kf_network", optimize=True)
    ]

    # initial condition
    IC = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"ca": exp(-20 * x), "cb": exp(-20 * x)},
        batch_size=DEFAULT_NUM_INITIAL_POINTS,
        batch_per_epoch=1,  # total nb of points=batch_size*batch_per_epoch
        parameterization={t_symbol: 0.0},
        fixed_dataset=True,
    )
    domain.add_constraint(IC, "IC")

    # boundary condition
    BC_1 = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"ca": 1, "cb": 1},
        batch_size=DEFAULT_NUM_BOUNDARY_POINTS // 2,
        batch_per_epoch=1,  # total nb of points=batch_size*batch_per_epoch
        parameterization={t_symbol: (0, 10)},
        criteria=Eq(x, 0),
        fixed_dataset=True,
    )
    domain.add_constraint(BC_1, "BC_1")

    # boundary condition
    BC_2 = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"ca": 0, "cb": 0},
        batch_size=DEFAULT_NUM_BOUNDARY_POINTS // 2,
        batch_per_epoch=1,  # total nb of points=batch_size*batch_per_epoch
        parameterization={t_symbol: (0, 10)},
        criteria=Eq(x, 1),
        fixed_dataset=True,
    )
    domain.add_constraint(BC_2, "BC_2")

    X, T, Ca, Cb = gen_traindata()
    # observations constraint
    outvar = {"ca": Ca, "cb": Cb}
    obs_constraint = PointwiseConstraint.from_numpy(
        nodes=nodes,
        invar={"x": X, "t": T},
        outvar=outvar,
        batch_size=X.shape[0],
    )
    domain.add_constraint(obs_constraint, "obs_constraint")

    # interior
    interior = PointwiseInteriorConstraint(
        nodes=all_nodes,
        geometry=geo,
        outvar={"eq1": 0, "eq2": 0},
        batch_size=DEFAULT_NUM_DOMAIN_POINTS,
        batch_per_epoch=1,  # total nb of points=batch_size*batch_per_epoch
        fixed_dataset=True,
        parameterization={t_symbol: (0, 10)},
    )
    domain.add_constraint(interior, "interior")

    monitor = PointwiseMonitor(
        invar={"x": X, "t": T},
        output_names=["D", "kf"],
        metrics={
            "mean_D": lambda var: torch.mean(var["D"]),
            "mean_kf": lambda var: torch.mean(var["kf"]),
        },
        nodes=param_nodes,
    )
    domain.add_monitor(monitor)

    solver = Solver(cfg, domain)

    start = time.time()
    solver.solve()
    end = time.time()
    print("Training time=", end - start)

    with open("monitors/mean_D.csv", "r") as f:
        last_line = f.readlines()[-1]
    pred_D = float(last_line.split(",")[-1])
    with open("monitors/mean_kf.csv", "r") as f:
        last_line = f.readlines()[-1]
    pred_kf = float(last_line.split(",")[-1])

    true_D = 0.002
    true_kf = 0.1

    print(
        "NVIDIA Modulus: estimated D=",
        pred_D,
        "l1re=",
        np.abs(pred_D - true_D) / np.abs(true_D),
        "l2re=",
        np.sqrt((pred_D - true_D) ** 2) / np.sqrt(true_D**2),
    )
    print(
        "NVIDIA Modulus: estimated kf=",
        pred_kf,
        "l1re=",
        np.abs(pred_kf - true_kf) / np.abs(true_kf),
        "l2re=",
        np.sqrt((pred_kf - true_kf) ** 2) / np.sqrt(true_kf**2),
    )


run()
