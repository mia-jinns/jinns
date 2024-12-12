import os

# os.environ['HYDRA_FULL_ERROR']='1'
seed = 1

import numpy as np
import matplotlib.pyplot as plt

# from PINNACLE
DEFAULT_NUM_DOMAIN_POINTS = 700
DEFAULT_NUM_BOUNDARY_POINTS = 200
DEFAULT_NUM_INITIAL_POINTS = 100

from scipy.io import loadmat


# Load training data
def load_training_data(num):
    data = loadmat("../../cylinder_nektar_wake.mat")
    U_star = data["U_star"]  # N x 2 x T
    P_star = data["p_star"]  # N x T
    t_star = data["t"]  # T x 1
    X_star = data["X_star"]  # N x 2
    N = X_star.shape[0]
    T = t_star.shape[0]
    # Rearrange Data
    XX = np.tile(X_star[:, 0:1], (1, T))  # N x T
    YY = np.tile(X_star[:, 1:2], (1, T))  # N x T
    TT = np.tile(t_star, (1, N)).T  # N x T
    UU = U_star[:, 0, :]  # N x T
    VV = U_star[:, 1, :]  # N x T
    PP = P_star  # N x T
    x = XX.flatten()[:, None]  # NT x 1
    y = YY.flatten()[:, None]  # NT x 1
    t = TT.flatten()[:, None]  # NT x 1
    u = UU.flatten()[:, None]  # NT x 1
    v = VV.flatten()[:, None]  # NT x 1
    p = PP.flatten()[:, None]  # NT x 1
    # training domain: X × Y = [1, 8] × [−2, 2] and T = [0, 7]
    data1 = np.concatenate([x, y, t, u, v, p], 1)
    data2 = data1[:, :][data1[:, 2] <= 7]
    data3 = data2[:, :][data2[:, 0] >= 1]
    data4 = data3[:, :][data3[:, 0] <= 8]
    data5 = data4[:, :][data4[:, 1] >= -2]
    data_domain = data5[:, :][data5[:, 1] <= 2]
    # choose number of training points: num =7000
    idx = np.random.choice(data_domain.shape[0], num, replace=False)
    x_train = data_domain[idx, 0:1]
    y_train = data_domain[idx, 1:2]
    t_train = data_domain[idx, 2:3]
    u_train = data_domain[idx, 3:4]
    v_train = data_domain[idx, 4:5]
    p_train = data_domain[idx, 5:6]
    return [x_train, y_train, t_train, u_train, v_train, p_train]


C1_init = 0.0
C2_init = 0.0

from sympy import Symbol, Function, Number, sin, Eq, exp, Lambda


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
    C1, C2 = Symbol("C1"), Symbol("C2")
    x, y, t_symbol = Symbol("x"), Symbol("y"), Symbol("t")
    geo = Rectangle((1.0, -2.0), (8.0, 2.0))
    domain = Domain()

    ns = NavierStokes(nu=C2, rho=C1, dim=2, time=True)

    model = instantiate_arch(
        input_keys=[Key("x"), Key("y"), Key("t")],
        output_keys=[Key("u"), Key("v"), Key("p")],
        cfg=cfg.arch.fully_connected,
        nr_layers=6,
        layer_size=50,
        activation_fn=torch.nn.Tanh(),
    )

    C1_model = FullyConnectedArch(
        input_keys=[Key("x"), Key("y"), Key("t")], output_keys=[Key("C1")]
    )
    C1_model._impl = OneParameterNN(C1_init)

    C2_model = FullyConnectedArch(
        input_keys=[Key("x"), Key("y"), Key("t")], output_keys=[Key("C2")]
    )
    C2_model._impl = OneParameterNN(C2_init)

    # NOTE about detaching names, read carefully:https://docs.nvidia.com/deeplearning/modulus/modulus-sym/user_guide/foundational/inverse_problem.html
    C1_node = C1_model.make_node(name="C1_network", optimize=True)
    C2_node = C2_model.make_node(name="C2_network", optimize=True)
    model_node = model.make_node(name="ns_network", optimize=True)
    ns_nodes = ns.make_nodes()
    all_nodes = ns_nodes + [model_node] + [C1_node] + [C2_node]
    nodes = [model_node]
    param_nodes = [C1_node] + [C2_node]

    [ob_x, ob_y, ob_t, ob_u, ob_v, ob_p] = load_training_data(num=7000)
    # observations constraint
    outvar = {"u": ob_u, "v": ob_v, "p": ob_p}
    obs_constraint = PointwiseConstraint.from_numpy(
        nodes=nodes,
        invar={"x": ob_x, "y": ob_y, "t": ob_t},
        outvar=outvar,
        batch_size=ob_x.shape[0],
    )
    domain.add_constraint(obs_constraint, "obs_constraint")

    # interior
    interior = PointwiseInteriorConstraint(
        nodes=all_nodes,
        geometry=geo,
        outvar={"continuity": 0, "momentum_x": 0, "momentum_y": 0},
        batch_size=DEFAULT_NUM_DOMAIN_POINTS,
        batch_per_epoch=1,  # total nb of points=batch_size*batch_per_epoch
        fixed_dataset=True,
        parameterization={t_symbol: (0, 7)},
    )
    domain.add_constraint(interior, "interior")

    monitor = PointwiseMonitor(
        invar={"x": ob_x, "y": ob_y, "t": ob_t},
        output_names=["C1", "C2"],
        metrics={
            "mean_C1": lambda var: torch.mean(var["C1"]),
            "mean_C2": lambda var: torch.mean(var["C2"]),
        },
        nodes=param_nodes,
    )
    domain.add_monitor(monitor)

    solver = Solver(cfg, domain)

    start = time.time()
    solver.solve()
    end = time.time()
    print("Training time=", end - start)

    with open("monitors/mean_C1.csv", "r") as f:
        last_line = f.readlines()[-1]
    pred_C1 = float(last_line.split(",")[-1])
    with open("monitors/mean_C2.csv", "r") as f:
        last_line = f.readlines()[-1]
    pred_C2 = float(last_line.split(",")[-1])

    true_C1 = 1.0
    true_C2 = 0.01

    print(
        "NVIDIA Modulus: estimated C1=",
        pred_C1,
        "l1re=",
        np.abs(pred_C1 - true_C1) / np.abs(true_C1),
        "l2re=",
        np.sqrt((pred_C1 - true_C1) ** 2) / np.sqrt(true_C1**2),
    )
    print(
        "NVIDIA Modulus: estimated C2=",
        pred_C2,
        "l1re=",
        np.abs(pred_C2 - true_C2) / np.abs(true_C2),
        "l2re=",
        np.sqrt((pred_C2 - true_C2) ** 2) / np.sqrt(true_C2**2),
    )


run()
