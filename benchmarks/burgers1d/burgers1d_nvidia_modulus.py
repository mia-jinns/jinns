import os

# os.environ['HYDRA_FULL_ERROR']='1'
seed = 1

import numpy as np
import matplotlib.pyplot as plt

# from PINNACLE
DEFAULT_NUM_DOMAIN_POINTS = 8192 // 2
DEFAULT_NUM_BOUNDARY_POINTS = 2048 // 2
DEFAULT_NUM_INITIAL_POINTS = 2048 // 2

ref_data = np.loadtxt("burgers1d.dat", comments="%").astype(np.float32)

from sympy import Symbol, Function, Number, sin
from modulus.sym.eq.pde import PDE


class BurgersEquation(PDE):
    name = "BurgersEquation"

    def __init__(self, c=1.0):

        # coordinates
        x = Symbol("x")

        # time
        t = Symbol("t")

        # make input variables
        input_variables = {"x": x, "t": t}

        # make u function
        u = Function("u")(*input_variables)

        # viscosity
        c = Number(c)

        # set equations
        self.equations = {}
        self.equations["burgers_equation"] = (
            u.diff(t, 1) + u * u.diff(x) - c * u.diff(x, 2)
        )


import modulus.sym
from modulus.sym.hydra import to_yaml
from modulus.distributed import DistributedManager
from modulus.sym.hydra import instantiate_arch, ModulusConfig
from modulus.sym.eq.phy_informer import PhysicsInformer
from modulus.models.mlp.fully_connected import FullyConnected
from modulus.sym.solver import Solver
from modulus.sym.domain import Domain
from modulus.sym.geometry.primitives_1d import Line1D
from modulus.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
)

from modulus.sym.domain.validator import PointwiseValidator
from modulus.sym.domain.monitor import PointwiseMonitor
from modulus.sym.key import Key

from modulus.sym.utils.benchmark import timeit

import torch

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
    x, t_symbol = Symbol("x"), Symbol("t")
    geo = Line1D(-1, 1)
    domain = Domain()

    be = BurgersEquation(c=0.01 / np.pi)

    model = instantiate_arch(
        input_keys=[Key("x"), Key("t")],
        output_keys=[Key("u")],
        cfg=cfg.arch.fully_connected,
        nr_layers=5,
        layer_size=100,
        activation_fn=torch.nn.Tanh(),
    )
    # print(model)

    nodes = be.make_nodes() + [model.make_node(name="be_network")]

    # initial condition
    IC = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"u": -sin(np.pi * x)},
        batch_size=DEFAULT_NUM_INITIAL_POINTS,
        batch_per_epoch=1,  # total nb of points=batch_size*batch_per_epoch
        parameterization={t_symbol: 0.0},
        fixed_dataset=True,
    )
    domain.add_constraint(IC, "IC")

    # boundary condition
    BC = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"u": 0},
        batch_size=DEFAULT_NUM_BOUNDARY_POINTS,
        batch_per_epoch=1,  # total nb of points=batch_size*batch_per_epoch
        parameterization={t_symbol: (0, 1)},
        fixed_dataset=True,
    )
    domain.add_constraint(BC, "BC")

    # interior
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"burgers_equation": 0},
        batch_size=DEFAULT_NUM_DOMAIN_POINTS,
        batch_per_epoch=1,  # total nb of points=batch_size*batch_per_epoch
        parameterization={t_symbol: (0, 1)},
        fixed_dataset=True,
    )
    domain.add_constraint(interior, "interior")

    ref_data_ = trans_time_data_to_dataset(ref_data, "../../burgers1d.dat", 2, 1)
    ref_data_ = ref_data_
    invar_numpy = {"x": ref_data_[:, 0:1], "t": ref_data_[:, 1:2]}
    true_outvar = {"u": ref_data_[:, 2:3]}
    validator = PointwiseValidator(
        nodes=nodes,
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
        # plt.scatter(
        #    x=data["arr_0"][()]["x"],
        #    y=data["arr_0"][()]["t"],
        #    c=data["arr_0"][()]["pred_u"],
        #    #cmap="viridis"
        # )
        # plt.colorbar()
        # plt.savefig("burgers_modulus_pred.png")
        # plt.scatter(
        #    x=data["arr_0"][()]["x"],
        #    y=data["arr_0"][()]["t"],
        #    c=data["arr_0"][()]["true_u"],
        #    #cmap="viridis"
        # )
        # plt.colorbar()
        # plt.savefig("burgers_modulus_true.png")


run()
