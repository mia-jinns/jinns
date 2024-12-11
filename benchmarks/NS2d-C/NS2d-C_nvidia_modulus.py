import os

# os.environ['HYDRA_FULL_ERROR']='1'
seed = 1

import numpy as np
import matplotlib.pyplot as plt

# from PINNACLE
DEFAULT_NUM_DOMAIN_POINTS = 8192 // 10
DEFAULT_NUM_BOUNDARY_POINTS = 2048 // 10

from sympy import Symbol, Function, Number, sin, Eq

import modulus.sym
from modulus.sym.hydra import to_yaml
from modulus.distributed import DistributedManager
from modulus.sym.hydra import instantiate_arch, ModulusConfig
from modulus.sym.eq.phy_informer import PhysicsInformer
from modulus.models.mlp.fully_connected import FullyConnected
from modulus.sym.eq.pdes.navier_stokes import NavierStokes
from modulus.sym.solver import Solver
from modulus.sym.domain import Domain
from modulus.sym.geometry.primitives_1d import Point1D
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
    x, y = Symbol("x"), Symbol("y")
    rec = Rectangle((0, 0), (1, 1))
    domain = Domain()

    nu = 0.01
    a = 4

    ns = NavierStokes(nu=nu, rho=1, dim=2, time=False)

    model = instantiate_arch(
        input_keys=[Key("x"), Key("y")],
        output_keys=[Key("u"), Key("v"), Key("p")],
        cfg=cfg.arch.fully_connected,
        nr_layers=5,
        layer_size=100,
        activation_fn=torch.nn.Tanh(),
    )
    # print(model)

    nodes = ns.make_nodes() + [model.make_node(name="ns_network")]

    # top wall
    top_wall_BC = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec,
        outvar={"u": a * x * (1 - x), "v": 0},
        batch_size=DEFAULT_NUM_BOUNDARY_POINTS // 4,  # NOTE divide by 4
        batch_per_epoch=1,  # total nb of points=batch_size*batch_per_epoch
        fixed_dataset=True,
        criteria=Eq(y, 1),
    )
    domain.add_constraint(top_wall_BC, "top_wall_BC")

    # other walls
    other_BC = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec,
        outvar={"u": 0, "v": 0},
        batch_size=3 * DEFAULT_NUM_BOUNDARY_POINTS // 4,
        batch_per_epoch=1,  # total nb of points=batch_size*batch_per_epoch
        fixed_dataset=True,
        criteria=y < 1,
    )
    domain.add_constraint(other_BC, "other_BC")

    # p0 constraint
    # Note that writing a PointwiseConstraint.from_numpy() with a single point
    # does not work (error in var_polyvtk)
    # outvar = {"p": np.zeros((1, 1))}
    # p_constraint = PointwiseConstraint.from_numpy(
    #    nodes=nodes,
    #    invar={"x": np.zeros((1, 1)), "y": np.zeros((1, 1))},
    #    outvar=outvar,
    #    batch_size=1,
    # )
    # domain.add_constraint(p_constraint, "p_constraint")
    # so we use an alternative way:
    point = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=Point1D(0.0, Parameterization({Parameter("y"): 0})),
        outvar={"p": 0.0},
        batch_size=2,  # don'k know why we need this=2 otherwise same error as above
        batch_per_epoch=2,
        fixed_dataset=True,
    )
    domain.add_constraint(point, "point")

    # interior
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=rec,
        outvar={"continuity": 0, "momentum_x": 0, "momentum_y": 0},
        batch_size=DEFAULT_NUM_DOMAIN_POINTS,
        batch_per_epoch=1,  # total nb of points=batch_size*batch_per_epoch
        fixed_dataset=True,
    )
    domain.add_constraint(interior, "interior")

    ref_data = np.loadtxt("../../lid_driven_a4.dat", comments="%").astype(np.float32)
    invar_numpy = {"x": ref_data[:, 0:1], "y": ref_data[:, 1:2]}
    true_outvar = {"u": ref_data[:, 2:3], "v": ref_data[:, 3:4], "p": ref_data[:, 4:5]}
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
            np.concatenate(
                [
                    data["arr_0"][()]["true_u"],
                    data["arr_0"][()]["true_v"],
                    data["arr_0"][()]["true_p"],
                ],
                axis=-1,
            ),
            np.concatenate(
                [
                    data["arr_0"][()]["pred_u"],
                    data["arr_0"][()]["pred_v"],
                    data["arr_0"][()]["pred_p"],
                ],
                axis=-1,
            ),
        )  # about indexing https://stackoverflow.com/a/37949466


run()
