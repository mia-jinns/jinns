from ._diffrax_solver import (
    SpatialDiscretisation,
    reaction_diffusion_2d_vector_field,
    laplacian,
    dirichlet_boundary_condition,
    neumann_boundary_condition,
    plot_diffrax_solution,
)
from ._sinuspinn import create_sinusPINN
from ._spectralpinn import create_spectralPINN
