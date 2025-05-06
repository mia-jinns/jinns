from ._diffrax_solver import (
    SpatialDiscretisation,
    reaction_diffusion_2d_vector_field,
    laplacian,
    dirichlet_boundary_condition,
    neumann_boundary_condition,
    plot_diffrax_solution,
)

__all__ = [
    "SpatialDiscretisation",
    "reaction_diffusion_2d_vector_field",
    "laplacian",
    "dirichlet_boundary_condition",
    "neumann_boundary_condition",
    "plot_diffrax_solution",
]
