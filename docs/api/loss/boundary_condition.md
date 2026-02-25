# Boundary condition

## Abstract Class

::: jinns.loss.BoundaryConditionAbstract
    options:
        members: False
        heading_level: 3

## Dirichlet

::: jinns.loss.Dirichlet
    options:
        members: False
        heading_level: 3

## Neumann

::: jinns.loss.Neumann
    options:
        members: False
        heading_level: 3

## Utility

If the boundary condition on all facets is equal you may want to decorate with the following function which takes care of the vectorization on the facet dimension.

::: jinns.loss.equation_on_all_facets_equal
