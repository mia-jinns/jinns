jinns.loss
================

The ``loss`` module is decomposed of different parts:

* `Loss ODE` points to a class that implements the loss for solving PINN problems involving ordinary differential equations

* `Loss PDE` points to classes that implements the loss for solving PINN problems involving stationary and non stationary partial differential equations

* `Dynamic Loss` points to classes which implement the mechanistic part of the above mentioned losses.
These classes have functions implementing the differential operator :math:`\mathcal{N}[u]`

* `Operators` points to functions which help implementing differential operators

* `Boundary Conditions` points to functions which help implementing the boundary conditions of partial differential equations



.. toctree::
   :maxdepth: 1
   :caption: Loss modules:

   loss_ode.rst
   loss_pde.rst
   dynamic_loss.rst
   boundary_conditions.rst
   operators.rst
