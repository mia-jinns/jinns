.. pinn_solver documentation master file, created by
   sphinx-quickstart on Thu May  4 21:13:02 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to jinn's documentation!
================================

Changelog:

* v0.3.1

    - Check is NaN parameter after a gradient step to be able to return the last valid parameters for debugging

    - Clean the main optimization loop. Correction: loss evaluation was performed with the parameters from the previous iteration and not the current one!

* v0.3.0

    - Optimization now starts with a call to `jinns.solve()` which replaces the class `PinnSolver`. This change is not backward compatible, check out the updated notebooks and documentation!

    - Loss weights are now totally modular (to ponderate vectorial outputs, to comply with SystemLossXXX, etc.)

    - Reimplementation in jinns of *Systems Biology: Identifiability Analysis and Parameter Identification via Systems-Biology-Informed Neural Networks*, Mitchell Daneker  et al., Computational Modeling of Signaling Networks, 2023

* v0.2.1

    - Introduce non homogeneous equation parameters

    - Improve dynamic losses with generic operators

    - Minor fixes and improvements

* v0.2.0

    - Implement a vanilla version of Resisual Adaptative Resampling scheme as proposed in *DeepXDE: A deep learning library for solving differential equations*, Lu Lu et al., SIAM Review, 2021

    - Implement metamodel learning, i.e., a parameter of the PDE/ODE is input of the neural  network and the dynamic loss is minimized for all the values of that parameter. For example, this is studied in *Surrogate modeling for fluid flows based  on physics-constrained deep learning without simulation data*, Sun et al. 2020

    - Implement seq2seq learning for ODE, based on the article *Characterizing possible failure modes in physics-informed neural networks*, Krishnapriyan et al., NeurIPS 2021

    - New notebooks to illustrate these approaches

    - Minor bug fixes and code improvements
    - Doc improvements

* v0.1.0

    * Initial release

.. toctree::
   :maxdepth: 1
   :caption: Documented modules

   solver.rst
   loss.rst
   data.rst
   utils.rst

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Package tutorials

   Notebooks/Tutorials/*

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Notebooks example

   Notebooks/ODE/*
   Notebooks/PDE/*

.. toctree::
   :maxdepth: 1
   :caption: Background

   math_pinn.rst
   fokker_planck.rst
   param_estim_pinn.rst



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
