.. pinn_solver documentation master file, created by
   sphinx-quickstart on Thu May  4 21:13:02 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to jinn's documentation!
================================

Changelog:

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
