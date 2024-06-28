.. pinn_solver documentation master file, created by
   sphinx-quickstart on Thu May  4 21:13:02 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to jinn's documentation!
================================

Changelog:

* v0.8.10:

    - Merge `[!44] <https://gitlab.com/mia_jinns/jinns/-/merge_requests/44>`_

* v0.8.9:

    - Merge `[!41] <https://gitlab.com/mia_jinns/jinns/-/merge_requests/41>`_
    which rewrite RAR sampling for the non-stationnary PDE case using cartesian
    product of time x space: `nt` and `n` can now be different for RAR, time
    and space are handled separately, all notebooks examples are updated.

* v0.8.8:

    - Add experimental PINN architectures and notebook to solve the Matérn SPDE!

* v0.8.7:

    - Merge `[!38] <https://gitlab.com/mia_jinns/jinns/-/merge_requests/38>`_
    Introduces `validation` argument in `solve()` for user-defined validation
    strategy with possible early stopping. The validation strategy are expected
    to be `eqx.Module`, and we implement the vanilla validation loss in `jinns.
    validation.ValidationLoss`. Users are free to develop their own validation
    modules inheriting from `jinns.validation.AbstractValidationModule`.
    See the tutorial notebook.

* v0.8.6:

    - Merge `[!37] <https://gitlab.com/mia_jinns/jinns/-/merge_requests/37>`_

* v0.8.5:

    - Merge `[!36] <https://gitlab.com/mia_jinns/jinns/-/merge_requests/36>`_

* v0.8.4:

    - Fix a bug: wrong argument in the wrapper function for heterogeneous parameter evaluation of a PDEStatio

* v0.8.3:

    - Add the possibility to load user-provided tables of parameters in DataGeneratorParameter and not only to randomly sample them.

* v0.8.2:

    - Fix a bug: it was not possible to jit a reloaded HyperPINN model.

* v0.8.1:

    - New feature: `save_pinn` and `load_pinn` in `jinns.utils` for pre-trained
    PINN models.

* v0.8.0:

    - Many performance optimizations and code improvements. All notebooks have been updated with much reduced computations times.

    - New way to handle observations with `DataGeneratorObservations` that behaves like traditional dataloaders in machine learning. Checkout the documentation and the various notebooks that use observations. The previous ways to handle observations is not supported anymore.

* v0.7.2:

    - Add check to avoid tricky broadcasting bug in observation loss in ODEs

* v0.7.1:

    - Refactoring and optimizing with `jax.tree_util.*` functions instead of `for` loops. Important speed improvement, in particular for SystemLossODE/SystemLossPDE which we recommend using instead of heavy vectorial LossODE/LossPDE.

* v0.7.0:

    - We introduce HyperPINNs (*HyperPINN: Learning parameterized differential equations with physics-informed hypernetworks*, Avila Belbute-Peres et al., 2021) for a more efficient metamodeling, have a look at the notebook!

    - We start providing functions for solving PDEs with diffrax and the line method. Check out `jinns.experimental`.

    - Diverse code improvements, code corrections and updated notebooks.

* v0.6.1:

    - Generalization and improvements of heterogeneous equation parameters (in particular, an equation parameter can now be set as a PINN output). Introduce a decorator for simpler and cleaner dynamic losses with heterogeneous parameters).

    - Inverse problems from the literature are coded in notebooks 2D_Poisson_inverse and 2D_Heat_inverse.

    - Diverse code improvements, code corrections and updated notebooks.

* v0.6.0:

    - Code refactoring: now we only pass `params` to the PINN. If we want to add some elements of `params["eq_params"]` as inputs of the PINN we can do so through a `input_transform(t, x, params)` function. See the Burger notebook for inverse problem.

    - Set gradients is now highly modular: for each term of the loss you can chose the parameter(s) w.r.t. which you want to compute the gradient

    - We provide two utility functions for optimization: delayed and alternate schedule for two sets of parameters

    - Diverse code improvements, code corrections and updated notebooks.

* v0.5.2:

    - Fix problem in observation loss term. Add tests for observation loss term.

* v0.5.1:

    - Fix problem in observation loss term. Add tests for observation loss term.

    - Diverse code improvements and updated notebooks.

* v0.5.0:

    - SPINNs are now implemented in `jinns` in many common use cases and their results have been added to the existing notebooks! SPINNs have been proposed in *Separable Physics-Informed Neural Networks*, Cho et al., 2023.

    - New convention to handle vectorial solutions: now, when we are working with PDEs, neural networks encoding solution of one dimension must have a trailing (1,) in its shape. Same thing for the returned value of dynamic losses: one must have a trailing (1,)

    - Diverse code improvements

* v0.4.2:

    - Critical bug correction concerning the manipulation of the optimizer's `opt_state` which caused weird failures in the optimization process

* v0.4.1:

    - Generalize heterogeneity for the equation parameters. It can now be an arbitrary function provided by the user and thus depend on covariables. Update the corresponding notebook.

* v0.4.0:

    - Introduce Sobolev regularization as defined in *Convergence and error analysis of PINNs*, Doumèche et al., 2023 with a new notebook reproducing the example from the paper.
    - Minor corrections and code improvements (clarification of variables names and dictionnary entries)

* v0.3.4

    - Correct bug of 1D omega boundary batch being shuffled

* v0.3.3

    - Support other `jaxopt` optimizers than just the stochastic gradient descent from `jaxopt.OptaxSolver`. E.g. LBFGS can now be used. The argument `optax_solver` from `solve()` is now renamed `optimizer`, this can cause some scripts to break

    - Add printing of loss value along the optimization process

    - Updated notebook: inverse problem in Burger equation

* v0.3.2

    - Improve `solve()` for storing loss values and parameters

    - Clean inverse problem notebooks. Add the updated notebook for joint estimation of viscosity and PINN in Burger equation

    - Minor fixes

* v0.3.1

    - Check if any NaN parameter after a gradient step to be able to return the last valid parameters for debugging

    - Clean the main optimization loop. Correction: loss evaluation was performed with the parameters from the previous iteration and not the current one

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
   experimental.rst
   validation.rst

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
   Notebooks/SPDE/*

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
