# Loss computations in jinns

This document gives details about the steps involved in loss computations in jinns.

## 1. Residuals for a single colocation point

First, the functions whose name ends in `_apply` from the file `jinns/loss/loss_utils.py` implement the formula to compute the loss value for a single colocation point, data point (for observations), Monte-Carlo sample (for normalization...). This means:

- `initial_condition_apply` implements $u(x, 0)-f^{ini}(x), \forall x\in\Omega$.
- `dynamic_loss_apply` implements $\mathcal{N}[u](x), \forall x\in\Omega$ or $\mathcal{N}[u](x, t), \forall (x, t)\in\Omega\times I$.
- `boundary_condition_apply` implements $\mathcal{B}[u](x), \forall x\in\Omega$ or $\mathcal{B}[u](x, t), \forall (x, t)\in\Omega\times I$.
- `observations_loss_apply` implements $u(x_i^{obs})-y_i^{obs}$ where $(x_i^{obs},y_i^{obs})$ is a pair of observation points.
- `normalization_loss_apply` implements $u(x) * \omega_i$ where $x$ is a Monte-Carlo sample and $\omega_i$ its weight.

There is no vmapping or reduction operation here. They will come shortly.

## 2. Transform and homogeneize the `_apply` functions

The next step consists in transforming and fixing all the heterogeneous arguments of the previous functions to give them all the signature: `(batch, params)-> value`. This is done the functions `_get_XXX_fun` (e.g. `_get_boundary_loss_fun`) in the files `_abstract_loss.py` or `_LossXDE.py`. This step then enables a very generic vmapping and reduction over all loss terms.

## 3. Vmap the functions

All the previously formed functions will then be vmapped (with different vmapping recipe). To do so:

- One the one hand, each of the main loss classes (`LossODE`, `LossPDEStatio` and `LossPDENonStatio`) implements a method `_prepare_loss_terms` which gathers, in a `LossXXXComponents` module, the previously formed functions, as well as the `batch` and possibly more arguments that parameterize the vmapping operation (let's name them `kwargs_vmapping`).
- On the other hand, each of the main loss classes has a `_vmap_loss_fun` attribute which is a `LossXXXComponents` which refers to a function that will perform the vmapping (one of the functions that start with `vmap_` in `loss/loss_utils.py`).

Then, in `_abstract_loss.py`, the function `_get_evaluate_by_terms_lambda` actually performs the vmapping via a tree mapping operation over the two elements listed above. This returns a `LossXXXComponents` where each leaf is the vmapped residual function as a function of the parameters (the batch is fixed), i.e., `lambda p: jax.vmap(fun, kwargs_vmapping)(batch, params=p)`.

## 4. Reduce the function

Each of the main loss classes has a `_reduction_functions` attribute which is a `LossXXXComponents` which implements a function that performs the reduction operation.

It is composed over the vmapped residuals of the previous step. In a similar manner we get a `LossXXXComponents` where each leaf is the **reduced** vmapped residual function as a function of the parameters, i.e., `lambda p: reduction_function(jax.vmap(fun, kwargs_vmapping))(batch, params=p)`. Again this is done in `_get_evaluate_by_terms_lambda`.

## 5. Compute the loss terms

This is the actual computation of the loss terms where the functions are evaluated for a set of parameters. This is done in `evaluate` method from `_abstract_loss.py`.


## 6. Ponderation and summation

This is done in `ponderate_and_sum_loss`.
