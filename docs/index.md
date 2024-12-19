# Welcome to jinns' documentation

**jinns** is a Python package for physics-informed neural networks (PINNs) in the [JAX](https://jax.readthedocs.io/en/latest/) ecosystem. It provides an intuitive and flexible interface for

 * forward problem: learning a PDE solution.
 * inverse problem: learning the parameters of a PDE.
 * meta-modeling: learning a family of PDE indexed by its parameters.

# Installation

```
pip install jinns
```

Requires Python 3.11+ and JAX 0.4.31+

JAX ecosystem: jinns depends on [Equinox](https://docs.kidger.site/equinox/) and [Optax](https://optax.readthedocs.io/en/latest/).

# Examples

Navigate the tutorials and example sections of the left, with many introductory notebooks, as well as advanced examples.

# News and updates

See the [changelog page](./changelog.md) for the last updates, features and bug fixes.

# Citing us

Please consider citing our work if you found it useful to yours, using this [ArXiV preprint](https://arxiv.org/abs/2412.14132)
```
@article{gangloff_jouvin2024jinns,
  title={jinns: a JAX Library for Physics-Informed Neural Networks},
  author={Gangloff, Hugo and Jouvin, Nicolas},
  journal={arXiv preprint arXiv:2412.14132},
  year={2024}
}
```
