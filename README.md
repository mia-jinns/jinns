jinns
=====

Physics Informed Neural Networks with JAX. **jinns** has been developed to estimate solutions to your ODE et PDE problems using neural networks.
**jinns** is built on JAX.

**jinns** specific points:

- **jinns** is coded with JAX as a backend: forward and backward autodiff, vmapping, jitting and more!

- We focus the development towards inverse problems and inference in mecanistic-statistical models

- [Separable PINN](https://openreview.net/pdf?id=dEySGIcDnI) are implemented

- Check out our various notebooks to get started with `jinns`

For more information, open an issue or contact us!

# Installation

Install the latest version with pip

```bash
pip install jinns
```

# Documentation

The project's documentation is available at [https://mia_jinns.gitlab.io/jinns/index.html](https://mia_jinns.gitlab.io/jinns/index.html)

Note that all the tests were performed on a rather small Nvidia T600 GPU, expect a substancial performance gain on bigger devices.

# Contributing

* First fork the library on Gitlab.

* Then clone and install the library in development mode with

```bash
pip install -e .
```

* Install pre-commit and run it.

```bash
pip install pre-commit
pre-commit install
```

* Open a merge request once you are done with your changes.

# Contributors & references

*Active*: Hugo Gangloff, Nicolas Jouvin
*Past*: Pierre Gloaguen, Charles Ollion, Achille Thin
