jinns
=====

Physics Informed Neural Networks with JAX. **jinns** has been developed to estimate solutions to your ODE et PDE problems using neural networks.
**jinns** is built on JAX.

**jinns** specific points:

- **jinns** is coded with JAX as a backend: forward and backward autodiff, vmapping, jitting and more!

- In **jinns**, we give the user maximum control on what is happening. We also keep the maths and computations visible and not hidden behind layers of code!

- In the near future, we want to focus the development on inverse problems and inference in mecanistic-statistical models

- [Separable PINNs](https://openreview.net/pdf?id=dEySGIcDnI) are implemented

- [Hyper PINNs](https://arxiv.org/pdf/2111.01008.pdf) are implemented

- Check out our various notebooks to get started with `jinns`

For more information, open an issue or contact us!

# Installation

Install the latest version with pip

```bash
pip install jinns
```

# Documentation

The project's documentation is available at [https://mia_jinns.gitlab.io/jinns/index.html](https://mia_jinns.gitlab.io/jinns/index.html)

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
