# Preconditionned optimization in Jinns

Preconditionned gradient descent computes a preconditionner on the euclidean gradient, in the form of a psd matrix. It comes in various flavors, based on the Hessian of the loss as in Newton's method, or an information matrix as in Natural Gradient. It is strongly advised to use preconditionned optimization when the network size is moderate (~1000) as it leads to robust performance improvements.

Mathematically, at iteration $t$ and a point $\nu^{(t)} \in \mathbb{R}^p$, the preconditionned gradient step $\eta^{(t)}$ is the solution of the linear
system

$$
    \eta^{(t)} = P(\nu^{(t)})^{-1} \nabla_{\nu} \mathcal{L}(\nu^{(t)}).
$$

Here, the preconditionned matrix $P(\nu^{(t)})$ is a $p\times p$ psd matrix, where $p$ is the number of trainable networks parameters. Depending on the method, stochastic variants and regularizations when solving the linear system may exists.

For now in jinns we implement in the form of `optax.GradientTransformExtraArgs`
 1. Quasi-Newton methods: BFGS and Broyden's algorithm
 2. Natural gradient descent: vanilla version (aka energy-based GD) with optional ridge regularization

## ssBFGS and ssBroyden

Self-scaled version of Broyden's and BFGS (Broyden–Fletcher–Goldfarb–Shanno) quasi-Newton methods for approximating the Hessian.

Our implementation is based on the [Scimba](https://www.scimba.org/) source code. The algorithms are described in [this article](https://arxiv.org/pdf/2405.04230).

::: jinns.optimizers.self_scaled_bfgs_or_broyden
    options:
        heading_level: 3

## Natural gradient

In natural gradient, the precondition matrix $P=G = [G_{ij}]_{i,j=1}^p$ is the so-called *Gram matrix* induced by the parametric mapping $\nu \mapsto \rho_\nu$.

\begin{align*}
    G_{kl} = \int \partial_{\nu_k} \rho_\nu(x) \partial_{\nu_l} \rho_\nu(x) \; \mathrm{d} x \\
    \hat{G}_{kl} \approx \frac{1}{n} \sum_{i=1}^n  \partial_{\nu_k} \rho_\nu(x_i) \partial_{\nu_l} \rho_\nu(x_i)
\end{align*}

This $L^2$ inner product is approximated empirically using the same collocation points as the one
used for the loss function.

In the PINNs context, the mapping $\rho_\nu(x)$ contains all the dynamic, BC, IC and
data-fitting terms  at point $x$ (when these terms are appropriate depending on the ODE/PDE problem):

$$
    \rho_\nu : x \mapsto (\mathcal{N}[u_\nu](x), \mathcal{B}[u_\nu](x), \ldots )
$$


### Vanilla Natural gradient 

::: jinns.optimizers.vanilla_ngd
    options:
        heading_level: 3


### Various flavour of NGD

Other regularization for natural gradients have been investigated, to name a few

 * [ANaGRAM](https://arxiv.org/pdf/2412.10782)
 * [Sketchy Natural Gradient](https://proceedings.mlr.press/v267/best-mckay25a.html) 
 * [Nyström Natural Gradient](https://arxiv.org/pdf/2505.11638v3)

We plan to implement them in the future in jinns.
