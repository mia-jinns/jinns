# Natural gradient

Natural gradient methods computes a preconditionner on the euclidean gradient, in
a similar fashion as in second order methods (e.g. Newton's method). It is strongly advised
to use it when the network size is moderate (~1000) as it leads to robust performance
improvements.

Mathematically, at a point $\nu \in \mathbb{R}^p$, the natural gradient $\eta$ is the solution of the linear
system

$$
    \eta = G^{-1} \nabla_{\nu} \mathcal{L}(\nu).
$$
The preconditionner matrix $G = [G_ij]_{i,j=1}^p$ is the so-called *Gram matrix* of a mapping $\nu \mapsto \rho_\nu$

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


## Vanilla Natural gradient 

::: jinns.optimizers.vanilla_ngd
    options:
        heading_level: 3


## Various flavour of NGD

Other regularization for natural gradients have been investigated, to name a few

 * [ANaGRAM](https://arxiv.org/pdf/2412.10782)
 * [Sketchy Natural Gradient](https://proceedings.mlr.press/v267/best-mckay25a.html) 
 * [Nyström Natural Gradient](https://arxiv.org/pdf/2505.11638v3)

We plan to implement them in the future in jinns.
