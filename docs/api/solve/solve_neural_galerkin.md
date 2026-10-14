# jinns' `solve_neural_galerkin()` function

Implement the Neural Galerkin approach defined in these two main references:

- Evolutional Deep Neural Network, Y. Du et al., 2021, [https://arxiv.org/abs/2103.09959](https://arxiv.org/abs/2103.09959).
- Neural semi-Lagrangian method for high-dimensional advection-diffusion problems, E. Franck et al., [https://arxiv.org/abs/2504.207152025](https://arxiv.org/abs/2504.207152025).

Briefly, when solving a PDE $\mathcal{N}_\theta[u_\nu(t,x)]=0$, the Neural Galerkin approach moves the time dependency into the PINN parameters such that we see the problem in the following form: $\mathcal{N}_\theta[u_{\nu_t}(x)]=0$. The Neural Galerkin approach then discretizes the time interval and constructs the set $(\nu_0,\dots,\nu_T)$ in two main steps:

1. Fit the initial condition to get the initial set of parameters $\nu_0$.
2. Resolve an ODE to get $\nu$ at each time step.

::: jinns.solve_neural_galerkin
    options:
        heading_level: 3