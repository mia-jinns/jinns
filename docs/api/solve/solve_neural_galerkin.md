# jinns' `solve_neural_galerkin()` function

Implement the Neural Galerkin approach defined in these two main references:

- Evolutional Deep Neural Network, Y. Du et al., 2021, [https://arxiv.org/abs/2103.09959](https://arxiv.org/abs/2103.09959).
- Neural semi-Lagrangian method for high-dimensional advection-diffusion problems, E. Franck et al., [https://arxiv.org/abs/2504.207152025](https://arxiv.org/abs/2504.207152025).

Briefly, when solving a spatio-temporal PDE 
$$\partial_t u(t,x) + \mathcal{N}_\theta[u](t,x)=0,
$$ 
the Neural Galerkin approach moves the time dependency into the PINN parameters: 
$$
{u_\nu(t,x) = u_{\nu(t)}(x)}.
$$
From this, it is possible to derive the following ODE on the parameters:
$$ 
 \left\{ \begin{align*} 
    & \nu(0) = \argmin_\eta \int_\Omega |u_\eta(x) - u_0(x)|^2 \, dx , \\
    & M(\nu) \frac{d \nu(t)}{dt} =  - L(\nu)
    \end{align*} 
\right. .
$$
with $M(\nu) = \int_\Omega \nabla_\nu u_{\nu(t)}(x) \otimes \nabla_\nu u_{\nu(t)}(x) \, dx $ and $L(\nu) = \int_\Omega \nabla_\nu u_{\nu(t)}(x) \mathcal{N}_\theta[u_{\nu(t)}](x) \, dx$.

Thus, it remains to solve the parameter ODE above, by integrating using an ODE solver with our favorite scheme. In the end, we can checkooint the solver to save at specific queried times, leaving us with a set $(\nu_0,\dots,\nu_T)$.

The method works in two steps:
1. Fit the initial condition to get the initial set of parameters $\nu_0$.
2. Resolve the ODE to get $\nu(t_i)$ at each queried time step.

::: jinns.solve_neural_galerkin
    options:
        heading_level: 3