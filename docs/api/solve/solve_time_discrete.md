# jinns' `solve_time_discrete()` function

The `solve_time_discrete()` function is the entry point for solving temporal PDEs with discrete time methods. Two are implemented: Discrete PINNs and Neural Galerkin.

## Discrete PINNs

Implement the Discrete PINNs approach defined in these two main references:

- Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations, M. Raissi et al., 2019, [https://www.sciencedirect.com/science/article/pii/S0021999118307125](https://www.sciencedirect.com/science/article/pii/S0021999118307125)
- Neural semi-Lagrangian method for high-dimensional advection-diffusion problems, E. Franck et al., [https://arxiv.org/abs/2504.207152025](https://arxiv.org/abs/2504.207152025).

In this approach, let us write the temporal PDE as: 

$$\partial_t u(t,x) = \mathcal{N}[u](t,x),
$$ 

and move the time dependency into the PINN parameters: 

$$
{u_\nu(t,x) = u_{\nu(t)}(x)}.
$$

Then, Discrete PINN relies on a time discretization $(t^n)_n$ and then on solving a small optimization problem (taking advantage on a warm start for weights and biases) at each time step to construct the weights and biases temporal dynamic. Note that the procedure is initialized by fitting the intial condition. For example, when using the explicit Euler scheme (or RK1), the transition from $\nu(t)$ to $\nu(t+1)$ is based on the approximation:

$$
\frac{u_{\nu(t+1)}(x)-u_{\nu(t)}(x)}{\Delta t}=\mathcal{N}[u_{\nu(t)}](x).
$$

Then this means that we must solve:

$$
\nu(t+1)\in\mathrm{argmin}_{\nu}\int_{\Omega}|u_{\nu}(x)-u_{\nu(t)}(x)+\Delta t \mathcal{N}[u_{\nu(t)}](x)|^2\mathrm{d}x.
$$

In the recommended RK4 scheme the update is based on the equation:

$$
u_{\nu(t+1)}(x) = u_{\nu(t)}(x) + \frac{\Delta t}{6}(k_1+2k_2+2k_3+k_4),
$$

where

$$
\begin{cases}
k_1=\mathcal{N}[u_{\nu(t)}](x),\\
k_2=\mathcal{N}[u_{\nu(t)} + \frac{\Delta t}{2}k_1](x)\triangleq\mathcal{N}[u_{\nu'(t)}],\\
k_3=\mathcal{N}[u_{\nu(t)} + \frac{\Delta t}{2}k_2](x)\triangleq\mathcal{N}[u_{\nu''(t)}],\\
k_4=\mathcal{N}[u_{\nu(t)} + \Delta t k_3](x)\triangleq\mathcal{N}[u_{\nu'''(t)}].\\
\end{cases}
$$

We then get $\nu(t+1)$ by successively solving 4 optimization problems :

First

$$
\nu'(t)\in\mathrm{argmin}_{\nu}\int_{\Omega}|u_{\nu}(x)-u_{\nu(t)}(x)+\frac{\Delta t}{2} \mathcal{N}[u_{\nu(t)}](x)|^2\mathrm{d}x.
$$

Then

$$
\nu''(t)\in\mathrm{argmin}_{\nu}\int_{\Omega}|u_{\nu}(x)-u_{\nu(t)}(x)+\frac{\Delta t}{2} \mathcal{N}[u_{\nu'(t)}](x)|^2\mathrm{d}x.
$$

Then

$$
\nu'''(t)\in\mathrm{argmin}_{\nu}\int_{\Omega}|u_{\nu}(x)-u_{\nu(t)}(x)+\Delta t \mathcal{N}[u_{\nu''(t)}](x)|^2\mathrm{d}x.
$$

And finally

$$
\nu(t+1)\in\mathrm{argmin}_{\nu}\int_{\Omega}|u_{\nu}(x)+\frac{\Delta t}{6}(\mathcal{N}[u_{\nu(t)}](x)+2\mathcal{N}[u_{\nu'(t)}](x)+2\mathcal{N}[u_{\nu''(t)}](x)+\mathcal{N}[u_{\nu'''(t)}](x))|^2\mathrm{d}x.
$$


## Neural Galerkin 

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

Thus, it remains to solve the parameter ODE above, by integrating using an ODE solver with our favorite scheme. In the end, we can checkpoint the solver to save at specific queried times, leaving us with a set $(\nu_0,\dots,\nu_T)$.

The method works in two steps:
1. Fit the initial condition to get the initial set of parameters $\nu_0$.
2. Resolve the ODE to get $\nu(t_i)$ at each queried time step.

::: jinns.solve_time_discrete
    options:
        heading_level: 3