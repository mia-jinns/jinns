# Mathematical foundations of physics-informed neural-networks (PINNs)


# Mathematical framework and notations
Recently, the machine learning litterature have been interested in tackling the problem of learning the solution of partial differential equation (PDE) thanks to parametric function - such as neural-networks - and a dedicated loss function representing the PDE dynamic. This methodology have been coined [physics-informed neural-networks](https://maziarraissi.github.io/PINNs/) (PINNS) in the litterature.

The **jinns** package implements this approach using the JAX ecosystem.

## Dynamic loss
A dynamical system is described by a differential operator $\mathcal{N}_\theta$
where $\theta$ represents the parameters of the system such as a diffusion
coefficient, a viscosity or a growth rate. A solution $u$ satisfies the identity

$$
\begin{equation*}
\mathcal{N}_\theta[u] = 0,
\end{equation*}
$$

with possible border an initial conditions depending on the type of equation
at hand.

The **jinns** package allows to tackle two different types of problems:

 1. **Forward problem :** for a given set of equation parameters $\theta$, find a parametric function $u_{\hat{\nu}}$ forming a good approximation of the solution $u$ in the sense of minimising some "physics-loss" that we precise in the following sections
    $$
    \hat{u} = u_{\hat{\nu}} \quad \textrm{with:} \quad \hat{\nu} \in \arg \min_{\nu} L_{PINN}(\nu),
    $$
  The physics-informed neural network (PINNS) litterature proposed to use neural networks with weights and bias $\nu$ to find the best candidate minimizing the loss. The training is usually done by stochastic gradient descent on mini-batches of collocation points in the domain of $u$. In addition, automatic differentiation may both be used for computing the differential operator $\mathcal{N}_\theta$ and loss' gradients with respect to $\nu$.

 2. **Inverse problem :** for a given set of observation of the dynamic $\mathcal{D} = \{ u_i \}_{i=1}^{n_{obs}}$, find the set of equation parameters $\theta$ that best fits the data. Thus, we have some combination of the physics-loss with a data-fitting term. The latter could be a standard MSE, or more refined losses such as the likelihood of a statistical model.

    $$
    (\hat{\nu}, \hat{\theta})  \in \arg \min_{\nu, \theta}  \left\{ L_{PINN}(\nu, \theta) + w_{obs} L_{obs}(\nu, \theta; \mathcal{D}) \right\},
    $$

 3. **Meta-modeling** The problem of meta-modeling consists in learning a function $u_{\nu}(theta)$ outputting approximate solution for any values $\theta$ (within a reasonable range). In this case, the training involves feeding the function training values of equation parameters $\{\theta_j\}$.

## Ordinary differential equation


## Partial differential equation


 Introducing some notations, we wish to learn a solution $u$ to a PDE driven by the a differential operator $\mathcal{N}_\theta$ on a space domain ${\Omega \subset \mathbb{R}^d}$, a time interval $I = [0, T]$, with possible border condition on $\partial \Omega$ and initial condition $u(0, x) = u_0(x)$.

$$
\begin{equation}
\begin{cases}
\tag{PDE}
& \mathcal{N}_\theta[u](t, x) = 0, \quad \forall  t, x \in I\times \Omega, & \textrm{(Dynamic)}\\
& u(0, \cdot) = u_0(x), \quad \forall x \in \omega & \textrm{(Initial condition)} \\
& \mathcal{B}[u](t, dx) = f(dx), \quad \forall dx \in \partial \Omega, \forall t \in I & \textrm{(Boundary condition)}
\end{cases}
\end{equation}
$$

In this case, the operator $\mathcal{N}_\theta$ is a differential operator involving partial derivatives w.r.t. $t$ and $x$ and depends on a set of equation parameters $\theta$. The operator $\mathcal{B}$ acts on $\partial \Omega$, the border of $\Omega$. The PDE is said to be stationnary (in time) if $u$ does not depend on $t$.

The physics-loss in this case is described through

$$
L_{PINN}(\nu, \theta) = \Vert  \mathcal{N}_\theta[u_\nu] \Vert^2_{dyn} + w_{init} \Vert u_{\nu}(\cdot, 0) - u_0 \Vert^2_{init} + w_b \Vert \mathcal{B}[u_{\nu}] - f \Vert^2_{border},
$$

where $w_b$ and $w_{init}$ are loss weights allowing to calibrate between the different terms. Here, the notation $\Vert \cdot \Vert^2$ corresponds to MSE computed on a discretization of the time interval $I$, the space $\Omega$ and its border $\partial \Omega$.
