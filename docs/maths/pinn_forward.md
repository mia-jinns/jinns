# Mathematical foundations of physics-informed neural-networks (PINNs)


# Mathematical framework and notations
Recently, the machine learning litterature have been interested in tackling the problem of learning the solution of partial differential equation (PDE) thanks to parametric function - such as neural-networks - and a dedicated loss function representing the PDE dynamic. This methodology have been coined [physics-informed neural-networks](https://maziarraissi.github.io/PINNs/) (PINNS) in the litterature.

The **jinns** package implements this approach using the JAX ecosystem.


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

The operator $\mathcal{N}_\theta$ is a differential operator involving partial derivatives w.r.t. $t$ and $x$ and depends on a set of equation parameters $\theta$. The operator $\mathcal{B}$ acts on $\partial \Omega$, the border of $\Omega$. The PDE is said to be stationnary (in time), if $u$ does not change with $t$.


The **jinns** package allows to tackle two different types of problems:

 1. **Forward problem :** for a given set of equation parameters $\theta$, find a parametric function $u_{\hat{\nu}}$ which is a good approximation of the solution $u$ in some sense made precise below.
 2. **Inverse problem :** for a given set of observation of the dynamic $\mathcal{D} = \{ u_{obs}(t_i, x_i))\}_{i=1}^{n_{obs}}$, find the set of equation parameters $\theta$ that best fits the data.

In forward problems, our goal is to learn a parametric function $u_\nu$ which approximates a solution of (PDE). The physics-informed neural network (PINNS) litterature proposed to use neural networks with weights and bias $\nu$ to find the best candidate minimizing the loss

$$
\hat{u} = u_{\hat{\nu}} \quad \textrm{with:} \quad \hat{\nu} \in \arg \min_{\nu} \left\{ L(\nu) = \Vert  \mathcal{N}_\theta[u_\nu] \Vert^2_{dyn} + w_{init} \Vert u_{\nu}(\cdot, 0) - u_0 \Vert^2_{init} + w_b \Vert \mathcal{B}[u_{\nu}] - f \Vert^2_{border} \right\},
$$

where the $(w_b, w_{init})$ are loss weights allowing to calibrate between the different terms. Here, the notation $\Vert \cdot \Vert^2$ corresponds to MSE computed on a discretization of the time interval $I$, the space $\Omega$ and its border $\partial \Omega$. The training may be done via stochastic gradient descent on batches of the training sets. In addition, automatic differentiation can be used both for computing the differential operator $\mathcal{N}_\theta$ and gradients with respect to $\nu$.

For inverse problem, we define a function $u_{\nu, \theta}$ and wish to find an estimate of the equation parameters $\hat{\theta}$ so that

$$
(\hat{\nu}, \hat{\theta})  \in \arg \min_{\nu, \theta}  \left\{ L(\nu, \theta) + w_{obs} \Vert u_{\nu, \theta} - u_{obs}\Vert_{\mathcal{D}}^2 \right\},
$$

**New:** The problem of meta-model learning a function $u_{\nu, \theta}$ giving an approximate solution for any values $\theta$ (within a reasonable range) is not now tackled by the package. In this case, the function is learnt over a grid of values $\{\theta_j\}$.


## Ordinary differential equation
TODO
