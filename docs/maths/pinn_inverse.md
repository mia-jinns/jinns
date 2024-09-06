# Parameter estimation in PINNs


In these notes we gives insight about joint parameter estimation in PINN
problems, i.e., we wish to estimate both the differential equation parameters
and its solution with a PINN. We recall the notations used in introductory sections:
 * the differential operator $\mathcal{N}_\theta[\cdot]$ with equation parameters $\theta$
 * the PINN $u_\nu$ with weight and biases $\nu$

The pinn framework aims at minimizing the dynamic loss
``` math
\begin{equation}
\vert \mathcal{N}_{\theta}[u_{\nu}](t, x) \vert^2,
\end{equation}
```
with some boundary/initial conditions

# Vanilla parameter estimation

A direct and quite successful approach has been proposed in _Physics-informed
neural networks: A deep learning framework for solving forward and inverse
problems involving nonlinear partial differential equations_, Raissi et al.,
2019. This procedure consists in a joint alternate minimization of the dynamic
loss with respect to $\theta$ and $\nu$,

``` math
\begin{equation}
\text{Alternate between }
$\hat{\theta}=\mathrm{argmin}_{\theta}\Vert\mathcal{N}_{\theta}[u_{\nu}](t,x)\Vert
\text{ and }
$\hat{\nu}=\mathrm{argmin}_{\nu}\Vert\mathcal{N}_{\theta}[u_{\nu}](t,x)\Vert,
\end{equation}
```

along with an additional mean square error term with respect to an available batch of
observations,

``` math
\begin{equation}
\mathrm{argmin}_{\nu} \sum_{i}(u_{\nu}(t_i,x_i) - u^{obs}(t_i,x_i))^2
\end{equation}
```

In some of our example notebooks we show how to perform joint estimation according to
this procedure.

# Vanilla parameter estimation + $\theta$ is fed in the PINN
TODO ?

# A full statistical model for parameter estimation
TODO ?
