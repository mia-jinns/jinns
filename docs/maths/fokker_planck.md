# Fokker-Planck

## Theory

$$
\begin{equation*}
\begin{cases}
    & {\mathrm d} X_t =  \mu(X_t, t) {\mathrm d}t + \sigma(X_t, t) {\mathrm d} W_t,\\
    & X_0 \sim u_0. \\
\end{cases}
\tag{SDE}
\end{equation*}
$$

The [Fokker-Planck equation](https://en.wikipedia.org/wiki/Fokker%E2%80%93Planck_equation) describes the evolution of $u(x,t)$ the law of $X_t$. We give the 1-dimensional formulation here but the d-dimensional fomulation can be found [on wikipedia](https://en.wikipedia.org/wiki/Fokker%E2%80%93Planck_equation#Higher_dimensions)

$$
\begin{equation*}
 \begin{cases}
      &\frac{\partial}{\partial t} u(x, t) = - \frac{\partial}{\partial x} \left[  \mu(x,t) u(x,t) \right] +   \frac{1}{2} \frac{\partial^2}{\partial x^2}  \left[  \sigma(x,t)^2  u(x,t) \right], \\
      &u(x,0)=u_0(x), \quad \textrm{(initial condition)} \\
      & \int_{\Omega} u(x, t) \mathrm d x = 1, \quad \textrm{(p.d.f. condition)}\\
  \end{cases} \qquad x\in \Omega \subset \mathbb{R}, t\in\mathbb{R}^+.
\end{equation*}
$$

Thus, we wish to solve FPE to learn $u_{\hat{\nu}}(x, t)$ with a PINN loss

$$
\begin{equation}
  L_{\textrm{FPE}}(\nu) = \Vert - \frac{\partial}{\partial t} u_\nu - \frac{\partial}{\partial x} \left[  \mu u_\nu \right] +   \frac{1}{2} \frac{\partial^2}{\partial x^2}  \left[  \sigma^2  u_\nu \right] \Vert_{dyn}^2 + w_t \Vert u_{\nu}(\cdot, 0) - u_0 \Vert_{temp}^2 + w_{pdf} \Vert \int_{\Omega} u_{\nu}(\cdot, x) \textrm{d} x - 1 \Vert_{pdf}^2.
\end{equation}
$$
If we wish to learn the stationary distribution, the $- \frac{\partial}{\partial t} u$ is set to 0 in the FPE loss.

!!! question "What about the border condition ?"

    There is no border condition on $\partial \Omega$ here, but an additional
    loss term: the network must learn a p.d.f. with normalization constant
    equals to 1 at any time $t$. This also prevents the neural-network from learning
    the trivial solution $u_\nu = 0$. The normalization constant at time $t$ is
    approximated via Monte-Carlo integration on $\Omega$.


## Examples

Several example in different dimension $d$, stationary or not, are presented in the `Notebooks/` folder.
