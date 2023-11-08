import jax
import jax.numpy as jnp
from jax import grad
from functools import partial


def _div(u, nn_params, eq_params, x, t=None):
    r"""
    Compute the divergence of a vector field :math:`\mathbf{u}` ie
    :math:`\nabla \cdot u(x)` with u a vector field from :math:`\mathbb{R}^n`
    to :math:`\mathbb{R}^n`
    """

    def scan_fun(_, i):
        if t is None:
            du_dxi = grad(
                lambda x, nn_params, eq_params: u(x, nn_params, eq_params)[i], 0
            )(x, nn_params, eq_params)[i]
        else:
            du_dxi = grad(
                lambda t, x, nn_params, eq_params: u(x, nn_params, eq_params)[i], 1
            )(x, nn_params, eq_params)[i]
        return _, du_dxi

    _, accu = jax.lax.scan(scan_fun, {}, jnp.arange(x.shape[0]))
    return jnp.sum(accu)


def _laplacian(u, nn_params, eq_params, x, t=None):
    r"""
    Compute the Laplacian of a scalar field u (from :math:`\mathbb{R}^n`
    to :math:`\mathbb{R}`) for x of arbitrary dimension ie
    :math:`\Delta u(x)=\nabla\cdot\nabla u(x)`
    For computational reason we do not compute the trace of the Hessian but
    we explicitly call the gradient twice
    """

    def scan_fun(_, i):
        if t is None:
            d2u_dxi2 = grad(
                lambda x, nn_params, eq_params: grad(u, 0)(x, nn_params, eq_params)[i],
                0,
            )(x, nn_params, eq_params)[i]
        else:
            d2u_dxi2 = grad(
                lambda t, x, nn_params, eq_params: grad(u, 1)(
                    t, x, nn_params, eq_params
                )[i],
                1,
            )(t, x, nn_params, eq_params)[i]
        return _, d2u_dxi2

    _, trace_hessian = jax.lax.scan(scan_fun, {}, jnp.arange(x.shape[0]))
    return jnp.sum(trace_hessian)


def _vectorial_laplacian(u, nn_params, eq_params, x, t=None, u_vec_ndim=None):
    r"""
    Compute the vectorial Laplacian of a vector field u (from :math:`\mathbb{R}^m`
    to :math:`\mathbb{R}^n`) for x of arbitrary dimension ie
    :math:`\Delta \mathbf{u}(x)=\nabla\cdot\nabla \mathbf{u}(x)`

    **Note:** We need to provide in u_vec_ndim the dimension of the vector
    :math:`\mathbf{u}(x)` if it is different than that of x
    """
    if u_vec_ndim is None:
        u_vec_ndim = x.shape[0]

    def scan_fun(_, j):
        # The loop over the components of u(x). We compute one Laplacian for
        # each of these components
        if t is None:
            uj = lambda x, nn_params, eq_params: u(x, nn_params, eq_params)[j]
        else:
            uj = lambda t, x, nn_params, eq_params: u(t, x, nn_params, eq_params)[j]
        lap_on_j = _laplacian(uj, nn_params, eq_params, x, t)
        return _, lap_on_j

    _, vec_lap = jax.lax.scan(scan_fun, {}, jnp.arange(u_vec_ndim))
    return vec_lap


def _u_dot_nabla_times_u(u, nn_params, eq_params, x, t=None):
    r"""
    Implement :math:`((\mathbf{u}\cdot\nabla)\mathbf{u})(x)` for x of arbitrary
    dimension. Note that :math:`\mathbf{u}` is a vector field from :math:`\mathbb{R}^n`
    to :math:`\mathbb{R}^n`
    Currently for `x.ndim=2`

    **Note:** We do not use loops but code explicitly the expression to avoid
    computing twice some terms
    """
    if x.shape[0] == 2:
        if t is None:
            ux = lambda x: u(x, nn_params, eq_params)[0]
            uy = lambda x: u(x, nn_params, eq_params)[1]

            dux_dx = lambda x: grad(ux, 0)(x)[0]
            dux_dy = lambda x: grad(ux, 0)(x)[1]

            duy_dx = lambda x: grad(uy, 0)(x)[0]
            duy_dy = lambda x: grad(uy, 0)(x)[1]

            return jnp.array(
                [
                    ux(x) * dux_dx(x) + uy(x) * dux_dy(x),
                    ux(x) * duy_dx(x) + uy(x) * duy_dy(x),
                ]
            )
        else:
            ux = lambda t, x: u(t, x, nn_params, eq_params)[0]
            uy = lambda t, x: u(t, x, nn_params, eq_params)[1]

            dux_dx = lambda t, x: grad(ux, 1)(t, x)[0]
            dux_dy = lambda t, x: grad(ux, 1)(t, x)[1]

            duy_dx = lambda t, x: grad(uy, 1)(t, x)[0]
            duy_dy = lambda t, x: grad(uy, 1)(t, x)[1]

            return jnp.array(
                [
                    ux(t, x) * dux_dx(t, x) + uy(t, x) * dux_dy(t, x),
                    ux(t, x) * duy_dx(t, x) + uy(t, x) * duy_dy(t, x),
                ]
            )
    else:
        raise NotImplementedError("x.ndim must be 2")


def _sobolev(u, m, statio=True):
    r"""
    Compute the Sobolev regularization of order m
    of a scalar field u (from :math:`\mathbb{R}^d1` to :math:`\mathbb{R}`)
    for x of arbitrary dimension i.e.
    :math:`\frac{1}{n_l}\sum_{l=1}^{n_l}\sum_{|\alpha|=1}^{m+1} ||\partial^{\alpha} u(x_l)||_2^2` where
    :math:`m\geq\max(d_1 // 2, K)` with `K` the order of the differential
    operator.

    This regularization is proposed in _Convergence and error analysis of
    PINNs_, Doumeche et al., 2023, https://arxiv.org/pdf/2305.01240.pdf
    """

    def jac_recursive(u, order, start):
        # Compute the derivative of order `start`
        if order == 0:
            return u
        elif start == 0:
            return jac_recursive(jax.jacrev(u), order - 1, start + 1)
        else:
            return jac_recursive(jax.jacfwd(u), order - 1, start + 1)

    if statio:
        return lambda x, nn_params, eq_params: jnp.sum(
            jac_recursive(lambda x: u(x, nn_params, eq_params), m + 1, 0)(x) ** 2
        )
    else:
        return lambda t, x, nn_params, eq_params: jnp.sum(
            jac_recursive(
                lambda tx: u(tx[0:1], tx[1:], nn_params, eq_params), m + 1, 0
            )(jnp.concatenate([t, x], axis=0))
            ** 2
        )
