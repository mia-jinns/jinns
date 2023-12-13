import jax
import jax.numpy as jnp
from jax import grad
from functools import partial
from jinns.utils._pinn import PINN
from jinns.utils._spinn import SPINN


def _div_rev(u, nn_params, eq_params, x, t=None):
    r"""
    Compute the divergence of a vector field :math:`\mathbf{u}`, i.e.,
    :math:`\nabla \cdot \mathbf{u}(\mathbf{x})` with :math:`\mathbf{u}` a vector
    field from :math:`\mathbb{R}^d` to :math:`\mathbb{R}^d`.
    The computation is done using backward AD
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


def _div_fwd(u, nn_params, eq_params, x, t=None):
    r"""
    Compute the divergence of a **batched** vector field :math:`\mathbf{u}`, i.e.,
    :math:`\nabla \cdot \mathbf{u}(\mathbf{x})` with :math:`\mathbf{u}` a vector
    field from :math:`\mathbb{R}^{b \times d}` to :math:`\mathbb{R}^{b \times b
    \times d}`. The result is then in :math:`\mathbb{R}^{b\times b}`.
    Because of the embedding that happens in SPINNs the
    computation is most efficient with forward AD. This is the idea behind Separable PINNs.
    This function is to be used in the context of SPINNs only.
    """

    def scan_fun(_, i):
        tangent_vec = jnp.repeat(
            jax.nn.one_hot(i, x.shape[-1])[None], x.shape[0], axis=0
        )
        if t is None:
            __, du_dxi = jax.jvp(
                lambda x: u(x, nn_params, eq_params)[..., i], (x,), (tangent_vec,)
            )
        else:
            __, du_dxi = jax.jvp(
                lambda x: u(t, x, nn_params, eq_params)[..., i], (x,), (tangent_vec,)
            )
        return _, du_dxi

    _, accu = jax.lax.scan(scan_fun, {}, jnp.arange(x.shape[1]))
    return jnp.sum(accu, axis=0)


def _laplacian_rev(u, nn_params, eq_params, x, t=None):
    r"""
    Compute the Laplacian of a scalar field :math:`u` (from :math:`\mathbb{R}^d`
    to :math:`\mathbb{R}`) for :math:`\mathbf{x}` of arbitrary dimension, i.e.,
    :math:`\Delta u(\mathbf{x})=\nabla\cdot\nabla u(\mathbf{x})`.
    The computation is done using backward AD.
    """

    # Note that the last dim of u is nec. 1
    if t is None:
        u_ = lambda x: u(x, nn_params, eq_params)[0]
    else:
        u_ = lambda t, x: u(t, x, nn_params, eq_params)[0]

    if t is None:
        return jnp.trace(jax.hessian(u_)(x))
    else:
        return jnp.trace(jax.hessian(u_, argnums=1)(t, x))

    # For a small d, we found out that trace of the Hessian is faster, but the
    # trick below for taking directly the diagonal elements might prove useful
    # in higher dimensions?

    # def scan_fun(_, i):
    #    if t is None:
    #        d2u_dxi2 = grad(
    #            lambda x: grad(u_, 0)(x)[i],
    #            0,
    #        )(
    #            x
    #        )[i]
    #    else:
    #        d2u_dxi2 = grad(
    #            lambda t, x: grad(u_, 1)(t, x)[i],
    #            1,
    #        )(
    #            t, x
    #        )[i]
    #    return _, d2u_dxi2

    # _, trace_hessian = jax.lax.scan(scan_fun, {}, jnp.arange(x.shape[0]))
    # return jnp.sum(trace_hessian)


def _laplacian_fwd(u, nn_params, eq_params, x, t=None):
    r"""
    Compute the Laplacian of a **batched** scalar field :math:`u`
    (from :math:`\mathbb{R}^{b\times d}` to :math:`\mathbb{R}^{b\times b}`)
    for :math:`\mathbf{x}` of arbitrary dimension :math:`d` with batch
    dimension :math:`b`.
    Because of the embedding that happens in SPINNs the
    computation is most efficient with forward AD. This is the idea behind Separable PINNs.
    This function is to be used in the context of SPINNs only.
    """

    def scan_fun(_, i):
        tangent_vec = jnp.repeat(
            jax.nn.one_hot(i, x.shape[-1])[None], x.shape[0], axis=0
        )

        if t is None:
            du_dxi_fun = lambda x: jax.jvp(
                lambda x: u(x, nn_params, eq_params)[..., 0], (x,), (tangent_vec,)
            )[
                1
            ]  # Note the indexing [..., 0]
            __, d2u_dxi2 = jax.jvp(du_dxi_fun, (x,), (tangent_vec,))
        else:
            du_dxi_fun = lambda x: jax.jvp(
                lambda x: u(t, x, nn_params, eq_params)[..., 0], (x,), (tangent_vec,)
            )[
                1
            ]  # Note the indexing [..., 0]
            __, d2u_dxi2 = jax.jvp(du_dxi_fun, (x,), (tangent_vec,))
        return _, d2u_dxi2

    _, trace_hessian = jax.lax.scan(scan_fun, {}, jnp.arange(x.shape[1]))
    return jnp.sum(trace_hessian, axis=0)


def _vectorial_laplacian(u, nn_params, eq_params, x, t=None, u_vec_ndim=None):
    r"""
    Compute the vectorial Laplacian of a vector field :math:`\mathbf{u}` (from
    :math:`\mathbb{R}^d`
    to :math:`\mathbb{R}^n`) for :math:`\mathbf{x}` of arbitrary dimension, i.e.,
    :math:`\Delta \mathbf{u}(\mathbf{x})=\nabla\cdot\nabla
    \mathbf{u}(\mathbf{x})`.

    **Note:** We need to provide `u_vec_ndim` the dimension of the vector
    :math:`\mathbf{u}(\mathbf{x})` if it is different than that of
    :math:`\mathbf{x}`.

    **Note:** `u` can be a SPINN, in this case, it corresponds to a vector
    field from (from :math:`\mathbb{R}^{b\times d}` to
    :math:`\mathbb{R}^{b\times b\times n}`) and forward mode AD is used.
    Technically, the return is of dimension :math:`n\times b \times b`.
    """
    if u_vec_ndim is None:
        u_vec_ndim = x.shape[0]

    def scan_fun(_, j):
        # The loop over the components of u(x). We compute one Laplacian for
        # each of these components
        # Note the expand_dims
        if isinstance(u, PINN):
            if t is None:
                uj = lambda x, nn_params, eq_params: jnp.expand_dims(
                    u(x, nn_params, eq_params)[j], axis=-1
                )
            else:
                uj = lambda t, x, nn_params, eq_params: jnp.expand_dims(
                    u(t, x, nn_params, eq_params)[j], axis=-1
                )
            lap_on_j = _laplacian_rev(uj, nn_params, eq_params, x, t)
        elif isinstance(u, SPINN):
            if t is None:
                uj = lambda x, nn_params, eq_params: jnp.expand_dims(
                    u(x, nn_params, eq_params)[..., j], axis=-1
                )
            else:
                uj = lambda t, x, nn_params, eq_params: jnp.expand_dims(
                    u(t, x, nn_params, eq_params)[..., j], axis=-1
                )
            lap_on_j = _laplacian_fwd(uj, nn_params, eq_params, x, t)

        return _, lap_on_j

    _, vec_lap = jax.lax.scan(scan_fun, {}, jnp.arange(u_vec_ndim))
    return vec_lap


def _u_dot_nabla_times_u_rev(u, nn_params, eq_params, x, t=None):
    r"""
    Implement :math:`((\mathbf{u}\cdot\nabla)\mathbf{u})(\mathbf{x})` for
    :math:`\mathbf{x}` of arbitrary
    dimension. :math:`\mathbf{u}` is a vector field from :math:`\mathbb{R}^n`
    to :math:`\mathbb{R}^n`. **Currently for** `x.ndim=2` **only**.
    The computation is done using backward AD.
    We do not use loops but code explicitly the expression to avoid
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


def _u_dot_nabla_times_u_fwd(u, nn_params, eq_params, x, t=None):
    r"""
    Implement :math:`((\mathbf{u}\cdot\nabla)\mathbf{u})(\mathbf{x})` for
    :math:`\mathbf{x}` of arbitrary dimension **with a batch dimension**.
    I.e., :math:`\mathbf{u}` is a vector field from :math:`\mathbb{R}^{b\times
    b}`
    to :math:`\mathbb{R}^{b\times b \times d}`. **Currently for** :math:`d=2`
    **only**.
    We do not use loops but code explicitly the expression to avoid
    computing twice some terms.
    Because of the embedding that happens in SPINNs the
    computation is most efficient with forward AD. This is the idea behind Separable PINNs.
    This function is to be used in the context of SPINNs only.
    """
    if x.shape[-1] == 2:
        tangent_vec_0 = jnp.repeat(jnp.array([1.0, 0.0])[None], x.shape[0], axis=0)
        tangent_vec_1 = jnp.repeat(jnp.array([0.0, 1.0])[None], x.shape[0], axis=0)
        if t is None:
            u_at_x, du_dx = jax.jvp(
                lambda x: u(x, nn_params, eq_params), (x,), (tangent_vec_0,)
            )  # thanks to forward AD this gets dux_dx and duy_dx in a vector
            # ie the derivatives of both components of u wrt x
            # this also gets the vector of u evaluated at x
            u_at_x, du_dy = jax.jvp(
                lambda x: u(x, nn_params, eq_params), (x,), (tangent_vec_1,)
            )  # thanks to forward AD this gets dux_dy and duy_dy in a vector
            # ie the derivatives of both components of u wrt y

        else:
            u_at_x, du_dx = jax.jvp(
                lambda x: u(t, x, nn_params, eq_params), (x,), (tangent_vec_0,)
            )
            u_at_x, du_dy = jax.jvp(
                lambda x: u(t, x, nn_params, eq_params), (x,), (tangent_vec_1,)
            )

        return jnp.stack(
            [
                u_at_x[..., 0] * du_dx[..., 0] + u_at_x[..., 1] * du_dy[..., 0],
                u_at_x[..., 0] * du_dx[..., 1] + u_at_x[..., 1] * du_dy[..., 1],
            ],
            axis=-1,
        )
    else:
        raise NotImplementedError("x.ndim must be 2")


def _sobolev(u, m, statio=True):
    r"""
    Compute the Sobolev regularization of order :math:`m`
    of a scalar field :math:`u` (from :math:`\mathbb{R}^{d}` to :math:`\mathbb{R}`)
    for :math:`\mathbf{x}` of arbitrary dimension :math:`d`, i.e.,
    :math:`\frac{1}{n_l}\sum_{l=1}^{n_l}\sum_{|\alpha|=1}^{m+1} ||\partial^{\alpha} u(x_l)||_2^2` where
    :math:`m\geq\max(d_1 // 2, K)` with :math:`K` the order of the differential
    operator.

    This regularization is proposed in *Convergence and error analysis of
    PINNs*, Doumeche et al., 2023, https://arxiv.org/pdf/2305.01240.pdf
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
