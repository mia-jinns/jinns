"""
Implements diverse operators for dynamic losses
"""

from typing import Literal

import jax
import jax.numpy as jnp
from jax import grad
import equinox as eqx
from jaxtyping import Float, Array
from jinns.parameters._params import Params


def divergence_rev(
    inputs: Float[Array, "dim"] | Float[Array, "1+dim"],
    u: eqx.Module,
    params: Params,
    eq_type: Literal["nonstatio_PDE", "statio_PDE"] = None,
) -> float:
    r"""
    Compute the divergence of a vector field $\mathbf{u}$, i.e.,
    $\nabla_\mathbf{x} \cdot \mathbf{u}(\mathrm{inputs})$ with $\mathbf{u}$ a vector
    field from $\mathbb{R}^d$ to $\mathbb{R}^d$ or $\mathbb{R}^{1+d}$
    to $\mathbb{R}^{1+d}$. Thus, this
    function can be used for stationary or non-stationary PINNs. In the first
    case $\mathrm{inputs}=\mathbf{x}$, in the second case
    case $\mathrm{inputs}=\mathbf{t,x}$.
    The computation is done using backward AD

    Parameters
    ----------
    inputs
        `x` or `t_x`
    u
        the PINN
    params
        the PINN parameters
    eq_type
        whether we consider a stationary or non stationary PINN. Most often we
        can know that by inspecting the `u` argument (PINN object). But if `u` is
        a function, we must set this attribute.
    """

    try:
        eq_type = u.eq_type
    except AttributeError:
        pass  # use the value passed as argument
    if eq_type is None:
        raise ValueError("eq_type could not be set!")

    def scan_fun(_, i):
        if eq_type == "nonstatio_PDE":
            du_dxi = grad(lambda inputs, params: u(inputs, params)[1 + i])(
                inputs, params
            )[1 + i]
        else:
            du_dxi = grad(lambda inputs, params: u(inputs, params)[i])(inputs, params)[
                i
            ]
        return _, du_dxi

    if eq_type == "nonstatio_PDE":
        _, accu = jax.lax.scan(scan_fun, {}, jnp.arange(inputs.shape[0] - 1))
    elif eq_type == "statio_PDE":
        _, accu = jax.lax.scan(scan_fun, {}, jnp.arange(inputs.shape[0]))
    else:
        raise ValueError("Unexpected u.eq_type!")
    return jnp.sum(accu)


def divergence_fwd(
    inputs: Float[Array, "batch_size dim"] | Float[Array, "batch_size 1+dim"],
    u: eqx.Module,
    params: Params,
    eq_type: Literal["nonstatio_PDE", "statio_PDE"] = None,
) -> Float[Array, "batch_size * (1+dim) 1"] | Float[Array, "batch_size * (dim) 1"]:
    r"""
    Compute the divergence of a **batched** vector field $\mathbf{u}$, i.e.,
    $\nabla_\mathbf{x} \cdot \mathbf{u}(\mathbf{x})$ with $\mathbf{u}$ a vector
    field from $\mathbb{R}^{b \times d}$ to $\mathbb{R}^{b \times b
    \times d}$ or from $\mathbb{R}^{b \times d+1}$ to $\mathbb{R}^{b \times b
    \times d+1}$. Thus, this
    function can be used for stationary or non-stationary PINNs.
    Because of the embedding that happens in SPINNs the
    computation is most efficient with forward AD. This is the idea behind
    Separable PINNs.

    !!! warning "Warning"

        This function is to be used in the context of SPINNs only.

    Parameters
    ----------
    inputs
        `x` or `t_x`
    u
        the PINN
    params
        the PINN parameters
    eq_type
        whether we consider a stationary or non stationary PINN. Most often we
        can know that by inspecting the `u` argument (PINN object). But if `u` is
        a function, we must set this attribute.
    """

    try:
        eq_type = u.eq_type
    except AttributeError:
        pass  # use the value passed as argument
    if eq_type is None:
        raise ValueError("eq_type could not be set!")

    def scan_fun(_, i):
        if eq_type == "nonstatio_PDE":
            tangent_vec = jnp.repeat(
                jax.nn.one_hot(i + 1, inputs.shape[-1])[None],
                inputs.shape[0],
                axis=0,
            )
            __, du_dxi = jax.jvp(
                lambda inputs: u(inputs, params)[..., 1 + i], (inputs,), (tangent_vec,)
            )
        else:
            tangent_vec = jnp.repeat(
                jax.nn.one_hot(i, inputs.shape[-1])[None],
                inputs.shape[0],
                axis=0,
            )
            __, du_dxi = jax.jvp(
                lambda inputs: u(inputs, params)[..., i], (inputs,), (tangent_vec,)
            )
        return _, du_dxi

    if eq_type == "nonstatio_PDE":
        _, accu = jax.lax.scan(scan_fun, {}, jnp.arange(inputs.shape[1] - 1))
    elif eq_type == "statio_PDE":
        _, accu = jax.lax.scan(scan_fun, {}, jnp.arange(inputs.shape[1]))
    else:
        raise ValueError("Unexpected u.eq_type!")
    return jnp.sum(accu, axis=0)


def laplacian_rev(
    inputs: Float[Array, "dim"] | Float[Array, "1+dim"],
    u: eqx.Module,
    params: Params,
    method: Literal["trace_hessian_x", "trace_hessian_t_x", "loop"] = "trace_hessian_x",
    eq_type: Literal["nonstatio_PDE", "statio_PDE"] = None,
) -> float:
    r"""
    Compute the Laplacian of a scalar field $u$ from $\mathbb{R}^d$
    to $\mathbb{R}$ or from $\mathbb{R}^{1+d}$ to $\mathbb{R}$, i.e., this
    function can be used for stationary or non-stationary PINNs. In the first
    case $\mathrm{inputs}=\mathbf{x}$ is of arbitrary dimension, i.e.,
    $\Delta_\mathbf{x} u(\mathbf{x})=\nabla_\mathbf{x}\cdot\nabla_\mathbf{x} u(\mathbf{x})$.
    In the second case $inputs=\mathbf{t,x}$, but we still compute
    $\Delta_\mathbf{x} u(\mathrm{inputs})$.
    The computation is done using backward AD.

    Parameters
    ----------
    inputs
        `x` or `t_x`
    u
        the PINN
    params
        the PINN parameters
    method
        how to compute the Laplacian. `"trace_hessian_x"` means that we take
        the trace of the Hessian matrix computed with `x` only (`t` is excluded
        from the beginning, we compute less derivatives at the price of a
        concatenation). `"trace_hessian_t_x"` means that the computation
        of the Hessian integrates `t` which is excluded at the end (we avoid a
        concatenate but we compute more derivatives). `"loop"` means that we
        directly compute the second order derivatives with a loop (we avoid
        non-diagonal derivatives at the cost of a loop).
    eq_type
        whether we consider a stationary or non stationary PINN. Most often we
        can know that by inspecting the `u` argument (PINN object). But if `u` is
        a function, we must set this attribute.
    """

    try:
        eq_type = u.eq_type
    except AttributeError:
        pass  # use the value passed as argument
    if eq_type is None:
        raise ValueError("eq_type could not be set!")

    if method == "trace_hessian_x":
        # NOTE we afford a concatenate here to avoid computing Hessian elements for
        # nothing. In case of simple derivatives we prefer the vectorial
        # computation and then discarding elements but for higher order derivatives
        # it might not be worth it. See other options below for computating the
        # Laplacian
        if eq_type == "nonstatio_PDE":
            u_ = lambda x: jnp.squeeze(
                u(jnp.concatenate([inputs[:1], x], axis=0), params)
            )
            return jnp.sum(jnp.diag(jax.hessian(u_)(inputs[1:])))
        if eq_type == "statio_PDE":
            u_ = lambda inputs: jnp.squeeze(u(inputs, params))
            return jnp.sum(jnp.diag(jax.hessian(u_)(inputs)))
        raise ValueError("Unexpected eq_type!")
    if method == "trace_hessian_t_x":
        # NOTE that it is unclear whether it is better to vectorially compute the
        # Hessian (despite a useless time dimension) as below
        if eq_type == "nonstatio_PDE":
            u_ = lambda inputs: jnp.squeeze(u(inputs, params))
            return jnp.sum(jnp.diag(jax.hessian(u_)(inputs))[1:])
        if eq_type == "statio_PDE":
            u_ = lambda inputs: jnp.squeeze(u(inputs, params))
            return jnp.sum(jnp.diag(jax.hessian(u_)(inputs)))
        raise ValueError("Unexpected eq_type!")

    if method == "loop":
        # For a small d, we found out that trace of the Hessian is faster, see
        # https://stackoverflow.com/questions/77517357/jax-grad-derivate-with-respect-an-specific-variable-in-a-matrix
        # but could the trick below for taking directly the diagonal elements
        # prove useful in higher dimensions?

        u_ = lambda inputs: u(inputs, params).squeeze()

        def scan_fun(_, i):
            if eq_type == "nonstatio_PDE":
                d2u_dxi2 = grad(
                    lambda inputs: grad(u_)(inputs)[1 + i],
                )(
                    inputs
                )[1 + i]
            else:
                d2u_dxi2 = grad(
                    lambda inputs: grad(u_, 0)(inputs)[i],
                    0,
                )(
                    inputs
                )[i]
            return _, d2u_dxi2

        if eq_type == "nonstatio_PDE":
            _, trace_hessian = jax.lax.scan(
                scan_fun, {}, jnp.arange(inputs.shape[0] - 1)
            )
        elif eq_type == "statio_PDE":
            _, trace_hessian = jax.lax.scan(scan_fun, {}, jnp.arange(inputs.shape[0]))
        else:
            raise ValueError("Unexpected eq_type!")
        return jnp.sum(trace_hessian)
    raise ValueError("Unexpected method argument!")


def laplacian_fwd(
    inputs: Float[Array, "batch_size 1+dim"] | Float[Array, "batch_size dim"],
    u: eqx.Module,
    params: Params,
    method: Literal["trace_hessian_t_x", "trace_hessian_x", "loop"] = "loop",
    eq_type: Literal["nonstatio_PDE", "statio_PDE"] = None,
) -> Float[Array, "batch_size * (1+dim) 1"] | Float[Array, "batch_size * (dim) 1"]:
    r"""
    Compute the Laplacian of a **batched** scalar field $u$
    from $\mathbb{R}^{b\times d}$ to $\mathbb{R}^{b\times b}$ or
    from $\mathbb{R}^{b\times (1 + d)}$ to $\mathbb{R}^{b\times b}$ or, i.e., this
    function can be used for stationary or non-stationary PINNs
    for $\mathbf{x}$ of arbitrary dimension $d$ or $1+d$ with batch
    dimension $b$.
    Because of the embedding that happens in SPINNs the
    computation is most efficient with forward AD. This is the idea behind
    Separable PINNs.

    !!! warning "Warning"

        This function is to be used in the context of SPINNs only.

    !!! warning "Warning"

        Because of the batch dimension, the current implementation of
        `method="trace_hessian_t_x"` or `method="trace_hessian_x"`
        should not be used except for debugging
        purposes. Indeed, computing the Hessian is very costly.

    Parameters
    ----------
    inputs
        `x` or `t_x`
    u
        the PINN
    params
        the PINN parameters
    method
        how to compute the Laplacian. `"trace_hessian_t_x"` means that the computation
        of the Hessian integrates `t` which is excluded at the end (**see
        Warning**). `"trace_hessian_x"` means an Hessian computation which
        excludes `t` (**see Warning**). `"loop"` means that we
        directly compute the second order derivatives with a loop (we avoid
        non-diagonal derivatives at the cost of a loop).
    eq_type
        whether we consider a stationary or non stationary PINN. Most often we
        can know that by inspecting the `u` argument (PINN object). But if `u` is
        a function, we must set this attribute.
    """

    try:
        eq_type = u.eq_type
    except AttributeError:
        pass  # use the value passed as argument
    if eq_type is None:
        raise ValueError("eq_type could not be set!")

    if method == "loop":

        def scan_fun(_, i):
            if eq_type == "nonstatio_PDE":
                tangent_vec = jnp.repeat(
                    jax.nn.one_hot(i + 1, inputs.shape[-1])[None],
                    inputs.shape[0],
                    axis=0,
                )
            else:
                tangent_vec = jnp.repeat(
                    jax.nn.one_hot(i, inputs.shape[-1])[None],
                    inputs.shape[0],
                    axis=0,
                )

            du_dxi_fun = lambda inputs: jax.jvp(
                lambda inputs: u(inputs, params),
                (inputs,),
                (tangent_vec,),
            )[1]
            __, d2u_dxi2 = jax.jvp(du_dxi_fun, (inputs,), (tangent_vec,))
            return _, d2u_dxi2

        if eq_type == "nonstatio_PDE":
            _, trace_hessian = jax.lax.scan(
                scan_fun, {}, jnp.arange(inputs.shape[-1] - 1)
            )
        elif eq_type == "statio_PDE":
            _, trace_hessian = jax.lax.scan(scan_fun, {}, jnp.arange(inputs.shape[-1]))
        else:
            raise ValueError("Unexpected eq_type!")
        return jnp.sum(trace_hessian, axis=0)
    if method == "trace_hessian_t_x":
        if eq_type == "nonstatio_PDE":
            # compute the Hessian including the batch dimension, get rid of the
            # (..,1,..) axis that is here because of the scalar output
            # if inputs.shape==(10,3) (1 for time, 2 for x_dim)
            # then r.shape=(10,10,10,1,10,3,10,3)
            # there are way too much derivatives!
            r = jax.hessian(u)(inputs, params).squeeze()
            # compute the traces by avoid the time derivatives
            # after that r.shape=(10,10,10,10)
            r = jnp.trace(r[..., :, 1:, :, 1:], axis1=-3, axis2=-1)
            # but then we are in a cartesian product, for each coordinate on
            # the first two dimensions we only want the trace at the same
            # coordinate on the last two dimensions
            # this is done easily with einsum but we need to automate the
            # formula according to the input dim
            res_dims = "".join([f"{chr(97 + d)}" for d in range(inputs.shape[-1])])
            lap = jnp.einsum(res_dims + "ii->" + res_dims, r)
            return lap[..., None]
        if eq_type == "statio_PDE":
            # compute the Hessian including the batch dimension, get rid of the
            # (..,1,..) axis that is here because of the scalar output
            # if inputs.shape==(10,2), r.shape=(10,10,1,10,2,10,2)
            # there are way too much derivatives!
            r = jax.hessian(u)(inputs, params).squeeze()
            # compute the traces, after that r.shape=(10,10,10,10)
            r = jnp.trace(r, axis1=-3, axis2=-1)
            # but then we are in a cartesian product, for each coordinate on
            # the first two dimensions we only want the trace at the same
            # coordinate on the last two dimensions
            # this is done easily with einsum but we need to automate the
            # formula according to the input dim
            res_dims = "".join([f"{chr(97 + d)}" for d in range(inputs.shape[-1])])
            lap = jnp.einsum(res_dims + "ii->" + res_dims, r)
            return lap[..., None]
        raise ValueError("Unexpected eq_type!")
    if method == "trace_hessian_x":
        if eq_type == "statio_PDE":
            # compute the Hessian including the batch dimension, get rid of the
            # (..,1,..) axis that is here because of the scalar output
            # if inputs.shape==(10,2), r.shape=(10,10,1,10,2,10,2)
            # there are way too much derivatives!
            r = jax.hessian(u)(inputs, params).squeeze()
            # compute the traces, after that r.shape=(10,10,10,10)
            r = jnp.trace(r, axis1=-3, axis2=-1)
            # but then we are in a cartesian product, for each coordinate on
            # the first two dimensions we only want the trace at the same
            # coordinate on the last two dimensions
            # this is done easily with einsum but we need to automate the
            # formula according to the input dim
            res_dims = "".join([f"{chr(97 + d)}" for d in range(inputs.shape[-1])])
            lap = jnp.einsum(res_dims + "ii->" + res_dims, r)
            return lap[..., None]
        raise ValueError("Unexpected eq_type!")
    raise ValueError("Unexpected method argument!")


def vectorial_laplacian_rev(
    inputs: Float[Array, "dim"] | Float[Array, "1+dim"],
    u: eqx.Module,
    params: Params,
    dim_out: int = None,
) -> Float[Array, "dim_out"]:
    r"""
    Compute the vectorial Laplacian of a vector field $\mathbf{u}$ from
    $\mathbb{R}^d$ to $\mathbb{R}^n$ or from $\mathbb{R}^{1+d}$ to
    $\mathbb{R}^n$, i.e., this
    function can be used for stationary or non-stationary PINNs. In the first
    case $\mathrm{inputs}=\mathbf{x}$ is of arbitrary dimension, i.e.,
    $\Delta_\mathbf{x} \mathbf{u}(\mathbf{x})=\nabla\cdot\nabla
    \mathbf{u}(\mathbf{x})$.
    In the second case $inputs=\mathbf{t,x}$, and we perform
    $\Delta_\mathbf{x} \mathbf{u}(\mathrm{inputs})=\nabla\cdot\nabla
    \mathbf{u}(\mathrm{inputs})$.

    Parameters
    ----------
    inputs
        `x` or `t_x`
    u
        the PINN
    params
        the PINN parameters
    dim_out
        Dimension of the vector $\mathbf{u}(\mathrm{inputs})$. This needs to be
        provided if it is different than that of $\mathrm{inputs}$.
    """
    if dim_out is None:
        dim_out = inputs.shape[0]

    def scan_fun(_, j):
        # The loop over the components of u(x). We compute one Laplacian for
        # each of these components
        # Note the jnp.expand_dims call
        uj = lambda inputs, params: jnp.expand_dims(u(inputs, params)[j], axis=-1)
        lap_on_j = laplacian_rev(inputs, uj, params, eq_type=u.eq_type)

        return _, lap_on_j

    _, vec_lap = jax.lax.scan(scan_fun, {}, jnp.arange(dim_out))
    return vec_lap


def vectorial_laplacian_fwd(
    inputs: Float[Array, "batch_size dim"] | Float[Array, "batch_size 1+dim"],
    u: eqx.Module,
    params: Params,
    dim_out: int = None,
) -> Float[Array, "batch_size * (1+dim) n"] | Float[Array, "batch_size * (dim) n"]:
    r"""
    Compute the vectorial Laplacian of a vector field $\mathbf{u}$ when
    `u` is a SPINN, in this case, it corresponds to a vector
    field from from $\mathbb{R}^{b\times d}$ to
    $\mathbb{R}^{b\times b\times n}$ or from $\mathbb{R}^{b\times 1+d}$ to
    $\mathbb{R}^{b\times b\times n}$, i.e., this
    function can be used for stationary or non-stationary PINNs.

    Forward mode AD is used.

    !!! warning "Warning"

        This function is to be used in the context of SPINNs only.

    Parameters
    ----------
    inputs
        `x` or `t_x`
    u
        the PINN
    params
        the PINN parameters
    dim_out
        the value of the output dimension ($n$ in the formula above). Must be
        set if different from $d$.
    """
    if dim_out is None:
        dim_out = inputs.shape[0]

    def scan_fun(_, j):
        # The loop over the components of u(x). We compute one Laplacian for
        # each of these components
        # Note the expand_dims
        uj = lambda inputs, params: jnp.expand_dims(u(inputs, params)[..., j], axis=-1)
        lap_on_j = laplacian_fwd(inputs, uj, params, eq_type=u.eq_type)

        return _, lap_on_j

    _, vec_lap = jax.lax.scan(scan_fun, {}, jnp.arange(dim_out))
    return jnp.moveaxis(vec_lap.squeeze(), 0, -1)


def _u_dot_nabla_times_u_rev(
    x: Float[Array, "2"], u: eqx.Module, params: Params
) -> Float[Array, "2"]:
    r"""
    Implement $((\mathbf{u}\cdot\nabla)\mathbf{u})(\mathbf{x})$ for
    $\mathbf{x}$ of arbitrary
    dimension. $\mathbf{u}$ is a vector field from $\mathbb{R}^n$
    to $\mathbb{R}^n$. **Currently for** `x.ndim=2` **only**.
    The computation is done using backward AD.
    We do not use loops but code explicitly the expression to avoid
    computing twice some terms
    """
    assert x.shape[0] == 2
    ux = lambda x: u(x, params)[0]
    uy = lambda x: u(x, params)[1]

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


def _u_dot_nabla_times_u_fwd(
    x: Float[Array, "batch_size 2"],
    u: eqx.Module,
    params: Params,
) -> Float[Array, "batch_size batch_size 2"]:
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
    assert x.shape[-1] == 2
    tangent_vec_0 = jnp.repeat(jnp.array([1.0, 0.0])[None], x.shape[0], axis=0)
    tangent_vec_1 = jnp.repeat(jnp.array([0.0, 1.0])[None], x.shape[0], axis=0)
    u_at_x, du_dx = jax.jvp(
        lambda x: u(x, params), (x,), (tangent_vec_0,)
    )  # thanks to forward AD this gets dux_dx and duy_dx in a vector
    # ie the derivatives of both components of u wrt x
    # this also gets the vector of u evaluated at x
    u_at_x, du_dy = jax.jvp(
        lambda x: u(x, params), (x,), (tangent_vec_1,)
    )  # thanks to forward AD this gets dux_dy and duy_dy in a vector
    # ie the derivatives of both components of u wrt y
    return jnp.stack(
        [
            u_at_x[..., 0] * du_dx[..., 0] + u_at_x[..., 1] * du_dy[..., 0],
            u_at_x[..., 0] * du_dx[..., 1] + u_at_x[..., 1] * du_dy[..., 1],
        ],
        axis=-1,
    )
