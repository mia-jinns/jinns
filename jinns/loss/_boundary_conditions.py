import jax
import jax.numpy as jnp
from jax import vmap, grad


def _compute_boundary_loss_statio(
    boundary_condition_type, f, border_batch, u, params, facet
):
    r"""A generic function that will compute the mini-batch MSE of a
    boundary condition in the stationary case, given by:

    .. math::
        D[u](\partial x) = f(\partial x), \forall \partial x \in \partial \Omega

    Where :math:`D[\cdot]` is a differential operator, possibly identity.


    Parameters
    ----------
    boundary_condition_type : a string
        a string defining the differential operator :math:`D[\cdot]`.
        Currently implements one of "Dirichlet" (:math:`D = Id`) and Von
        Neuman (:math:`D[\cdot] = \nabla \cdot n`) where :math:`n` is the
        unitary outgoing vector normal to :math:`\partial\Omega`
    f :
        the function to be matched in the boundary condition. It should have
        one argument only (other are ignored).
    border_batch : jnp.array
        the mini-batch on :math:`\partial \Omega`
    u :
        a PINN
    params:
        The dictionary of parameters of the model.
        Typically, it is a dictionary of
        dictionaries: `eq_params` and `nn_params``, respectively the
        differential equation parameters and the neural network parameter
    facet:
        An integer which represents the id of the facet which is currently
        considered (in the order provided wy the DataGenerator which is fixed)

    Returns
    -------
    scalar
        the MSE computed on `border_batch`
    """
    if boundary_condition_type.lower() in "dirichlet":
        mse = boundary_dirichlet_statio(f, border_batch, u, params)
    elif any(
        [
            boundary_condition_type.lower() in s
            for s in ["von neumann", "vn", "vonneumann"]
        ]
    ):
        mse = boundary_neumann_statio(f, border_batch, u, params, facet)
    return mse


def _compute_boundary_loss_nonstatio(
    boundary_condition_type, f, times_batch, border_batch, u, params, facet
):
    r"""A generic function that will compute the mini-batch MSE of a
    boundary condition in the non-stationary case, given by:

    .. math::
        D[u](t, \partial x) = f(\partial x), \forall t \in I, \forall \partial
        x \in \partial \Omega

    Where :math:`D[\cdot]` is a differential operator, possibly identity.


    Parameters
    ----------
    boundary_condition_type : a string
        a string defining the differential operator :math:`D[\cdot]`.
        Currently implements one of "Dirichlet" (:math:`D = Id`) and Von
        Neuman (:math:`D[\cdot] = \nabla \cdot n`) where :math:`n` is the
        outgoing unitary vector normal to :math:`\partial\Omega`
    f : a callable
        the function to be matched in the boundary condition. It should have
        one argument only (other are ignored).
    border_batch : jnp.array
        the mini-batch on :math:`\partial \Omega`
    u :
        a PINN
    params
        The dictionary of parameters of the model.
        Typically, it is a dictionary of
        dictionaries: `eq_params` and `nn_params``, respectively the
        differential equation parameters and the neural network parameter
    facet:
        An integer which represents the id of the facet which is currently
        considered (in the order provided wy the DataGenerator which is fixed)

    Returns
    -------
    scalar
        the MSE computed on `border_batch`
    """
    if boundary_condition_type.lower() in "dirichlet":
        mse = boundary_dirichlet_nonstatio(f, times_batch, border_batch, u, params)
    elif any(
        [
            boundary_condition_type.lower() in s
            for s in ["von neumann", "vn", "vonneumann"]
        ]
    ):
        mse = boundary_neumann_nonstatio(f, times_batch, border_batch, u, params, facet)
    return mse


def boundary_dirichlet_statio(f, border_batch, u, params):
    r"""
    This omega boundary condition enforces a solution that is equal to f on
    border batch.

    Parameters
    ----------
    f:
        the constraint function
    border_batch
        A mini_batch of points
    u
        The PINN
    params
        The dictionary of parameters of the model.
        Typically, it is a dictionary of
        dictionaries: `eq_params` and `nn_params``, respectively the
        differential equation parameters and the neural network parameter
    """
    v_u_boundary = vmap(
        lambda dx: u(
            dx,
            u_params=params["nn_params"],
            eq_params=jax.lax.stop_gradient(params["eq_params"]),
        )
        - f(dx),
        (0),
        0,
    )

    mse_u_boundary = jnp.mean((v_u_boundary(border_batch)) ** 2)
    return mse_u_boundary


def boundary_neumann_statio(f, border_batch, u, params, facet):
    r"""
    This omega boundary condition enforces a solution where :math:`\nabla u\cdot
    n` is equal to `f` on omega borders. :math:`n` is the unitary
    outgoing vector normal at border :math:`\partial\Omega`.

    Parameters
    ----------
    f
        the constraint function

    border_batch
        A mini_batch of points
    u
        The PINN
    params
        The dictionary of parameters of the model.
        Typically, it is a dictionary of
        dictionaries: `eq_params` and `nn_params``, respectively the
        differential equation parameters and the neural network parameter
    facet:
        An integer which represents the id of the facet which is currently
        considered (in the order provided wy the DataGenerator which is fixed)
    """
    # We resort to the shape of the border_batch to determine the dimension as
    # described in the border_batch function
    if jnp.squeeze(border_batch).ndim == 0:  # case 1D borders (just a scalar)
        n = jnp.array([1, -1])  # the unit vectors normal to the two borders

    else:  # case 2D borders (because 3D borders are not supported yet)
        # they are in the order: left, right, bottom, top so we give the normal
        # outgoing vectors accordingly with shape in concordance with
        # border_batch shape (batch_size, ndim, nfacets)
        n = jnp.array([[-1, 1, 0, 0], [0, 0, -1, 1]])

    v_neumann = vmap(
        lambda dx: jnp.dot(
            grad(u, 0)(
                dx, params["nn_params"], jax.lax.stop_gradient(params["eq_params"])
            ),
            n[..., facet],
        )
        - f(dx),
        0,
    )
    mse_u_boundary = jnp.mean((v_neumann(border_batch)) ** 2)
    return mse_u_boundary


def boundary_dirichlet_nonstatio(f, times_batch, omega_border_batch, u, params):
    """
    This omega boundary condition enforces a solution that is equal to f
    at times_batch x omega borders

    Parameters
    ----------
    f
        the constraint function
    times_batch
        A mini-batch of time points
    border_batch
        A mini_batch of border points
    u
        The PINN
    params
        The dictionary of parameters of the model.
        Typically, it is a dictionary of
        dictionaries: `eq_params` and `nn_params``, respectively the
        differential equation parameters and the neural network parameter
    """
    tile_omega_border_batch = jnp.tile(
        omega_border_batch, reps=(times_batch.shape[0], 1)
    )

    def rep_times(k):
        return jnp.repeat(times_batch, k, axis=0)

    v_u_boundary = vmap(
        lambda t, dx: u(
            t,
            dx,
            u_params=params["nn_params"],
            eq_params=jax.lax.stop_gradient(params["eq_params"]),
        )
        - f(dx),
        (0, 0),
        0,
    )

    mse_u_boundary = jnp.mean(
        (v_u_boundary(rep_times(omega_border_batch.shape[0]), tile_omega_border_batch))
        ** 2
    )
    return mse_u_boundary


def boundary_neumann_nonstatio(f, times_batch, omega_border_batch, u, params, facet):
    r"""
    This omega boundary condition enforces a solution where :math:`\nabla u\cdot
    n` is equal to `f` at time_batch x omega borders. :math:`n` is the unitary
    outgoing vector normal at border :math:`\partial\Omega`.

    Parameters
    ----------
    f:
        the constraint function
    times_batch
        A mini-batch of time points
    border_batch
        A mini_batch of border points
    u
        The PINN
    params
        The dictionary of parameters of the model.
        Typically, it is a dictionary of
        dictionaries: `eq_params` and `nn_params``, respectively the
        differential equation parameters and the neural network parameter
    facet:
        An integer which represents the id of the facet which is currently
        considered (in the order provided wy the DataGenerator which is fixed)
    """
    tile_omega_border_batch = jnp.tile(
        omega_border_batch, reps=(times_batch.shape[0], 1)
    )

    def rep_times(k):
        return jnp.repeat(times_batch, k, axis=0)

    # We resort to the shape of the border_batch to determine the dimension as
    # described in the border_batch function
    if jnp.squeeze(omega_border_batch).ndim == 0:  # case 1D borders (just a scalar)
        n = jnp.array([1, -1])  # the unit vectors normal to the two borders

    else:  # case 2D borders (because 3D borders are not supported yet)
        # they are in the order: left, right, bottom, top so we give the normal
        # outgoing vectors accordingly with shape in concordance with
        # border_batch shape (batch_size, ndim, nfacets)
        n = jnp.array([[-1, 1, 0, 0], [0, 0, -1, 1]])

    v_neumann = vmap(
        lambda t, dx: jnp.dot(
            grad(u, 1)(
                t, dx, params["nn_params"], jax.lax.stop_gradient(params["eq_params"])
            ),
            n[..., facet],
        )
        - f(dx),
        0,
        0,
    )
    mse_u_boundary = jnp.mean(
        (v_neumann(rep_times(omega_border_batch.shape[0]), tile_omega_border_batch))
        ** 2
    )

    return mse_u_boundary
