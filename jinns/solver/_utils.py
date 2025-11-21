import jax
import jax.numpy as jnp
import equinox as eqx

from jinns.data._DataGeneratorODE import DataGeneratorODE
from jinns.data._CubicMeshPDEStatio import CubicMeshPDEStatio
from jinns.data._CubicMeshPDENonStatio import CubicMeshPDENonStatio
from jinns.data._DataGeneratorParameter import DataGeneratorParameter


def _check_batch_size(other_data, main_data, attr_name):
    if isinstance(main_data, DataGeneratorODE):
        if main_data.temporal_batch_size is not None:
            if getattr(other_data, attr_name) != main_data.temporal_batch_size:
                raise ValueError(
                    f"{other_data.__class__}.{attr_name} must be equal"
                    f" to {main_data.__class__}.temporal_batch_size for correct"
                    " vectorization"
                )
        else:
            if main_data.nt is not None:
                if getattr(other_data, attr_name) != main_data.nt:
                    raise ValueError(
                        f"{other_data.__class__}.{attr_name} must be equal"
                        f" to {main_data.__class__}.nt for correct"
                        " vectorization"
                    )
    if isinstance(main_data, CubicMeshPDEStatio) and not isinstance(
        main_data, CubicMeshPDENonStatio
    ):
        if main_data.omega_batch_size is not None:
            if getattr(other_data, attr_name) != main_data.omega_batch_size:
                raise ValueError(
                    f"{other_data.__class__}.{attr_name} must be equal"
                    f" to {main_data.__class__}.omega_batch_size for correct"
                    " vectorization"
                )
        else:
            if main_data.n is not None:
                if getattr(other_data, attr_name) != main_data.n:
                    raise ValueError(
                        f"{other_data.__class__}.{attr_name} must be equal"
                        f" to {main_data.__class__}.n for correct"
                        " vectorization"
                    )
        if main_data.omega_border_batch_size is not None:
            if getattr(other_data, attr_name) != main_data.omega_border_batch_size:
                raise ValueError(
                    f"{other_data.__class__}.{attr_name} must be equal"
                    f" to {main_data.__class__}.omega_border_batch_size for correct"
                    " vectorization"
                )
        else:
            if main_data.nb is not None:
                if getattr(other_data, attr_name) != main_data.nb:
                    raise ValueError(
                        f"{other_data.__class__}.{attr_name} must be equal"
                        f" to {main_data.__class__}.nb for correct"
                        " vectorization"
                    )
    if isinstance(main_data, CubicMeshPDENonStatio):
        if main_data.domain_batch_size is not None:
            if getattr(other_data, attr_name) != main_data.domain_batch_size:
                raise ValueError(
                    f"{other_data.__class__}.{attr_name} must be equal"
                    f" to {main_data.__class__}.domain_batch_size for correct"
                    " vectorization"
                )
        else:
            if main_data.n is not None:
                if getattr(other_data, attr_name) != main_data.n:
                    raise ValueError(
                        f"{other_data.__class__}.{attr_name} must be equal"
                        f" to {main_data.__class__}.n for correct"
                        " vectorization"
                    )
        if main_data.border_batch_size is not None:
            if getattr(other_data, attr_name) != main_data.border_batch_size:
                raise ValueError(
                    f"{other_data.__class__}.{attr_name} must be equal"
                    f" to {main_data.__class__}.border_batch_size for correct"
                    " vectorization"
                )
        else:
            if main_data.nb is not None:
                if main_data.dim > 1 and getattr(other_data, attr_name) != (
                    main_data.nb // 2**main_data.dim
                ):
                    raise ValueError(
                        f"{other_data.__class__}.{attr_name} must be equal"
                        f" to ({main_data.__class__}.nb // 2**{main_data.__class__}.dim)"
                        " for correct vectorization"
                    )
        if main_data.initial_batch_size is not None:
            if getattr(other_data, attr_name) != main_data.initial_batch_size:
                raise ValueError(
                    f"{other_data.__class__}.{attr_name} must be equal"
                    f" to {main_data.__class__}.initial_batch_size for correct"
                    " vectorization"
                )
        else:
            if main_data.ni is not None:
                if getattr(other_data, attr_name) != main_data.ni:
                    raise ValueError(
                        f"{other_data.__class__}.{attr_name} must be equal"
                        f" to {main_data.__class__}.ni for correct"
                        " vectorization"
                    )
    if isinstance(main_data, DataGeneratorParameter):
        if main_data.param_batch_size is not None:
            if getattr(other_data, attr_name) != main_data.param_batch_size:
                raise ValueError(
                    f"{other_data.__class__}.{attr_name} must be equal"
                    f" to {main_data.__class__}.param_batch_size for correct"
                    " vectorization"
                )
        else:
            if main_data.n is not None:
                if getattr(other_data, attr_name) != main_data.n:
                    raise ValueError(
                        f"{other_data.__class__}.{attr_name} must be equal"
                        f" to {main_data.__class__}.n for correct"
                        " vectorization"
                    )


def _init_stored_weights_terms(loss, n_iter):
    return eqx.tree_at(
        lambda pt: jax.tree.leaves(
            pt, is_leaf=lambda x: x is not None and eqx.is_inexact_array(x)
        ),
        loss.loss_weights,
        tuple(
            jnp.zeros((n_iter))
            for n in range(
                len(
                    jax.tree.leaves(
                        loss.loss_weights,
                        is_leaf=lambda x: x is not None and eqx.is_inexact_array(x),
                    )
                )
            )
        ),
    )


def _init_stored_params(tracked_params, params, n_iter):
    return jax.tree_util.tree_map(
        lambda tracked_param, param: (
            jnp.zeros((n_iter,) + jnp.asarray(param).shape)
            if tracked_param is not None
            else None
        ),
        tracked_params,
        params,
        is_leaf=lambda x: x is None,  # None values in tracked_params will not
        # be traversed. Thus the user can provide something like `tracked_params = jinns.parameters.Params(
        # nn_params=None, eq_params={"nu": True})` while init_params.nn_params
        # being a complex data structure
    )
