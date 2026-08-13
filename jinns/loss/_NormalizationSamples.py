"""
Implement the equinox Module for the normalization samples
"""

from typing import Literal, get_args, cast
from dataclasses import InitVar
import jax
from jaxtyping import Float, Array, PRNGKeyArray
import jax.numpy as jnp
import equinox as eqx

from jinns.data._Batchs import PDENonStatioBatch, PDEStatioBatch
from jinns.data._CubicMeshPDEStatio import CubicMeshPDEStatio
from jinns.data._RARParameters import RARParameters
from jinns.nn._hyperpinn import HyperPINN
from jinns.nn._spinn import SPINN
from jinns.parameters._params import Params

# if TYPE_CHECKING:
from jinns.loss._LossPDE import _LossPDEAbstract

AvailableNormSamplesAndWeightsUpdateMethods = Literal["RAR", "resample"]


class NormalizationSamples(eqx.Module):
    """
    Module for the normalization samples used in norm_loss computations

    Parameters
    ----------
    samples : Float[Array, " nb_samples dimension"]
        Monte-Carlo sample points for computing the
        normalization constant.
    weights : Float[Array, " nb_samples"] | float | int
        The importance sampling weights for Monte-Carlo integration of the
        normalization constant. Must be provided if `samples` is provided.
        `weights` should be broadcastble to `samples`.
        Alternatively, the user can pass a float or an integer that will be
        made broadcastable to `samples`.
        These corresponds to the weights $w_k = \frac{1}{q(x_k)}$ where
        $q(\cdot)$ is the proposal p.d.f. and $x_k$ are the Monte-Carlo samples.
        **If using SPINN, `weights` must be a scalar**.
    min_pts : tuple[Float, ...]
        A tuple of minimum values of the domain along each dimension. For a sampling
        in `n` dimension, this represents $(x_{1, min}, x_{2,min}, ...,
        x_{n, min})$
    max_pts : tuple[Float, ...]
        A tuple of maximum values of the domain along each dimension. For a sampling
        in `n` dimension, this represents $(x_{1, max}, x_{2,max}, ...,
        x_{n,max})$
    update_samples_and_weights_method : AvailableNormSamplesAndWeightsUpdateMethods | None
        XXX
    rar_parameters: RARParameters | None
        XXX
    u_type: InitVar[str]
        XXX
    max_time_slices : int, default=100
        The maximum number of time points in the Cartesian product with the
        omega points to create the set of collocation points upon which the
        normalization constant is computed.
        Only used with LossPDENonStatio
    max_samples_omega : int, default=1000
        The maximum number of omega points in the Cartesian product with the
        time points to create the set of collocation points upon which the
        normalization constant is computed.
        Only used with LossPDENonStatio
    """

    samples: Float[Array, " nb_samples dimension"] = eqx.field(kw_only=True)
    weights: Float[Array, " nb_samples"] = eqx.field(kw_only=True)
    min_pts: tuple[float, ...] = eqx.field(kw_only=True)
    max_pts: tuple[float, ...] = eqx.field(kw_only=True)
    update_samples_and_weights_method: (
        AvailableNormSamplesAndWeightsUpdateMethods | None
    ) = eqx.field(kw_only=True, default=None, static=True)
    rar_parameters: RARParameters | None = eqx.field(kw_only=True, default=None)
    max_samples_omega: int = eqx.field(static=True)
    max_time_slices: int = eqx.field(static=True)
    dim: int = eqx.field(static=True, init=False)

    u_type: InitVar[str]

    def __init__(
        self,
        samples: Float[Array, " nb_samples dimension"],
        weights: Float[Array, " nb_samples"] | float | int,
        min_pts: tuple[float, ...],
        max_pts: tuple[float, ...],
        update_samples_and_weights_method=None,
        rar_parameters=None,
        u_type="PINN",
        max_time_slices: int = 100,
        max_samples_omega: int = 1000,
    ):
        if (
            update_samples_and_weights_method is not None
            and update_samples_and_weights_method
            not in get_args(AvailableNormSamplesAndWeightsUpdateMethods)
        ):
            raise ValueError(
                f"{update_samples_and_weights_method=} is not a valid method"
            )
        if update_samples_and_weights_method == "RAR" and rar_parameters is None:
            raise ValueError("rar_parameters attribute must be set!")
        self.update_samples_and_weights_method = update_samples_and_weights_method
        self.samples = samples
        self.min_pts = min_pts
        self.max_pts = max_pts
        self.dim = len(min_pts)
        if u_type == "PINN":
            if isinstance(weights, (int, float)):
                weights = jnp.array(weights) * jnp.ones((samples.shape[0],))
            elif isinstance(weights, Array) and (
                weights.shape == (1,) or weights.ndim == 0
            ):
                # if user provided weights=jnp.array([0.1]) or
                # jnp.array(0.1)...
                weights = weights * jnp.ones((samples.shape[0],))
            if not (weights.shape[0] == samples.shape[0]):
                raise ValueError(
                    "weights and samples must have the same leading dimension"
                )
            self.weights = weights
        elif u_type == "SPINN":
            if not (
                isinstance(weights, (int, float))
                or (isinstance(weights, Array) and weights.squeeze().shape == ())
            ):
                raise ValueError("weights must be scalar when using SPINN")
            self.weights = jnp.array(weights)

        # with the variables below we avoid memory overflow since a cartesian
        # product is taken
        self.max_time_slices = max_time_slices
        self.max_samples_omega = max_samples_omega
        if self.samples.shape[0] > self.max_samples_omega:
            raise ValueError(
                "Number of norm_samples is bigger than max_samples_omega"
                " attribute of LossPDENonStatio. Increase max_samples_omega or reduce the"
                " number of norm_samples. This check has been set to avoid memory explosion"
                " in normalization loss computation."
            )

    # def update_samples_and_weights(
    #     self,
    #     iteration_nb: int,
    #     data: CubicMeshPDEStatio | CubicMeshPDENonStatio,
    #     params: Params[Array],
    #     key: PRNGKeyArray,
    # ) -> Self:
    #     """
    #     Update the weights and samples according to a predefined scheme

    #     """
    #     old_samples = self.samples
    #     old_weights = self.weights

    #     if self.update_samples_and_weights_method == "resample":
    #         # Simple resampling in the domain
    #         if len(data.min_pts) == 1: # 1D case
    #             key, subkey = jax.random.split(key)
    #             new_samples = jax.random.uniform(
    #                 subkey,
    #                 shape=(old_samples.shape[0], 1),
    #                 minval=data.min_pts[0],
    #                 maxval=data.max_pts[0],
    #             )
    #         elif len(data.min_pts) == 2: # 2D case
    #             key, subkey1, subkey2 = jax.random.split(key, 3)
    #             new_samples = jnp.stack(
    #                 [
    #                     jax.random.uniform(subkey1, shape=(old_samples.shape[0],),
    #                         minval=data.min_pts[0], maxval=data.max_pts[0]),
    #                     jax.random.uniform(subkey2, shape=(old_samples.shape[0],),
    #                         minval=data.min_pts[1], maxval=data.max_pts[1]),
    #                 ],
    #                 axis=1,
    #             )
    #         new_weights = old_weights
    #     elif self.update_samples_and_weights_method == "RAR":
    #         pass
    #     else:
    #         raise ValueError("update_samples_and_weights_method for "
    #                          "samples and weights not implemented")

    #     # At iteration 0 we do not do any update
    #     (new_samples, new_weights) = jax.lax.cond(
    #         iteration_nb == 0,
    #         lambda _: (old_samples, old_weights),
    #         lambda _: (new_samples, new_weights),
    #         None,
    #     )
    #     return eqx.tree_at(
    #         lambda pt: (pt.samples, pt.weights), self, (new_samples, new_weights)
    #     )


def _rar_step_true_norm_samples(
    loss: _LossPDEAbstract,
    params: Params[Array],
    batch: PDEStatioBatch | PDENonStatioBatch,
    key: PRNGKeyArray,
    i: int,
) -> _LossPDEAbstract:
    assert loss.norm_samples is not None
    assert loss.norm_samples.rar_parameters is not None

    if isinstance(loss.u, HyperPINN) or isinstance(loss.u, SPINN):
        raise NotImplementedError("RAR not implemented for hyperPINN and SPINN")

    # the signature we get from tree.reduce is Array | int
    # we are sure this is Array so we use the cast to get rid of int
    res = cast(
        Array,
        jax.tree.reduce(
            jnp.add,
            loss.values_and_grad_per_sample(
                params, loss._get_normalization_loss_batch(batch)
            )[0].dyn_loss,
            0,
        ),
    )
    res = jnp.atleast_2d(res)
    res = jnp.sum(res**2, axis=-1)
    res_abs = jnp.abs(res)
    norm_samples_size = res_abs.shape[0]

    # Here we create the novelty that be incorporated to the norm samples
    novelty_sample_size = round(
        norm_samples_size * loss.norm_samples.rar_parameters.novelty_proportion
    )
    key, subkey = jax.random.split(key, loss.norm_samples.dim + 1)
    new_samples = CubicMeshPDEStatio.sample_in_omega_domain(
        keys=[subkey] if loss.norm_samples.dim == 1 else subkey,
        sample_size=novelty_sample_size,
        dim=loss.norm_samples.dim,
        method="uniform",
        min_pts=loss.norm_samples.min_pts,
        max_pts=loss.norm_samples.max_pts,
    )

    # RAR-G
    ## Select the m points with higher dynamic loss, they will be conserved
    if loss.norm_samples.rar_parameters.method == "G":
        keep_idx = jnp.argsort(res_abs, descending=True)[
            : round(
                norm_samples_size
                * (1 - loss.norm_samples.rar_parameters.novelty_proportion)
            )
        ]
        weights = loss.norm_samples.weights
    # RAR-D
    elif loss.norm_samples.rar_parameters.method == "D":
        assert loss.norm_samples.rar_parameters.k is not None
        assert loss.norm_samples.rar_parameters.c is not None

        res_normalized = res_abs / (jnp.max(res_abs) + 1e-6)
        prop_weights = (
            res_normalized**loss.norm_samples.rar_parameters.k
            + loss.norm_samples.rar_parameters.c
        )
        weights = prop_weights / jnp.sum(prop_weights)
        keep_idx = jax.random.choice(
            key,
            a=jnp.arange(res_abs.shape[0]),
            shape=(
                round(
                    norm_samples_size
                    * (1 - loss.norm_samples.rar_parameters.novelty_proportion)
                ),
            ),
            replace=False,
            p=weights.flatten(),
        )
    else:
        raise ValueError("Unknown RAR method")

    ## Introduce novelty samples with the novelty_sample_size samples that have been sampled
    arr = jnp.concatenate([loss.norm_samples.samples[keep_idx], new_samples], axis=0)
    loss = eqx.tree_at(
        lambda pt: (pt.norm_samples.samples, pt.norm_samples.weights),
        loss,
        (arr, weights),
    )
    return loss

    # new_omega = data.omega.at[
    #         data.curr_omega_idx : data.curr_omega_idx * data.omega_batch_size
    #     ].set(  # type: ignore
    #         batch.domain_batch
    #     )
    # else:
    #     new_omega = batch.domain_batch
    # data = eqx.tree_at(lambda m: m.omega, data, new_omega)
    # if isinstance(data, CubicMeshPDEStatio) and not isinstance(
    #     data, CubicMeshPDENonStatio
    # ):
    #     key, *subkeys = jax.random.split(key, data.dim + 1)
    #     new_samples = data.sample_in_omega_domain(subkeys, novelty_sample_size)
    # elif isinstance(data, CubicMeshPDENonStatio):
    #     key, subkey = jax.random.split(key)
    #     new_samples_times = data.sample_in_time_domain(subkey, novelty_sample_size)
    #     if data.dim == 1:
    #         key, subkeys = jax.random.split(key, 2)
    #     else:
    #         key, *subkeys = jax.random.split(key, data.dim + 1)
    #     new_samples_omega = data.sample_in_omega_domain(subkeys, novelty_sample_size)
    #     new_samples = jnp.concatenate([new_samples_times, new_samples_omega], axis=1)
    # else:
    #     raise ValueError("Wrong DataGenerator type")
    # if param_data is not None:
    #     key, subkey = jax.random.split(key)
    #     _, _param_n_samples = param_data.generate_data(subkey, batch_size)
    #     new_param_samples = DGParams(_param_n_samples, "DGParams")

    ## Introduce novelty samples with the novelty_sample_size samples that have been sampled
    ## Begin (Update the batch with the novelty)
    ### for each param with a jax.tree.map
    # if param_data is not None:
    #     param_batch = jax.tree.map(
    #         lambda b, new_b: jnp.concatenate([b[keep_idx], new_b], axis=0),
    #         batch.param_batch_dict,
    #         new_param_samples,
    #     )
    # else:
    #     param_batch = None
    # if isinstance(batch, ODEBatch) and isinstance(data, DataGeneratorODE):
    #     arr = jnp.concatenate([batch.temporal_batch[keep_idx], new_samples], axis=0)
    #     batch = eqx.tree_at(lambda pt: pt.temporal_batch, batch, arr)

    #     # Also add the new points ie update the fixed datasets of the DGs
    #     if data.temporal_batch_size is not None:
    #         new_times = data.times.at[
    #             data.curr_time_idx : data.curr_time_idx * data.temporal_batch_size
    #         ].set(  # type: ignore
    #             batch.temporal_batch
    #         )
    #     else:
    #         new_times = batch.temporal_batch
    #     data = eqx.tree_at(lambda m: m.times, data, new_times)
    # elif isinstance(batch, PDEStatioBatch) or isinstance(batch, PDENonStatioBatch):
    #     arr = jnp.concatenate([batch.domain_batch[keep_idx], new_samples], axis=0)
    #     batch = eqx.tree_at(lambda pt: pt.domain_batch, batch, arr)
    #     # Also add the new points ie update the fixed datasets of the DGs
    #     if isinstance(data, CubicMeshPDEStatio) and not isinstance(
    #         data, CubicMeshPDENonStatio
    #     ):
    #         if data.omega_batch_size is not None:
    #             new_omega = data.omega.at[
    #                 data.curr_omega_idx : data.curr_omega_idx * data.omega_batch_size
    #             ].set(  # type: ignore
    #                 batch.domain_batch
    #             )
    #         else:
    #             new_omega = batch.domain_batch
    #         data = eqx.tree_at(lambda m: m.omega, data, new_omega)
    #     elif isinstance(data, CubicMeshPDENonStatio):
    #         if data.domain_batch_size is not None:
    #             new_domain = data.domain.at[
    #                 data.curr_domain_idx : data.curr_domain_idx * data.domain_batch_size
    #             ].set(  # type: ignore
    #                 batch.domain_batch
    #             )
    #         else:
    #             new_domain = batch.domain_batch
    #         data = eqx.tree_at(lambda m: m.domain, data, new_domain)
    # else:
    #     raise ValueError
    # ## Here is the batch we will return
    # batch = append_param_batch(batch, param_batch)
    # ## End (Update the batch with the novelty)

    # # update RAR parameters for all cases
    # data = eqx.tree_at(lambda m: m.rar_parameters._rar_iter_from_last_sampling, data, 0)

    # # NOTE must return data to be correctly updated because we cannot
    # # have side effects in this function that will be jitted
    # return data, param_data, batch


def _rar_step_false_norm_samples(
    loss: _LossPDEAbstract,
    params: Params[Array],
    batch: PDEStatioBatch | PDENonStatioBatch,
    key: PRNGKeyArray,
    i: int,
) -> _LossPDEAbstract:
    assert loss.norm_samples is not None
    assert loss.norm_samples.rar_parameters is not None  # for type checker

    # Add 1 only if we are after the burn in period
    increment = jax.lax.cond(
        i <= loss.norm_samples.rar_parameters.start_iter,
        lambda: 0,
        lambda: 1,
    )

    new_rar_iter_from_last_sampling = (
        loss.norm_samples.rar_parameters._rar_iter_from_last_sampling + increment
    )
    loss = eqx.tree_at(
        lambda m: m.norm_samples.rar_parameters._rar_iter_from_last_sampling,
        loss,
        new_rar_iter_from_last_sampling,
    )
    return loss
