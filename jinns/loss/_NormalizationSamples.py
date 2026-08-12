"""
Implement the equinox Module for the normalization samples
"""

from typing import Literal, get_args
from dataclasses import InitVar
from jaxtyping import Float, Array
import jax.numpy as jnp
import equinox as eqx

from jinns.data._RARParameters import RARParameters

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
