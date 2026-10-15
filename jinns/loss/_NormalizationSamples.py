"""
Implement the equinox Module for the normalization samples
"""

from __future__ import (
    annotations,
)  # https://docs.python.org/3/library/typing.html#constant
from typing import Literal, get_args, Any
from dataclasses import InitVar
import jax
from jaxtyping import Float, Array, PRNGKeyArray
import jax.numpy as jnp
import equinox as eqx

from jinns.data._CubicMeshPDEStatio import CubicMeshPDEStatio

AvailableNormSamplesAndWeightsUpdateMethods = Literal["resample"]


class NormalizationSamples(eqx.Module):
    r"""
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
        The strategy to update the normalization samples. Currently the only implemented strategy
        is resampling the colocation points on the same domain. Default is None.
    u_type: InitVar[Literal["PINN", "SPINN"]], default="PINN"
        The type of PINN architecture that is used for the PINN.
    time_slices : int | Float[Array, " n_time_slices 1"] | None, default=None
        This argument needs to be specified for computing the normalization loss for LossPDENonStatio.
        If an integer is passed, then the time collocation points used for integration are chosen
        by taking the `time_slices` first time collocation points from the domain batch
        (they can then be different at each gradient step).
        If a 1D array is passed, then `time_slices` is the array of time collocation points
        used for the integration (and remain fixed over the whole optimization process).
        Recall that the normalization batch is computed as a cartesian product between
        a set of time collocation points (chosen depending on `time_slices`) and the sample of space
        collocation points `NormalizationSamples.samples`. Therefore, the user should carefully
        watch the memory complexity of the problem.
        This argument should remain None for stationary losses.
    max_samples_omega : int, default=1000
        The maximum number of omega points in the Cartesian product with the
        time points to create the set of collocation points upon which the
        normalization constant is computed.
        Only used with LossPDENonStatio
    method_kwargs : dict[Any, Any] | None, default=None
        The hyperparameters needed for the sample and weight update passed as a dictionary.
        For `update_samples_and_weights_method="resample"`, the hyperparameter that must be passed
        is `resample_every`, the value matching this key is a positive integer.
    """

    samples: Float[Array, " nb_samples dimension"] = eqx.field(kw_only=True)
    weights: Float[Array, " nb_samples"] = eqx.field(kw_only=True)
    min_pts: tuple[float, ...] = eqx.field(kw_only=True)
    max_pts: tuple[float, ...] = eqx.field(kw_only=True)
    update_samples_and_weights_method: (
        AvailableNormSamplesAndWeightsUpdateMethods | None
    ) = eqx.field(kw_only=True, default=None, static=True)

    time_slices: int | Float[Array, " n_time_slices 1"] | None = eqx.field(
        static=True, kw_only=True, default=None
    )
    max_samples_omega: int = eqx.field(static=True, kw_only=True)

    dim: int = eqx.field(static=True, init=False, kw_only=True)
    method_kwargs: dict[Any, Any] | None = eqx.field(static=True, kw_only=True)

    u_type: InitVar[Literal["PINN", "SPINN"]]

    def __init__(
        self,
        samples: Float[Array, " nb_samples dimension"],
        weights: Float[Array, " nb_samples"] | float | int,
        min_pts: tuple[float, ...],
        max_pts: tuple[float, ...],
        update_samples_and_weights_method=None,
        u_type="PINN",
        time_slices: int | Float[Array, " n_time_slices 1"] | None = None,
        max_samples_omega: int = 1000,
        method_kwargs: dict[Any, Any] | None = None,
    ):
        if (
            update_samples_and_weights_method is not None
            and update_samples_and_weights_method
            not in get_args(AvailableNormSamplesAndWeightsUpdateMethods)
        ):
            raise ValueError(
                f"{update_samples_and_weights_method=} is not a valid method"
            )
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

        self.max_samples_omega = max_samples_omega

        if time_slices is not None:
            if isinstance(time_slices, int):
                self.time_slices = time_slices
            else:
                # this is an array
                if time_slices.ndim == 1:
                    self.time_slices = time_slices[:, None]
                else:
                    # this is a Float[Array, "n_time_slices 1"]
                    self.time_slices = time_slices
            # with the variables below we avoid memory overflow since a cartesian
            # product is taken
            if self.samples.shape[0] > self.max_samples_omega:
                raise ValueError(
                    "Number of norm_samples is bigger than max_samples_omega"
                    " attribute of LossPDENonStatio. Increase max_samples_omega or reduce the"
                    " number of norm_samples. This check has been set to avoid memory explosion"
                    " in normalization loss computation."
                )
        else:
            # assign None
            self.time_slices = time_slices

        self.method_kwargs = method_kwargs

    def update_samples_and_weights(
        self,
        iteration_nb: int,
        key: PRNGKeyArray,
    ) -> NormalizationSamples:
        """
        Update the weights and samples according to a predefined scheme

        """

        if self.update_samples_and_weights_method == "resample":
            assert self.method_kwargs is not None
            # At iteration 0 we do not do any update
            (new_samples, new_weights) = jax.lax.cond(
                iteration_nb % self.method_kwargs["resample_every"] == 0,
                lambda _: self.resample(key),
                lambda _: (self.samples, self.weights),
                key,
            )
            return eqx.tree_at(
                lambda pt: (pt.samples, pt.weights), self, (new_samples, new_weights)
            )
        else:
            return self

    def resample(self, key: PRNGKeyArray):
        old_samples = self.samples
        old_weights = self.weights
        if self.dim == 1:
            key, subkey = jax.random.split(key, 2)
            subkey = [subkey]
        else:
            key, *subkey = jax.random.split(key, self.dim + 1)
        new_samples = CubicMeshPDEStatio.sample_in_omega_domain(
            keys=subkey,
            sample_size=old_samples.shape[0],
            dim=self.dim,
            method="uniform",
            min_pts=self.min_pts,
            max_pts=self.max_pts,
        )
        return new_samples, old_weights
