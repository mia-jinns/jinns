import jax.numpy as jnp
import jax

from jinns.data._utils import append_param_batch
from jinns.data._DataGeneratorParameter import DataGeneratorParameter
from jinns.data._CubicMeshPDENonStatio import CubicMeshPDENonStatio
from jinns.data._ACMPDENonStatio import ACMPDENonStatio
from jinns.data._AbstractDataGenerator import AbstractDataGenerator
from jinns.data._Batchs import PDENonStatioBatch

from jaxtyping import Array, Float, Key
from typing import Literal
import equinox as eqx


class RarDataGenerator(AbstractDataGenerator):
    """
    RarDataGenerator implement RAR sampling strategies:
    key: Key
    domain_dg: Datagenerator of domain, initial and border data
    params_dg: Datagenerator of parameters
    current_batch: the current batch for the RAR sampling strategy
    batch_size: the common batchsize
    resample_every: number of iterations before we trigger RAR
    proportion: proportion of the previous batch to be resampled using RAR
    loss: loss object
    current_params: optimization parameters at the current iteration
    """

    key: Key = eqx.field(kw_only=True)
    domain_dg: CubicMeshPDENonStatio | ACMPDENonStatio = eqx.field(kw_only=True)
    params_dg: DataGeneratorParameter = eqx.field(kw_only=True)
    resampling_strategy: Literal["RAR-D", "RAR-G", "R3"] = eqx.field(
        kw_only=True, static=True, default="RAR-D"
    )
    resampling_period: int = eqx.field(kw_only=True)
    warmup_iterations: int = eqx.field(
        kw_only=True
    )  ## train the PINN before triggering the rar

    current_residuals: Float[Array, "n 1"] | None = eqx.field(
        init=False
    )  ## residuals at iteration it
    current_sample: PDENonStatioBatch = eqx.field(
        init=False
    )  ## residuals for which we get the residual

    batch_size: int = eqx.field(init=False, static=True)
    in_warmup_phase: bool = eqx.field(init=False)
    warmup_counter: int = eqx.field(init=False)
    resampling_counter: int = eqx.field(init=False)

    ## HyperParameters
    proportion: float = eqx.field(
        kw_only=True, static=True
    )  ## balance between and residual based distribution
    k: float = eqx.field(
        kw_only=True, static=True
    )  ## Control how the distribution sharpness
    c: float = eqx.field(
        kw_only=True, static=True
    )  ## Balance with a uniform distribution

    def __post_init__(self):
        """
        After initialization the currentbatch is a uniform or quasirandom sample
        """
        self.batch_size = (
            self.domain_dg.domain_batch_size
            if isinstance(self.domain_dg, CubicMeshPDENonStatio)
            else self.domain_dg.n
        )

        ## Initializing current_residuals and current_sample with the same shape
        self.current_residuals = jnp.zeros((self.batch_size, 1))
        self.current_sample = append_param_batch(
            self.domain_dg.get_batch()[1], self.params_dg.get_batch()[1]
        )

        self.in_warmup_phase = True
        self.warmup_counter = 0
        self.resampling_counter = 0
        assert 0 <= self.proportion <= 1
        assert self.k > 0
        assert self.c >= 0

    def update_current_residuals_and_sample(self, residuals, sample):
        """
        The residuals are compute on a newly generated uniform data at each iteration.
        res_sample is generated outside
        """
        new = eqx.tree_at(
            lambda pt: (pt.current_residuals, pt.current_sample),
            self,
            (residuals, sample),
        )
        return new

    def retain_sample(self, key: Key):
        """
        Sample a proportion of points from the current_batch according to the absolute value of their residuals
        add points
        """
        retain_indexes = self._get_indexes(key)

        res_batch = PDENonStatioBatch(
            domain_batch=self.current_sample.domain_batch[retain_indexes],
            border_batch=None,
            initial_batch=None,
            param_batch_dict={
                "D": self.current_sample.param_batch_dict["D"][retain_indexes],
                "r": self.current_sample.param_batch_dict["r"][retain_indexes],
            },
        )
        return res_batch

    def re_sample(self, key):
        """
        Resample the batch using RAR strategy and return a new instance with updated state.
        """
        key, *subkeys = jax.random.split(key, 3)

        # Fresh sample in ["uniform", "sobol", "halton"] for the domain and the parameters.
        domain_dg, domain_batch = self.domain_dg.get_batch()
        params_dg, params_batch = self.params_dg.get_batch()
        batch = append_param_batch(domain_batch, params_batch)

        # Step 1: Retain points from the sample used to compute residuals
        retained_sample = self.retain_sample(subkeys[0])

        # Step 2: Choose new points to mix with RAR points
        indexes = jax.random.choice(
            subkeys[1],
            a=self.batch_size,
            shape=(self.batch_size - retained_sample.domain_batch.shape[0],),
            replace=False,
        )

        # Step 3: Concatenate RAR and new samples
        new_batch = PDENonStatioBatch(
            domain_batch=jnp.concatenate(
                [retained_sample.domain_batch, batch.domain_batch[indexes]], axis=0
            ),
            border_batch=batch.border_batch,
            initial_batch=batch.initial_batch,
            param_batch_dict={
                "D": jnp.concatenate(
                    [
                        retained_sample.param_batch_dict["D"],
                        batch.param_batch_dict["D"][indexes],
                    ]
                ),
                "r": jnp.concatenate(
                    [
                        retained_sample.param_batch_dict["r"],
                        batch.param_batch_dict["r"][indexes],
                    ]
                ),
            },
        )
        new = eqx.tree_at(
            lambda pt: (
                pt.key,
                pt.domain_dg,
                pt.params_dg,
                pt.current_sample,
                pt.resampling_counter,
            ),
            self,
            (key, domain_dg, params_dg, new_batch, 0),
        )
        return new, new_batch

    def get_batch(self):
        domain_dg, domain_batch = self.domain_dg.get_batch()
        params_dg, param_batch = self.params_dg.get_batch()
        batch = append_param_batch(domain_batch, param_batch)

        def warmup_phase(_):
            def ongoing(_):
                new_self = eqx.tree_at(
                    lambda pt: (pt.domain_dg, pt.params_dg, pt.warmup_counter),
                    self,
                    (domain_dg, params_dg, self.warmup_counter + 1),
                )
                return new_self, batch

            def switch_to_rar(_):
                jax.debug.print("🚀 RAR triggered !!")
                new_self = eqx.tree_at(
                    lambda pt: (
                        pt.domain_dg,
                        pt.params_dg,
                        pt.in_warmup_phase,
                        pt.current_sample,
                    ),
                    self,
                    (domain_dg, params_dg, False, batch),
                )
                return new_self, batch

            return jax.lax.cond(
                self.warmup_counter < self.warmup_iterations,
                ongoing,
                switch_to_rar,
                operand=None,
            )

        def rar_phase(_):
            def no_resample(_):
                new_self = eqx.tree_at(
                    lambda pt: (pt.domain_dg, pt.params_dg, pt.resampling_counter),
                    self,
                    (domain_dg, params_dg, self.resampling_counter + 1),
                )
                return new_self, self.current_sample

            def do_resample(_):
                return self.re_sample(self.key)

            return jax.lax.cond(
                self.resampling_counter < self.resampling_period,
                no_resample,
                do_resample,
                operand=None,
            )

        return jax.lax.cond(self.in_warmup_phase, warmup_phase, rar_phase, operand=None)

    def _get_indexes(self, key):
        """
        This method returns the indexes of the selected points to add based on the following strategies:
        1. RAR-G: Greedy RAR
        2. RAR-D: Distributional RAR
        3. R3: Retain-Resample-Release
        """
        n = self.batch_size
        res_abs = (
            jnp.absolute(self.current_residuals)
            if self.current_residuals is not None
            else None
        )  ## current residuals
        if res_abs is not None:
            ## Check shapes: res_abs should be of shape (n, 1)
            assert res_abs.shape == (n, 1)
            if self.resampling_strategy == "RAR-G":
                ## return a proportion of the maximal points
                return jnp.argsort(res_abs.flatten(), descending=True)[
                    : int(n * self.proportion)
                ]
            elif self.resampling_strategy == "RAR-D":
                res_normalized = res_abs / (
                    jnp.max(res_abs) + 1e-10
                )  # avoid division by zero
                prop_weights = res_normalized**self.k + self.c
                weights = prop_weights / jnp.sum(prop_weights)
                return jax.random.choice(
                    key,
                    a=jnp.arange(res_abs.shape[0]),
                    shape=(int(n * self.proportion),),
                    replace=False,
                    p=weights.flatten(),
                )
            ## R3 Is not jax compatible for now
            # elif self.resampling_strategy=="R3":
            #     ## To avoid dynamic shape we do the following:
            #     threshold = jnp.mean(res_abs)
            #     mask = res_abs > threshold
            #     indices = jnp.arange(res_abs.shape[0])
            #     return indices[mask.flatten()]
            else:
                raise ValueError(
                    f"The resampling strategy {self.resampling_strategy} is not implemented."
                )
        else:
            raise ValueError("Cannot perform RAR sampling - residulas not computed yet")
