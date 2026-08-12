""" """

from typing import Literal
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array


class RARParameters(eqx.Module):
    """
    Dataclass to specify the Residual Adaptative Resampling procedure

    RAR methods as inspired from https://arxiv.org/pdf/2207.10289
    However the critical difference is that the dataset size is fixed. So new points replace others


    Parameters
    ----------
    update_every : int
        the number of gradient steps taken between
        each update of collocation points in the RAR algo.
    novelty_proportion : float
        the proportion of the batchsize which is replaced
        by new samples at each RAR step
    method : Literal['G', 'D']
        - either "G" for RAR-G, ie, new points replace the batch points with the lowest dynamic loss
        - either "D" for RAR-D, ie, new points replace the batch points that have not been selected when
        `resampling batch_size * (1 - novelty_proportion)` points among the batch points with weigths given
        by the formula (2) in the article below. In this case, RARParameters must specify the keys 'k' and 'c'
        with float values as defined in the formula.
    start_iter : int, default=0
        the iteration at which we start the RAR sampling scheme (we first have a "burn-in" period).
    k : Array | None
        the float value of k, only for RAR-D. When no prior information the article recommends k=2.0
    c : Array | None
        the float value of c, only for RAR-D. When no prior information the article recommends c=0.0
    """

    update_every: int = eqx.field(static=True, kw_only=True)
    novelty_proportion: float = eqx.field(static=True, kw_only=True)
    method: Literal["G", "D"] = eqx.field(static=True, kw_only=True)
    start_iter: int = eqx.field(default=0, static=True, kw_only=True)
    k: Array | None = eqx.field(default=None, static=True, kw_only=True)
    c: Array | None = eqx.field(default=None, static=True, kw_only=True)

    _rar_iter_from_last_sampling: int = eqx.field(init=False)

    def __post_init__(self):
        if self.method == "D" and (self.k is None or self.c is None):
            raise ValueError("k and c must be specified for RAR-D")
        if self.novelty_proportion < 0 or self.novelty_proportion > 1:
            raise ValueError("novelty_proportion must be in [0; 1]")
        if self.k is not None and self.c is not None:
            self.k = jnp.array(self.k)
            self.c = jnp.array(self.c)

        self._rar_iter_from_last_sampling = 0
