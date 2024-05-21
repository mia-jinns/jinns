import jax
import equinox as eqx
import jax.numpy as jnp
from jinns.utils._pinn import PINN


def almost_zero_init(weight: jax.Array, key: jax.random.PRNGKey) -> jax.Array:
    out, in_ = weight.shape
    stddev = 1e-2
    return stddev * jax.random.normal(key, shape=(out, in_))


class _SpectralPINN(eqx.Module):
    """
    A specific PINN whose acrhitecture is similar to spectral method for simulation of a spatial field
    (Chilès and Delfiner, 2012) - a single layer with cos() activation function and sum for last layer
    """

    layers_pinn: list
    nbands: int

    def __init__(self, key, list_layers_pinn, nbands):
        """
        Parameters
        ----------
        key
            A jax random key
        list_layers_pinn
            A list as eqx_list in jinns' PINN utility for the main PINN
        nbands
            Number of spectral bands (i.e., neurones in the single layer of the PINN)
        """
        self.nbands = nbands
        self.layers_pinn = []
        for l in list_layers_pinn:
            if len(l) == 1:
                self.layers_pinn.append(l[0])
            else:
                key, subkey = jax.random.split(key, 2)
                self.layers_pinn.append(l[0](*l[1:], key=subkey))

    def __call__(self, x):
        # forward pass through the actual PINN
        for layer in self.layers_pinn:
            x = layer(x)

        return jnp.sqrt(2 / self.nbands) * jnp.sum(x)


class spectralPINN(PINN):
    """
    MUST inherit from PINN to pass all the checks

    HOWEVER we dot not bother with reimplementing anything
    """

    def __init__(self, key, list_layers_pinn, nbands):
        super().__init__({}, jnp.s_[...], "statio_PDE", None, None, None)
        key, subkey = jax.random.split(key, 2)
        _pinn = _SpectralPINN(subkey, list_layers_pinn, nbands)

        self.params, self.static = eqx.partition(_pinn, eqx.is_inexact_array)

    def init_params(self):
        return self.params

    def __call__(self, x, params):
        try:
            model = eqx.combine(params["nn_params"], self.static)
        except (KeyError, TypeError) as e:  # give more flexibility
            model = eqx.combine(params, self.static)
        # model = eqx.tree_at(lambda m:
        #        m.layers_pinn[0].bias,
        #        model,
        #        model.layers_pinn[0].bias % (2 *
        #            jnp.pi)
        #        )
        res = model(x)
        if not res.shape:
            return jnp.expand_dims(res, axis=-1)
        return res


def create_spectralPINN(key, list_layers_pinn, nbands):
    """ """
    u = spectralPINN(key, list_layers_pinn, nbands)
    return u
