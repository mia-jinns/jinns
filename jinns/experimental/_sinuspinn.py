import jax
import equinox as eqx
import jax.numpy as jnp
from jinns.utils._pinn import PINN


def almost_zero_init(weight: jax.Array, key: jax.random.PRNGKey) -> jax.Array:
    out, in_ = weight.shape
    stddev = 1e-2
    return stddev * jax.random.normal(key, shape=(out, in_))


class _SinusPINN(eqx.Module):
    """
    A specific PINN whose layers are x_sin2x functions whose frequencies are
    determined by an other network
    """

    layers_pinn: list
    layers_aux_nn: list

    def __init__(self, key, list_layers_pinn, list_layers_aux_nn):
        """
        Parameters
        ----------
        key
            A jax random key
        list_layers_pinn
            A list as eqx_list in jinns' PINN utility for the main PINN
        list_layers_aux_nn
            A list as eqx_list in jinns' PINN utility for the network which outputs
            the PINN's activation frequencies
        """
        self.layers_pinn = []
        for l in list_layers_pinn:
            if len(l) == 1:
                self.layers_pinn.append(l[0])
            else:
                key, subkey = jax.random.split(key, 2)
                self.layers_pinn.append(l[0](*l[1:], key=subkey))
        self.layers_aux_nn = []
        for idx, l in enumerate(list_layers_aux_nn):
            if len(l) == 1:
                self.layers_aux_nn.append(l[0])
            else:
                key, subkey = jax.random.split(key, 2)
                linear_layer = l[0](*l[1:], key=subkey)
                key, subkey = jax.random.split(key, 2)
                linear_layer = eqx.tree_at(
                    lambda l: l.weight,
                    linear_layer,
                    almost_zero_init(linear_layer.weight, subkey),
                )
                if (idx == len(list_layers_aux_nn) - 1) or (
                    idx == len(list_layers_aux_nn) - 2
                ):
                    # for the last layer: almost 0 weights and 0.5 bias
                    linear_layer = eqx.tree_at(
                        lambda l: l.bias,
                        linear_layer,
                        0.5 * jnp.ones(linear_layer.bias.shape),
                    )
                else:
                    # for the other previous layers:
                    # almost 0 weight and 0 bias
                    linear_layer = eqx.tree_at(
                        lambda l: l.bias,
                        linear_layer,
                        jnp.zeros(linear_layer.bias.shape),
                    )
                self.layers_aux_nn.append(linear_layer)

                ## init to zero the frequency network except last biases
                # key, subkey = jax.random.split(key, 2)
                # _pinn = init_linear_weight(_pinn, almost_zero_init, subkey)
                # key, subkey = jax.random.split(key, 2)
                # _pinn = init_linear_bias(_pinn, zero_init, subkey)
                # print(_pinn)
                # print(jax.tree_util.tree_leaves(_pinn, is_leaf=lambda
                #    p:not isinstance(p,eqx.nn.Linear))[0].layers_aux_nn[-1].bias)
                # _pinn = eqx.tree_at(lambda p:_pinn.layers_aux_nn[-1].bias, 0.5 *
                #        jnp.ones(_pinn.layers_aux_nn[-1].bias.shape))
                #        #, is_leaf=lambda
                #        #p:not isinstance(p, eqx.nn.Linear))

    def __call__(self, x):
        x_ = x.copy()
        # forward pass in the network which determines the freq
        for layer in self.layers_aux_nn:
            x_ = layer(x_)
        freq_list = jnp.clip(jnp.square(x_), a_min=1e-4, a_max=5)
        x_ = x.copy()
        # forward pass through the actual PINN
        for idx, layer in enumerate(self.layers_pinn):
            if idx % 2 == 0:
                # Currently: every two layer we have an activation
                # requiring a frequency
                x_ = layer(x_)
            else:
                x_ = layer(x_, freq_list[(idx - 1) // 2])
        return x_


class sinusPINN(PINN):
    """
    MUST inherit from PINN to pass all the checks

    HOWEVER we dot not bother with reimplementing anything
    """

    def __init__(self, key, list_layers_pinn, list_layers_aux_nn):
        super().__init__({}, jnp.s_[...], "statio_PDE", None, None, None)
        key, subkey = jax.random.split(key, 2)
        _pinn = _SinusPINN(subkey, list_layers_pinn, list_layers_aux_nn)

        self.params, self.static = eqx.partition(_pinn, eqx.is_inexact_array)

    def init_params(self):
        return self.params

    def __call__(self, x, params):
        try:
            model = eqx.combine(params["nn_params"], self.static)
        except (KeyError, TypeError) as e:  # give more flexibility
            model = eqx.combine(params, self.static)
        res = model(x)
        if not res.shape:
            return jnp.expand_dims(res, axis=-1)
        return res


def create_sinusPINN(key, list_layers_pinn, list_layers_aux_nn):
    """ """
    u = sinusPINN(key, list_layers_pinn, list_layers_aux_nn)
    return u
