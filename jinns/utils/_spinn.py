import jax
import jax.numpy as jnp
import equinox as eqx


class _SPINN(eqx.Module):
    """
    Construct a Separable PINN as proposed in
    Cho et al., _Separable Physics-Informed Neural Networks_, NeurIPS, 2023
    """

    layers: list
    separated_mlp: list
    d: int
    r: int
    m: int

    def __init__(self, key, d, r, eqx_list, m=1):
        """
        Parameters
        ----------
        key
            A jax random key
        d
            An integer. The number of dimensions to treat separately
        r
            An integer. The dimension of the embedding
        eqx_list
            A list of list of successive equinox modules and activation functions to
            describe *each separable PINN architecture*.
            The inner lists have the eqx module or
            axtivation function as first item, other items represents arguments
            that could be required (eg. the size of the layer).
            __Note:__ the `key` argument need not be given.
            Thus typical example is `eqx_list=
            [[eqx.nn.Linear, d, 20],
                [jax.nn.tanh],
                [eqx.nn.Linear, 20, 20],
                [jax.nn.tanh],
                [eqx.nn.Linear, 20, 20],
                [jax.nn.tanh],
                [eqx.nn.Linear, 20, r]
            ]`
        """
        keys = jax.random.split(key, 8)

        self.d = d
        self.r = r
        self.m = m

        self.separated_mlp = []
        for d in range(self.d):
            self.layers = []
            for l in eqx_list:
                if len(l) == 1:
                    self.layers.append(l[0])
                else:
                    key, subkey = jax.random.split(key, 2)
                    self.layers.append(l[0](*l[1:], key=subkey))
            self.separated_mlp.append(self.layers)

    def __call__(self, t, x):
        if t is not None:
            dimensions = jnp.concatenate([t, x.flatten()], axis=0)
        else:
            dimensions = jnp.concatenate([x.flatten()], axis=0)
        outputs = []
        for d in range(self.d):
            t_ = dimensions[d][None]
            for layer in self.separated_mlp[d]:
                t_ = layer(t_)
            outputs += [t_]
        return jnp.asarray(outputs)


class SPINN:
    """
    Basically a wrapper around the `__call__` function to be able to give a type to
    our former `self.u`
    The function create_SPINN has the role to population the `__call__` function
    """

    def __init__(self, key, d, r, eqx_list, m=1):
        self.d, self.r, self.m = d, r, m
        _spinn = _SPINN(key, d, r, eqx_list, m)
        self.params, self.static = eqx.partition(_spinn, eqx.is_inexact_array)

    def init_params(self):
        return self.params

    def __call__(self, *args, **kwargs):
        return self.apply_fn(self, *args, **kwargs)

    def _eval_nn(self, res):
        """
        common content of apply_fn put here in order to factorize code
        """
        a = ", ".join([f"{chr(97 + d)}z" for d in range(res.shape[1])])
        b = "".join([f"{chr(97 + d)}" for d in range(res.shape[1])])
        res = jnp.stack(
            [
                jnp.einsum(
                    f"{a} -> {b}",
                    *(
                        res[:, d, m * self.r : (m + 1) * self.r]
                        for d in range(res.shape[1])
                    ),
                )
                for m in range(self.m)
            ],
            axis=-1,
        )  # compute each output dimension

        # force (1,) output for non vectorial solution (consistency)
        if len(res.shape) == self.d:
            return jnp.expand_dims(res, axis=-1)
        else:
            return res


def create_SPINN(key, d, r, eqx_list, eq_type, m=1):
    """
    Utility function to create a SPINN neural network with the equinox
    library.

    *Note* that a SPINN is not vmapped from the outside and expects batch of the
    same size for each input. It outputs directly a solution of shape
    (batchsize, batchsize). See the paper for more details.

    Parameters
    ----------
    key
        A jax random key that will be used to initialize the network parameters
    d
        An integer. The number of dimensions to treat separately
    r
        An integer. The dimension of the embedding
    eqx_list
        A list of list of successive equinox modules and activation functions to
        describe *each separable PINN architecture*.
        The inner lists have the eqx module or
        axtivation function as first item, other items represents arguments
        that could be required (eg. the size of the layer).
        __Note:__ the `key` argument need not be given.
        Thus typical example is `eqx_list=
        [[eqx.nn.Linear, d, 20],
        [jax.nn.tanh],
        [eqx.nn.Linear, 20, 20],
        [jax.nn.tanh],
        [eqx.nn.Linear, 20, 20],
        [jax.nn.tanh],
        [eqx.nn.Linear, 20, r]
        ]`
    eq_type
        A string with three possibilities.
        "ODE": the PINN is called with one input `t`.
        "statio_PDE": the PINN is called with one input `x`, `x`
        can be high dimensional.
        "nonstatio_PDE": the PINN is called with two inputs `t` and `x`, `x`
        can be high dimensional.
        **Note: the input dimension as given in eqx_list has to match the sum
        of the dimension of `t` + the dimension of `x` + the number of
        parameters in `eq_params` if with_eq_params is `True` (see below)**
    m
        An integer. The output dimension of the neural network. According to
        the SPINN article, a total embedding dimension of `r*m` is defined. We
        then sum groups of `r` embedding dimensions to compute each output.
        Default is 1.


    Returns
    -------
    init_fn
        A function which (re-)initializes the SPINN parameters with the provided
        jax random key
    apply_fn
        A function to apply the neural network on given inputs for given
        parameters. A typical call will be of the form `u(t, nn_params)` for
        ODE or `u(t, x, nn_params)` for nD PDEs (`x` being multidimensional)
        or even `u(t, x, nn_params, eq_params)` if with_eq_params is `True`

    Raises
    ------
    RuntimeError
        If the parameter value for eq_type is not in `["ODE", "statio_PDE",
        "nonstatio_PDE"]` and for various failing checks
    """

    if eq_type not in ["ODE", "statio_PDE", "nonstatio_PDE"]:
        raise RuntimeError("Wrong parameter value for eq_type")

    try:
        nb_inputs_declared = eqx_list[0][1]  # normally we look for 2nd ele of 1st layer
    except IndexError:
        nb_inputs_declared = eqx_list[1][
            1
        ]  # but we can have, eg, a flatten first layer
    if nb_inputs_declared != 1:
        raise ValueError("Input dim must be set to 1 in SPINN!")

    try:
        nb_outputs_declared = eqx_list[-1][2]  # normally we look for 3rd ele of
        # last layer
    except IndexError:
        nb_outputs_declared = eqx_list[-2][2]
        # but we can have, eg, a `jnp.exp` last layer
    if nb_outputs_declared != r * m:
        raise ValueError("Output dim must be set to r * m in SPINN!")

    if d > 24:
        raise ValueError(
            "Too many dimensions, not enough letters" " available in jnp.einsum"
        )

    if eq_type == "statio_PDE":

        def apply_fn(self, x, u_params, eq_params=None):
            spinn = eqx.combine(u_params, self.static)
            v_model = jax.vmap(spinn, (0))
            res = v_model(t=None, x=x)
            return self._eval_nn(res)

    elif eq_type == "nonstatio_PDE":

        def apply_fn(self, t, x, u_params, eq_params=None):
            spinn = eqx.combine(u_params, self.static)
            v_model = jax.vmap(spinn, ((0, 0)))
            res = v_model(t, x)
            return self._eval_nn(res)

    else:
        raise RuntimeError("Wrong parameter value for eq_type")

    spinn = SPINN(key, d, r, eqx_list, m)
    spinn.apply_fn = apply_fn

    return spinn
