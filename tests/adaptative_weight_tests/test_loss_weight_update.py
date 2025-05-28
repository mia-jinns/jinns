"""
Test the soft adapt implementation
https://docs.nvidia.com/deeplearning/physicsnemo/physicsnemo-sym/user_guide/theory/advanced_schemes.html#softadapt
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import jinns


def test_weight_history():
    """
    With a soft_adapt and two terms in the loss and a loss that will permanently be equal to a
    constant value we know that the history of loss weights should be full of
    0.5 values
    """
    loss_weights = jinns.loss.LossWeightsPDENonStatio(
        dyn_loss=jnp.array(1.0),
        initial_condition=jnp.array(1.0),
        update_method="soft_adapt",
    )

    key = jax.random.PRNGKey(2)
    eqx_list = (
        (eqx.nn.Linear, 2, 32),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 32, 32),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 32, 32),
        (jax.nn.tanh,),
        (eqx.nn.Linear, 32, 1),
    )
    key, subkey = jax.random.split(key)
    u, init_nn_params = jinns.nn.PINN_MLP.create(
        key=subkey, eqx_list=eqx_list, eq_type="nonstatio_PDE"
    )

    n = 2500
    ni = 200
    nb = 200
    dim = 1
    xmin = -1
    xmax = 1
    tmin = 0
    tmax = 1
    method = "uniform"

    key, subkey = jax.random.split(key)
    train_data = jinns.data.CubicMeshPDENonStatio(
        key=subkey,
        n=n,
        nb=nb,
        ni=ni,
        dim=dim,
        min_pts=(xmin,),
        max_pts=(xmax,),
        tmin=tmin,
        tmax=tmax,
        method=method,
    )

    nu = 1 / (100 * jnp.pi)
    init_params = jinns.parameters.Params(
        nn_params=init_nn_params, eq_params={"nu": nu}
    )

    def u0(x):
        return 0.6

    class DummyLoss(jinns.loss.PDENonStatio):
        def equation(self, t_x, u, params):
            return 0.6

    dummy_loss = jinns.loss.BurgersEquation(Tmax=10)

    loss = jinns.loss.LossPDENonStatio(
        u=u,
        loss_weights=loss_weights,
        dynamic_loss=dummy_loss,
        initial_condition_fun=u0,
        params=init_params,
    )

    params = init_params

    tx = optax.adamw(learning_rate=1e-3)
    n_iter = 10
    params, total_loss_list, loss_by_term_dict, _, _, _, _, stored_lw, _, _ = (
        jinns.solve(
            init_params=params, data=train_data, optimizer=tx, loss=loss, n_iter=n_iter
        )
    )
    response1 = jnp.array(
        [
            1.0,
            0.8277719,
            0.637744,
            0.5815456,
            0.55772996,
            0.5572184,
            0.5907892,
            0.6587647,
            0.7124741,
            0.7012826,
        ]
    )
    response2 = jnp.array(
        [
            1.0,
            0.17222811,
            0.362256,
            0.41845444,
            0.44227007,
            0.4427816,
            0.40921077,
            0.3412353,
            0.28752592,
            0.2987174,
        ]
    )
    assert jnp.allclose(stored_lw.dyn_loss, response1)
    assert jnp.allclose(stored_lw.initial_condition, response2)
    assert stored_lw.boundary_loss is None
    assert stored_lw.observations is None
    assert stored_lw.norm_loss is None
