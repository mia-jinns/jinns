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

    n = 1
    ni = 1
    nb = None
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
    train_data = eqx.tree_at(lambda pt: pt.domain, train_data, jnp.array([[0.0, 1.0]]))
    train_data = eqx.tree_at(lambda pt: pt.initial, train_data, jnp.array([[1.0]]))

    nu = 1 / (100 * jnp.pi)
    init_params = jinns.parameters.Params(
        nn_params=init_nn_params, eq_params={"nu": nu}
    )

    def u0(x):
        return 0.6

    class DummyLoss(jinns.loss.PDENonStatio):
        def equation(self, t_x, u, params):
            return (0.6 - u(t_x, params).squeeze())[None]

    dummy_loss = DummyLoss(Tmax=1)

    loss = jinns.loss.LossPDENonStatio(
        u=u,
        loss_weights=loss_weights,
        update_weight_method="soft_adapt",
        dynamic_loss=dummy_loss,
        initial_condition_fun=u0,
        params=init_params,
    )

    params = init_params

    tx = optax.adamw(learning_rate=1e-3)
    n_iter = 10
    key, subkey = jax.random.split(key)
    params, total_loss_list, loss_by_term_dict, _, _, _, _, stored_lw, _, _ = (
        jinns.solve(
            init_params=params,
            data=train_data,
            optimizer=tx,
            loss=loss,
            n_iter=n_iter,
            key=subkey,
        )
    )
    response1 = jnp.array([1.0] + [0.5] * 9)
    response2 = jnp.array([1.0] + [0.5] * 9)

    assert jnp.allclose(stored_lw.dyn_loss, response1)
    assert jnp.allclose(stored_lw.initial_condition, response2)
    assert stored_lw.boundary_loss is None
    assert stored_lw.observations is None
    assert stored_lw.norm_loss is None


def test_loss_value():
    """
    The loss should have the same initial value for initial parameters using the same wights
    """

    loss_weights = jinns.loss.LossWeightsPDENonStatio(
        dyn_loss=jnp.array(1.0),
        initial_condition=jnp.array(1.0),
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

    n = 1
    ni = 1
    nb = None
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
    train_data = eqx.tree_at(lambda pt: pt.domain, train_data, jnp.array([[0.0, 1.0]]))
    train_data = eqx.tree_at(lambda pt: pt.initial, train_data, jnp.array([[1.0]]))

    nu = 1 / (100 * jnp.pi)
    init_params = jinns.parameters.Params(
        nn_params=init_nn_params, eq_params={"nu": nu}
    )

    def u0(x):
        return 0.6

    class DummyLoss(jinns.loss.PDENonStatio):
        def equation(self, t_x, u, params):
            return (0.6 - u(t_x, params).squeeze())[None]

    dummy_loss = DummyLoss(Tmax=1)

    loss_kwargs = {
        "u": u,
        "loss_weights": loss_weights,
        "dynamic_loss": dummy_loss,
        "initial_condition_fun": u0,
        "params": init_params,
    }

    loss = jinns.loss.LossPDENonStatio(**loss_kwargs)

    loss_SA = jinns.loss.LossPDENonStatio(
        **(loss_kwargs | {"update_weight_method": "softadapt"})
    )

    loss_LRA = jinns.loss.LossPDENonStatio(
        **(loss_kwargs | {"update_weight_method": "lr_annealing"})
    )

    losses_and_grad = jax.value_and_grad(loss, 0, has_aux=True)
    losses_and_grad_SA = jax.value_and_grad(loss_SA, 0, has_aux=True)
    losses_and_grad_LRA = jax.value_and_grad(loss_LRA, 0, has_aux=True)

    losses, _ = losses_and_grad(init_params, train_data.get_batch()[1])
    losses_SA, _ = losses_and_grad_SA(init_params, train_data.get_batch()[1])
    losses_LRA, _ = losses_and_grad_LRA(init_params, train_data.get_batch()[1])

    reduced_tree = jax.tree.map(
        lambda *args: jnp.all(args[:-1] == args[1:]),  # using transitivity of =
        losses[1],
        losses_SA[1],
        losses_LRA[1],
    )
    assert jax.tree.reduce(jnp.logical_and, reduced_tree, initializer=jnp.array(True))
