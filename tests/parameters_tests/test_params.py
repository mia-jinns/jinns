    )
    param_train_data, param_batch = param_train_data.get_batch()

    params = _update_eq_params(params, param_batch)

    assert params.eq_params.nu.shape == (10, 1)
