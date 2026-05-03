# Optax optimizers

Jinns is readily compatible with Optax optimizers like so

```python
...  # define your PINN problem
optimizer = optax.adam(1e-2)
jinns.solve(
    ...,
    optimizer=optimizer
)
```

For `GradientTransformExtraArgs` which require extra arguments to their`update()` function, see the notebook [Tutorial Forward Problem](Notebooks/Tutorials/implementing_your_own_PDE_problem) for an example on how to pass these extra arguments in jinns.