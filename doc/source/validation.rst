jinns.validation
=================

The `jinns` package allows you to implement your own validation strategy.
The latter should be an `eqx.module` (see [Equinox main page](https://docs.kidger.site/equinox/api/module/module/)) which is a PyTree with a
`__call__(self, params)` method and a `call_every` attribute. The abstract
interface is implemented in :obj:`~jinns.validation._validation.AbstractValidationModule` as

```
class AbstractValidationModule(eqx.Module):
    """Abstract class representing interface for any validation module. It must
    1. have a `call_every` attribute.
    2. implement a __call__ returning (AbstractValidationModule, Bool, Array)
    """

    call_every: eqx.AbstractVar[Int]  # Mandatory for all validation step,
    # it tells that the validation step is performed every call_every
    # iterations.

    @abc.abstractmethod
    def __call__(
        self, params: PyTree
    ) -> tuple["AbstractValidationModule", Bool, Array]:
        raise NotImplementedError
```

**Note**: the `__call__` should return a size 3 tuple `(validation [eqx.Module], early_stop [bool], validation_criterion [Array])`

.. automodule:: jinns.validation
   :members:
   :imported-members:
   :inherited-members:
   :special-members: __init__
   :undoc-members:
   :show-inheritance:
   :private-members:
