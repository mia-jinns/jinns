"""
Formalize the data structure for the parameters
"""

from dataclasses import fields
from typing import Generic, TypeVar
import equinox as eqx
from jaxtyping import Array, PyTree

T = TypeVar("T")  # the generic type for what is in the Params PyTree because we
# have possibly Params of Arrays, boolean, ...

### NOTE
### We are taking derivatives with respect to Params eqx.Modules.
### This has been shown to behave weirdly if some fields of eqx.Modules have
### been set as `field(init=False)`, we then should never create such fields in
### jinns' Params modules.
### We currently have silenced the warning related to this (see jinns.__init__
### see https://github.com/patrick-kidger/equinox/pull/1043/commits/f88e62ab809140334c2f987ed13eff0d80b8be13


class EqParamsMeta(type):
    """
    We finally came up with a Metaclass pattern to handle the fact that we want
    one and only one type to be created for EqParams.
    If we were to create a new **class type** (despite same name) each time we
    create a new Params object, nothing would be broadcastble in terms of jax
    tree utils operations and this would be useless. The difficulty comes from
    the fact that we need to instanciate from this same class at different
    moments of the jinns workflow eg: parameter creation, derivative keys
    creations, tracked parameter designation, etc. (ie. each time a Params
    class is instanciated whatever its usage, we need the same EqParams class
    to be instanciated)

    This is inspired by the Singleton pattern in Python
    (https://stackoverflow.com/a/10362179)

    Here we need the call of a metaclass because (https://stackoverflow.com/a/45536640):
    Metaclasses implement how the class will behave (not the instance). So when you look at the instance creation:
    `x = Foo()`
    This literally "calls" the class Foo. That's why __call__ of the metaclass
    is invoked before the __new__ and
    __init__ methods of your class initialize the instance.
    Other viewpoint: Metaclasses,as well as classes making use of those
    metaclasses, are created when the lines of code containing
    the class statement body is executed
    """

    def __init__(self, *args, **kwargs):
        super(EqParamsMeta, self).__init__(*args, **kwargs)
        self._EqParamsClass = None

    def __call__(
        self, d: dict[str, Array], class_name: str | None = None
    ) -> eqx.Module:
        """
        Notably, once the class template is registered (after the first call to
        EqParams()), all calls with different keys in `d` will fail.
        On the other hand, passing a class name is a way to force to recreate
        the self._EqParamsClass type
        """
        if self._EqParamsClass is None and class_name is not None:
            self._EqParamsClass = type(
                class_name,
                (eqx.Module,),
                {"__annotations__": {k: type(v) for k, v in d.items()}},
            )
        try:
            return self._EqParamsClass(**d)
        except TypeError as _:
            print(
                "EqParams has been init for the parameters "
                f"{tuple(k for k in self._EqParamsClass.__annotations__.keys())}"
                f" but now an instanciation is resquested with keys={tuple(k for k in d.keys())}"
                " which results in an error"
            )
            raise ValueError

    def clear(cls) -> None:
        """
        Mainly for pytest where stuff is not complety reset after tests
        Taken from https://stackoverflow.com/a/50065732
        """
        cls._EqParamsClass = None


class EqParams(metaclass=EqParamsMeta):
    pass


class Params(eqx.Module, Generic[T]):
    """
    The equinox module for the parameters

    Parameters
    ----------
    nn_params : PyTree[T]
        A PyTree of the non-static part of the PINN eqx.Module, i.e., the
        parameters of the PINN
    eq_params : PyTree[T]
        A PyTree of the equation parameters. For retrocompatibility and
        verbosity issue this can be provided as formerly ie as
        a dictionary of the equation parameters where keys are the parameter name,
        values are their corresponding value
    """

    nn_params: PyTree[T] = eqx.field(kw_only=True, default=None)
    eq_params: PyTree[T] = eqx.field(
        kw_only=True,
        default=None,
        converter=lambda x: EqParams(x, "EqParams")
        if isinstance(x, dict)
        else x,  # the first call here is critical
    )


def _update_eq_params(
    params: Params[Array],
    eq_param_batch: PyTree[Array],
) -> Params:
    """
    Update params.eq_params with a batch of eq_params for given key(s)
    """

    param_names_to_update = tuple(f.name for f in fields(eq_param_batch))
    params = eqx.tree_at(
        lambda p: p.eq_params,
        params,
        eqx.tree_at(
            lambda pt: tuple(
                getattr(pt, f.name)
                for f in fields(pt)
                if f.name in param_names_to_update
            ),
            params.eq_params,
            tuple(getattr(eq_param_batch, f) for f in param_names_to_update),
        ),
    )

    return params


def _get_vmap_in_axes_params(
    eq_param_batch: eqx.Module, params: Params[Array]
) -> tuple[Params[int | None] | None]:
    """
    Return the input vmap axes when there is batch(es) of parameters to vmap
    over. The latter are designated by keys in eq_params_batch_dict.
    If eq_params_batch_dict is None (i.e. no additional parameter batch), we
    return (None,).

    Note that we return a Params PyTree with an integer to designate the
    vmapped axis or None if there is not
    """
    if eq_param_batch is None:
        return (None,)
    # We use pytree indexing of vmapped axes and vmap on axis
    # 0 of the eq_parameters for which we have a batch
    # this is for a fine-grained vmaping
    # scheme over the params
    param_names_to_vmap = tuple(f.name for f in fields(eq_param_batch))
    vmap_axes_dict = {
        k.name: (0 if k.name in param_names_to_vmap else None)
        for k in fields(params.eq_params)
    }
    eq_param_vmap_axes = type(params.eq_params)(**vmap_axes_dict)
    vmap_in_axes_params = (
        Params(
            nn_params=None,
            eq_params=eq_param_vmap_axes,
        ),
    )
    return vmap_in_axes_params
