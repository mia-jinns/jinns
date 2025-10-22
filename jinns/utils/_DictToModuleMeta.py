from typing import Any
import equinox as eqx

from jinns.utils._ItemizableModule import ItemizableModule


class DictToModuleMeta(type):
    """
    A Metaclass based solution to handle the fact that we only
    want one type to be created for EqParams.
    If we were to create a new **class type** (despite same name) each time we
    create a new Params object, nothing would be broadcastable in terms of jax
    tree utils operations and this would be useless. The difficulty comes from
    the fact that we need to instanciate from this same class at different
    moments of the jinns workflow eg: parameter creation, derivative keys
    creations, tracked parameter designation, etc. (ie. each time a Params
    class is instanciated whatever its usage, we need the same EqParams class
    to be instanciated)

    This is inspired by the Singleton pattern in Python
    (https://stackoverflow.com/a/10362179)

    Here we need the call of a metaclass because as explained in
     https://stackoverflow.com/a/45536640). To quote from the answer
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
        super(DictToModuleMeta, self).__init__(*args, **kwargs)
        self._class = None

    def __call__(self, d: dict[str, Any], class_name: str | None = None) -> eqx.Module:
        """
        Notably, once the class template is registered (after the first call to
        EqParams()), all calls with different keys in `d` will fail.
        """
        if self._class is None and class_name is not None:
            self._class = type(
                class_name,
                (ItemizableModule,),
                {"__annotations__": {k: type(v) for k, v in d.items()}},
            )
        try:
            return self._class(**d)  # type: ignore
        except TypeError as _:
            print(
                "DictToModuleMeta has been created with the fields"
                f"{tuple(k for k in self._class.__annotations__.keys())}"
                f" but an instanciation is resquested with fields={tuple(k for k in d.keys())}"
                " which results in an error"
            )
            raise ValueError

    def clear(cls) -> None:
        """
        The current Metaclass implementation freezes the list of equation parameters inside a Python session;
        only one EqParams annotation can exist at a given time. Use `EqParams.clear()`  to reset.
        Also useful for pytest where stuff is not complety reset after tests
        Taken from https://stackoverflow.com/a/50065732
        """
        cls._class = None
