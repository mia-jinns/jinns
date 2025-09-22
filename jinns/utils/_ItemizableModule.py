from dataclasses import fields
from typing import Any, ItemsView
import equinox as eqx


class ItemizableModule(eqx.Module):
    def items(self) -> ItemsView[str, Any]:
        """
        For the dataclass to be iterated like a dictionary.
        Practical and retrocompatible with old code when loss components were
        dictionaries

        About the type hint: https://stackoverflow.com/questions/73022688/type-annotation-for-dict-items
        """
        return {
            field.name: getattr(self, field.name)
            for field in fields(self)
            if getattr(self, field.name) is not None
        }.items()
