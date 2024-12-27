from typing import Any, Callable, Type, TypeVar

T = TypeVar("T")


__REGISTRIES__ = [
    "tasks",
    "models",
    "metrics",
    "datasets",
]


class Registry:
    def __init__(self, registry_type: str):
        """Registries class."""
        if registry_type not in __REGISTRIES__:
            raise ValueError(
                f"Found 'registry type={registry_type}', "
                f"expected {__REGISTRIES__}."
            )
        self.registry_type = registry_type
        self.elements_dict = {}

    def register_element(self, name: str = None) -> Callable:
        """Register an element
        """
        def decorator(fn: Callable) -> Callable:
            key = name if name is not None else fn.__name__
            if key in self.elements_dict:
                raise ValueError(
                    f"An entry is already registered under the name '{key}': "
                    f"{self.elements_dict[key]}"
                )
            self.elements_dict[key] = fn
            return fn
        return decorator

    def get_element_reference(self, name: str) -> Type[T]:
        """Get element reference from its name
        """
        return self.elements_dict[name]
    
    def get_element_instance(self, name: str, **params: Any) -> T:
        """Get element instance from its name and parameters
        """
        return self.get_element_reference(name)(**params)
    
    def get_element_names(self) -> list:
        """Get names of all registered elements
        """
        return sorted(list(self.elements_dict.keys()))