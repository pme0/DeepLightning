from typing import Callable


__ModelRegistry__ = [
    "mobilenetv3"
]


def register_model() -> Callable:
    def wrapper(fn: Callable) -> Callable:
        name = fn.__name__
        if name in __ModelRegistry__:
            raise ValueError(f"A model is already registered with name '{name}'.")
        __ModelRegistry__[name] = fn
        return fn
    return wrapper


class ModelRegistry:
    models = {}
    
    @classmethod
    def register(cls, name):
        def wrapper(model_class):
            cls.models[name] = model_class
            return model_class
        return wrapper
    
    @classmethod
    def get_model(cls, name):
        return cls.models.get(name, None)