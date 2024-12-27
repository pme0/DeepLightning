from collections.abc import MutableMapping
from dataclasses import fields, is_dataclass
import inspect
from typing import Any


def flatten_dict(d: MutableMapping, parent_key: str = '', sep: str ='.') -> MutableMapping:
    """Recursively flattens a dictionary into a flat dictionary."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))

    return dict(items)


def flatten_dataclass(obj: Any, prefix: str = '') -> dict[str, Any]:
    """Recursively flattens a dataclass into a flat dictionary with dotted keys."""
    flat_dict = {}

    if is_dataclass(obj):
        for field in fields(obj):
            value = getattr(obj, field.name)
            key = f'{prefix}{field.name}' if prefix else field.name
            if is_dataclass(value):
                flat_dict.update(flatten_dataclass(value, prefix=f'{key}.'))
            else:
                flat_dict[key] = value
    else:
        flat_dict[prefix] = obj

    return flat_dict


def exists(x) -> bool:
    return x is not None


def public_attributes(d) -> dict:
    return {
        key: val 
        for key, val in d.__dict__.items() 
        if not key.startswith("__")
    }


def get_nested_attr(obj, attr_path: str):
    """Recursively get a nested attribute using dot notation."""
    attr_list = attr_path.split('.')
    for attr in attr_list:
        obj = getattr(obj, attr)
    return obj


def get_num_args(fn):
    args = inspect.getfullargspec(fn).args
    n = len(args)
    if "self" in args:
        return n - 1
    return n