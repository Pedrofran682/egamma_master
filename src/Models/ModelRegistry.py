import importlib
import inspect
import logging
import pkgutil
from typing import Any, Callable, Dict, List, Optional, Type, Union

import torch.nn as nn

import src.Models.egamma as egamma_pkg

log = logging.getLogger(__name__)


class ModelRegistry:
    """Registry providing automatic discovery and instantiation of neural network models."""

    _models: Dict[str, Type[nn.Module]] = {}
    _discovered: bool = False

    @classmethod
    def discover_models(cls, package: Any = egamma_pkg, force_reload: bool = False) -> None:
        """Scan package modules and register all nn.Module subclasses by their class name.

        Args:
            package: Python package or module containing model implementations.
            force_reload: If True, forces re-scanning of the package even if already discovered.
        """
        if cls._discovered and not force_reload:
            return

        if not hasattr(package, "__path__"):
            log.warning(f"Package {package} has no __path__ attribute; skipping auto-discovery.")
            cls._discovered = True
            return

        for _, module_name, _ in pkgutil.iter_modules(package.__path__):
            full_module_name = f"{package.__name__}.{module_name}"
            try:
                module = importlib.import_module(full_module_name)
            except Exception as e:
                log.error(f"Failed to import module {full_module_name}: {e}")
                continue

            for _, obj in inspect.getmembers(module, inspect.isclass):
                if (
                    issubclass(obj, nn.Module)
                    and obj is not nn.Module
                    and obj.__module__ == full_module_name
                ):
                    cls.register(obj)

        cls._discovered = True
        log.info(f"ModelRegistry discovered {len(cls._models)} models: {list(cls._models.keys())}")

    @classmethod
    def register(
        cls,
        model_class: Optional[Type[nn.Module]] = None,
        name: Optional[str] = None,
    ) -> Union[Type[nn.Module], Callable[[Type[nn.Module]], Type[nn.Module]]]:
        """Register a model class using its class name or an explicit alias.

        Can be used as a function or a decorator.

        Args:
            model_class: Model class subclassing nn.Module.
            name: Optional explicit registration name. If omitted, uses model_class.__name__.

        Returns:
            Registered model class or decorator wrapper.
        """
        def decorator(subclass: Type[nn.Module]) -> Type[nn.Module]:
            reg_name = name or subclass.__name__
            cls._models[reg_name] = subclass
            return subclass

        if model_class is not None:
            return decorator(model_class)
        return decorator

    @classmethod
    def get_class(cls, name: str) -> Type[nn.Module]:
        """Retrieve model class by class name or legacy short tag.

        Args:
            name: Model class name (e.g., 'ModelV1') or legacy tag (e.g., 'V1').

        Returns:
            Model class subclassing nn.Module.

        Raises:
            ValueError: If the requested model cannot be found in the registry.
        """
        if not cls._discovered:
            cls.discover_models()

        if name in cls._models:
            return cls._models[name]

        # Support fallback for legacy tags like 'V1' -> 'ModelV1'
        if f"Model{name}" in cls._models:
            return cls._models[f"Model{name}"]

        available_models = sorted(list(cls._models.keys()))
        raise ValueError(
            f"Model '{name}' not found in ModelRegistry. Available models: {available_models}"
        )

    @classmethod
    def create(cls, name: str, *args: Any, **kwargs: Any) -> nn.Module:
        """Instantiate a registered model by name.

        Args:
            name: Model class name or legacy tag.
            *args: Positional arguments forwarded to model __init__.
            **kwargs: Keyword arguments forwarded to model __init__.

        Returns:
            Instantiated nn.Module.
        """
        model_cls = cls.get_class(name)
        return model_cls(*args, **kwargs)

    @classmethod
    def list_models(cls) -> List[str]:
        """List all registered model class names.

        Returns:
            Sorted list of model names.
        """
        if not cls._discovered:
            cls.discover_models()
        return sorted(list(cls._models.keys()))

    @classmethod
    def contains(cls, name: str) -> bool:
        """Check whether a model name or tag is registered.

        Args:
            name: Model name or tag to check.

        Returns:
            True if registered, False otherwise.
        """
        if not cls._discovered:
            cls.discover_models()
        return name in cls._models or f"Model{name}" in cls._models
