import logging
from typing import Type
from .base import ModelAdapter
from .default import DefaultAdapter

logger = logging.getLogger(__name__)

_ADAPTERS: dict[str, Type[ModelAdapter]] = {}


def register_adapter(name: str, adapter_cls: Type[ModelAdapter]):
    """Register an adapter class."""
    _ADAPTERS[name] = adapter_cls


def get_adapter(model_alias: str, model_def: dict | None = None) -> ModelAdapter:
    """Get adapter instance for a model.
    
    Checks model definition for explicit 'adapter' field, falls back to default.
    """
    adapter_name = None
    if model_def:
        adapter_name = model_def.get("adapter")
    
    if adapter_name and adapter_name in _ADAPTERS:
        logger.debug(f"Using adapter '{adapter_name}' for model '{model_alias}'")
        return _ADAPTERS[adapter_name]()
    
    return DefaultAdapter()


# Auto-register adapters when module loads
def _register_builtins():
    from .default import DefaultAdapter
    from .pygmalion import PygmalionAdapter
    
    register_adapter("default", DefaultAdapter)
    register_adapter("pygmalion", PygmalionAdapter)
    register_adapter("mythomax", PygmalionAdapter)  # Alias


_register_builtins()
