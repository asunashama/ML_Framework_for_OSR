from typing import Any, Callable, Dict, List, Optional


class Registry:
    """Generic component registry for extensible tasks, backbones, and OSR methods."""

    def __init__(self, name: str):
        self._name = name
        self._items: Dict[str, Any] = {}

    def register(self, name: Optional[str] = None) -> Callable:
        def decorator(obj: Any) -> Any:
            key = (name or obj.__name__).lower()
            self._items[key] = obj
            return obj

        return decorator

    def get(self, name: str) -> Any:
        key = name.lower()
        if key not in self._items:
            available = ", ".join(sorted(self._items.keys()))
            raise KeyError(f"Unknown {self._name} '{name}'. Available: [{available}]")
        return self._items[key]

    def list_available(self) -> List[str]:
        return sorted(self._items.keys())


# Global registries
# TASK_REGISTRY reserves extension points for:
# - 'sei_osr': Specific Emitter Identification (individual emitter OSR)
# - 'amc_osr': Automatic Modulation / Regime Recognition OSR (future extension)
TASK_REGISTRY = Registry("task")
BACKBONE_REGISTRY = Registry("backbone")
METHOD_REGISTRY = Registry("method")
