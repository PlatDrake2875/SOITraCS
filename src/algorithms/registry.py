"""Algorithm registration and discovery."""

from typing import Dict, Type, Optional
from .base import BaseAlgorithm


class AlgorithmRegistry:
    """Registry for algorithm classes."""

    def __init__(self) -> None:
        self._algorithms: Dict[str, Type[BaseAlgorithm]] = {}

    def register(self, name: str, algorithm_class: Type[BaseAlgorithm]) -> None:
        """
        Register an algorithm class.

        Args:
            name: Algorithm identifier (e.g., "cellular_automata")
            algorithm_class: The algorithm class
        """
        self._algorithms[name] = algorithm_class

    def get(self, name: str) -> Optional[Type[BaseAlgorithm]]:
        """
        Get an algorithm class by name.

        Args:
            name: Algorithm identifier

        Returns:
            Algorithm class or None if not found
        """
        return self._algorithms.get(name)

    def list_algorithms(self) -> list[str]:
        """Get list of registered algorithm names."""
        return list(self._algorithms.keys())

    def is_registered(self, name: str) -> bool:
        """Check if an algorithm is registered."""
        return name in self._algorithms


# Global registry instance
_registry: Optional[AlgorithmRegistry] = None


def get_algorithm_registry() -> AlgorithmRegistry:
    """Get the global algorithm registry."""
    global _registry
    if _registry is None:
        _registry = AlgorithmRegistry()
        _register_default_algorithms(_registry)
    return _registry


def register_algorithm(name: str, algorithm_class: Type[BaseAlgorithm]) -> None:
    """Register an algorithm with the global registry."""
    get_algorithm_registry().register(name, algorithm_class)


def _register_default_algorithms(registry: AlgorithmRegistry) -> None:
    """Register all built-in algorithms."""
    # Import and register each algorithm
    try:
        from .cellular_automata import CellularAutomataAlgorithm
        registry.register("cellular_automata", CellularAutomataAlgorithm)
    except ImportError:
        pass

    try:
        from .sotl import SOTLAlgorithm
        registry.register("sotl", SOTLAlgorithm)
    except ImportError:
        pass

    try:
        from .aco import ACOAlgorithm
        registry.register("aco", ACOAlgorithm)
    except ImportError:
        pass

    try:
        from .pso import PSOAlgorithm
        registry.register("pso", PSOAlgorithm)
    except ImportError:
        pass

    try:
        from .som import SOMAlgorithm
        registry.register("som", SOMAlgorithm)
    except ImportError:
        pass

    try:
        from .marl import MARLAlgorithm
        registry.register("marl", MARLAlgorithm)
    except ImportError:
        pass
