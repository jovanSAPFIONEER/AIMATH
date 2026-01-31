"""
Solver Registry - Central management of available solvers.

Provides a factory interface to get solvers by name and
manages solver registration.
"""

from typing import Dict, Optional, Type
import logging

from .solver_base import BaseSolver, SolverResult
from .symbolic_solver import SymbolicSolver
from .numerical_solver import NumericalSolver
from .llm_solver import LLMSolver

logger = logging.getLogger(__name__)


class SolverRegistry:
    """
    Registry for mathematical solvers.
    
    Manages available solvers and provides factory methods
    to instantiate them by name.
    
    Example:
        >>> registry = SolverRegistry()
        >>> solver = registry.get("symbolic")
        >>> result = solver.solve(problem)
    """
    
    # Default registered solvers
    _default_solvers: Dict[str, Type[BaseSolver]] = {
        "symbolic": SymbolicSolver,
        "numerical": NumericalSolver,
        "llm": LLMSolver,
    }
    
    def __init__(self):
        """Initialize registry with default solvers."""
        self._solvers: Dict[str, Type[BaseSolver]] = dict(self._default_solvers)
        self._instances: Dict[str, BaseSolver] = {}
    
    def register(self, name: str, solver_class: Type[BaseSolver]) -> None:
        """
        Register a new solver.
        
        Args:
            name: Name to register solver under
            solver_class: Solver class (must inherit BaseSolver)
        """
        if not issubclass(solver_class, BaseSolver):
            raise TypeError(
                f"Solver must inherit from BaseSolver, got {type(solver_class)}"
            )
        
        self._solvers[name] = solver_class
        logger.info(f"Registered solver: {name}")
    
    def get(self, name: str, **kwargs) -> BaseSolver:
        """
        Get a solver instance by name.
        
        Args:
            name: Solver name
            **kwargs: Arguments to pass to solver constructor
            
        Returns:
            Solver instance
            
        Raises:
            KeyError: If solver not found
        """
        if name not in self._solvers:
            available = list(self._solvers.keys())
            raise KeyError(
                f"Solver '{name}' not found. Available: {available}"
            )
        
        # Create new instance with kwargs
        return self._solvers[name](**kwargs)
    
    def get_cached(self, name: str, **kwargs) -> BaseSolver:
        """
        Get a cached solver instance.
        
        Returns same instance for repeated calls with same name.
        
        Args:
            name: Solver name
            **kwargs: Arguments for first instantiation
            
        Returns:
            Solver instance (cached)
        """
        if name not in self._instances:
            self._instances[name] = self.get(name, **kwargs)
        return self._instances[name]
    
    def list_solvers(self) -> list[dict]:
        """
        List all registered solvers with metadata.
        
        Returns:
            List of solver info dicts
        """
        result = []
        for name, cls in self._solvers.items():
            result.append({
                'name': name,
                'class': cls.__name__,
                'trust_level': cls.trust_level,
                'description': cls.__doc__.split('\n')[1].strip() if cls.__doc__ else '',
            })
        return result
    
    def get_by_trust(self, min_trust: float = 0.0) -> list[str]:
        """
        Get solvers with trust level at or above threshold.
        
        Args:
            min_trust: Minimum trust level (0-1)
            
        Returns:
            List of solver names
        """
        return [
            name for name, cls in self._solvers.items()
            if cls.trust_level >= min_trust
        ]


# Global registry instance
_global_registry = SolverRegistry()


def get_solver(name: str, **kwargs) -> BaseSolver:
    """
    Get a solver by name from global registry.
    
    Convenience function for common use case.
    
    Args:
        name: Solver name ("symbolic", "numerical", "llm")
        **kwargs: Solver constructor arguments
        
    Returns:
        Solver instance
    """
    return _global_registry.get_cached(name, **kwargs)


def register_solver(name: str, solver_class: Type[BaseSolver]) -> None:
    """
    Register a solver in global registry.
    
    Args:
        name: Name to register under
        solver_class: Solver class
    """
    _global_registry.register(name, solver_class)


def list_solvers() -> list[dict]:
    """List all available solvers."""
    return _global_registry.list_solvers()
