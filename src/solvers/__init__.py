"""Solvers module - Multiple solving strategies for consensus."""

from .solver_base import BaseSolver, SolverResult
from .symbolic_solver import SymbolicSolver
from .numerical_solver import NumericalSolver
from .llm_solver import LLMSolver
from .solver_registry import get_solver, register_solver, SolverRegistry

__all__ = [
    "BaseSolver",
    "SolverResult",
    "SymbolicSolver",
    "NumericalSolver", 
    "LLMSolver",
    "get_solver",
    "register_solver",
    "SolverRegistry",
]
