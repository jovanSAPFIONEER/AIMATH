"""Core module - Main engine and type definitions."""

from .types import (
    MathProblem,
    Solution,
    VerificationResult,
    Explanation,
    ConfidenceLevel,
    DifficultyLevel,
)
from .engine import MathEngine

__all__ = [
    "MathProblem",
    "Solution", 
    "VerificationResult",
    "Explanation",
    "ConfidenceLevel",
    "DifficultyLevel",
    "MathEngine",
]
