"""
Base Solver - Abstract interface for mathematical solvers.

All solvers must implement this interface to participate in
multi-path solving and consensus verification.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional
from enum import Enum
import logging

from ..core.types import MathProblem, SolutionStep

logger = logging.getLogger(__name__)


class SolverStatus(Enum):
    """Status of a solver result."""
    SUCCESS = "success"
    PARTIAL = "partial"      # Some parts solved
    FAILED = "failed"
    TIMEOUT = "timeout"
    UNSUPPORTED = "unsupported"  # Problem type not supported


@dataclass
class SolverResult:
    """
    Result from a solver attempt.
    
    Attributes:
        status: Whether solving succeeded
        answer: The computed answer
        steps: Solution steps (if available)
        method_name: Name of the solving method used
        confidence: Self-reported confidence (0-1)
        computation_time_ms: Time taken
        error: Error message if failed
        metadata: Additional solver-specific data
    """
    status: SolverStatus
    answer: Optional[Any] = None
    steps: list[SolutionStep] = field(default_factory=list)
    method_name: str = "unknown"
    confidence: float = 0.0
    computation_time_ms: float = 0.0
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    
    @property
    def is_success(self) -> bool:
        """Check if solving was successful."""
        return self.status == SolverStatus.SUCCESS
    
    def __repr__(self) -> str:
        status_str = "✓" if self.is_success else "✗"
        return f"SolverResult({status_str} {self.method_name}: {self.answer})"


class BaseSolver(ABC):
    """
    Abstract base class for mathematical solvers.
    
    All solvers must implement:
    - solve(): Main solving method
    - can_solve(): Check if problem is solvable by this solver
    - trust_level: How much to trust this solver's results
    
    Trust Hierarchy:
        HIGHEST: Formal theorem provers (Lean, Z3)
        HIGH: Symbolic computation (SymPy)
        MEDIUM: Numerical methods (NumPy/SciPy)
        LOW: LLM-based solving (requires verification)
    """
    
    # Class-level properties
    name: str = "base"
    trust_level: float = 0.5  # 0-1, higher = more trustworthy
    timeout_seconds: float = 10.0
    
    def __init__(self, timeout: Optional[float] = None):
        """
        Initialize solver.
        
        Args:
            timeout: Maximum time for solving (seconds)
        """
        if timeout is not None:
            self.timeout_seconds = timeout
    
    @abstractmethod
    def solve(self, problem: MathProblem) -> SolverResult:
        """
        Solve the given mathematical problem.
        
        Args:
            problem: The problem to solve
            
        Returns:
            SolverResult with answer and metadata
        """
        pass
    
    @abstractmethod
    def can_solve(self, problem: MathProblem) -> bool:
        """
        Check if this solver can handle the given problem.
        
        Args:
            problem: The problem to check
            
        Returns:
            True if solver can attempt this problem type
        """
        pass
    
    def get_steps(self, problem: MathProblem, answer: Any) -> list[SolutionStep]:
        """
        Generate solution steps (optional).
        
        Override in subclasses that can provide step-by-step solutions.
        
        Args:
            problem: The original problem
            answer: The computed answer
            
        Returns:
            List of solution steps
        """
        return []
    
    def validate_answer(self, problem: MathProblem, answer: Any) -> bool:
        """
        Validate that an answer is reasonable.
        
        Basic sanity checks. Override for solver-specific validation.
        
        Args:
            problem: The original problem
            answer: The answer to validate
            
        Returns:
            True if answer passes basic validation
        """
        if answer is None:
            return False
        
        # Check for NaN, inf
        try:
            import math
            if isinstance(answer, (int, float)):
                if math.isnan(answer) or math.isinf(answer):
                    return False
        except (TypeError, ValueError):
            pass
        
        return True
    
    def _create_error_result(self, error: str) -> SolverResult:
        """Create a failed result with error message."""
        return SolverResult(
            status=SolverStatus.FAILED,
            method_name=self.name,
            error=error,
        )
    
    def _create_timeout_result(self) -> SolverResult:
        """Create a timeout result."""
        return SolverResult(
            status=SolverStatus.TIMEOUT,
            method_name=self.name,
            error=f"Solver timed out after {self.timeout_seconds}s",
        )
    
    def _create_unsupported_result(self, reason: str) -> SolverResult:
        """Create an unsupported problem result."""
        return SolverResult(
            status=SolverStatus.UNSUPPORTED,
            method_name=self.name,
            error=f"Problem not supported: {reason}",
        )
