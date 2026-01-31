"""
AIMATH - AI Math Verification & Discovery Tool

A rigorous mathematical verification system that helps everyone from
amateurs to professionals solve, verify, and discover mathematical concepts
with built-in anti-hallucination mechanisms.

Example Usage:
    >>> from aimath import MathEngine
    >>> engine = MathEngine()
    >>> result = engine.solve("x^2 - 5x + 6 = 0")
    >>> print(result.solutions)  # [2, 3]

    >>> from aimath import ProofAssistant
    >>> prover = ProofAssistant()
    >>> theorem = prover.state_theorem("sum_of_angles", "Triangle angles sum to 180°")
"""

__version__ = "1.0.0"
__author__ = "Jovan"
__license__ = "MIT"

# Lazy imports to avoid circular dependencies and missing modules
def __getattr__(name):
    """Lazy import of submodules."""
    if name == "MathEngine":
        from .core.engine import MathEngine
        return MathEngine
    elif name == "ProofAssistant":
        from .proof_assistant import ProofAssistant
        return ProofAssistant
    elif name == "Theorem":
        from .proof_assistant import Theorem
        return Theorem
    elif name == "Proof":
        from .proof_assistant import Proof
        return Proof
    elif name == "Proposition":
        from .proof_assistant import Proposition
        return Proposition
    elif name == "ProofVerifier":
        from .proof_assistant import ProofVerifier
        return ProofVerifier
    elif name == "SymbolicSolver":
        from .solvers.symbolic_solver import SymbolicSolver
        return SymbolicSolver
    elif name == "NumericalSolver":
        from .solvers.numerical_solver import NumericalSolver
        return NumericalSolver
    elif name == "ExplanationEngine":
        from .explanation.explanation_engine import ExplanationEngine
        return ExplanationEngine
    elif name == "Verifier":
        from .verification.verifier import Verifier
        return Verifier
    raise AttributeError(f"module 'aimath' has no attribute '{name}'")

__all__ = [
    "__version__",
    "__author__",
    "__license__",
    "MathEngine",
    "ProofAssistant",
    "Theorem",
    "Proof",
    "Proposition",
    "ProofVerifier",
    "SymbolicSolver",
    "NumericalSolver",
    "ExplanationEngine",
    "Verifier",
]
