"""
AI Math Verification & Discovery Tool

A rigorous mathematical verification system with anti-hallucination 
mechanisms and genuine explanation quality enforcement.
"""

__version__ = "0.1.0"
__author__ = "AI Math Team"

# Lazy imports to avoid circular dependencies
def __getattr__(name):
    if name == "MathEngine":
        from aimath.core.engine import MathEngine
        return MathEngine
    elif name == "ProofAssistant":
        from aimath.proof_assistant import ProofAssistant
        return ProofAssistant
    elif name == "solvers":
        from aimath import solvers
        return solvers
    elif name == "verifiers":
        from aimath import verification
        return verification
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
