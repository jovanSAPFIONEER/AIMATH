"""Verification module - Anti-hallucination verification system."""

from .verifier import Verifier
from .substitution import SubstitutionChecker
from .counterexample import CounterexampleSearcher
from .consensus import ConsensusChecker
from .confidence import ConfidenceScorer
from .formal_prover import FormalProver

__all__ = [
    "Verifier",
    "SubstitutionChecker",
    "CounterexampleSearcher",
    "ConsensusChecker",
    "ConfidenceScorer",
    "FormalProver",
]
