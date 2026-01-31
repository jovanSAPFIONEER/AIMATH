"""
Explanation module - Quality-enforced mathematical explanations.

This module prevents classic AI explanation failures:
- No hand-waving ("it's complex...")
- No skipped steps ("obviously...")
- No undefined jargon
- No circular definitions
- Mandatory "why" for every step
"""

from .explanation_engine import ExplanationEngine
from .quality_checker import QualityChecker, CLEARScore
from .anti_patterns import AntiPatternDetector
from .expansion_engine import ExpansionEngine
from .difficulty_adapter import DifficultyAdapter

__all__ = [
    "ExplanationEngine",
    "QualityChecker",
    "CLEARScore",
    "AntiPatternDetector",
    "ExpansionEngine",
    "DifficultyAdapter",
]
