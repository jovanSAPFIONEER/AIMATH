"""
MathClaw Discovery Layer

This module orchestrates autonomous mathematical discovery.
It connects LLM conjecture generation with AIMATH verification.

Key principle: ONLY verified results enter the knowledge base.
No unverified LLM claims are trusted.
"""

from .conjecture_generator import ConjectureGenerator
from .verification_bridge import VerificationBridge, VerificationResult
from .theorem_store import TheoremStore
from .knowledge_exporter import KnowledgeExporter
from .discovery_engine import DiscoveryEngine

__all__ = [
    'ConjectureGenerator',
    'VerificationBridge',
    'VerificationResult',
    'TheoremStore',
    'KnowledgeExporter',
    'DiscoveryEngine',
]
