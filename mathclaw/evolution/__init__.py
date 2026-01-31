"""
MathClaw Evolution Layer

This module provides controlled self-improvement capabilities.
Only TEXT-based evolution (prompts, strategies) is allowed.
Code generation is FORBIDDEN.

The evolution layer can ONLY modify:
- Prompt templates (text)
- Strategy weights (numbers)
- Domain selection (configuration)

It can NEVER modify:
- Python code
- Security layer
- Verification layer
"""

from .strategy_store import StrategyStore, Strategy
from .prompt_mutator import PromptMutator, MutationResult
from .domain_selector import DomainSelector, MathDomain
from .success_tracker import SuccessTracker, DiscoveryRecord

__all__ = [
    'StrategyStore',
    'Strategy',
    'PromptMutator',
    'MutationResult',
    'DomainSelector',
    'MathDomain',
    'SuccessTracker',
    'DiscoveryRecord',
]
