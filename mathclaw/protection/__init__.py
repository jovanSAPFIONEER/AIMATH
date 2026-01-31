"""
MathClaw Protection Layer

This module provides code integrity protection to prevent the evolution
engine from corrupting critical system files.

All components in this layer are FROZEN - they must never be modified.
"""

from .frozen_registry import FrozenRegistry, FrozenFileViolation
from .checksum_guardian import ChecksumGuardian, IntegrityViolation
from .rollback_manager import RollbackManager, RollbackError
from .health_checker import HealthChecker, HealthCheckFailed

__all__ = [
    'FrozenRegistry',
    'FrozenFileViolation',
    'ChecksumGuardian',
    'IntegrityViolation',
    'RollbackManager',
    'RollbackError',
    'HealthChecker',
    'HealthCheckFailed',
]
