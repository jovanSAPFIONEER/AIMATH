"""
Frozen Registry - Defines which files can NEVER be modified.

This module maintains a list of critical files that must remain unchanged.
The evolution engine is forbidden from modifying these files.

FROZEN: This file must NEVER be modified by the evolution engine.
"""

from pathlib import Path
from typing import List, Set
from dataclasses import dataclass


class FrozenFileViolation(Exception):
    """Raised when an attempt is made to modify a frozen file."""
    
    def __init__(self, filepath: str, operation: str = "modify"):
        self.filepath = filepath
        self.operation = operation
        super().__init__(f"FROZEN FILE VIOLATION: Cannot {operation} '{filepath}'")


@dataclass
class FrozenFile:
    """Represents a frozen file with metadata."""
    path: Path
    reason: str
    category: str  # 'security', 'verification', 'protection', 'core'


class FrozenRegistry:
    """
    Registry of files that must never be modified by the evolution engine.
    
    Philosophy: The trust layer must be immutable. If the evolution engine
    could modify verification code, it could bypass all safety checks.
    
    Frozen categories:
    1. Security layer - Input validation, safe parsing, sandboxing
    2. Verification layer - Mathematical proof checking
    3. Protection layer - This registry, checksums, rollback
    4. Core types - Data structures that everything depends on
    
    Example:
        >>> registry = FrozenRegistry()
        >>> registry.is_frozen("mathclaw/security/safe_parser.py")
        True
        >>> registry.is_frozen("mathclaw/evolution/strategy.py")
        False
    """
    
    def __init__(self, base_path: Path = None):
        """
        Initialize registry.
        
        Args:
            base_path: Root path of the project (auto-detected if None)
        """
        self.base_path = base_path or Path(__file__).parent.parent.parent
        self._frozen_files: Set[Path] = set()
        self._frozen_patterns: List[str] = []
        
        # Register all frozen files
        self._register_frozen_files()
    
    def _register_frozen_files(self) -> None:
        """Register all frozen files and patterns."""
        
        # ═══════════════════════════════════════════════════════════════
        # SECURITY LAYER - FROZEN
        # ═══════════════════════════════════════════════════════════════
        self._add_frozen_directory(
            "mathclaw/security",
            "Security layer - prevents code injection",
            "security"
        )
        
        # ═══════════════════════════════════════════════════════════════
        # PROTECTION LAYER - FROZEN
        # ═══════════════════════════════════════════════════════════════
        self._add_frozen_directory(
            "mathclaw/protection",
            "Protection layer - maintains integrity",
            "protection"
        )
        
        # ═══════════════════════════════════════════════════════════════
        # VERIFICATION LAYER (AIMATH) - FROZEN
        # ═══════════════════════════════════════════════════════════════
        self._add_frozen_directory(
            "aimath/verification",
            "Verification layer - mathematical truth checking",
            "verification"
        )
        
        # ═══════════════════════════════════════════════════════════════
        # PROOF ASSISTANT - FROZEN
        # ═══════════════════════════════════════════════════════════════
        self._add_frozen_directory(
            "aimath/proof_assistant",
            "Proof assistant - formal verification",
            "verification"
        )
        
        # ═══════════════════════════════════════════════════════════════
        # CORE TYPES - FROZEN
        # ═══════════════════════════════════════════════════════════════
        self._add_frozen_file(
            "aimath/core/types.py",
            "Core data types - foundational structures",
            "core"
        )
        
        # ═══════════════════════════════════════════════════════════════
        # SPECIFIC CRITICAL FILES
        # ═══════════════════════════════════════════════════════════════
        critical_files = [
            ("aimath/solvers/conjecture_tester.py", "Conjecture verification"),
            ("aimath/core/engine.py", "Main orchestration engine"),
        ]
        
        for filepath, reason in critical_files:
            self._add_frozen_file(filepath, reason, "core")
    
    def _add_frozen_directory(self, rel_path: str, reason: str, category: str) -> None:
        """Add all Python files in a directory as frozen."""
        dir_path = self.base_path / rel_path
        if dir_path.exists():
            for py_file in dir_path.rglob("*.py"):
                self._frozen_files.add(py_file.resolve())
        
        # Also add pattern for future files
        self._frozen_patterns.append(rel_path)
    
    def _add_frozen_file(self, rel_path: str, reason: str, category: str) -> None:
        """Add a specific file as frozen."""
        file_path = (self.base_path / rel_path).resolve()
        self._frozen_files.add(file_path)
    
    def is_frozen(self, filepath: str | Path) -> bool:
        """
        Check if a file is frozen.
        
        Args:
            filepath: Path to check (relative or absolute)
            
        Returns:
            True if file is frozen
        """
        path = Path(filepath)
        
        # Make absolute if relative
        if not path.is_absolute():
            path = (self.base_path / path).resolve()
        else:
            path = path.resolve()
        
        # Check exact match
        if path in self._frozen_files:
            return True
        
        # Check patterns
        rel_path = str(path.relative_to(self.base_path)) if self._is_under_base(path) else ""
        rel_path = rel_path.replace("\\", "/")  # Normalize for Windows
        
        for pattern in self._frozen_patterns:
            if rel_path.startswith(pattern):
                return True
        
        return False
    
    def _is_under_base(self, path: Path) -> bool:
        """Check if path is under base_path."""
        try:
            path.relative_to(self.base_path)
            return True
        except ValueError:
            return False
    
    def check_write_allowed(self, filepath: str | Path) -> None:
        """
        Check if writing to a file is allowed.
        
        Args:
            filepath: Path to check
            
        Raises:
            FrozenFileViolation: If file is frozen
        """
        if self.is_frozen(filepath):
            raise FrozenFileViolation(str(filepath), "write")
    
    def check_delete_allowed(self, filepath: str | Path) -> None:
        """
        Check if deleting a file is allowed.
        
        Args:
            filepath: Path to check
            
        Raises:
            FrozenFileViolation: If file is frozen
        """
        if self.is_frozen(filepath):
            raise FrozenFileViolation(str(filepath), "delete")
    
    def get_frozen_files(self) -> List[Path]:
        """Get list of all frozen files."""
        return sorted(list(self._frozen_files))
    
    def get_frozen_patterns(self) -> List[str]:
        """Get list of frozen directory patterns."""
        return self._frozen_patterns.copy()
    
    def summarize(self) -> dict:
        """Get summary of frozen files."""
        return {
            'total_frozen_files': len(self._frozen_files),
            'frozen_patterns': self._frozen_patterns,
            'categories': {
                'security': len([f for f in self._frozen_files if 'security' in str(f)]),
                'protection': len([f for f in self._frozen_files if 'protection' in str(f)]),
                'verification': len([f for f in self._frozen_files if 'verification' in str(f) or 'proof' in str(f)]),
                'core': len([f for f in self._frozen_files if 'core' in str(f)]),
            }
        }


# ═══════════════════════════════════════════════════════════════
# Global registry instance
# ═══════════════════════════════════════════════════════════════

_registry = None

def get_registry() -> FrozenRegistry:
    """Get the global frozen file registry."""
    global _registry
    if _registry is None:
        _registry = FrozenRegistry()
    return _registry


def is_frozen(filepath: str | Path) -> bool:
    """Check if a file is frozen."""
    return get_registry().is_frozen(filepath)


def check_write_allowed(filepath: str | Path) -> None:
    """Check if writing to a file is allowed."""
    get_registry().check_write_allowed(filepath)
