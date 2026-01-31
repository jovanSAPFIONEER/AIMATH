"""
Checksum Guardian - Monitor file integrity of frozen files.

This module computes and verifies checksums of all frozen files.
If any checksum mismatches, the system HALTS to prevent corruption.

FROZEN: This file must NEVER be modified by the evolution engine.
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from .frozen_registry import FrozenRegistry, get_registry


class IntegrityViolation(Exception):
    """Raised when file integrity check fails."""
    
    def __init__(self, filepath: str, expected: str, actual: str):
        self.filepath = filepath
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"INTEGRITY VIOLATION: '{filepath}' has been modified!\n"
            f"  Expected: {expected[:16]}...\n"
            f"  Actual:   {actual[:16]}..."
        )


@dataclass
class FileChecksum:
    """Checksum record for a file."""
    filepath: str
    checksum: str
    size: int
    modified: str  # ISO format timestamp
    

class ChecksumGuardian:
    """
    Monitors and verifies file integrity.
    
    On initialization, computes SHA-256 checksums of all frozen files.
    On each verify() call, recomputes and compares checksums.
    If ANY mismatch is detected, raises IntegrityViolation.
    
    Example:
        >>> guardian = ChecksumGuardian()
        >>> guardian.initialize()  # Compute baseline checksums
        >>> 
        >>> # Later, during agent loop:
        >>> guardian.verify()  # Raises if any frozen file changed
    """
    
    CHECKSUM_FILE = ".mathclaw/checksums.json"
    
    def __init__(
        self,
        registry: FrozenRegistry = None,
        base_path: Path = None,
    ):
        """
        Initialize guardian.
        
        Args:
            registry: Frozen file registry
            base_path: Project root path
        """
        self.registry = registry or get_registry()
        self.base_path = base_path or Path(__file__).parent.parent.parent
        self.checksum_path = self.base_path / self.CHECKSUM_FILE
        
        self._checksums: Dict[str, FileChecksum] = {}
        self._initialized = False
    
    def initialize(self, force: bool = False) -> int:
        """
        Initialize checksums for all frozen files.
        
        Args:
            force: If True, recompute even if checksums exist
            
        Returns:
            Number of files checksummed
        """
        # Load existing checksums if available
        if not force and self.checksum_path.exists():
            self._load_checksums()
            self._initialized = True
            return len(self._checksums)
        
        # Compute checksums for all frozen files
        frozen_files = self.registry.get_frozen_files()
        
        for filepath in frozen_files:
            if filepath.exists():
                checksum = self._compute_checksum(filepath)
                stat = filepath.stat()
                
                self._checksums[str(filepath)] = FileChecksum(
                    filepath=str(filepath),
                    checksum=checksum,
                    size=stat.st_size,
                    modified=datetime.fromtimestamp(stat.st_mtime).isoformat(),
                )
        
        # Save checksums
        self._save_checksums()
        self._initialized = True
        
        return len(self._checksums)
    
    def verify(self, halt_on_violation: bool = True) -> List[Tuple[str, str, str]]:
        """
        Verify all frozen files against stored checksums.
        
        Args:
            halt_on_violation: If True, raise exception on first violation
            
        Returns:
            List of violations: [(filepath, expected, actual), ...]
            
        Raises:
            IntegrityViolation: If halt_on_violation and violation found
        """
        if not self._initialized:
            self.initialize()
        
        violations = []
        
        for filepath_str, record in self._checksums.items():
            filepath = Path(filepath_str)
            
            if not filepath.exists():
                # File deleted - this is a violation
                if halt_on_violation:
                    raise IntegrityViolation(filepath_str, record.checksum, "FILE_DELETED")
                violations.append((filepath_str, record.checksum, "FILE_DELETED"))
                continue
            
            # Compute current checksum
            current = self._compute_checksum(filepath)
            
            if current != record.checksum:
                if halt_on_violation:
                    raise IntegrityViolation(filepath_str, record.checksum, current)
                violations.append((filepath_str, record.checksum, current))
        
        return violations
    
    def verify_single(self, filepath: str | Path) -> bool:
        """
        Verify a single file's integrity.
        
        Args:
            filepath: File to verify
            
        Returns:
            True if file matches stored checksum
        """
        filepath = Path(filepath).resolve()
        filepath_str = str(filepath)
        
        if filepath_str not in self._checksums:
            return True  # Not a tracked file
        
        if not filepath.exists():
            return False
        
        current = self._compute_checksum(filepath)
        return current == self._checksums[filepath_str].checksum
    
    def update_checksum(self, filepath: str | Path) -> None:
        """
        Update checksum for a file (USE WITH EXTREME CAUTION).
        
        This should only be used during legitimate updates by developers,
        NEVER by the evolution engine.
        
        Args:
            filepath: File to update checksum for
        """
        filepath = Path(filepath).resolve()
        
        if not filepath.exists():
            raise FileNotFoundError(f"Cannot update checksum: {filepath}")
        
        checksum = self._compute_checksum(filepath)
        stat = filepath.stat()
        
        self._checksums[str(filepath)] = FileChecksum(
            filepath=str(filepath),
            checksum=checksum,
            size=stat.st_size,
            modified=datetime.fromtimestamp(stat.st_mtime).isoformat(),
        )
        
        self._save_checksums()
    
    def get_status(self) -> Dict:
        """Get guardian status."""
        return {
            'initialized': self._initialized,
            'tracked_files': len(self._checksums),
            'checksum_file': str(self.checksum_path),
            'checksum_file_exists': self.checksum_path.exists(),
        }
    
    def _compute_checksum(self, filepath: Path) -> str:
        """Compute SHA-256 checksum of a file."""
        sha256 = hashlib.sha256()
        
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        
        return sha256.hexdigest()
    
    def _save_checksums(self) -> None:
        """Save checksums to disk."""
        self.checksum_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'version': 1,
            'created': datetime.now().isoformat(),
            'checksums': {
                k: {
                    'checksum': v.checksum,
                    'size': v.size,
                    'modified': v.modified,
                }
                for k, v in self._checksums.items()
            }
        }
        
        self.checksum_path.write_text(json.dumps(data, indent=2))
    
    def _load_checksums(self) -> None:
        """Load checksums from disk."""
        if not self.checksum_path.exists():
            return
        
        try:
            data = json.loads(self.checksum_path.read_text())
            
            for filepath, record in data.get('checksums', {}).items():
                self._checksums[filepath] = FileChecksum(
                    filepath=filepath,
                    checksum=record['checksum'],
                    size=record['size'],
                    modified=record['modified'],
                )
        except Exception:
            # Start fresh if load fails
            self._checksums = {}


# ═══════════════════════════════════════════════════════════════
# Global guardian instance
# ═══════════════════════════════════════════════════════════════

_guardian = None

def get_guardian() -> ChecksumGuardian:
    """Get the global checksum guardian."""
    global _guardian
    if _guardian is None:
        _guardian = ChecksumGuardian()
    return _guardian


def verify_integrity() -> List[Tuple[str, str, str]]:
    """Verify integrity of all frozen files."""
    guardian = get_guardian()
    if not guardian._initialized:
        guardian.initialize()
    return guardian.verify(halt_on_violation=True)


def initialize_checksums(force: bool = False) -> int:
    """Initialize or reload checksums."""
    return get_guardian().initialize(force=force)
