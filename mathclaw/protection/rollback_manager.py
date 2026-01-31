"""
Rollback Manager - State recovery system.

This module provides the ability to rollback to known-good states
when the system detects corruption or failures.

FROZEN: This file must NEVER be modified by the evolution engine.
"""

import shutil
import json
from pathlib import Path
from typing import Optional, List, Dict
from dataclasses import dataclass
from datetime import datetime
import subprocess


class RollbackError(Exception):
    """Raised when rollback fails."""
    pass


@dataclass
class Snapshot:
    """Represents a system state snapshot."""
    id: str
    timestamp: str
    description: str
    git_commit: Optional[str]
    files_backed_up: int


class RollbackManager:
    """
    Manages system state snapshots and rollback.
    
    Strategy:
    1. If git is available, use git commits/branches for rollback
    2. Otherwise, use file-based snapshots
    
    The evolution layer should create a snapshot before any changes.
    If health checks fail, rollback to the previous snapshot.
    
    Example:
        >>> manager = RollbackManager()
        >>> 
        >>> # Before making changes:
        >>> snapshot_id = manager.create_snapshot("Before strategy update")
        >>> 
        >>> # If something goes wrong:
        >>> manager.rollback(snapshot_id)
    """
    
    SNAPSHOTS_DIR = ".mathclaw/snapshots"
    MAX_SNAPSHOTS = 10
    
    def __init__(self, base_path: Path = None):
        """
        Initialize rollback manager.
        
        Args:
            base_path: Project root path
        """
        self.base_path = base_path or Path(__file__).parent.parent.parent
        self.snapshots_dir = self.base_path / self.SNAPSHOTS_DIR
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        
        self._has_git = self._check_git()
        self._snapshots: Dict[str, Snapshot] = {}
        
        self._load_snapshot_index()
    
    def _check_git(self) -> bool:
        """Check if git is available and this is a git repo."""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', '--git-dir'],
                cwd=self.base_path,
                capture_output=True,
                text=True,
            )
            return result.returncode == 0
        except Exception:
            return False
    
    def create_snapshot(self, description: str = "") -> str:
        """
        Create a snapshot of the current state.
        
        Args:
            description: Human-readable description
            
        Returns:
            Snapshot ID
        """
        timestamp = datetime.now()
        snapshot_id = timestamp.strftime("%Y%m%d_%H%M%S")
        
        git_commit = None
        files_backed_up = 0
        
        if self._has_git:
            # Use git for snapshot
            git_commit = self._git_snapshot(snapshot_id, description)
        else:
            # Use file-based snapshot
            files_backed_up = self._file_snapshot(snapshot_id)
        
        snapshot = Snapshot(
            id=snapshot_id,
            timestamp=timestamp.isoformat(),
            description=description,
            git_commit=git_commit,
            files_backed_up=files_backed_up,
        )
        
        self._snapshots[snapshot_id] = snapshot
        self._save_snapshot_index()
        self._cleanup_old_snapshots()
        
        return snapshot_id
    
    def rollback(self, snapshot_id: str = None) -> bool:
        """
        Rollback to a previous snapshot.
        
        Args:
            snapshot_id: ID of snapshot to rollback to (latest if None)
            
        Returns:
            True if rollback successful
            
        Raises:
            RollbackError: If rollback fails
        """
        if not snapshot_id:
            # Get most recent snapshot
            if not self._snapshots:
                raise RollbackError("No snapshots available")
            snapshot_id = sorted(self._snapshots.keys())[-1]
        
        if snapshot_id not in self._snapshots:
            raise RollbackError(f"Snapshot not found: {snapshot_id}")
        
        snapshot = self._snapshots[snapshot_id]
        
        if self._has_git and snapshot.git_commit:
            return self._git_rollback(snapshot.git_commit)
        else:
            return self._file_rollback(snapshot_id)
    
    def get_snapshots(self) -> List[Snapshot]:
        """Get list of available snapshots."""
        return sorted(self._snapshots.values(), key=lambda s: s.timestamp, reverse=True)
    
    def get_latest_snapshot(self) -> Optional[Snapshot]:
        """Get the most recent snapshot."""
        snapshots = self.get_snapshots()
        return snapshots[0] if snapshots else None
    
    def _git_snapshot(self, snapshot_id: str, description: str) -> str:
        """Create a git-based snapshot."""
        try:
            # Stash any uncommitted changes
            subprocess.run(
                ['git', 'stash', 'push', '-m', f'mathclaw_snapshot_{snapshot_id}'],
                cwd=self.base_path,
                capture_output=True,
            )
            
            # Get current commit hash
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                cwd=self.base_path,
                capture_output=True,
                text=True,
            )
            
            if result.returncode == 0:
                return result.stdout.strip()
            
            return None
            
        except Exception as e:
            # Fall back to file snapshot
            return None
    
    def _git_rollback(self, commit: str) -> bool:
        """Rollback using git."""
        try:
            # First, stash current changes
            subprocess.run(
                ['git', 'stash', 'push', '-m', 'mathclaw_pre_rollback'],
                cwd=self.base_path,
                capture_output=True,
            )
            
            # Reset to the commit
            result = subprocess.run(
                ['git', 'checkout', commit, '--', '.'],
                cwd=self.base_path,
                capture_output=True,
                text=True,
            )
            
            if result.returncode != 0:
                raise RollbackError(f"Git rollback failed: {result.stderr}")
            
            return True
            
        except RollbackError:
            raise
        except Exception as e:
            raise RollbackError(f"Git rollback error: {e}")
    
    def _file_snapshot(self, snapshot_id: str) -> int:
        """Create a file-based snapshot of evolvable files."""
        snapshot_dir = self.snapshots_dir / snapshot_id
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        
        # Directories that can evolve (and thus need backup)
        evolvable_dirs = [
            'mathclaw/evolution',
            'mathclaw/discovery',
        ]
        
        files_backed_up = 0
        
        for rel_dir in evolvable_dirs:
            src_dir = self.base_path / rel_dir
            if not src_dir.exists():
                continue
            
            dst_dir = snapshot_dir / rel_dir
            dst_dir.mkdir(parents=True, exist_ok=True)
            
            for py_file in src_dir.glob('*.py'):
                dst_file = dst_dir / py_file.name
                shutil.copy2(py_file, dst_file)
                files_backed_up += 1
        
        # Also backup database if exists
        db_file = self.base_path / 'mathclaw.db'
        if db_file.exists():
            shutil.copy2(db_file, snapshot_dir / 'mathclaw.db')
            files_backed_up += 1
        
        return files_backed_up
    
    def _file_rollback(self, snapshot_id: str) -> bool:
        """Rollback using file-based snapshot."""
        snapshot_dir = self.snapshots_dir / snapshot_id
        
        if not snapshot_dir.exists():
            raise RollbackError(f"Snapshot directory not found: {snapshot_dir}")
        
        # Restore evolvable directories
        evolvable_dirs = [
            'mathclaw/evolution',
            'mathclaw/discovery',
        ]
        
        for rel_dir in evolvable_dirs:
            src_dir = snapshot_dir / rel_dir
            if not src_dir.exists():
                continue
            
            dst_dir = self.base_path / rel_dir
            dst_dir.mkdir(parents=True, exist_ok=True)
            
            for py_file in src_dir.glob('*.py'):
                dst_file = dst_dir / py_file.name
                shutil.copy2(py_file, dst_file)
        
        # Restore database if in snapshot
        db_snapshot = snapshot_dir / 'mathclaw.db'
        if db_snapshot.exists():
            shutil.copy2(db_snapshot, self.base_path / 'mathclaw.db')
        
        return True
    
    def _cleanup_old_snapshots(self) -> None:
        """Remove old snapshots beyond the limit."""
        if len(self._snapshots) <= self.MAX_SNAPSHOTS:
            return
        
        # Sort by timestamp and remove oldest
        sorted_ids = sorted(self._snapshots.keys())
        to_remove = sorted_ids[:-self.MAX_SNAPSHOTS]
        
        for snapshot_id in to_remove:
            # Remove snapshot directory
            snapshot_dir = self.snapshots_dir / snapshot_id
            if snapshot_dir.exists():
                shutil.rmtree(snapshot_dir, ignore_errors=True)
            
            # Remove from index
            del self._snapshots[snapshot_id]
        
        self._save_snapshot_index()
    
    def _save_snapshot_index(self) -> None:
        """Save snapshot index to disk."""
        index_file = self.snapshots_dir / 'index.json'
        
        data = {
            'snapshots': {
                sid: {
                    'timestamp': s.timestamp,
                    'description': s.description,
                    'git_commit': s.git_commit,
                    'files_backed_up': s.files_backed_up,
                }
                for sid, s in self._snapshots.items()
            }
        }
        
        index_file.write_text(json.dumps(data, indent=2))
    
    def _load_snapshot_index(self) -> None:
        """Load snapshot index from disk."""
        index_file = self.snapshots_dir / 'index.json'
        
        if not index_file.exists():
            return
        
        try:
            data = json.loads(index_file.read_text())
            
            for sid, info in data.get('snapshots', {}).items():
                self._snapshots[sid] = Snapshot(
                    id=sid,
                    timestamp=info['timestamp'],
                    description=info.get('description', ''),
                    git_commit=info.get('git_commit'),
                    files_backed_up=info.get('files_backed_up', 0),
                )
        except Exception:
            pass  # Start fresh if load fails


# ═══════════════════════════════════════════════════════════════
# Global manager instance
# ═══════════════════════════════════════════════════════════════

_manager = None

def get_manager() -> RollbackManager:
    """Get the global rollback manager."""
    global _manager
    if _manager is None:
        _manager = RollbackManager()
    return _manager


def create_snapshot(description: str = "") -> str:
    """Create a system snapshot."""
    return get_manager().create_snapshot(description)


def rollback(snapshot_id: str = None) -> bool:
    """Rollback to a snapshot."""
    return get_manager().rollback(snapshot_id)
