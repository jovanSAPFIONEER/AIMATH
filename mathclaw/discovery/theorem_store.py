"""
Theorem Store - Persistent storage for proven mathematical results.

Only VERIFIED results enter this store.
This is the source of truth for MathClaw's discoveries.

Each theorem includes:
- The statement
- Proof method
- Timestamp
- Full verification log
"""

import sqlite3
import hashlib
import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class Theorem:
    """A proven mathematical theorem."""
    id: Optional[int]
    statement: str
    natural_language: str
    domain: str
    proof_method: str
    proof_steps: List[str]
    discovered_at: str
    strategy_id: str
    variables: List[str]
    assumptions: List[str]
    verification_time_ms: float
    checksum: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


class TheoremStore:
    """
    Persistent store for proven theorems.
    
    This is the verified knowledge base of MathClaw.
    Only results that pass verification enter this store.
    
    Features:
    - SQLite storage with checksum integrity
    - Domain-based organization
    - Search and query capabilities
    - Export to various formats
    
    Example:
        >>> store = TheoremStore()
        >>> 
        >>> # Add a proven theorem
        >>> theorem = store.add_theorem(
        ...     statement="sin(x)**2 + cos(x)**2 = 1",
        ...     natural_language="Pythagorean identity",
        ...     domain="trigonometry",
        ...     proof_method="symbolic_simplification",
        ...     proof_steps=["..."],
        ...     strategy_id="trig_identity",
        ... )
        >>> 
        >>> # Query theorems
        >>> trig = store.get_by_domain("trigonometry")
    """
    
    def __init__(self, db_path: Path = None):
        """
        Initialize the theorem store.
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path or Path(__file__).parent.parent.parent / 'mathclaw_theorems.db'
        self._init_database()
    
    def _init_database(self) -> None:
        """Initialize the database schema."""
        conn = sqlite3.connect(self.db_path)
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS theorems (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                statement TEXT NOT NULL UNIQUE,
                natural_language TEXT,
                domain TEXT,
                proof_method TEXT,
                proof_steps TEXT,
                discovered_at TEXT NOT NULL,
                strategy_id TEXT,
                variables TEXT,
                assumptions TEXT,
                verification_time_ms REAL,
                checksum TEXT NOT NULL UNIQUE
            )
        ''')
        
        # Create indexes
        conn.execute('CREATE INDEX IF NOT EXISTS idx_domain ON theorems(domain)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_proof_method ON theorems(proof_method)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_discovered ON theorems(discovered_at)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_checksum ON theorems(checksum)')
        
        conn.commit()
        conn.close()
    
    def add_theorem(
        self,
        statement: str,
        natural_language: str = None,
        domain: str = "general",
        proof_method: str = None,
        proof_steps: List[str] = None,
        strategy_id: str = None,
        variables: List[str] = None,
        assumptions: List[str] = None,
        verification_time_ms: float = 0,
    ) -> Optional[Theorem]:
        """
        Add a proven theorem to the store.
        
        Args:
            statement: The theorem statement
            natural_language: Human-readable description
            domain: Mathematical domain
            proof_method: How it was proven
            proof_steps: Steps of the proof
            strategy_id: Strategy that discovered it
            variables: Variables involved
            assumptions: Domain assumptions
            verification_time_ms: Time to verify
            
        Returns:
            Theorem object with ID, or None if duplicate
        """
        # Normalize statement
        statement = self._normalize_statement(statement)
        
        # Compute checksum
        checksum = self._compute_checksum(statement)
        
        # Check if already exists
        if self.exists(statement):
            return None
        
        discovered_at = datetime.now().isoformat()
        
        conn = sqlite3.connect(self.db_path)
        
        try:
            cursor = conn.execute('''
                INSERT INTO theorems 
                (statement, natural_language, domain, proof_method, proof_steps,
                 discovered_at, strategy_id, variables, assumptions,
                 verification_time_ms, checksum)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                statement,
                natural_language,
                domain,
                proof_method,
                json.dumps(proof_steps or []),
                discovered_at,
                strategy_id,
                json.dumps(variables or []),
                json.dumps(assumptions or []),
                verification_time_ms,
                checksum,
            ))
            
            theorem_id = cursor.lastrowid
            conn.commit()
            
            return Theorem(
                id=theorem_id,
                statement=statement,
                natural_language=natural_language or statement,
                domain=domain,
                proof_method=proof_method,
                proof_steps=proof_steps or [],
                discovered_at=discovered_at,
                strategy_id=strategy_id,
                variables=variables or [],
                assumptions=assumptions or [],
                verification_time_ms=verification_time_ms,
                checksum=checksum,
            )
            
        except sqlite3.IntegrityError:
            return None
        finally:
            conn.close()
    
    def exists(self, statement: str) -> bool:
        """Check if a theorem already exists."""
        statement = self._normalize_statement(statement)
        checksum = self._compute_checksum(statement)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            'SELECT 1 FROM theorems WHERE checksum = ? LIMIT 1',
            (checksum,)
        )
        exists = cursor.fetchone() is not None
        conn.close()
        
        return exists
    
    def get_by_id(self, theorem_id: int) -> Optional[Theorem]:
        """Get a theorem by ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            'SELECT * FROM theorems WHERE id = ?',
            (theorem_id,)
        )
        row = cursor.fetchone()
        conn.close()
        
        return self._row_to_theorem(row) if row else None
    
    def get_by_domain(self, domain: str, limit: int = 100) -> List[Theorem]:
        """Get theorems by domain."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute('''
            SELECT * FROM theorems 
            WHERE domain = ?
            ORDER BY discovered_at DESC
            LIMIT ?
        ''', (domain, limit))
        
        theorems = [self._row_to_theorem(row) for row in cursor.fetchall()]
        conn.close()
        
        return theorems
    
    def get_recent(self, limit: int = 50) -> List[Theorem]:
        """Get most recently discovered theorems."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute('''
            SELECT * FROM theorems 
            ORDER BY discovered_at DESC
            LIMIT ?
        ''', (limit,))
        
        theorems = [self._row_to_theorem(row) for row in cursor.fetchall()]
        conn.close()
        
        return theorems
    
    def search(self, query: str, limit: int = 50) -> List[Theorem]:
        """Search theorems by statement or description."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute('''
            SELECT * FROM theorems 
            WHERE statement LIKE ? OR natural_language LIKE ?
            ORDER BY discovered_at DESC
            LIMIT ?
        ''', (f'%{query}%', f'%{query}%', limit))
        
        theorems = [self._row_to_theorem(row) for row in cursor.fetchall()]
        conn.close()
        
        return theorems
    
    def get_all(self, limit: int = 1000) -> List[Theorem]:
        """Get all theorems."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute('''
            SELECT * FROM theorems 
            ORDER BY discovered_at DESC
            LIMIT ?
        ''', (limit,))
        
        theorems = [self._row_to_theorem(row) for row in cursor.fetchall()]
        conn.close()
        
        return theorems
    
    def count(self) -> int:
        """Get total number of theorems."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute('SELECT COUNT(*) FROM theorems')
        count = cursor.fetchone()[0]
        conn.close()
        return count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about stored theorems."""
        conn = sqlite3.connect(self.db_path)
        
        # Total count
        cursor = conn.execute('SELECT COUNT(*) FROM theorems')
        total = cursor.fetchone()[0]
        
        # By domain
        cursor = conn.execute('''
            SELECT domain, COUNT(*) FROM theorems GROUP BY domain
        ''')
        by_domain = {row[0]: row[1] for row in cursor.fetchall()}
        
        # By proof method
        cursor = conn.execute('''
            SELECT proof_method, COUNT(*) FROM theorems GROUP BY proof_method
        ''')
        by_method = {row[0]: row[1] for row in cursor.fetchall()}
        
        # By strategy
        cursor = conn.execute('''
            SELECT strategy_id, COUNT(*) FROM theorems GROUP BY strategy_id
        ''')
        by_strategy = {row[0]: row[1] for row in cursor.fetchall()}
        
        # Average verification time
        cursor = conn.execute('SELECT AVG(verification_time_ms) FROM theorems')
        avg_time = cursor.fetchone()[0] or 0
        
        # Most recent
        cursor = conn.execute('''
            SELECT discovered_at FROM theorems 
            ORDER BY discovered_at DESC LIMIT 1
        ''')
        most_recent_row = cursor.fetchone()
        most_recent = most_recent_row[0] if most_recent_row else None
        
        conn.close()
        
        return {
            'total_theorems': total,
            'by_domain': by_domain,
            'by_proof_method': by_method,
            'by_strategy': by_strategy,
            'avg_verification_time_ms': round(avg_time, 2),
            'most_recent_discovery': most_recent,
        }
    
    def _normalize_statement(self, statement: str) -> str:
        """Normalize a theorem statement for comparison."""
        # Remove extra whitespace
        import re
        statement = re.sub(r'\s+', ' ', statement.strip())
        
        # Normalize spacing around operators
        statement = re.sub(r'\s*=\s*', ' = ', statement)
        statement = re.sub(r'\s*\+\s*', ' + ', statement)
        statement = re.sub(r'\s*-\s*', ' - ', statement)
        statement = re.sub(r'\s*\*\s*', '*', statement)
        
        return statement.strip()
    
    def _compute_checksum(self, statement: str) -> str:
        """Compute checksum for a theorem statement."""
        return hashlib.sha256(statement.encode()).hexdigest()
    
    def _row_to_theorem(self, row) -> Theorem:
        """Convert database row to Theorem object."""
        return Theorem(
            id=row[0],
            statement=row[1],
            natural_language=row[2],
            domain=row[3],
            proof_method=row[4],
            proof_steps=json.loads(row[5]) if row[5] else [],
            discovered_at=row[6],
            strategy_id=row[7],
            variables=json.loads(row[8]) if row[8] else [],
            assumptions=json.loads(row[9]) if row[9] else [],
            verification_time_ms=row[10],
            checksum=row[11],
        )
