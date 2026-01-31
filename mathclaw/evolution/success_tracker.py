"""
Success Tracker - Records all discovery attempts and learns from them.

This module tracks every conjecture attempted, whether it succeeded or failed,
and what strategies/domains were used. This data drives evolution.

This module is EVOLVABLE - analysis methods can be improved.
"""

import sqlite3
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class DiscoveryStatus(Enum):
    """Status of a discovery attempt."""
    PROVEN = "proven"           # Verified as true
    DISPROVEN = "disproven"     # Found counterexample
    PLAUSIBLE = "plausible"     # Passed tests but not proven
    UNKNOWN = "unknown"         # Couldn't determine
    ERROR = "error"             # Processing error


@dataclass
class DiscoveryRecord:
    """Record of a discovery attempt."""
    id: Optional[int]
    conjecture: str
    status: DiscoveryStatus
    strategy_id: str
    domain: str
    verification_method: Optional[str]
    counterexample: Optional[str]
    execution_time_ms: float
    timestamp: str
    notes: Optional[str] = None


class SuccessTracker:
    """
    Tracks discovery attempts and analyzes patterns.
    
    Stores every conjecture attempt with:
    - The conjecture itself
    - Result (proven/disproven/unknown)
    - Strategy used
    - Domain explored
    - Time taken
    
    This data is used to:
    - Update strategy success rates
    - Identify promising patterns
    - Avoid repeating failures
    
    Example:
        >>> tracker = SuccessTracker()
        >>> 
        >>> # Record an attempt
        >>> record = tracker.record_attempt(
        ...     conjecture="sin(x)^2 + cos(x)^2 = 1",
        ...     status=DiscoveryStatus.PROVEN,
        ...     strategy_id="trig_identity",
        ...     domain="trigonometry",
        ...     execution_time_ms=150.5
        ... )
        >>> 
        >>> # Analyze patterns
        >>> stats = tracker.get_strategy_analysis()
    """
    
    def __init__(self, db_path: Path = None):
        """
        Initialize tracker.
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path or Path(__file__).parent.parent.parent / 'mathclaw.db'
        self._init_database()
    
    def _init_database(self) -> None:
        """Initialize the database schema."""
        conn = sqlite3.connect(self.db_path)
        
        # Discovery records table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS discovery_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conjecture TEXT NOT NULL,
                status TEXT NOT NULL,
                strategy_id TEXT,
                domain TEXT,
                verification_method TEXT,
                counterexample TEXT,
                execution_time_ms REAL,
                timestamp TEXT NOT NULL,
                notes TEXT
            )
        ''')
        
        # Proven theorems table (only verified discoveries)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS theorems (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                statement TEXT NOT NULL UNIQUE,
                proof_method TEXT,
                domain TEXT,
                discovered_at TEXT NOT NULL,
                strategy_id TEXT,
                checksum TEXT
            )
        ''')
        
        # Create indexes for common queries
        conn.execute('CREATE INDEX IF NOT EXISTS idx_status ON discovery_records(status)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_strategy ON discovery_records(strategy_id)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_domain ON discovery_records(domain)')
        
        conn.commit()
        conn.close()
    
    def record_attempt(
        self,
        conjecture: str,
        status: DiscoveryStatus,
        strategy_id: str,
        domain: str,
        verification_method: str = None,
        counterexample: str = None,
        execution_time_ms: float = 0,
        notes: str = None,
    ) -> DiscoveryRecord:
        """
        Record a discovery attempt.
        
        Args:
            conjecture: The conjecture tested
            status: Result of verification
            strategy_id: Strategy that generated it
            domain: Mathematical domain
            verification_method: How it was verified
            counterexample: If disproven, the counterexample
            execution_time_ms: Time taken
            notes: Additional notes
            
        Returns:
            DiscoveryRecord with assigned ID
        """
        timestamp = datetime.now().isoformat()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute('''
            INSERT INTO discovery_records 
            (conjecture, status, strategy_id, domain, verification_method,
             counterexample, execution_time_ms, timestamp, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            conjecture,
            status.value,
            strategy_id,
            domain,
            verification_method,
            counterexample,
            execution_time_ms,
            timestamp,
            notes,
        ))
        
        record_id = cursor.lastrowid
        conn.commit()
        
        # If proven, also add to theorems table
        if status == DiscoveryStatus.PROVEN:
            self._add_theorem(conn, conjecture, verification_method, domain, strategy_id)
        
        conn.close()
        
        return DiscoveryRecord(
            id=record_id,
            conjecture=conjecture,
            status=status,
            strategy_id=strategy_id,
            domain=domain,
            verification_method=verification_method,
            counterexample=counterexample,
            execution_time_ms=execution_time_ms,
            timestamp=timestamp,
            notes=notes,
        )
    
    def _add_theorem(
        self,
        conn: sqlite3.Connection,
        statement: str,
        proof_method: str,
        domain: str,
        strategy_id: str,
    ) -> None:
        """Add a proven theorem to the theorems table."""
        import hashlib
        
        checksum = hashlib.sha256(statement.encode()).hexdigest()
        
        try:
            conn.execute('''
                INSERT OR IGNORE INTO theorems 
                (statement, proof_method, domain, discovered_at, strategy_id, checksum)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                statement,
                proof_method,
                domain,
                datetime.now().isoformat(),
                strategy_id,
                checksum,
            ))
        except sqlite3.IntegrityError:
            pass  # Already exists
    
    def get_recent_records(self, limit: int = 100) -> List[DiscoveryRecord]:
        """Get most recent discovery records."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute('''
            SELECT id, conjecture, status, strategy_id, domain,
                   verification_method, counterexample, execution_time_ms,
                   timestamp, notes
            FROM discovery_records
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (limit,))
        
        records = [self._row_to_record(row) for row in cursor.fetchall()]
        conn.close()
        
        return records
    
    def get_proven_theorems(self, domain: str = None, limit: int = 100) -> List[Dict]:
        """Get proven theorems."""
        conn = sqlite3.connect(self.db_path)
        
        if domain:
            cursor = conn.execute('''
                SELECT statement, proof_method, domain, discovered_at, strategy_id
                FROM theorems
                WHERE domain = ?
                ORDER BY discovered_at DESC
                LIMIT ?
            ''', (domain, limit))
        else:
            cursor = conn.execute('''
                SELECT statement, proof_method, domain, discovered_at, strategy_id
                FROM theorems
                ORDER BY discovered_at DESC
                LIMIT ?
            ''', (limit,))
        
        theorems = [
            {
                'statement': row[0],
                'proof_method': row[1],
                'domain': row[2],
                'discovered_at': row[3],
                'strategy_id': row[4],
            }
            for row in cursor.fetchall()
        ]
        
        conn.close()
        return theorems
    
    def has_been_tried(self, conjecture: str) -> bool:
        """Check if a conjecture has already been tried."""
        # Normalize the conjecture for comparison
        normalized = conjecture.strip().lower()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            'SELECT 1 FROM discovery_records WHERE LOWER(conjecture) = ? LIMIT 1',
            (normalized,)
        )
        exists = cursor.fetchone() is not None
        conn.close()
        
        return exists
    
    def get_stats(self) -> Dict[str, Any]:
        """Get overall statistics."""
        conn = sqlite3.connect(self.db_path)
        
        # Total counts by status
        cursor = conn.execute('''
            SELECT status, COUNT(*) FROM discovery_records GROUP BY status
        ''')
        status_counts = {row[0]: row[1] for row in cursor.fetchall()}
        
        # Total theorems
        cursor = conn.execute('SELECT COUNT(*) FROM theorems')
        theorem_count = cursor.fetchone()[0]
        
        # Average execution time
        cursor = conn.execute('''
            SELECT AVG(execution_time_ms) FROM discovery_records
        ''')
        avg_time = cursor.fetchone()[0] or 0
        
        # Records per domain
        cursor = conn.execute('''
            SELECT domain, COUNT(*), SUM(CASE WHEN status = 'proven' THEN 1 ELSE 0 END)
            FROM discovery_records
            GROUP BY domain
        ''')
        domain_stats = {
            row[0]: {'attempts': row[1], 'proven': row[2]}
            for row in cursor.fetchall()
        }
        
        conn.close()
        
        total = sum(status_counts.values())
        proven = status_counts.get('proven', 0)
        
        return {
            'total_attempts': total,
            'proven_count': proven,
            'disproven_count': status_counts.get('disproven', 0),
            'plausible_count': status_counts.get('plausible', 0),
            'error_count': status_counts.get('error', 0),
            'success_rate': proven / total if total > 0 else 0,
            'theorem_count': theorem_count,
            'avg_execution_time_ms': round(avg_time, 2),
            'domain_stats': domain_stats,
        }
    
    def get_strategy_analysis(self) -> Dict[str, Dict]:
        """Analyze success rates by strategy."""
        conn = sqlite3.connect(self.db_path)
        
        cursor = conn.execute('''
            SELECT 
                strategy_id,
                COUNT(*) as attempts,
                SUM(CASE WHEN status = 'proven' THEN 1 ELSE 0 END) as proven,
                SUM(CASE WHEN status = 'disproven' THEN 1 ELSE 0 END) as disproven,
                AVG(execution_time_ms) as avg_time
            FROM discovery_records
            GROUP BY strategy_id
        ''')
        
        analysis = {}
        for row in cursor.fetchall():
            strategy_id = row[0]
            attempts = row[1]
            proven = row[2]
            
            analysis[strategy_id] = {
                'attempts': attempts,
                'proven': proven,
                'disproven': row[3],
                'success_rate': proven / attempts if attempts > 0 else 0,
                'avg_time_ms': round(row[4] or 0, 2),
            }
        
        conn.close()
        return analysis
    
    def export_theorems_markdown(self, output_path: Path = None) -> str:
        """Export proven theorems to markdown."""
        theorems = self.get_proven_theorems(limit=1000)
        
        lines = [
            "# MathClaw Discovered Theorems",
            f"\n*Generated: {datetime.now().isoformat()}*\n",
            f"**Total Theorems: {len(theorems)}**\n",
        ]
        
        # Group by domain
        by_domain: Dict[str, List] = {}
        for t in theorems:
            domain = t['domain'] or 'general'
            if domain not in by_domain:
                by_domain[domain] = []
            by_domain[domain].append(t)
        
        for domain, domain_theorems in sorted(by_domain.items()):
            lines.append(f"\n## {domain.replace('_', ' ').title()}\n")
            
            for t in domain_theorems:
                lines.append(f"### {t['statement']}")
                lines.append(f"- **Proof method:** {t['proof_method'] or 'N/A'}")
                lines.append(f"- **Discovered:** {t['discovered_at'][:10]}")
                lines.append(f"- **Strategy:** {t['strategy_id'] or 'N/A'}")
                lines.append("")
        
        content = '\n'.join(lines)
        
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(content)
        
        return content
    
    def _row_to_record(self, row) -> DiscoveryRecord:
        """Convert database row to DiscoveryRecord."""
        return DiscoveryRecord(
            id=row[0],
            conjecture=row[1],
            status=DiscoveryStatus(row[2]),
            strategy_id=row[3],
            domain=row[4],
            verification_method=row[5],
            counterexample=row[6],
            execution_time_ms=row[7],
            timestamp=row[8],
            notes=row[9] if len(row) > 9 else None,
        )
