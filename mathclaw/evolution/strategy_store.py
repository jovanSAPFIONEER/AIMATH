"""
Strategy Store - Manages mathematical discovery strategies.

Strategies are TEXT-ONLY templates that guide conjecture generation.
They can be evolved (mutated) but NEVER contain executable code.

This module is EVOLVABLE - the evolution engine can modify strategy
weights and add new text-based strategies.
"""

import json
import sqlite3
from pathlib import Path
from typing import List, Optional, Dict
from dataclasses import dataclass, field
from datetime import datetime
import random


@dataclass
class Strategy:
    """A mathematical discovery strategy."""
    id: str
    name: str
    description: str
    prompt_template: str  # TEXT ONLY - no code
    domain: str  # 'algebra', 'calculus', 'number_theory', etc.
    success_rate: float = 0.0
    total_uses: int = 0
    total_successes: int = 0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    is_baseline: bool = False  # Baseline strategies can't be deleted


class StrategyStore:
    """
    Stores and manages discovery strategies.
    
    Strategies are prompt templates that guide the LLM to generate
    mathematical conjectures. They are pure TEXT - no code.
    
    Example:
        >>> store = StrategyStore()
        >>> strategy = store.get_random_strategy(domain='calculus')
        >>> prompt = strategy.prompt_template.format(
        ...     known_theorems=["sin^2 + cos^2 = 1"],
        ...     target_domain="trigonometry"
        ... )
    """
    
    # Default baseline strategies that always exist
    BASELINE_STRATEGIES = [
        Strategy(
            id="generalize_identity",
            name="Generalize Known Identity",
            description="Take a known identity and generalize parameters",
            prompt_template="""You are a mathematical researcher. Your task is to propose NEW mathematical identities.

Known identity: {known_identity}

Generate a plausible GENERALIZATION of this identity by:
1. Replacing specific numbers with variables
2. Adding parameters
3. Extending to related functions

Output ONLY the mathematical identity in the form: LHS = RHS
Use standard mathematical notation (sin, cos, exp, log, sqrt, etc.)

Example input: sin(x)^2 + cos(x)^2 = 1
Example output: sin(n*x)^2 + cos(n*x)^2 = 1

Your generalized identity:""",
            domain="general",
            is_baseline=True,
        ),
        Strategy(
            id="combine_functions",
            name="Combine Functions",
            description="Create identities by combining different function types",
            prompt_template="""You are exploring mathematical relationships between different functions.

Function types to combine: {function_types}

Generate a plausible identity that relates these function types.
The identity should be non-trivial but mathematically plausible.

Output ONLY the identity in the form: LHS = RHS

Examples:
- exp(i*x) = cos(x) + i*sin(x)
- log(x*y) = log(x) + log(y)
- sin(2*x) = 2*sin(x)*cos(x)

Your proposed identity:""",
            domain="general",
            is_baseline=True,
        ),
        Strategy(
            id="series_pattern",
            name="Series Pattern Discovery",
            description="Find patterns in infinite series and products",
            prompt_template="""You are investigating infinite series and their closed forms.

Starting pattern: {starting_series}

Generate a plausible identity relating a series to a closed form.
Consider:
- Taylor series
- Fourier series
- Zeta-like series
- Product formulas

Output ONLY the identity in the form: Sum/Product = ClosedForm

Examples:
- Sum(1/n^2, n=1..inf) = pi^2/6
- Sum(x^n/n!, n=0..inf) = exp(x)

Your proposed identity:""",
            domain="analysis",
            is_baseline=True,
        ),
        Strategy(
            id="number_theory_conjecture",
            name="Number Theory Conjecture",
            description="Generate conjectures about integers and primes",
            prompt_template="""You are exploring number-theoretic relationships.

Focus area: {focus_area}

Generate a plausible conjecture about integers, primes, or divisibility.
The conjecture should be:
- Precise and testable
- Non-trivial
- Related to known number theory

Output the conjecture as a mathematical statement.

Examples:
- For all primes p > 2: p = 4k+1 or p = 4k+3
- gcd(n, m) * lcm(n, m) = n * m
- For n > 1: Sum of divisors of n > n (for abundant numbers)

Your conjecture:""",
            domain="number_theory",
            is_baseline=True,
        ),
        Strategy(
            id="trig_identity",
            name="Trigonometric Identity",
            description="Discover trigonometric identities",
            prompt_template="""You are exploring trigonometric relationships.

Known identities for inspiration:
{known_identities}

Generate a NEW trigonometric identity involving:
- sin, cos, tan, cot, sec, csc
- Multiple angles (2x, 3x, nx)
- Sum/difference formulas
- Product-to-sum or sum-to-product

Output ONLY the identity in the form: LHS = RHS

Your proposed identity:""",
            domain="trigonometry",
            is_baseline=True,
        ),
        Strategy(
            id="integral_closed_form",
            name="Integral Closed Form",
            description="Find closed forms for definite integrals",
            prompt_template="""You are searching for beautiful integral identities.

Inspiration: {inspiration}

Generate a plausible definite integral with a closed-form result.
The integral should be:
- Convergent
- Have a "nice" closed form involving pi, e, log, etc.

Output in the form: Integral(f(x), x, a, b) = result

Examples:
- Integral(exp(-x^2), x, 0, inf) = sqrt(pi)/2
- Integral(sin(x)/x, x, 0, inf) = pi/2
- Integral(log(x)/(1+x^2), x, 0, 1) = -pi*log(2)/4

Your integral identity:""",
            domain="calculus",
            is_baseline=True,
        ),
    ]
    
    def __init__(self, db_path: Path = None):
        """
        Initialize strategy store.
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path or Path(__file__).parent.parent.parent / 'mathclaw.db'
        self._init_database()
        self._ensure_baseline_strategies()
    
    def _init_database(self) -> None:
        """Initialize the database schema."""
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS strategies (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                prompt_template TEXT NOT NULL,
                domain TEXT NOT NULL,
                success_rate REAL DEFAULT 0.0,
                total_uses INTEGER DEFAULT 0,
                total_successes INTEGER DEFAULT 0,
                created_at TEXT,
                is_baseline INTEGER DEFAULT 0
            )
        ''')
        conn.commit()
        conn.close()
    
    def _ensure_baseline_strategies(self) -> None:
        """Ensure all baseline strategies exist."""
        for strategy in self.BASELINE_STRATEGIES:
            if not self.get_strategy(strategy.id):
                self.add_strategy(strategy)
    
    def add_strategy(self, strategy: Strategy) -> None:
        """Add a new strategy."""
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            INSERT OR REPLACE INTO strategies 
            (id, name, description, prompt_template, domain, success_rate, 
             total_uses, total_successes, created_at, is_baseline)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            strategy.id,
            strategy.name,
            strategy.description,
            strategy.prompt_template,
            strategy.domain,
            strategy.success_rate,
            strategy.total_uses,
            strategy.total_successes,
            strategy.created_at,
            1 if strategy.is_baseline else 0,
        ))
        conn.commit()
        conn.close()
    
    def get_strategy(self, strategy_id: str) -> Optional[Strategy]:
        """Get a strategy by ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            'SELECT * FROM strategies WHERE id = ?',
            (strategy_id,)
        )
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return self._row_to_strategy(row)
        return None
    
    def get_all_strategies(self, domain: str = None) -> List[Strategy]:
        """Get all strategies, optionally filtered by domain."""
        conn = sqlite3.connect(self.db_path)
        
        if domain:
            cursor = conn.execute(
                'SELECT * FROM strategies WHERE domain = ? OR domain = "general"',
                (domain,)
            )
        else:
            cursor = conn.execute('SELECT * FROM strategies')
        
        rows = cursor.fetchall()
        conn.close()
        
        return [self._row_to_strategy(row) for row in rows]
    
    def get_random_strategy(self, domain: str = None, weighted: bool = True) -> Strategy:
        """
        Get a random strategy.
        
        Args:
            domain: Optional domain filter
            weighted: If True, weight by success rate
            
        Returns:
            A randomly selected strategy
        """
        strategies = self.get_all_strategies(domain)
        
        if not strategies:
            # Return a baseline strategy
            return self.BASELINE_STRATEGIES[0]
        
        if weighted and any(s.total_uses > 0 for s in strategies):
            # Weight by success rate (with exploration bonus for unused)
            weights = []
            for s in strategies:
                if s.total_uses == 0:
                    # Exploration bonus for unused strategies
                    weights.append(0.5)
                else:
                    # Success rate + small baseline
                    weights.append(s.success_rate + 0.1)
            
            return random.choices(strategies, weights=weights, k=1)[0]
        else:
            return random.choice(strategies)
    
    def record_result(self, strategy_id: str, success: bool) -> None:
        """
        Record the result of using a strategy.
        
        Args:
            strategy_id: ID of strategy used
            success: Whether conjecture was proven
        """
        conn = sqlite3.connect(self.db_path)
        
        # Update counts
        conn.execute('''
            UPDATE strategies 
            SET total_uses = total_uses + 1,
                total_successes = total_successes + ?
            WHERE id = ?
        ''', (1 if success else 0, strategy_id))
        
        # Update success rate
        conn.execute('''
            UPDATE strategies
            SET success_rate = CAST(total_successes AS REAL) / total_uses
            WHERE id = ? AND total_uses > 0
        ''', (strategy_id,))
        
        conn.commit()
        conn.close()
    
    def delete_strategy(self, strategy_id: str) -> bool:
        """
        Delete a strategy (baseline strategies cannot be deleted).
        
        Returns:
            True if deleted, False if baseline or not found
        """
        strategy = self.get_strategy(strategy_id)
        
        if not strategy or strategy.is_baseline:
            return False
        
        conn = sqlite3.connect(self.db_path)
        conn.execute('DELETE FROM strategies WHERE id = ?', (strategy_id,))
        conn.commit()
        conn.close()
        
        return True
    
    def get_stats(self) -> Dict:
        """Get strategy statistics."""
        strategies = self.get_all_strategies()
        
        return {
            'total_strategies': len(strategies),
            'baseline_count': sum(1 for s in strategies if s.is_baseline),
            'evolved_count': sum(1 for s in strategies if not s.is_baseline),
            'total_uses': sum(s.total_uses for s in strategies),
            'total_successes': sum(s.total_successes for s in strategies),
            'avg_success_rate': (
                sum(s.success_rate for s in strategies) / len(strategies)
                if strategies else 0
            ),
            'best_strategy': max(strategies, key=lambda s: s.success_rate).name if strategies else None,
            'domains': list(set(s.domain for s in strategies)),
        }
    
    def _row_to_strategy(self, row) -> Strategy:
        """Convert database row to Strategy object."""
        return Strategy(
            id=row[0],
            name=row[1],
            description=row[2],
            prompt_template=row[3],
            domain=row[4],
            success_rate=row[5],
            total_uses=row[6],
            total_successes=row[7],
            created_at=row[8],
            is_baseline=bool(row[9]),
        )
