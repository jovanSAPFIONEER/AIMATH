"""
Domain Selector - Chooses which mathematical domain to explore.

This module selects which area of mathematics to focus on based on:
- Historical success rates
- Exploration vs exploitation balance
- User preferences

This module is EVOLVABLE - weights can be adjusted.
"""

import random
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path


class MathDomain(Enum):
    """Mathematical domains for exploration."""
    ALGEBRA = "algebra"
    CALCULUS = "calculus"
    TRIGONOMETRY = "trigonometry"
    NUMBER_THEORY = "number_theory"
    ANALYSIS = "analysis"
    LINEAR_ALGEBRA = "linear_algebra"
    COMBINATORICS = "combinatorics"
    GEOMETRY = "geometry"
    SPECIAL_FUNCTIONS = "special_functions"
    GENERAL = "general"


@dataclass
class DomainStats:
    """Statistics for a mathematical domain."""
    domain: MathDomain
    total_attempts: int = 0
    total_successes: int = 0
    success_rate: float = 0.0
    weight: float = 1.0  # Selection weight
    last_explored: Optional[str] = None


class DomainSelector:
    """
    Selects mathematical domains for exploration.
    
    Uses a multi-armed bandit approach:
    - Exploitation: Focus on domains with high success rates
    - Exploration: Occasionally try underexplored domains
    
    Example:
        >>> selector = DomainSelector()
        >>> domain = selector.select()
        >>> print(f"Exploring: {domain.value}")
        >>> 
        >>> # After result:
        >>> selector.record_result(domain, success=True)
    """
    
    STATS_FILE = ".mathclaw/domain_stats.json"
    
    # Exploration probability (epsilon in epsilon-greedy)
    EXPLORATION_RATE = 0.15
    
    def __init__(self, storage_path: Path = None):
        """
        Initialize domain selector.
        
        Args:
            storage_path: Path to persist statistics
        """
        self.base_path = storage_path or Path(__file__).parent.parent.parent
        self.stats_path = self.base_path / self.STATS_FILE
        
        self._stats: Dict[MathDomain, DomainStats] = {}
        self._init_stats()
        self._load_stats()
    
    def _init_stats(self) -> None:
        """Initialize stats for all domains."""
        for domain in MathDomain:
            self._stats[domain] = DomainStats(domain=domain)
    
    def select(
        self,
        exclude: List[MathDomain] = None,
        force_exploration: bool = False,
    ) -> MathDomain:
        """
        Select a domain to explore.
        
        Args:
            exclude: Domains to exclude from selection
            force_exploration: If True, always explore (ignore success rates)
            
        Returns:
            Selected domain
        """
        exclude = exclude or []
        available = [d for d in MathDomain if d not in exclude]
        
        if not available:
            available = list(MathDomain)
        
        # Epsilon-greedy selection
        if force_exploration or random.random() < self.EXPLORATION_RATE:
            # Exploration: random selection weighted by inverse attempts
            # (prefer underexplored domains)
            weights = []
            for domain in available:
                stats = self._stats[domain]
                # Inverse of attempts + 1 (so never zero)
                weight = 1.0 / (stats.total_attempts + 1)
                weights.append(weight)
            
            return random.choices(available, weights=weights, k=1)[0]
        else:
            # Exploitation: weighted by success rate
            weights = []
            for domain in available:
                stats = self._stats[domain]
                # Success rate + small baseline
                weight = stats.success_rate + 0.1
                weights.append(weight * stats.weight)
            
            return random.choices(available, weights=weights, k=1)[0]
    
    def record_result(self, domain: MathDomain, success: bool) -> None:
        """
        Record the result of exploring a domain.
        
        Args:
            domain: Domain that was explored
            success: Whether exploration was successful
        """
        from datetime import datetime
        
        stats = self._stats[domain]
        stats.total_attempts += 1
        
        if success:
            stats.total_successes += 1
        
        # Update success rate
        stats.success_rate = stats.total_successes / stats.total_attempts
        stats.last_explored = datetime.now().isoformat()
        
        self._save_stats()
    
    def get_stats(self) -> Dict[str, DomainStats]:
        """Get all domain statistics."""
        return {d.value: self._stats[d] for d in MathDomain}
    
    def get_best_domain(self) -> MathDomain:
        """Get the domain with highest success rate."""
        best = max(
            self._stats.values(),
            key=lambda s: s.success_rate if s.total_attempts > 0 else 0
        )
        return best.domain
    
    def get_least_explored(self) -> MathDomain:
        """Get the least explored domain."""
        least = min(
            self._stats.values(),
            key=lambda s: s.total_attempts
        )
        return least.domain
    
    def set_weight(self, domain: MathDomain, weight: float) -> None:
        """
        Set the selection weight for a domain.
        
        Higher weight = more likely to be selected during exploitation.
        
        Args:
            domain: Domain to adjust
            weight: New weight (must be positive)
        """
        if weight <= 0:
            raise ValueError("Weight must be positive")
        
        self._stats[domain].weight = weight
        self._save_stats()
    
    def get_summary(self) -> Dict:
        """Get a summary of domain exploration."""
        total_attempts = sum(s.total_attempts for s in self._stats.values())
        total_successes = sum(s.total_successes for s in self._stats.values())
        
        return {
            'total_attempts': total_attempts,
            'total_successes': total_successes,
            'overall_success_rate': total_successes / total_attempts if total_attempts > 0 else 0,
            'best_domain': self.get_best_domain().value if total_attempts > 0 else None,
            'least_explored': self.get_least_explored().value,
            'domains': {
                d.value: {
                    'attempts': s.total_attempts,
                    'successes': s.total_successes,
                    'success_rate': round(s.success_rate, 3),
                    'weight': s.weight,
                }
                for d, s in self._stats.items()
            }
        }
    
    def _save_stats(self) -> None:
        """Save statistics to disk."""
        self.stats_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            d.value: {
                'total_attempts': s.total_attempts,
                'total_successes': s.total_successes,
                'success_rate': s.success_rate,
                'weight': s.weight,
                'last_explored': s.last_explored,
            }
            for d, s in self._stats.items()
        }
        
        self.stats_path.write_text(json.dumps(data, indent=2))
    
    def _load_stats(self) -> None:
        """Load statistics from disk."""
        if not self.stats_path.exists():
            return
        
        try:
            data = json.loads(self.stats_path.read_text())
            
            for domain_name, stats in data.items():
                try:
                    domain = MathDomain(domain_name)
                    self._stats[domain] = DomainStats(
                        domain=domain,
                        total_attempts=stats.get('total_attempts', 0),
                        total_successes=stats.get('total_successes', 0),
                        success_rate=stats.get('success_rate', 0.0),
                        weight=stats.get('weight', 1.0),
                        last_explored=stats.get('last_explored'),
                    )
                except ValueError:
                    pass  # Unknown domain, skip
        except Exception:
            pass  # Start fresh if load fails
