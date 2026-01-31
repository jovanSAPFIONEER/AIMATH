"""
Rate Limiter - Protect against runaway API costs and resource exhaustion.

This module implements token bucket rate limiting for API calls and
resource-intensive operations.

FROZEN: This file must NEVER be modified by the evolution engine.
"""

import time
import threading
from typing import Optional, Dict
from dataclasses import dataclass, field
from datetime import datetime, date
import json
from pathlib import Path


class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded."""
    
    def __init__(self, message: str, retry_after: float = 0):
        self.message = message
        self.retry_after = retry_after
        super().__init__(f"Rate limit: {message} (retry after {retry_after:.1f}s)")


@dataclass
class UsageStats:
    """Track usage statistics."""
    requests_today: int = 0
    tokens_today: int = 0
    cost_today: float = 0.0
    last_request: Optional[float] = None
    date: str = field(default_factory=lambda: str(date.today()))
    
    def reset_if_new_day(self):
        """Reset counters if it's a new day."""
        today = str(date.today())
        if self.date != today:
            self.requests_today = 0
            self.tokens_today = 0
            self.cost_today = 0.0
            self.date = today


class RateLimiter:
    """
    Token bucket rate limiter with daily budgets.
    
    Features:
    - Requests per minute limiting
    - Daily request/token/cost budgets
    - Exponential backoff on errors
    - Persistent usage tracking
    
    Example:
        >>> limiter = RateLimiter(requests_per_minute=10, daily_budget=1.0)
        >>> 
        >>> # Before each API call:
        >>> limiter.acquire()  # Blocks if rate limited
        >>> 
        >>> # After API call:
        >>> limiter.record_usage(tokens=1500, cost=0.002)
    """
    
    # Cost estimates per 1K tokens (USD) - conservative estimates
    COST_PER_1K_TOKENS = {
        'openai': 0.002,      # GPT-4 average
        'anthropic': 0.003,   # Claude average  
        'gemini': 0.001,      # Gemini Pro
        'ollama': 0.0,        # Local, free
        'default': 0.002,
    }
    
    def __init__(
        self,
        requests_per_minute: int = 10,
        daily_request_limit: int = 1000,
        daily_token_limit: int = 1_000_000,
        daily_budget: float = 5.0,  # USD
        storage_path: Optional[Path] = None,
    ):
        """
        Initialize rate limiter.
        
        Args:
            requests_per_minute: Max requests per minute
            daily_request_limit: Max requests per day
            daily_token_limit: Max tokens per day
            daily_budget: Max cost per day in USD
            storage_path: Path to persist usage stats
        """
        self.requests_per_minute = requests_per_minute
        self.daily_request_limit = daily_request_limit
        self.daily_token_limit = daily_token_limit
        self.daily_budget = daily_budget
        
        # Token bucket state
        self.tokens = float(requests_per_minute)
        self.last_refill = time.time()
        self.refill_rate = requests_per_minute / 60.0  # tokens per second
        
        # Backoff state
        self.consecutive_errors = 0
        self.backoff_until = 0.0
        
        # Usage tracking
        self.stats = UsageStats()
        
        # Persistence
        self.storage_path = storage_path
        if storage_path:
            self._load_stats()
        
        # Thread safety
        self._lock = threading.Lock()
    
    def acquire(self, timeout: float = 60.0) -> bool:
        """
        Acquire permission to make a request.
        
        Blocks until a token is available or timeout.
        
        Args:
            timeout: Max time to wait in seconds
            
        Returns:
            True if acquired, False if timeout
            
        Raises:
            RateLimitExceeded: If daily budget exceeded
        """
        start_time = time.time()
        
        while True:
            with self._lock:
                # Check daily limits
                self.stats.reset_if_new_day()
                
                if self.stats.requests_today >= self.daily_request_limit:
                    raise RateLimitExceeded(
                        f"Daily request limit ({self.daily_request_limit}) exceeded",
                        retry_after=self._seconds_until_midnight()
                    )
                
                if self.stats.cost_today >= self.daily_budget:
                    raise RateLimitExceeded(
                        f"Daily budget (${self.daily_budget:.2f}) exceeded",
                        retry_after=self._seconds_until_midnight()
                    )
                
                # Check backoff
                now = time.time()
                if now < self.backoff_until:
                    wait_time = self.backoff_until - now
                    if time.time() - start_time + wait_time > timeout:
                        return False
                    # Release lock while waiting
                else:
                    # Refill tokens
                    self._refill_tokens()
                    
                    if self.tokens >= 1.0:
                        self.tokens -= 1.0
                        self.stats.requests_today += 1
                        self.stats.last_request = now
                        return True
            
            # Wait before retrying
            elapsed = time.time() - start_time
            if elapsed >= timeout:
                return False
            
            # Sleep a bit before checking again
            sleep_time = min(0.1, timeout - elapsed)
            time.sleep(sleep_time)
    
    def record_usage(
        self,
        tokens: int = 0,
        cost: float = None,
        provider: str = 'default',
        success: bool = True,
    ) -> None:
        """
        Record usage after an API call.
        
        Args:
            tokens: Number of tokens used
            cost: Actual cost (if known), otherwise estimated
            provider: LLM provider name
            success: Whether the call succeeded
        """
        with self._lock:
            self.stats.reset_if_new_day()
            
            # Record tokens
            self.stats.tokens_today += tokens
            
            # Record cost
            if cost is not None:
                self.stats.cost_today += cost
            elif tokens > 0:
                # Estimate cost
                rate = self.COST_PER_1K_TOKENS.get(provider, self.COST_PER_1K_TOKENS['default'])
                estimated_cost = (tokens / 1000) * rate
                self.stats.cost_today += estimated_cost
            
            # Handle success/failure for backoff
            if success:
                self.consecutive_errors = 0
                self.backoff_until = 0.0
            else:
                self.consecutive_errors += 1
                # Exponential backoff: 1s, 2s, 4s, 8s, ... max 60s
                backoff = min(60, 2 ** (self.consecutive_errors - 1))
                self.backoff_until = time.time() + backoff
            
            # Persist stats
            if self.storage_path:
                self._save_stats()
    
    def get_stats(self) -> Dict:
        """Get current usage statistics."""
        with self._lock:
            self.stats.reset_if_new_day()
            return {
                'requests_today': self.stats.requests_today,
                'tokens_today': self.stats.tokens_today,
                'cost_today': round(self.stats.cost_today, 4),
                'daily_request_limit': self.daily_request_limit,
                'daily_token_limit': self.daily_token_limit,
                'daily_budget': self.daily_budget,
                'requests_remaining': self.daily_request_limit - self.stats.requests_today,
                'budget_remaining': round(self.daily_budget - self.stats.cost_today, 4),
                'tokens_available': self.tokens,
                'in_backoff': time.time() < self.backoff_until,
                'consecutive_errors': self.consecutive_errors,
            }
    
    def wait_if_needed(self) -> float:
        """
        Calculate and wait for rate limit if needed.
        
        Returns:
            Time waited in seconds
        """
        with self._lock:
            now = time.time()
            
            # Check backoff first
            if now < self.backoff_until:
                wait_time = self.backoff_until - now
                time.sleep(wait_time)
                return wait_time
            
            # Check token bucket
            self._refill_tokens()
            
            if self.tokens < 1.0:
                # Need to wait for refill
                wait_time = (1.0 - self.tokens) / self.refill_rate
                time.sleep(wait_time)
                return wait_time
            
            return 0.0
    
    def reset_backoff(self) -> None:
        """Manually reset backoff state."""
        with self._lock:
            self.consecutive_errors = 0
            self.backoff_until = 0.0
    
    def _refill_tokens(self) -> None:
        """Refill token bucket based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        
        # Add tokens based on time elapsed
        self.tokens = min(
            float(self.requests_per_minute),  # Cap at max
            self.tokens + elapsed * self.refill_rate
        )
        
        self.last_refill = now
    
    def _seconds_until_midnight(self) -> float:
        """Calculate seconds until midnight (budget reset)."""
        now = datetime.now()
        midnight = datetime(now.year, now.month, now.day) + timedelta(days=1)
        return (midnight - now).total_seconds()
    
    def _load_stats(self) -> None:
        """Load stats from persistent storage."""
        try:
            if self.storage_path and self.storage_path.exists():
                data = json.loads(self.storage_path.read_text())
                self.stats = UsageStats(**data)
                self.stats.reset_if_new_day()
        except Exception:
            pass  # Start fresh if load fails
    
    def _save_stats(self) -> None:
        """Save stats to persistent storage."""
        try:
            if self.storage_path:
                self.storage_path.parent.mkdir(parents=True, exist_ok=True)
                data = {
                    'requests_today': self.stats.requests_today,
                    'tokens_today': self.stats.tokens_today,
                    'cost_today': self.stats.cost_today,
                    'last_request': self.stats.last_request,
                    'date': self.stats.date,
                }
                self.storage_path.write_text(json.dumps(data, indent=2))
        except Exception:
            pass  # Don't fail on save errors


# Fix missing import
from datetime import timedelta


# ═══════════════════════════════════════════════════════════════
# Convenience functions
# ═══════════════════════════════════════════════════════════════

_default_limiter = None

def get_limiter(
    requests_per_minute: int = 10,
    daily_budget: float = 5.0,
) -> RateLimiter:
    """Get or create default rate limiter."""
    global _default_limiter
    if _default_limiter is None:
        storage = Path(__file__).parent.parent.parent / '.mathclaw' / 'rate_limit_stats.json'
        _default_limiter = RateLimiter(
            requests_per_minute=requests_per_minute,
            daily_budget=daily_budget,
            storage_path=storage,
        )
    return _default_limiter


def acquire_rate_limit(timeout: float = 60.0) -> bool:
    """Acquire permission to make an API call."""
    return get_limiter().acquire(timeout)


def record_api_usage(tokens: int = 0, cost: float = None, success: bool = True) -> None:
    """Record API usage."""
    get_limiter().record_usage(tokens=tokens, cost=cost, success=success)
