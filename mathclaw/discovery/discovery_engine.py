"""
Discovery Engine - The autonomous discovery orchestrator.

This is the heart of MathClaw's autonomous operation.
It coordinates:
- Conjecture generation (LLM)
- Verification (AIMATH)
- Learning (Evolution layer)
- Storage (Theorem store)

Key principle: Continuous learning from verified results.
"""

import time
import threading
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class EngineState(Enum):
    """State of the discovery engine."""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"


@dataclass
class DiscoveryStats:
    """Statistics for the current discovery session."""
    session_start: str
    total_attempts: int
    proven: int
    disproven: int
    plausible: int
    errors: int
    total_time_seconds: float
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        return self.proven / self.total_attempts if self.total_attempts > 0 else 0


class DiscoveryEngine:
    """
    Autonomous mathematical discovery engine.
    
    Orchestrates the discovery loop:
    1. Select domain and strategy
    2. Generate conjecture using LLM
    3. Verify using AIMATH
    4. Store if proven
    5. Update strategy weights
    6. Repeat
    
    Example:
        >>> from mathclaw.providers import OpenAIProvider
        >>> 
        >>> engine = DiscoveryEngine(
        ...     llm_provider=OpenAIProvider(api_key="..."),
        ... )
        >>> 
        >>> # Start autonomous discovery
        >>> engine.start()
        >>> 
        >>> # Check status
        >>> print(engine.stats)
        >>> 
        >>> # Stop
        >>> engine.stop()
    """
    
    # Default configuration
    DEFAULT_CONFIG = {
        'min_interval_seconds': 2,      # Minimum time between attempts
        'max_attempts_per_hour': 100,   # Rate limit
        'verification_timeout': 30,     # Seconds
        'batch_size': 1,                # Conjectures per iteration
        'checkpoint_interval': 10,      # Attempts between checkpoints
    }
    
    def __init__(
        self,
        llm_provider,
        config: Dict[str, Any] = None,
        db_path: Path = None,
    ):
        """
        Initialize the discovery engine.
        
        Args:
            llm_provider: LLM provider for conjecture generation
            config: Configuration overrides
            db_path: Path to SQLite database
        """
        from ..evolution import StrategyStore, DomainSelector, SuccessTracker
        from .conjecture_generator import ConjectureGenerator
        from .verification_bridge import VerificationBridge, VerificationStatus
        from .theorem_store import TheoremStore
        
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        self.db_path = db_path or Path(__file__).parent.parent.parent / 'mathclaw.db'
        
        # Initialize components
        self.strategy_store = StrategyStore(self.db_path)
        self.domain_selector = DomainSelector(self.db_path)
        self.success_tracker = SuccessTracker(self.db_path)
        self.theorem_store = TheoremStore()
        self.verification_bridge = VerificationBridge()
        
        self.conjecture_generator = ConjectureGenerator(
            llm_provider=llm_provider,
            strategy_store=self.strategy_store,
            domain_selector=self.domain_selector,
        )
        
        # Engine state
        self._state = EngineState.IDLE
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        
        # Session stats
        self._stats = None
        self._callbacks: Dict[str, Callable] = {}
    
    @property
    def state(self) -> EngineState:
        """Get current engine state."""
        return self._state
    
    @property
    def stats(self) -> Optional[DiscoveryStats]:
        """Get current session statistics."""
        return self._stats
    
    def start(self, blocking: bool = False) -> None:
        """
        Start autonomous discovery.
        
        Args:
            blocking: If True, run in current thread
        """
        if self._state == EngineState.RUNNING:
            return
        
        self._stop_event.clear()
        self._pause_event.set()  # Not paused
        self._state = EngineState.RUNNING
        
        # Initialize session stats
        from datetime import datetime
        self._stats = DiscoveryStats(
            session_start=datetime.now().isoformat(),
            total_attempts=0,
            proven=0,
            disproven=0,
            plausible=0,
            errors=0,
            total_time_seconds=0,
        )
        
        if blocking:
            self._discovery_loop()
        else:
            self._thread = threading.Thread(target=self._discovery_loop)
            self._thread.daemon = True
            self._thread.start()
    
    def stop(self) -> None:
        """Stop autonomous discovery."""
        self._state = EngineState.STOPPING
        self._stop_event.set()
        
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=10)
        
        self._state = EngineState.IDLE
    
    def pause(self) -> None:
        """Pause discovery."""
        self._pause_event.clear()
        self._state = EngineState.PAUSED
    
    def resume(self) -> None:
        """Resume discovery."""
        self._pause_event.set()
        self._state = EngineState.RUNNING
    
    def on_discovery(self, callback: Callable) -> None:
        """Register callback for new discoveries."""
        self._callbacks['discovery'] = callback
    
    def on_attempt(self, callback: Callable) -> None:
        """Register callback for each attempt."""
        self._callbacks['attempt'] = callback
    
    def _discovery_loop(self) -> None:
        """Main discovery loop."""
        from .verification_bridge import VerificationStatus
        from ..evolution.success_tracker import DiscoveryStatus
        
        start_time = time.time()
        attempts_this_hour = 0
        hour_start = time.time()
        
        while not self._stop_event.is_set():
            # Wait if paused
            self._pause_event.wait()
            
            if self._stop_event.is_set():
                break
            
            # Rate limiting
            if attempts_this_hour >= self.config['max_attempts_per_hour']:
                time_elapsed = time.time() - hour_start
                if time_elapsed < 3600:
                    time.sleep(3600 - time_elapsed)
                hour_start = time.time()
                attempts_this_hour = 0
            
            try:
                # Run one discovery iteration
                self._run_iteration()
                
                attempts_this_hour += 1
                self._stats.total_attempts += 1
                self._stats.total_time_seconds = time.time() - start_time
                
            except Exception as e:
                self._stats.errors += 1
                print(f"[DiscoveryEngine] Error in iteration: {e}")
            
            # Checkpoint
            if self._stats.total_attempts % self.config['checkpoint_interval'] == 0:
                self._save_checkpoint()
            
            # Minimum interval between attempts
            time.sleep(self.config['min_interval_seconds'])
    
    def _run_iteration(self) -> None:
        """Run a single discovery iteration."""
        from .verification_bridge import VerificationStatus
        from ..evolution.success_tracker import DiscoveryStatus
        
        # Generate conjecture
        conjecture = self.conjecture_generator.generate(
            context={
                'recent_theorems': [
                    t.statement for t in self.theorem_store.get_recent(5)
                ]
            }
        )
        
        if not conjecture:
            return
        
        # Fire attempt callback
        if 'attempt' in self._callbacks:
            self._callbacks['attempt'](conjecture)
        
        # Verify
        result = self.verification_bridge.verify(
            conjecture.statement,
            variables=conjecture.variables,
            assumptions=conjecture.assumptions,
            timeout=self.config['verification_timeout'],
        )
        
        # Map verification status to discovery status
        status_map = {
            VerificationStatus.PROVEN: DiscoveryStatus.PROVEN,
            VerificationStatus.DISPROVEN: DiscoveryStatus.DISPROVEN,
            VerificationStatus.PLAUSIBLE: DiscoveryStatus.PLAUSIBLE,
            VerificationStatus.UNKNOWN: DiscoveryStatus.UNKNOWN,
            VerificationStatus.ERROR: DiscoveryStatus.ERROR,
            VerificationStatus.TIMEOUT: DiscoveryStatus.ERROR,
        }
        
        discovery_status = status_map.get(result.status, DiscoveryStatus.UNKNOWN)
        
        # Record the attempt
        self.success_tracker.record_attempt(
            conjecture=conjecture.statement,
            status=discovery_status,
            strategy_id=conjecture.strategy_id,
            domain=conjecture.domain,
            verification_method=result.method,
            counterexample=result.counterexample,
            execution_time_ms=result.execution_time_ms,
        )
        
        # Update stats
        if result.status == VerificationStatus.PROVEN:
            self._stats.proven += 1
            
            # Store the theorem
            self.theorem_store.add_theorem(
                statement=conjecture.statement,
                natural_language=conjecture.natural_language,
                domain=conjecture.domain,
                proof_method=result.method,
                proof_steps=result.proof_steps,
                strategy_id=conjecture.strategy_id,
                variables=conjecture.variables,
                assumptions=conjecture.assumptions,
                verification_time_ms=result.execution_time_ms,
            )
            
            # Fire discovery callback
            if 'discovery' in self._callbacks:
                self._callbacks['discovery'](conjecture, result)
            
            # Update strategy success
            self.strategy_store.record_result(
                conjecture.strategy_id,
                success=True,
                domain=conjecture.domain,
            )
            
            # Update domain success
            self.domain_selector.update_success_rate(
                conjecture.domain,
                success=True,
            )
            
        elif result.status == VerificationStatus.DISPROVEN:
            self._stats.disproven += 1
            
            self.strategy_store.record_result(
                conjecture.strategy_id,
                success=False,
                domain=conjecture.domain,
            )
            
            self.domain_selector.update_success_rate(
                conjecture.domain,
                success=False,
            )
            
        elif result.status == VerificationStatus.PLAUSIBLE:
            self._stats.plausible += 1
            
        else:
            self._stats.errors += 1
    
    def _save_checkpoint(self) -> None:
        """Save checkpoint state."""
        # The SQLite databases already persist state
        # This is for any additional state tracking
        pass
    
    def run_once(self) -> Dict[str, Any]:
        """
        Run a single discovery attempt.
        
        Useful for testing or manual operation.
        
        Returns:
            Dict with attempt details
        """
        from .verification_bridge import VerificationStatus
        
        # Generate
        conjecture = self.conjecture_generator.generate()
        
        if not conjecture:
            return {'success': False, 'error': 'Failed to generate conjecture'}
        
        # Verify
        result = self.verification_bridge.verify(
            conjecture.statement,
            variables=conjecture.variables,
            timeout=self.config['verification_timeout'],
        )
        
        # Store if proven
        if result.status == VerificationStatus.PROVEN:
            self.theorem_store.add_theorem(
                statement=conjecture.statement,
                natural_language=conjecture.natural_language,
                domain=conjecture.domain,
                proof_method=result.method,
                proof_steps=result.proof_steps,
                strategy_id=conjecture.strategy_id,
                variables=conjecture.variables,
                verification_time_ms=result.execution_time_ms,
            )
        
        return {
            'success': True,
            'conjecture': conjecture.statement,
            'domain': conjecture.domain,
            'strategy': conjecture.strategy_id,
            'verification_status': result.status.value,
            'proof_method': result.method,
            'counterexample': result.counterexample,
            'execution_time_ms': result.execution_time_ms,
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get engine status."""
        return {
            'state': self._state.value,
            'session_stats': {
                'total_attempts': self._stats.total_attempts if self._stats else 0,
                'proven': self._stats.proven if self._stats else 0,
                'success_rate': self._stats.success_rate if self._stats else 0,
                'runtime_seconds': self._stats.total_time_seconds if self._stats else 0,
            } if self._stats else None,
            'total_theorems': self.theorem_store.count(),
            'strategy_count': len(self.strategy_store.get_all_strategies()),
            'config': self.config,
        }
