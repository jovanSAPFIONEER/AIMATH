"""
MathClaw - Autonomous Mathematical Discovery Engine

An AI system that autonomously discovers and proves mathematical theorems,
with built-in protections against hallucinations and self-corruption.

Architecture:
    ┌─────────────────────────────────────────────────────────┐
    │                     MathClaw Core                        │
    ├─────────────────────────────────────────────────────────┤
    │  Security Layer      │  Protection Layer                │
    │  ├─ InputValidator   │  ├─ FrozenRegistry               │
    │  ├─ SafeParser       │  ├─ ChecksumGuardian             │
    │  ├─ Sandbox          │  ├─ RollbackManager              │
    │  └─ RateLimiter      │  └─ HealthChecker                │
    ├─────────────────────────────────────────────────────────┤
    │  Evolution Layer     │  Discovery Layer                 │
    │  ├─ StrategyStore    │  ├─ ConjectureGenerator          │
    │  ├─ PromptMutator    │  ├─ VerificationBridge           │
    │  ├─ DomainSelector   │  ├─ TheoremStore                 │
    │  └─ SuccessTracker   │  └─ DiscoveryEngine              │
    ├─────────────────────────────────────────────────────────┤
    │  API Layer                                               │
    │  ├─ LLM Providers (OpenAI, Anthropic, Gemini, Ollama)   │
    │  ├─ Configuration                                        │
    │  └─ CLI Interface                                        │
    └─────────────────────────────────────────────────────────┘

Key Principles:
    1. NO HALLUCINATIONS: Only verified results enter the knowledge base
    2. NO SELF-CORRUPTION: Security/protection layers are immutable
    3. SAFE EVOLUTION: Only text prompts evolve, never code
    4. USER CONTROL: Uses user's own API keys, respects rate limits

Quick Start:
    from mathclaw import MathClaw
    
    # Create engine with your API key
    claw = MathClaw(openai_api_key="sk-...")
    
    # Run a single discovery
    result = claw.discover_one()
    
    # Start autonomous discovery
    claw.start()
    
    # Check status
    print(claw.status())
    
    # Export discoveries
    claw.export("discoveries.md")

CLI Usage:
    mathclaw start --provider openai
    mathclaw discover --count 5
    mathclaw theorems --limit 20
    mathclaw export --format markdown

Copyright 2024 - MathClaw Project
"""

__version__ = '1.0.0'
__author__ = 'MathClaw Project'

# High-level imports for convenience
from pathlib import Path
from typing import Optional, Dict, Any


class MathClaw:
    """
    High-level interface to MathClaw.
    
    Provides a simple API for mathematical discovery.
    
    Example:
        >>> claw = MathClaw(openai_api_key="sk-...")
        >>> 
        >>> # Single discovery
        >>> result = claw.discover_one()
        >>> print(result)
        >>> 
        >>> # Start autonomous mode
        >>> claw.start()
        >>> 
        >>> # Export results
        >>> claw.export("theorems.md")
    """
    
    def __init__(
        self,
        openai_api_key: str = None,
        anthropic_api_key: str = None,
        google_api_key: str = None,
        provider: str = "auto",
        model: str = None,
        config: Dict[str, Any] = None,
    ):
        """
        Initialize MathClaw.
        
        Args:
            openai_api_key: OpenAI API key
            anthropic_api_key: Anthropic API key
            google_api_key: Google API key
            provider: LLM provider ('openai', 'anthropic', 'gemini', 'ollama', 'auto')
            model: Model to use (optional)
            config: Additional configuration
        """
        from .api import get_provider, MathClawConfig
        from .discovery import DiscoveryEngine, TheoremStore, KnowledgeExporter
        
        # Set API keys in environment if provided
        import os
        if openai_api_key:
            os.environ['OPENAI_API_KEY'] = openai_api_key
        if anthropic_api_key:
            os.environ['ANTHROPIC_API_KEY'] = anthropic_api_key
        if google_api_key:
            os.environ['GOOGLE_API_KEY'] = google_api_key
        
        # Load config
        self.config = MathClawConfig.load()
        
        # Override with constructor args
        if config:
            for key, value in config.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
        
        # Get LLM provider
        self._llm = get_provider(
            provider_type=provider if provider != "auto" else None,
            model=model,
        )
        
        if not self._llm:
            raise RuntimeError(
                "No LLM provider available. Please provide an API key:\n"
                "  MathClaw(openai_api_key='sk-...')\n"
                "  MathClaw(anthropic_api_key='sk-ant-...')\n"
                "  MathClaw(google_api_key='...')"
            )
        
        # Initialize engine
        self._engine = DiscoveryEngine(
            llm_provider=self._llm,
            config={
                'min_interval_seconds': self.config.min_interval_seconds,
                'max_attempts_per_hour': self.config.max_attempts_per_hour,
                'verification_timeout': self.config.verification_timeout,
            }
        )
        
        # Initialize stores
        self._theorem_store = TheoremStore()
        self._exporter = KnowledgeExporter(self._theorem_store)
    
    def discover_one(self, domain: str = None) -> Dict[str, Any]:
        """
        Run a single discovery attempt.
        
        Args:
            domain: Mathematical domain to focus on (optional)
            
        Returns:
            Dict with discovery result
        """
        return self._engine.run_once()
    
    def start(self, blocking: bool = False) -> None:
        """
        Start autonomous discovery.
        
        Args:
            blocking: If True, run in current thread (blocks)
        """
        self._engine.start(blocking=blocking)
    
    def stop(self) -> None:
        """Stop autonomous discovery."""
        self._engine.stop()
    
    def pause(self) -> None:
        """Pause discovery."""
        self._engine.pause()
    
    def resume(self) -> None:
        """Resume discovery."""
        self._engine.resume()
    
    def status(self) -> Dict[str, Any]:
        """Get current status."""
        return self._engine.get_status()
    
    def theorems(self, limit: int = 50, domain: str = None):
        """
        Get discovered theorems.
        
        Args:
            limit: Maximum number to return
            domain: Filter by domain
            
        Returns:
            List of Theorem objects
        """
        if domain:
            return self._theorem_store.get_by_domain(domain, limit=limit)
        return self._theorem_store.get_recent(limit=limit)
    
    def export(
        self,
        output_path: str,
        format: str = "markdown",
        domain: str = None,
    ) -> str:
        """
        Export theorems to file.
        
        Args:
            output_path: Path to output file
            format: 'markdown', 'latex', 'json', or 'jupyter'
            domain: Filter by domain
            
        Returns:
            Content as string
        """
        path = Path(output_path)
        
        if format == "markdown":
            return self._exporter.to_markdown(path, domain=domain)
        elif format == "latex":
            return self._exporter.to_latex(path, domain=domain)
        elif format == "json":
            return self._exporter.to_json(path, domain=domain)
        elif format == "jupyter":
            return self._exporter.to_jupyter(path, domain=domain)
        else:
            raise ValueError(f"Unknown format: {format}")
    
    @property
    def provider_name(self) -> str:
        """Get the name of the LLM provider."""
        return self._llm.name
    
    @property
    def theorem_count(self) -> int:
        """Get total number of proven theorems."""
        return self._theorem_store.count()
    
    def on_discovery(self, callback) -> None:
        """Register callback for new discoveries."""
        self._engine.on_discovery(callback)
    
    def health_check(self) -> Dict[str, Any]:
        """Run system health check."""
        from .protection import HealthChecker
        checker = HealthChecker()
        return checker.run_full_check()


# Module-level exports
__all__ = [
    'MathClaw',
    '__version__',
]
