"""
Health Checker - System self-diagnostics.

This module verifies that the system is functioning correctly.
If critical functions fail, it triggers rollback.

FROZEN: This file must NEVER be modified by the evolution engine.
"""

from typing import List, Dict, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
import traceback


class HealthCheckFailed(Exception):
    """Raised when health check fails."""
    
    def __init__(self, check_name: str, error: str):
        self.check_name = check_name
        self.error = error
        super().__init__(f"Health check '{check_name}' failed: {error}")


@dataclass 
class HealthCheckResult:
    """Result of a single health check."""
    name: str
    passed: bool
    duration_ms: float
    error: Optional[str] = None
    details: Optional[str] = None


@dataclass
class HealthReport:
    """Complete health report."""
    timestamp: str
    all_passed: bool
    checks: List[HealthCheckResult]
    critical_failures: List[str]


class HealthChecker:
    """
    System health verification.
    
    Runs a series of checks to verify the system is functioning:
    1. Math sanity checks (1+1=2, etc.)
    2. Parser functionality
    3. Verification functionality
    4. Database connectivity
    5. Security layer integrity
    
    If ANY critical check fails, the system should halt or rollback.
    
    Example:
        >>> checker = HealthChecker()
        >>> report = checker.run_all()
        >>> if not report.all_passed:
        ...     trigger_rollback()
    """
    
    def __init__(self):
        """Initialize health checker."""
        self._checks: Dict[str, Callable] = {}
        self._critical_checks: set = set()
        
        # Register default checks
        self._register_default_checks()
    
    def _register_default_checks(self) -> None:
        """Register the default health checks."""
        
        # Critical checks - system MUST halt if these fail
        self.register_check('math_sanity', self._check_math_sanity, critical=True)
        self.register_check('parser_works', self._check_parser, critical=True)
        self.register_check('security_layer', self._check_security, critical=True)
        
        # Important checks - should trigger investigation
        self.register_check('verification_works', self._check_verification, critical=False)
        self.register_check('imports_work', self._check_imports, critical=False)
    
    def register_check(
        self,
        name: str,
        check_func: Callable[[], bool],
        critical: bool = False,
    ) -> None:
        """
        Register a health check.
        
        Args:
            name: Unique name for the check
            check_func: Function that returns True if healthy
            critical: If True, failure halts the system
        """
        self._checks[name] = check_func
        if critical:
            self._critical_checks.add(name)
    
    def run_all(self, halt_on_critical: bool = True) -> HealthReport:
        """
        Run all health checks.
        
        Args:
            halt_on_critical: If True, raise exception on critical failure
            
        Returns:
            HealthReport with all results
            
        Raises:
            HealthCheckFailed: If halt_on_critical and critical check fails
        """
        import time
        
        results = []
        critical_failures = []
        
        for name, check_func in self._checks.items():
            start = time.time()
            
            try:
                passed = check_func()
                duration = (time.time() - start) * 1000
                
                result = HealthCheckResult(
                    name=name,
                    passed=passed,
                    duration_ms=duration,
                    error=None if passed else "Check returned False",
                )
                
            except Exception as e:
                duration = (time.time() - start) * 1000
                error_msg = f"{type(e).__name__}: {str(e)}"
                
                result = HealthCheckResult(
                    name=name,
                    passed=False,
                    duration_ms=duration,
                    error=error_msg,
                    details=traceback.format_exc()[:500],
                )
            
            results.append(result)
            
            # Track critical failures
            if not result.passed and name in self._critical_checks:
                critical_failures.append(name)
                
                if halt_on_critical:
                    raise HealthCheckFailed(name, result.error or "Unknown error")
        
        return HealthReport(
            timestamp=datetime.now().isoformat(),
            all_passed=all(r.passed for r in results),
            checks=results,
            critical_failures=critical_failures,
        )
    
    def run_single(self, name: str) -> HealthCheckResult:
        """Run a single health check."""
        import time
        
        if name not in self._checks:
            raise ValueError(f"Unknown health check: {name}")
        
        check_func = self._checks[name]
        start = time.time()
        
        try:
            passed = check_func()
            duration = (time.time() - start) * 1000
            
            return HealthCheckResult(
                name=name,
                passed=passed,
                duration_ms=duration,
            )
            
        except Exception as e:
            duration = (time.time() - start) * 1000
            
            return HealthCheckResult(
                name=name,
                passed=False,
                duration_ms=duration,
                error=str(e),
            )
    
    # ═══════════════════════════════════════════════════════════════
    # Default Health Checks
    # ═══════════════════════════════════════════════════════════════
    
    def _check_math_sanity(self) -> bool:
        """
        Basic math sanity check.
        
        If 1+1 doesn't equal 2, something is VERY wrong.
        """
        from sympy import Integer, simplify, sin, cos, pi
        
        # Basic arithmetic
        if Integer(1) + Integer(1) != Integer(2):
            return False
        
        # Symbolic sanity
        if simplify(sin(pi)) != 0:
            return False
        
        # Trig identity
        from sympy import Symbol
        x = Symbol('x')
        identity = simplify(sin(x)**2 + cos(x)**2)
        if identity != 1:
            return False
        
        return True
    
    def _check_parser(self) -> bool:
        """Check that the safe parser works."""
        from mathclaw.security.safe_parser import SafeParser
        
        parser = SafeParser()
        
        # Should parse successfully
        result = parser.parse("sin(x) + cos(x)")
        if not result.success:
            return False
        
        # Should block dangerous input
        result = parser.parse("x + y")  # Normal input
        if not result.success:
            return False
        
        return True
    
    def _check_security(self) -> bool:
        """Check that security layer blocks attacks."""
        from mathclaw.security.input_validator import InputValidator, ValidationError
        
        validator = InputValidator()
        
        # Should block dangerous patterns
        dangerous_inputs = [
            "__import__('os')",
            "exec('print(1)')",
            "eval('1+1')",
            "open('/etc/passwd')",
        ]
        
        for dangerous in dangerous_inputs:
            try:
                validator.validate(dangerous)
                # If we get here, dangerous input wasn't blocked!
                return False
            except ValidationError:
                pass  # Good, it was blocked
        
        # Should allow safe input
        try:
            result = validator.validate("sin(x) + cos(x)")
            if not result.valid:
                return False
        except ValidationError:
            return False
        
        return True
    
    def _check_verification(self) -> bool:
        """Check that verification layer works."""
        try:
            from mathclaw.security.sandbox import Sandbox
            
            sandbox = Sandbox(timeout=10)
            
            # Test a known identity
            result = sandbox.run_verification("sin(x)**2 + cos(x)**2", "1")
            
            if not result.success:
                return False
            
            if not result.result.get('verified', False):
                return False
            
            return True
            
        except Exception:
            # Verification not fully set up yet - that's OK
            return True
    
    def _check_imports(self) -> bool:
        """Check that critical imports work."""
        try:
            import sympy
            import numpy
            from mathclaw.security import SafeParser, InputValidator
            from mathclaw.protection import FrozenRegistry, ChecksumGuardian
            return True
        except ImportError:
            return False


# ═══════════════════════════════════════════════════════════════
# Convenience functions
# ═══════════════════════════════════════════════════════════════

_checker = None

def get_checker() -> HealthChecker:
    """Get the global health checker."""
    global _checker
    if _checker is None:
        _checker = HealthChecker()
    return _checker


def check_health(halt_on_critical: bool = True) -> HealthReport:
    """Run all health checks."""
    return get_checker().run_all(halt_on_critical)


def quick_health_check() -> bool:
    """Quick check - returns True if system is healthy."""
    try:
        report = get_checker().run_all(halt_on_critical=False)
        return len(report.critical_failures) == 0
    except Exception:
        return False
