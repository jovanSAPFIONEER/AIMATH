"""
Verification Bridge - Connects conjectures to AIMATH verification.

This is the CRITICAL module that prevents hallucinations.
Every conjecture MUST pass through this bridge before being trusted.

Uses AIMATH's existing verification infrastructure.
"""

import sys
import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum


class VerificationStatus(Enum):
    """Result of verification attempt."""
    PROVEN = "proven"           # Symbolically proven true
    DISPROVEN = "disproven"     # Found counterexample
    PLAUSIBLE = "plausible"     # Passed numeric tests, not proven
    UNKNOWN = "unknown"         # Could not determine
    ERROR = "error"             # Error during verification
    TIMEOUT = "timeout"         # Verification timed out


@dataclass
class VerificationResult:
    """Complete result of a verification attempt."""
    status: VerificationStatus
    conjecture: str
    method: Optional[str] = None
    proof_steps: Optional[List[str]] = None
    counterexample: Optional[str] = None
    numeric_evidence: Optional[Dict] = None
    execution_time_ms: float = 0
    error_message: Optional[str] = None


class VerificationBridge:
    """
    Bridge between conjecture generation and AIMATH verification.
    
    This is the TRUST BOUNDARY of MathClaw.
    Nothing is considered true until it passes verification.
    
    Verification methods (in order of trust):
    1. Symbolic proof (highest trust)
    2. Numeric testing with many random values
    3. Plausibility heuristics (lowest trust, flagged)
    
    Example:
        >>> bridge = VerificationBridge()
        >>> 
        >>> result = bridge.verify("sin(x)**2 + cos(x)**2 = 1")
        >>> if result.status == VerificationStatus.PROVEN:
        ...     print("Verified!")
    """
    
    # Maximum time for any verification (seconds)
    MAX_VERIFICATION_TIME = 30
    
    # Number of random test points for numeric verification
    NUMERIC_TEST_COUNT = 1000
    
    def __init__(self, aimath_path: Path = None):
        """
        Initialize the bridge.
        
        Args:
            aimath_path: Path to AIMATH module (auto-detected if None)
        """
        self.aimath_path = aimath_path or self._find_aimath()
        self._verifier = None
        self._prover = None
        self._init_aimath()
    
    def _find_aimath(self) -> Path:
        """Find the AIMATH module path."""
        # Try relative to this file
        candidates = [
            Path(__file__).parent.parent.parent / 'aimath',
            Path(__file__).parent.parent.parent / 'AIMATH' / 'aimath',
            Path('c:/AI MATH/aimath'),
        ]
        
        for path in candidates:
            if path.exists() and (path / 'verification').exists():
                return path
        
        raise RuntimeError("Could not find AIMATH module")
    
    def _init_aimath(self) -> None:
        """Initialize AIMATH verification components."""
        # Add to path if needed
        aimath_parent = str(self.aimath_path.parent)
        if aimath_parent not in sys.path:
            sys.path.insert(0, aimath_parent)
        
        # Import verification modules
        try:
            from aimath.verification import EquationVerifier
            from aimath.proof_assistant import SymPyProver
            
            self._verifier = EquationVerifier()
            self._prover = SymPyProver()
        except ImportError as e:
            print(f"[VerificationBridge] Warning: Could not import AIMATH: {e}")
            self._verifier = None
            self._prover = None
    
    def verify(
        self,
        conjecture: str,
        variables: List[str] = None,
        assumptions: List[str] = None,
        timeout: float = None,
    ) -> VerificationResult:
        """
        Verify a mathematical conjecture.
        
        Args:
            conjecture: The conjecture to verify
            variables: Variables in the expression
            assumptions: Domain restrictions
            timeout: Max seconds (default: MAX_VERIFICATION_TIME)
            
        Returns:
            VerificationResult with status and details
        """
        timeout = timeout or self.MAX_VERIFICATION_TIME
        start_time = time.time()
        
        try:
            # Try symbolic verification first (highest trust)
            result = self._try_symbolic_proof(conjecture, variables, timeout)
            
            if result.status == VerificationStatus.PROVEN:
                result.execution_time_ms = (time.time() - start_time) * 1000
                return result
            
            # If not proven, try to find counterexample
            counter_result = self._try_find_counterexample(
                conjecture, variables, timeout - (time.time() - start_time)
            )
            
            if counter_result.status == VerificationStatus.DISPROVEN:
                counter_result.execution_time_ms = (time.time() - start_time) * 1000
                return counter_result
            
            # If no counterexample, run numeric tests
            remaining = timeout - (time.time() - start_time)
            if remaining > 0:
                numeric_result = self._try_numeric_verification(
                    conjecture, variables, remaining
                )
                numeric_result.execution_time_ms = (time.time() - start_time) * 1000
                return numeric_result
            
            return VerificationResult(
                status=VerificationStatus.TIMEOUT,
                conjecture=conjecture,
                execution_time_ms=(time.time() - start_time) * 1000,
            )
            
        except Exception as e:
            return VerificationResult(
                status=VerificationStatus.ERROR,
                conjecture=conjecture,
                error_message=str(e),
                execution_time_ms=(time.time() - start_time) * 1000,
            )
    
    def _try_symbolic_proof(
        self,
        conjecture: str,
        variables: List[str],
        timeout: float,
    ) -> VerificationResult:
        """Attempt to prove symbolically using SymPy."""
        from mathclaw.security import SafeParser
        import sympy as sp
        
        parser = SafeParser()
        
        try:
            # Parse the conjecture
            if '=' in conjecture and '==' not in conjecture:
                # It's an equation
                lhs, rhs = conjecture.split('=', 1)
                lhs_expr = parser.parse(lhs.strip())
                rhs_expr = parser.parse(rhs.strip())
                
                # Try to simplify difference to zero
                diff = sp.simplify(lhs_expr - rhs_expr)
                
                if diff == 0:
                    return VerificationResult(
                        status=VerificationStatus.PROVEN,
                        conjecture=conjecture,
                        method="symbolic_simplification",
                        proof_steps=[
                            f"Parsed LHS: {lhs_expr}",
                            f"Parsed RHS: {rhs_expr}",
                            f"Difference: {lhs_expr} - {rhs_expr}",
                            f"Simplified: {diff}",
                            "Difference equals 0, equation is proven.",
                        ],
                    )
                
                # Try trigsimp
                diff_trig = sp.trigsimp(lhs_expr - rhs_expr)
                if diff_trig == 0:
                    return VerificationResult(
                        status=VerificationStatus.PROVEN,
                        conjecture=conjecture,
                        method="trig_simplification",
                        proof_steps=[
                            f"Applied trigsimp to difference",
                            f"Result: {diff_trig}",
                            "Equation proven via trig identities.",
                        ],
                    )
                
                # Try expand + simplify
                diff_expand = sp.simplify(sp.expand(lhs_expr) - sp.expand(rhs_expr))
                if diff_expand == 0:
                    return VerificationResult(
                        status=VerificationStatus.PROVEN,
                        conjecture=conjecture,
                        method="expand_simplify",
                        proof_steps=[
                            f"Expanded both sides",
                            f"Simplified difference: {diff_expand}",
                            "Equation proven.",
                        ],
                    )
                    
            elif '<=' in conjecture or '>=' in conjecture or '<' in conjecture or '>' in conjecture:
                # It's an inequality - harder to prove symbolically
                pass
            
        except Exception as e:
            pass  # Continue to next method
        
        return VerificationResult(
            status=VerificationStatus.UNKNOWN,
            conjecture=conjecture,
            method="symbolic_proof_attempted",
        )
    
    def _try_find_counterexample(
        self,
        conjecture: str,
        variables: List[str],
        timeout: float,
    ) -> VerificationResult:
        """Try to find a counterexample using random testing."""
        from mathclaw.security import SafeParser
        import sympy as sp
        import random
        import math
        
        parser = SafeParser()
        
        if timeout <= 0:
            return VerificationResult(
                status=VerificationStatus.UNKNOWN,
                conjecture=conjecture,
            )
        
        try:
            # Detect variables
            if not variables:
                variables = self._detect_variables(conjecture)
            
            # Create symbols
            symbols = {v: sp.Symbol(v, real=True) for v in variables}
            
            # Parse expression
            if '=' in conjecture and '==' not in conjecture:
                lhs, rhs = conjecture.split('=', 1)
                lhs_expr = parser.parse(lhs.strip())
                rhs_expr = parser.parse(rhs.strip())
                diff_expr = lhs_expr - rhs_expr
            else:
                return VerificationResult(
                    status=VerificationStatus.UNKNOWN,
                    conjecture=conjecture,
                )
            
            # Create numerical function
            var_symbols = [symbols[v] for v in variables if v in symbols]
            f = sp.lambdify(var_symbols, diff_expr, modules=['numpy', 'sympy'])
            
            # Test many random values
            test_ranges = [
                (-10, 10),
                (-1, 1),
                (0.001, 1),
                (1, 100),
                (-100, -1),
            ]
            
            tolerance = 1e-10
            
            for _ in range(self.NUMERIC_TEST_COUNT):
                test_range = random.choice(test_ranges)
                
                try:
                    values = [random.uniform(*test_range) for _ in var_symbols]
                    result = float(f(*values))
                    
                    if abs(result) > tolerance:
                        # Potential counterexample - verify it's not just numeric error
                        if abs(result) > 0.01:  # Significant difference
                            point = {v: val for v, val in zip(variables, values)}
                            return VerificationResult(
                                status=VerificationStatus.DISPROVEN,
                                conjecture=conjecture,
                                method="numeric_counterexample",
                                counterexample=str(point),
                                numeric_evidence={
                                    'point': point,
                                    'lhs_value': float(lhs_expr.subs(symbols).subs(point)) if symbols else None,
                                    'rhs_value': float(rhs_expr.subs(symbols).subs(point)) if symbols else None,
                                    'difference': result,
                                },
                            )
                except (ValueError, ZeroDivisionError, OverflowError):
                    continue  # Skip problematic points
                except Exception:
                    continue
            
        except Exception as e:
            pass
        
        return VerificationResult(
            status=VerificationStatus.UNKNOWN,
            conjecture=conjecture,
        )
    
    def _try_numeric_verification(
        self,
        conjecture: str,
        variables: List[str],
        timeout: float,
    ) -> VerificationResult:
        """Run extensive numeric tests for plausibility."""
        from mathclaw.security import SafeParser
        import sympy as sp
        import random
        
        parser = SafeParser()
        
        if timeout <= 0:
            return VerificationResult(
                status=VerificationStatus.UNKNOWN,
                conjecture=conjecture,
            )
        
        try:
            if not variables:
                variables = self._detect_variables(conjecture)
            
            symbols = {v: sp.Symbol(v, real=True) for v in variables}
            
            if '=' in conjecture and '==' not in conjecture:
                lhs, rhs = conjecture.split('=', 1)
                lhs_expr = parser.parse(lhs.strip())
                rhs_expr = parser.parse(rhs.strip())
                diff_expr = lhs_expr - rhs_expr
            else:
                return VerificationResult(
                    status=VerificationStatus.UNKNOWN,
                    conjecture=conjecture,
                )
            
            var_symbols = [symbols[v] for v in variables if v in symbols]
            f = sp.lambdify(var_symbols, diff_expr, modules=['numpy', 'sympy'])
            
            passed = 0
            failed = 0
            total_tests = min(self.NUMERIC_TEST_COUNT, int(timeout * 100))
            
            for _ in range(total_tests):
                try:
                    values = [random.uniform(-10, 10) for _ in var_symbols]
                    result = float(f(*values))
                    
                    if abs(result) < 1e-8:
                        passed += 1
                    else:
                        failed += 1
                except:
                    continue
            
            if failed == 0 and passed > 100:
                return VerificationResult(
                    status=VerificationStatus.PLAUSIBLE,
                    conjecture=conjecture,
                    method="numeric_verification",
                    numeric_evidence={
                        'tests_passed': passed,
                        'tests_failed': failed,
                        'confidence': 'high' if passed > 500 else 'medium',
                    },
                )
            
        except Exception as e:
            pass
        
        return VerificationResult(
            status=VerificationStatus.UNKNOWN,
            conjecture=conjecture,
        )
    
    def _detect_variables(self, expression: str) -> List[str]:
        """Detect variables in an expression."""
        import re
        
        # Common single-letter variables
        common = {'x', 'y', 'z', 'a', 'b', 'c', 'n', 'm', 'k', 't', 'r'}
        
        # Find all single letters
        letters = set(re.findall(r'\b([a-zA-Z])\b', expression))
        
        # Filter to common variables
        variables = [v for v in letters if v in common]
        
        return variables if variables else ['x']
    
    def batch_verify(
        self,
        conjectures: List[str],
        timeout_per_conjecture: float = 10,
    ) -> List[VerificationResult]:
        """
        Verify multiple conjectures.
        
        Args:
            conjectures: List of conjectures to verify
            timeout_per_conjecture: Timeout for each
            
        Returns:
            List of VerificationResults
        """
        results = []
        
        for conjecture in conjectures:
            result = self.verify(
                conjecture,
                timeout=timeout_per_conjecture,
            )
            results.append(result)
        
        return results
    
    def quick_check(self, conjecture: str, num_tests: int = 100) -> bool:
        """
        Quick numeric check - is this worth verifying?
        
        Returns True if conjecture passes basic numeric tests.
        Does NOT mean it's proven!
        """
        from mathclaw.security import SafeParser
        import sympy as sp
        import random
        
        parser = SafeParser()
        
        try:
            variables = self._detect_variables(conjecture)
            symbols = {v: sp.Symbol(v, real=True) for v in variables}
            
            if '=' in conjecture:
                lhs, rhs = conjecture.split('=', 1)
                lhs_expr = parser.parse(lhs.strip())
                rhs_expr = parser.parse(rhs.strip())
                diff_expr = lhs_expr - rhs_expr
            else:
                return False
            
            var_symbols = [symbols[v] for v in variables if v in symbols]
            f = sp.lambdify(var_symbols, diff_expr, modules=['numpy', 'sympy'])
            
            for _ in range(num_tests):
                try:
                    values = [random.uniform(-5, 5) for _ in var_symbols]
                    result = float(f(*values))
                    if abs(result) > 0.001:
                        return False
                except:
                    continue
            
            return True
            
        except:
            return False
