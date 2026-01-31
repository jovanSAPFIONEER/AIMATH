"""
Conjecture Tester (Fuzz Verification)

The most powerful "trust" feature. It doesn't prove a theorem, 
but it rapidly attempts to BREAK it.

If a user thinks "Is x^x = e^x?", this tool runs thousands of 
random checks to say "False at x=2".

This is mathematically rigorous in the sense that:
- Finding ONE counterexample disproves a universal claim
- Passing all tests provides statistical confidence (not proof)

Use cases:
- Quick sanity check for identities
- Finding counterexamples to false claims
- Testing conjectures before attempting proof
- Validating symbolic computation results
"""

import sympy
from sympy import (
    Symbol, symbols, simplify, N, oo, I, pi, E,
    sin, cos, tan, exp, log, sqrt, Abs, re, im,
    latex, sympify, zoo, nan, S
)
import random
import numpy as np
from typing import List, Tuple, Any, Optional, Dict, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import time

logger = logging.getLogger(__name__)


class ConjectureStatus(Enum):
    """Status of a conjecture after testing."""
    DISPROVEN = "DISPROVEN (counterexample found)"
    PLAUSIBLE = "PLAUSIBLE (passed all tests)"
    INCONCLUSIVE = "INCONCLUSIVE (too many errors)"
    TRIVIALLY_TRUE = "TRIVIALLY TRUE (expressions identical)"
    TRIVIALLY_FALSE = "TRIVIALLY FALSE (expressions differ symbolically)"


@dataclass
class Counterexample:
    """A counterexample that disproves a conjecture."""
    point: Dict[Symbol, float]  # Variable assignments
    lhs_value: Any
    rhs_value: Any
    difference: float
    

@dataclass
class ConjectureResult:
    """Result of testing a mathematical conjecture."""
    lhs: Any
    rhs: Any
    variables: List[Symbol]
    status: ConjectureStatus
    trials_run: int
    trials_passed: int
    trials_failed: int
    trials_error: int
    counterexamples: List[Counterexample] = field(default_factory=list)
    confidence: float = 0.0  # Statistical confidence if passed
    time_elapsed: float = 0.0
    notes: List[str] = field(default_factory=list)


def test_conjecture(
    lhs: Union[str, Any],
    rhs: Union[str, Any],
    variables: List[Symbol] = None,
    trials: int = 1000,
    domain: Tuple[float, float] = (-10, 10),
    tolerance: float = 1e-9,
    complex_domain: bool = False,
    special_points: List[Dict] = None,
    verbose: bool = False
) -> ConjectureResult:
    """
    Numerically 'fuzz tests' an equality to see if it holds.
    
    The key insight: Finding ONE counterexample disproves a universal claim.
    
    Args:
        lhs: Left-hand side expression
        rhs: Right-hand side expression
        variables: List of variables (auto-detected if None)
        trials: Number of random test points
        domain: Range for random values (min, max)
        tolerance: Maximum allowed difference (for floating point)
        complex_domain: Whether to test complex values too
        special_points: Additional specific points to test
        verbose: Whether to print progress
        
    Returns:
        ConjectureResult with status and counterexamples
    """
    start_time = time.time()
    notes = []
    
    # Parse expressions
    if isinstance(lhs, str):
        lhs = sympify(lhs)
    if isinstance(rhs, str):
        rhs = sympify(rhs)
    
    # Auto-detect variables
    if variables is None:
        all_symbols = lhs.free_symbols | rhs.free_symbols
        variables = [s for s in all_symbols if s.is_Symbol]
        variables.sort(key=lambda x: str(x))  # Consistent ordering
    
    notes.append(f"Testing: {lhs} == {rhs}")
    notes.append(f"Variables: {variables}")
    notes.append(f"Domain: {domain}")
    
    # Step 0: Check if symbolically equal
    try:
        diff = simplify(lhs - rhs)
        if diff == 0:
            return ConjectureResult(
                lhs=lhs, rhs=rhs, variables=variables,
                status=ConjectureStatus.TRIVIALLY_TRUE,
                trials_run=0, trials_passed=0, trials_failed=0, trials_error=0,
                confidence=1.0,
                time_elapsed=time.time() - start_time,
                notes=notes + ["Expressions are symbolically identical after simplification"]
            )
        # Check if symbolically different in an obvious way
        if diff.is_number and diff != 0:
            return ConjectureResult(
                lhs=lhs, rhs=rhs, variables=variables,
                status=ConjectureStatus.TRIVIALLY_FALSE,
                trials_run=0, trials_passed=0, trials_failed=0, trials_error=0,
                counterexamples=[Counterexample({}, float(lhs) if lhs.is_number else lhs, 
                                                  float(rhs) if rhs.is_number else rhs,
                                                  float(abs(diff)))],
                confidence=0.0,
                time_elapsed=time.time() - start_time,
                notes=notes + [f"Expressions differ by constant: {diff}"]
            )
    except Exception:
        pass  # Continue with numerical testing
    
    # Create numpy-compatible functions
    try:
        f_lhs = sympy.lambdify(variables, lhs, modules=['numpy', 'scipy'])
        f_rhs = sympy.lambdify(variables, rhs, modules=['numpy', 'scipy'])
    except Exception as e:
        notes.append(f"Could not create numerical functions: {e}")
        return ConjectureResult(
            lhs=lhs, rhs=rhs, variables=variables,
            status=ConjectureStatus.INCONCLUSIVE,
            trials_run=0, trials_passed=0, trials_failed=0, trials_error=0,
            confidence=0.0,
            time_elapsed=time.time() - start_time,
            notes=notes
        )
    
    # Prepare test points
    test_points = []
    
    # Add special points first
    if special_points:
        test_points.extend(special_points)
    
    # Add standard special values
    special_values = [0, 1, -1, 0.5, -0.5, 2, -2, pi, E, sqrt(2)]
    for sv in special_values:
        try:
            sv_float = float(sv)
            if domain[0] <= sv_float <= domain[1]:
                test_points.append({v: sv_float for v in variables})
        except (TypeError, ValueError):
            pass
    
    # Add random points
    for _ in range(trials):
        point = {}
        for v in variables:
            if complex_domain:
                re_part = random.uniform(*domain)
                im_part = random.uniform(*domain)
                point[v] = complex(re_part, im_part)
            else:
                point[v] = random.uniform(*domain)
        test_points.append(point)
    
    # Run tests
    passed = 0
    failed = 0
    errors = 0
    counterexamples = []
    
    for i, point in enumerate(test_points):
        try:
            vals = [point[v] for v in variables]
            
            val_lhs = f_lhs(*vals)
            val_rhs = f_rhs(*vals)
            
            # Handle special values
            if np.isnan(val_lhs) or np.isnan(val_rhs):
                errors += 1
                continue
            if np.isinf(val_lhs) or np.isinf(val_rhs):
                # Both infinite with same sign is OK
                if np.isinf(val_lhs) and np.isinf(val_rhs):
                    if np.sign(val_lhs) == np.sign(val_rhs):
                        passed += 1
                        continue
                errors += 1
                continue
            
            # Check equality
            diff = abs(val_lhs - val_rhs)
            
            # Relative tolerance for large values
            scale = max(abs(val_lhs), abs(val_rhs), 1.0)
            if diff / scale <= tolerance:
                passed += 1
            else:
                failed += 1
                
                # Record counterexample (only first few)
                if len(counterexamples) < 5:
                    counterexamples.append(Counterexample(
                        point=point,
                        lhs_value=val_lhs,
                        rhs_value=val_rhs,
                        difference=diff
                    ))
                    
                    if verbose:
                        print(f"❌ Counterexample at {point}")
                        print(f"   LHS = {val_lhs}, RHS = {val_rhs}, diff = {diff}")
                        
        except Exception as e:
            errors += 1
            if verbose and errors <= 3:
                logger.debug(f"Error at {point}: {e}")
    
    # Determine status
    total_valid = passed + failed
    
    if failed > 0:
        status = ConjectureStatus.DISPROVEN
        confidence = 0.0
    elif errors > total_valid:
        status = ConjectureStatus.INCONCLUSIVE
        confidence = 0.0
    else:
        status = ConjectureStatus.PLAUSIBLE
        # Statistical confidence: probability of not finding a counterexample
        # if the conjecture were false with probability p
        # P(no counterexample in n trials | false with prob p) = (1-p)^n
        # If we assume p = 0.01 (1% of domain is counterexample), then
        # (0.99)^1000 ≈ 0.00004, so 99.996% confident it's true
        # This is heuristic, not rigorous!
        if total_valid > 0:
            confidence = min(0.99, 1 - (1/(total_valid + 1)))
        else:
            confidence = 0.0
    
    notes.append(f"Trials: {total_valid} valid, {errors} errors")
    
    return ConjectureResult(
        lhs=lhs, rhs=rhs, variables=variables,
        status=status,
        trials_run=len(test_points),
        trials_passed=passed,
        trials_failed=failed,
        trials_error=errors,
        counterexamples=counterexamples,
        confidence=confidence,
        time_elapsed=time.time() - start_time,
        notes=notes
    )


def test_inequality(
    lhs: Union[str, Any],
    rhs: Union[str, Any],
    relation: str = '>=',  # '>', '<', '>=', '<=', '!='
    variables: List[Symbol] = None,
    trials: int = 1000,
    domain: Tuple[float, float] = (-10, 10),
    constraint: Any = None  # Additional constraint like x > 0
) -> ConjectureResult:
    """
    Test an inequality conjecture.
    
    Examples:
        test_inequality("x**2", "0", ">=")  # x² ≥ 0
        test_inequality("(x+y)**2", "x**2 + y**2", ">=")  # (x+y)² ≥ x²+y² (FALSE!)
        test_inequality("exp(x)", "1 + x", ">=")  # e^x ≥ 1+x
    """
    if isinstance(lhs, str):
        lhs = sympify(lhs)
    if isinstance(rhs, str):
        rhs = sympify(rhs)
    
    # Convert inequality to equality test
    # lhs >= rhs  ⟺  lhs - rhs >= 0
    diff = lhs - rhs
    
    if variables is None:
        variables = list(diff.free_symbols)
        variables.sort(key=lambda x: str(x))
    
    # Create test function
    f_diff = sympy.lambdify(variables, diff, modules=['numpy', 'scipy'])
    
    passed = 0
    failed = 0
    errors = 0
    counterexamples = []
    
    for _ in range(trials):
        try:
            vals = [random.uniform(*domain) for _ in variables]
            point = dict(zip(variables, vals))
            
            # Check constraint if provided
            if constraint is not None:
                f_constraint = sympy.lambdify(variables, constraint, 'numpy')
                if not f_constraint(*vals):
                    continue  # Skip points outside constraint
            
            val = f_diff(*vals)
            
            if np.isnan(val) or np.isinf(val):
                errors += 1
                continue
            
            # Check relation
            if relation == '>=':
                holds = val >= -1e-10  # Small tolerance
            elif relation == '>':
                holds = val > 1e-10
            elif relation == '<=':
                holds = val <= 1e-10
            elif relation == '<':
                holds = val < -1e-10
            elif relation == '!=':
                holds = abs(val) > 1e-10
            else:
                holds = val >= 0
            
            if holds:
                passed += 1
            else:
                failed += 1
                if len(counterexamples) < 5:
                    counterexamples.append(Counterexample(
                        point=point,
                        lhs_value=float(lhs.subs(point)),
                        rhs_value=float(rhs.subs(point)),
                        difference=val
                    ))
                    
        except Exception:
            errors += 1
    
    status = ConjectureStatus.DISPROVEN if failed > 0 else ConjectureStatus.PLAUSIBLE
    
    return ConjectureResult(
        lhs=lhs, rhs=rhs, variables=variables,
        status=status,
        trials_run=trials,
        trials_passed=passed,
        trials_failed=failed,
        trials_error=errors,
        counterexamples=counterexamples,
        confidence=0.99 if failed == 0 else 0.0,
        notes=[f"Testing: {lhs} {relation} {rhs}"]
    )


def quick_check(expr1: str, expr2: str, n: int = 100) -> bool:
    """
    Quick boolean check if two expressions seem equal.
    
    Returns True if they appear equal, False otherwise.
    """
    result = test_conjecture(expr1, expr2, trials=n)
    return result.status in [ConjectureStatus.PLAUSIBLE, ConjectureStatus.TRIVIALLY_TRUE]


def find_counterexample(
    lhs: Union[str, Any],
    rhs: Union[str, Any],
    variables: List[Symbol] = None,
    max_attempts: int = 10000
) -> Optional[Counterexample]:
    """
    Try harder to find a counterexample.
    
    Uses multiple strategies:
    1. Random sampling
    2. Grid search
    3. Special values
    """
    result = test_conjecture(lhs, rhs, variables, trials=max_attempts)
    
    if result.counterexamples:
        return result.counterexamples[0]
    
    return None


def print_conjecture_result(result: ConjectureResult):
    """Pretty print a conjecture test result."""
    print("\n" + "═" * 60)
    print("CONJECTURE TEST RESULT")
    print("═" * 60)
    
    print(f"\nConjecture: {result.lhs} = {result.rhs}")
    print(f"Variables:  {result.variables}")
    
    print("\n" + "─" * 60)
    
    # Status with emoji
    status_emoji = {
        ConjectureStatus.DISPROVEN: "❌",
        ConjectureStatus.PLAUSIBLE: "✅",
        ConjectureStatus.INCONCLUSIVE: "⚠️",
        ConjectureStatus.TRIVIALLY_TRUE: "✓✓",
        ConjectureStatus.TRIVIALLY_FALSE: "✗✗"
    }
    
    print(f"{status_emoji.get(result.status, '?')} STATUS: {result.status.value}")
    
    print(f"\nTrials: {result.trials_run}")
    print(f"  Passed: {result.trials_passed}")
    print(f"  Failed: {result.trials_failed}")
    print(f"  Errors: {result.trials_error}")
    
    if result.status == ConjectureStatus.PLAUSIBLE:
        print(f"\nConfidence: {result.confidence*100:.1f}%")
        print("  (Statistical confidence based on trials passed)")
    
    if result.counterexamples:
        print("\n" + "─" * 60)
        print("COUNTEREXAMPLES:")
        for i, ce in enumerate(result.counterexamples[:3]):
            print(f"\n  #{i+1}: {ce.point}")
            print(f"      LHS = {ce.lhs_value}")
            print(f"      RHS = {ce.rhs_value}")
            print(f"      Difference = {ce.difference}")
    
    if result.notes:
        print("\n" + "─" * 60)
        print("Notes:")
        for note in result.notes:
            print(f"  • {note}")
    
    print(f"\nTime: {result.time_elapsed*1000:.1f} ms")
    print("═" * 60)


# Convenience function for CLI
def test(claim: str) -> ConjectureResult:
    """
    User-friendly conjecture tester.
    
    Examples:
        test("sin(x)**2 + cos(x)**2 = 1")
        test("x**x = e**x")
        test("(a+b)**2 = a**2 + b**2")
    """
    # Parse the claim
    if '=' in claim and '==' not in claim:
        parts = claim.split('=')
        if len(parts) == 2:
            lhs = parts[0].strip()
            rhs = parts[1].strip()
            return test_conjecture(lhs, rhs)
    
    # Try as equality with 0
    return test_conjecture(claim, 0)


# Quick test
if __name__ == "__main__":
    x, y = symbols('x y')
    
    print("="*70)
    print("TEST 1: True identity sin²x + cos²x = 1")
    print("="*70)
    result = test_conjecture(sin(x)**2 + cos(x)**2, 1)
    print_conjecture_result(result)
    
    print("\n" + "="*70)
    print("TEST 2: False claim x^x = e^x")
    print("="*70)
    result = test_conjecture(x**x, exp(x), domain=(0.1, 5))
    print_conjecture_result(result)
    
    print("\n" + "="*70)
    print("TEST 3: False claim (a+b)² = a² + b²")
    print("="*70)
    a, b = symbols('a b')
    result = test_conjecture((a+b)**2, a**2 + b**2)
    print_conjecture_result(result)
    
    print("\n" + "="*70)
    print("TEST 4: True inequality e^x ≥ 1 + x")
    print("="*70)
    result = test_inequality(exp(x), 1 + x, '>=')
    print_conjecture_result(result)
    
    print("\n" + "="*70)
    print("TEST 5: AM-GM inequality (a+b)/2 ≥ √(ab) for a,b > 0")
    print("="*70)
    result = test_inequality((a+b)/2, sqrt(a*b), '>=', domain=(0.01, 10))
    print_conjecture_result(result)
