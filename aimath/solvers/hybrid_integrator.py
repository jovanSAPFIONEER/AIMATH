"""
Hybrid Integrator

Combines multiple integration strategies:
1. Symbolic integration (SymPy) - fast, exact
2. Database lookup - instant for known integrals
3. Numeric + Constant Recognition - the "magic" fallback

This is what makes AIMATH a research-grade tool.
"""

import sympy
from sympy import (
    Symbol, symbols, sqrt, log, atan, sin, cos, exp,
    pi, oo, Integral, sympify, latex, N, simplify, nsimplify
)
from scipy.integrate import quad, dblquad
from scipy import special
import numpy as np
from typing import Optional, Union, Tuple, List, Any
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

# Import our modules
try:
    from aimath.solvers.constant_recognizer import recognize_constant, RecognitionResult
    from aimath.solvers.integral_database import lookup_integral, IntegralEntry, INTEGRAL_DB
except ImportError:
    from .constant_recognizer import recognize_constant, RecognitionResult
    from .integral_database import lookup_integral, IntegralEntry, INTEGRAL_DB


@dataclass
class IntegrationResult:
    """Result of hybrid integration."""
    original: str
    result: Any  # SymPy expression or float
    result_latex: str
    numerical_value: float
    method: str  # 'symbolic', 'database', 'numeric_recognition', 'numeric_only'
    confidence: str  # 'exact', 'high', 'medium', 'low'
    steps: List[str] = field(default_factory=list)
    is_closed_form: bool = True
    recognition_info: Optional[RecognitionResult] = None
    database_entry: Optional[IntegralEntry] = None


def hybrid_integrate(
    expr: Union[str, Any],
    var: Union[str, Symbol] = None,
    lower: Any = None,
    upper: Any = None,
    precision: float = 1e-10,
    timeout: float = 30.0
) -> IntegrationResult:
    """
    Hybrid integration combining symbolic, database, and numeric approaches.
    
    Args:
        expr: Expression to integrate (string or SymPy)
        var: Integration variable (default: x)
        lower: Lower limit (None for indefinite)
        upper: Upper limit (None for indefinite)
        precision: Tolerance for constant recognition
        timeout: Max time for symbolic integration
        
    Returns:
        IntegrationResult with solution and metadata
    """
    steps = []
    
    # Parse inputs
    if isinstance(expr, str):
        expr = sympify(expr)
    
    if var is None:
        var = Symbol('x')
    elif isinstance(var, str):
        var = Symbol(var)
    
    if lower is not None:
        lower = sympify(lower)
    if upper is not None:
        upper = sympify(upper)
    
    is_definite = lower is not None and upper is not None
    
    original_str = f"∫"
    if is_definite:
        original_str = f"∫_{{{lower}}}^{{{upper}}} {latex(expr)} d{var}"
    else:
        original_str = f"∫ {latex(expr)} d{var}"
    
    steps.append(f"Problem: {original_str}")
    
    # ═══════════════════════════════════════════════════════════════
    # STEP 1: Try Database Lookup (instant)
    # ═══════════════════════════════════════════════════════════════
    if is_definite:
        steps.append("\n📚 Step 1: Checking integral database...")
        db_result = lookup_integral(expr, var, lower, upper)
        
        if db_result:
            steps.append(f"  ✅ Found in database: {db_result.name}")
            steps.append(f"  Category: {db_result.category}")
            steps.append(f"  Difficulty: {db_result.difficulty}")
            steps.append(f"  Techniques: {', '.join(db_result.techniques)}")
            if db_result.reference:
                steps.append(f"  Reference: {db_result.reference}")
            
            numerical = float(N(db_result.result, 15))
            
            return IntegrationResult(
                original=original_str,
                result=db_result.result,
                result_latex=latex(db_result.result),
                numerical_value=numerical,
                method='database',
                confidence='exact',
                steps=steps,
                is_closed_form=True,
                database_entry=db_result
            )
        else:
            steps.append("  → Not found in database")
    
    # ═══════════════════════════════════════════════════════════════
    # STEP 2: Try Symbolic Integration (SymPy)
    # ═══════════════════════════════════════════════════════════════
    steps.append("\n🔢 Step 2: Attempting symbolic integration...")
    
    try:
        if is_definite:
            symbolic_result = sympy.integrate(expr, (var, lower, upper))
        else:
            symbolic_result = sympy.integrate(expr, var)
        
        # Check if SymPy actually computed it (vs returning unevaluated)
        if not symbolic_result.has(Integral):
            steps.append(f"  ✅ Symbolic integration successful!")
            steps.append(f"  Result: {symbolic_result}")
            
            numerical = float(N(symbolic_result, 15)) if is_definite else None
            
            return IntegrationResult(
                original=original_str,
                result=symbolic_result,
                result_latex=latex(symbolic_result),
                numerical_value=numerical if numerical else 0,
                method='symbolic',
                confidence='exact',
                steps=steps,
                is_closed_form=True
            )
        else:
            steps.append("  ⚠️ SymPy returned unevaluated integral")
            steps.append("  → Symbolic engine cannot find antiderivative")
    
    except Exception as e:
        steps.append(f"  ❌ Symbolic integration failed: {e}")
    
    # ═══════════════════════════════════════════════════════════════
    # STEP 3: Numeric Integration + Constant Recognition (The Magic!)
    # ═══════════════════════════════════════════════════════════════
    if is_definite:
        steps.append("\n🔮 Step 3: Hybrid Numeric-Symbolic approach...")
        steps.append("  (This is where the magic happens!)")
        
        try:
            # Convert to numpy function
            f_numpy = sympy.lambdify(var, expr, modules=['numpy', 'scipy'])
            
            # Handle infinite limits
            a = float(lower) if lower != -oo and lower != oo else (-np.inf if lower == -oo else np.inf)
            b = float(upper) if upper != -oo and upper != oo else (-np.inf if upper == -oo else np.inf)
            
            # Numerical integration
            numerical_value, error = quad(f_numpy, a, b)
            
            steps.append(f"  Numerical value: {numerical_value:.15f}")
            steps.append(f"  Integration error: {error:.2e}")
            
            # Try to recognize the constant
            steps.append("\n🔍 Step 4: Scanning for mathematical constants...")
            recognition = recognize_constant(numerical_value, precision)
            
            if recognition:
                steps.append(f"  ✅ Pattern found!")
                steps.append(f"  Recognized as: {recognition.recognized_form}")
                steps.append(f"  Method: {recognition.method}")
                steps.append(f"  Match error: {recognition.error:.2e}")
                
                # Verify the match
                verified_value = float(N(recognition.recognized_form, 20))
                match_error = abs(verified_value - numerical_value)
                
                if match_error < precision:
                    steps.append(f"  ✓ Verified: numerical ≈ symbolic (error: {match_error:.2e})")
                    
                    return IntegrationResult(
                        original=original_str,
                        result=recognition.recognized_form,
                        result_latex=recognition.latex,
                        numerical_value=numerical_value,
                        method='numeric_recognition',
                        confidence=recognition.confidence,
                        steps=steps,
                        is_closed_form=True,
                        recognition_info=recognition
                    )
            
            # No pattern found - return numerical result
            steps.append("  → No exact pattern found")
            steps.append("  Returning numerical approximation")
            
            return IntegrationResult(
                original=original_str,
                result=numerical_value,
                result_latex=f"{numerical_value:.10f}",
                numerical_value=numerical_value,
                method='numeric_only',
                confidence='medium',
                steps=steps,
                is_closed_form=False
            )
            
        except Exception as e:
            steps.append(f"  ❌ Numerical integration failed: {e}")
            
            return IntegrationResult(
                original=original_str,
                result=Integral(expr, (var, lower, upper)),
                result_latex=latex(Integral(expr, (var, lower, upper))),
                numerical_value=0,
                method='failed',
                confidence='none',
                steps=steps,
                is_closed_form=False
            )
    
    # Indefinite integral that SymPy couldn't solve
    steps.append("\n❌ Could not find closed-form antiderivative")
    
    return IntegrationResult(
        original=original_str,
        result=Integral(expr, var),
        result_latex=latex(Integral(expr, var)),
        numerical_value=0,
        method='failed',
        confidence='none',
        steps=steps,
        is_closed_form=False
    )


def solve_integral(expr_str: str) -> IntegrationResult:
    """
    User-friendly function to solve an integral from string.
    
    Examples:
        solve_integral("sin(x)/x from 0 to oo")
        solve_integral("exp(-x**2) from -oo to oo")
        solve_integral("atan(sqrt(x**2+2))/((x**2+1)*sqrt(x**2+2)) from 0 to 1")
    """
    import re
    
    # Parse "expr from a to b" format
    match = re.match(r'(.+?)\s+from\s+(.+?)\s+to\s+(.+?)$', expr_str.strip(), re.IGNORECASE)
    
    if match:
        expr = sympify(match.group(1))
        lower = sympify(match.group(2))
        upper = sympify(match.group(3))
        
        # Find variable
        free_vars = expr.free_symbols
        var = list(free_vars)[0] if free_vars else Symbol('x')
        
        return hybrid_integrate(expr, var, lower, upper)
    else:
        # Indefinite integral
        expr = sympify(expr_str)
        free_vars = expr.free_symbols
        var = list(free_vars)[0] if free_vars else Symbol('x')
        
        return hybrid_integrate(expr, var)


def print_integration_result(result: IntegrationResult):
    """Pretty print an integration result."""
    print("\n" + "═" * 60)
    print("INTEGRATION RESULT")
    print("═" * 60)
    
    for step in result.steps:
        print(step)
    
    print("\n" + "─" * 60)
    print(f"ANSWER: {result.result}")
    print(f"LaTeX:  {result.result_latex}")
    
    if result.numerical_value:
        print(f"Decimal: ≈ {result.numerical_value:.15f}")
    
    print(f"\nMethod: {result.method}")
    print(f"Confidence: {result.confidence}")
    print(f"Closed form: {'Yes' if result.is_closed_form else 'No'}")
    
    if result.database_entry:
        print(f"\n📚 From: {result.database_entry.name}")
        print(f"   Techniques: {', '.join(result.database_entry.techniques)}")
    
    print("═" * 60)


# Quick test
if __name__ == "__main__":
    x = Symbol('x')
    
    print("\n" + "="*70)
    print("TEST 1: Ahmed's Integral (the one that started it all!)")
    print("="*70)
    
    ahmed = atan(sqrt(x**2 + 2)) / ((x**2 + 1) * sqrt(x**2 + 2))
    result = hybrid_integrate(ahmed, x, 0, 1)
    print_integration_result(result)
    
    print("\n" + "="*70)
    print("TEST 2: Gaussian Integral")
    print("="*70)
    
    result = hybrid_integrate(exp(-x**2), x, -oo, oo)
    print_integration_result(result)
    
    print("\n" + "="*70)
    print("TEST 3: Simple integral (should use symbolic)")
    print("="*70)
    
    result = hybrid_integrate(x**2, x, 0, 1)
    print_integration_result(result)
    
    print("\n" + "="*70)
    print("TEST 4: Dirichlet Integral")
    print("="*70)
    
    result = hybrid_integrate(sin(x)/x, x, 0, oo)
    print_integration_result(result)
