"""
Constant Recognizer (Inverse Symbolic Calculator)

Takes a numerical value and attempts to recognize it as a combination
of famous mathematical constants (π, e, √2, ln(2), etc.).

This is the "magic" that turns 0.514041895890... into 5π²/96.
"""

import sympy
from sympy import (
    nsimplify, pi, E, log, sqrt, atan, Rational,
    GoldenRatio, EulerGamma, Catalan, zeta, S
)
from sympy.core.numbers import Float
from typing import Optional, List, Tuple, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class RecognitionResult:
    """Result of constant recognition attempt."""
    original_value: float
    recognized_form: Any  # SymPy expression or None
    latex: str
    confidence: str  # 'exact', 'high', 'medium', 'low'
    error: float
    method: str


# Famous mathematical constants to search for
BASIC_CONSTANTS = [pi, E, sqrt(2), sqrt(3), log(2)]

EXTENDED_CONSTANTS = [
    pi, E, 
    sqrt(2), sqrt(3), sqrt(5),
    log(2), log(3), log(10),
    GoldenRatio,  # φ = (1+√5)/2
    EulerGamma,   # γ ≈ 0.5772
    Catalan,      # G ≈ 0.9159 (Catalan's constant)
]

# Common forms to try: value might be a*const^n/b
COMMON_FORMS = [
    # Powers of pi
    (pi, 1), (pi, 2), (pi, 3), (pi, 4),
    (pi**2, 1),
    # Square roots
    (sqrt(2), 1), (sqrt(3), 1), (sqrt(5), 1),
    # Logarithms
    (log(2), 1), (log(2), 2),
    # Combinations
    (pi * sqrt(2), 1), (pi * sqrt(3), 1),
]


def recognize_constant(
    value: float, 
    precision: float = 1e-10,
    use_extended: bool = True,
    max_denominator: int = 1000
) -> Optional[RecognitionResult]:
    """
    Attempts to convert a float into an exact symbolic form 
    involving pi, e, sqrt(2), etc.
    
    Args:
        value: The numerical value to recognize
        precision: Tolerance for matching
        use_extended: Whether to use extended constant set
        max_denominator: Maximum denominator for rational coefficients
        
    Returns:
        RecognitionResult if found, None otherwise
    """
    if value == 0:
        return RecognitionResult(
            original_value=0,
            recognized_form=S.Zero,
            latex="0",
            confidence='exact',
            error=0.0,
            method='trivial'
        )
    
    constants = EXTENDED_CONSTANTS if use_extended else BASIC_CONSTANTS
    
    # Method 1: Direct nsimplify (SymPy's built-in)
    result = _try_nsimplify(value, constants, precision)
    if result:
        return result
    
    # Method 2: Check common forms manually
    result = _try_common_forms(value, precision, max_denominator)
    if result:
        return result
    
    # Method 3: Check if it's a rational multiple of pi^n
    result = _try_pi_multiple(value, precision, max_denominator)
    if result:
        return result
    
    # Method 4: Check zeta values
    result = _try_zeta_values(value, precision, max_denominator)
    if result:
        return result
    
    return None


def _try_nsimplify(value: float, constants: List, precision: float) -> Optional[RecognitionResult]:
    """Use SymPy's nsimplify with given constants."""
    try:
        exact_form = nsimplify(value, constants=constants, tolerance=precision)
        
        # Validate: check if they are actually close
        computed = float(exact_form.evalf())
        error = abs(computed - value)
        
        if error < precision:
            return RecognitionResult(
                original_value=value,
                recognized_form=exact_form,
                latex=sympy.latex(exact_form),
                confidence='exact' if error < 1e-14 else 'high',
                error=error,
                method='nsimplify'
            )
    except Exception as e:
        logger.debug(f"nsimplify failed: {e}")
    
    return None


def _try_common_forms(value: float, precision: float, max_denom: int) -> Optional[RecognitionResult]:
    """Check if value = (a/b) * constant for common constants."""
    
    for const, power in COMMON_FORMS:
        const_val = float(const.evalf())
        if const_val == 0:
            continue
            
        # value = (a/b) * const  =>  a/b = value / const
        ratio = value / const_val
        
        # Try to find a simple fraction
        frac = _find_simple_fraction(ratio, max_denom, precision / const_val)
        if frac:
            a, b = frac
            exact_form = Rational(a, b) * const
            error = abs(float(exact_form.evalf()) - value)
            
            if error < precision:
                return RecognitionResult(
                    original_value=value,
                    recognized_form=exact_form,
                    latex=sympy.latex(exact_form),
                    confidence='exact' if error < 1e-14 else 'high',
                    error=error,
                    method='common_form'
                )
    
    return None


def _try_pi_multiple(value: float, precision: float, max_denom: int) -> Optional[RecognitionResult]:
    """Check if value is a rational multiple of π, π², π³, etc."""
    
    for n in range(1, 5):  # π¹ through π⁴
        pi_n = float((pi ** n).evalf())
        ratio = value / pi_n
        
        frac = _find_simple_fraction(ratio, max_denom, precision / pi_n)
        if frac:
            a, b = frac
            exact_form = Rational(a, b) * pi ** n
            error = abs(float(exact_form.evalf()) - value)
            
            if error < precision:
                return RecognitionResult(
                    original_value=value,
                    recognized_form=exact_form,
                    latex=sympy.latex(exact_form),
                    confidence='exact' if error < 1e-14 else 'high',
                    error=error,
                    method=f'pi^{n}_multiple'
                )
    
    return None


def _try_zeta_values(value: float, precision: float, max_denom: int) -> Optional[RecognitionResult]:
    """Check if value involves Riemann zeta values ζ(2), ζ(3), ζ(4), etc."""
    
    # ζ(2) = π²/6, ζ(4) = π⁴/90, ζ(3) ≈ 1.202 (Apéry's constant)
    zeta_vals = [
        (zeta(2), "zeta(2)"),
        (zeta(3), "zeta(3)"),  # Apéry's constant
        (zeta(4), "zeta(4)"),
        (zeta(5), "zeta(5)"),
    ]
    
    for z, name in zeta_vals:
        z_val = float(z.evalf())
        ratio = value / z_val
        
        frac = _find_simple_fraction(ratio, max_denom, precision / z_val)
        if frac:
            a, b = frac
            exact_form = Rational(a, b) * z
            error = abs(float(exact_form.evalf()) - value)
            
            if error < precision:
                return RecognitionResult(
                    original_value=value,
                    recognized_form=exact_form,
                    latex=sympy.latex(exact_form),
                    confidence='exact' if error < 1e-14 else 'high',
                    error=error,
                    method=f'{name}_multiple'
                )
    
    return None


def _find_simple_fraction(value: float, max_denom: int, precision: float) -> Optional[Tuple[int, int]]:
    """
    Find integers a, b such that a/b ≈ value with |a/b - value| < precision.
    Uses continued fraction expansion.
    """
    if abs(value) < precision:
        return (0, 1)
    
    # Handle negative values
    sign = 1 if value > 0 else -1
    value = abs(value)
    
    # Continued fraction approach
    best_a, best_b = 0, 1
    best_error = abs(value)
    
    for b in range(1, max_denom + 1):
        a = round(value * b)
        if a > 0:
            error = abs(a / b - value)
            if error < best_error:
                best_error = error
                best_a, best_b = a, b
                
                if error < precision:
                    return (sign * best_a, best_b)
    
    if best_error < precision:
        return (sign * best_a, best_b)
    
    return None


def identify_constant(value: float, verbose: bool = False) -> str:
    """
    User-friendly function to identify a constant.
    Returns a formatted string description.
    """
    result = recognize_constant(value)
    
    if result is None:
        return f"Could not identify {value} as a known constant"
    
    output = []
    output.append(f"Input:      {result.original_value}")
    output.append(f"Recognized: {result.recognized_form}")
    output.append(f"LaTeX:      {result.latex}")
    output.append(f"Confidence: {result.confidence}")
    output.append(f"Error:      {result.error:.2e}")
    output.append(f"Method:     {result.method}")
    
    return "\n".join(output)


# Quick test
if __name__ == "__main__":
    # Ahmed's integral value
    test_values = [
        (0.514041895890073, "Ahmed's integral (should be 5π²/96)"),
        (3.141592653589793, "π"),
        (1.6449340668482264, "π²/6 = ζ(2)"),
        (1.2020569031595942, "ζ(3) - Apéry's constant"),
        (0.6931471805599453, "ln(2)"),
        (1.4142135623730951, "√2"),
    ]
    
    for val, desc in test_values:
        print(f"\n{'='*50}")
        print(f"Testing: {desc}")
        print(f"{'='*50}")
        result = recognize_constant(val)
        if result:
            print(f"✅ Found: {result.recognized_form}")
            print(f"   LaTeX: {result.latex}")
            print(f"   Error: {result.error:.2e}")
        else:
            print(f"❌ Not recognized")
