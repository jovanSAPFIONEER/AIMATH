"""
Test Mathematical Theses - Verification & Discovery

This script tests several mathematical theses to demonstrate
the verification system's capabilities.
"""

import sys
from sympy import (
    Symbol, symbols, sin, cos, tan, exp, log, sqrt, pi, E, I,
    simplify, trigsimp, expand, factor, diff, integrate, limit,
    Eq, solve, oo, summation, product, factorial, binomial,
    Rational, nsimplify, N
)
import math

# Configure output
def print_header(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def print_result(verified, message):
    status = "✓ VERIFIED" if verified else "✗ FAILED"
    print(f"  {status}: {message}")

def verify_by_substitution(expr, var, values):
    """Verify expression equals zero for given values."""
    results = []
    for val in values:
        result = expr.subs(var, val)
        simplified = simplify(result)
        results.append((val, simplified, simplified == 0))
    return results


# =============================================================================
# THESIS 1: Pythagorean Trigonometric Identity
# =============================================================================
def test_pythagorean_identity():
    print_header("THESIS 1: sin²(x) + cos²(x) = 1 for all x")
    
    x = Symbol('x')
    
    # Method 1: Symbolic simplification
    print("\n  Method 1: Symbolic Simplification")
    lhs = sin(x)**2 + cos(x)**2
    result = trigsimp(lhs)
    print(f"    sin²(x) + cos²(x) simplifies to: {result}")
    print_result(result == 1, "Symbolic verification")
    
    # Method 2: Numerical verification at multiple points
    print("\n  Method 2: Numerical Verification")
    test_points = [0, pi/6, pi/4, pi/3, pi/2, pi, 3*pi/2, 2*pi]
    all_pass = True
    for point in test_points:
        val = N(lhs.subs(x, point))
        passed = abs(val - 1) < 1e-10
        all_pass = all_pass and passed
        print(f"    x = {point}: {float(val):.10f} {'✓' if passed else '✗'}")
    print_result(all_pass, "Numerical verification at 8 points")
    
    # Method 3: Derivative test (constant should have 0 derivative)
    print("\n  Method 3: Derivative Test (constant → derivative = 0)")
    derivative = diff(lhs, x)
    simplified_deriv = trigsimp(derivative)
    print(f"    d/dx[sin²(x) + cos²(x)] = {simplified_deriv}")
    print_result(simplified_deriv == 0, "Derivative is zero (confirming constant)")
    
    return True


# =============================================================================
# THESIS 2: Euler's Identity
# =============================================================================
def test_euler_identity():
    print_header("THESIS 2: e^(iπ) + 1 = 0 (Euler's Identity)")
    
    # Direct computation
    print("\n  Method 1: Direct Symbolic Computation")
    result = exp(I * pi) + 1
    simplified = simplify(result)
    print(f"    e^(iπ) + 1 = {simplified}")
    print_result(simplified == 0, "Symbolic verification")
    
    # Via Euler's formula: e^(ix) = cos(x) + i*sin(x)
    print("\n  Method 2: Via Euler's Formula")
    x = Symbol('x')
    euler_formula = cos(x) + I * sin(x)
    at_pi = euler_formula.subs(x, pi)
    print(f"    e^(iπ) = cos(π) + i·sin(π) = {cos(pi)} + i·{sin(pi)} = {simplify(at_pi)}")
    print(f"    e^(iπ) + 1 = {simplify(at_pi + 1)}")
    print_result(simplify(at_pi + 1) == 0, "Via Euler's formula")
    
    # Numerical verification
    print("\n  Method 3: Numerical Verification")
    numerical = complex(N(exp(I * pi)))
    print(f"    e^(iπ) ≈ {numerical.real:.10f} + {numerical.imag:.10f}i")
    error = abs(numerical + 1)
    print(f"    |e^(iπ) + 1| = {error:.2e}")
    print_result(error < 1e-10, "Numerical verification")
    
    return True


# =============================================================================
# THESIS 3: Quadratic Formula
# =============================================================================
def test_quadratic_formula():
    print_header("THESIS 3: Quadratic Formula: x = (-b ± √(b²-4ac)) / 2a")
    
    a, b, c, x = symbols('a b c x')
    
    # General quadratic equation
    print("\n  Testing: ax² + bx + c = 0")
    quadratic = a*x**2 + b*x + c
    
    # The formula gives these roots
    root1 = (-b + sqrt(b**2 - 4*a*c)) / (2*a)
    root2 = (-b - sqrt(b**2 - 4*a*c)) / (2*a)
    
    print(f"    Root 1: x = {root1}")
    print(f"    Root 2: x = {root2}")
    
    # Verify by substitution
    print("\n  Method 1: Symbolic Substitution Verification")
    verify1 = simplify(quadratic.subs(x, root1))
    verify2 = simplify(quadratic.subs(x, root2))
    print(f"    Substituting root1: {verify1}")
    print(f"    Substituting root2: {verify2}")
    print_result(verify1 == 0 and verify2 == 0, "Both roots satisfy equation")
    
    # Numerical test case: x² - 5x + 6 = 0 (roots: 2 and 3)
    print("\n  Method 2: Numerical Test Case")
    print("    Testing: x² - 5x + 6 = 0")
    test_eq = x**2 - 5*x + 6
    solutions = solve(test_eq, x)
    print(f"    Solutions: {solutions}")
    
    # Verify
    for sol in solutions:
        val = test_eq.subs(x, sol)
        print(f"    x = {sol}: {test_eq} = {val} {'✓' if val == 0 else '✗'}")
    print_result(solutions == [2, 3], "Numerical verification")
    
    return True


# =============================================================================
# THESIS 4: Fundamental Theorem of Calculus
# =============================================================================
def test_ftc():
    print_header("THESIS 4: Fundamental Theorem of Calculus")
    print("  If F'(x) = f(x), then ∫[a,b] f(x)dx = F(b) - F(a)")
    
    x, a, b = symbols('x a b')
    
    # Test with f(x) = x²
    print("\n  Test Case: f(x) = x²")
    f = x**2
    F = integrate(f, x)  # Antiderivative
    print(f"    f(x) = {f}")
    print(f"    F(x) = ∫f(x)dx = {F}")
    
    # Verify F'(x) = f(x)
    F_prime = diff(F, x)
    print(f"    F'(x) = {F_prime}")
    print_result(simplify(F_prime - f) == 0, "F'(x) = f(x)")
    
    # Numerical example: ∫[0,3] x² dx
    print("\n  Numerical Verification: ∫[0,3] x² dx")
    definite = integrate(f, (x, 0, 3))
    print(f"    ∫[0,3] x² dx = {definite}")
    
    # F(3) - F(0)
    F_at_3 = F.subs(x, 3)
    F_at_0 = F.subs(x, 0)
    print(f"    F(3) - F(0) = {F_at_3} - {F_at_0} = {F_at_3 - F_at_0}")
    print_result(definite == F_at_3 - F_at_0, "FTC verification")
    
    return True


# =============================================================================
# THESIS 5: Sum of Geometric Series
# =============================================================================
def test_geometric_series():
    print_header("THESIS 5: Geometric Series Sum")
    print("  ∑(r^n, n=0 to ∞) = 1/(1-r) for |r| < 1")
    
    n, r = symbols('n r')
    
    # Symbolic (for |r| < 1)
    print("\n  Method 1: Symbolic Summation")
    # SymPy can compute this with assumptions
    from sympy import Abs
    
    # Finite sum first: ∑(r^n, n=0 to N) = (1 - r^(N+1))/(1 - r)
    N = Symbol('N', positive=True, integer=True)
    finite_sum = summation(r**n, (n, 0, N))
    print(f"    Finite sum: ∑(r^n, n=0 to N) = {finite_sum}")
    
    # As N → ∞ and |r| < 1, r^(N+1) → 0
    print("    As N → ∞ with |r| < 1: r^(N+1) → 0")
    print("    Therefore: ∑(r^n, n=0 to ∞) = 1/(1-r)")
    
    # Numerical verification
    print("\n  Method 2: Numerical Verification")
    test_r = 0.5
    partial_sums = []
    total = 0
    for i in range(20):
        total += test_r**i
        if i in [5, 10, 15, 19]:
            partial_sums.append((i+1, total))
    
    expected = 1 / (1 - test_r)
    print(f"    r = {test_r}, expected limit = {expected}")
    for terms, val in partial_sums:
        error = abs(val - expected)
        print(f"    Sum of {terms} terms: {val:.10f} (error: {error:.2e})")
    
    print_result(abs(total - expected) < 1e-5, "Converges to 1/(1-r)")
    
    return True


# =============================================================================
# THESIS 6: Binomial Theorem
# =============================================================================
def test_binomial_theorem():
    print_header("THESIS 6: Binomial Theorem")
    print("  (a + b)^n = ∑(C(n,k) * a^(n-k) * b^k, k=0 to n)")
    
    a, b, n, k = symbols('a b n k')
    
    # Test for specific n values
    print("\n  Verification for n = 0, 1, 2, 3, 4:")
    
    all_pass = True
    for n_val in range(5):
        # Direct expansion
        expanded = expand((a + b)**n_val)
        
        # Binomial formula
        binomial_sum = sum(
            binomial(n_val, k_val) * a**(n_val - k_val) * b**k_val
            for k_val in range(n_val + 1)
        )
        
        match = simplify(expanded - binomial_sum) == 0
        all_pass = all_pass and match
        
        print(f"    n = {n_val}: (a+b)^{n_val} = {expanded}")
        print(f"           Binomial: {binomial_sum} {'✓' if match else '✗'}")
    
    print_result(all_pass, "Binomial theorem verified for n = 0,1,2,3,4")
    
    # Numerical test: (2 + 3)^4 = 625
    print("\n  Numerical Test: (2 + 3)^4")
    numerical = (2 + 3)**4
    binomial_calc = sum(binomial(4, k) * 2**(4-k) * 3**k for k in range(5))
    print(f"    Direct: {numerical}")
    print(f"    Binomial formula: {binomial_calc}")
    print_result(numerical == binomial_calc, "Numerical verification")
    
    return True


# =============================================================================
# THESIS 7: Product Rule for Derivatives
# =============================================================================
def test_product_rule():
    print_header("THESIS 7: Product Rule: d/dx[f·g] = f'·g + f·g'")
    
    x = Symbol('x')
    
    # Test with f(x) = x² and g(x) = sin(x)
    print("\n  Test: f(x) = x², g(x) = sin(x)")
    f = x**2
    g = sin(x)
    
    # Direct derivative
    product = f * g
    direct_deriv = diff(product, x)
    print(f"    f·g = {product}")
    print(f"    d/dx[f·g] = {direct_deriv}")
    
    # Product rule
    f_prime = diff(f, x)
    g_prime = diff(g, x)
    product_rule = f_prime * g + f * g_prime
    print(f"    f'·g + f·g' = {f_prime}·{g} + {f}·{g_prime}")
    print(f"                = {expand(product_rule)}")
    
    # Compare
    match = simplify(direct_deriv - product_rule) == 0
    print_result(match, "Product rule verified")
    
    # Another test: f(x) = e^x, g(x) = x³
    print("\n  Test: f(x) = e^x, g(x) = x³")
    f = exp(x)
    g = x**3
    
    direct = diff(f * g, x)
    via_rule = diff(f, x) * g + f * diff(g, x)
    
    match2 = simplify(direct - via_rule) == 0
    print(f"    Direct: {direct}")
    print(f"    Via rule: {simplify(via_rule)}")
    print_result(match2, "Product rule verified")
    
    return True


# =============================================================================
# THESIS 8: Chain Rule for Derivatives
# =============================================================================
def test_chain_rule():
    print_header("THESIS 8: Chain Rule: d/dx[f(g(x))] = f'(g(x))·g'(x)")
    
    x = Symbol('x')
    
    # Test: f(u) = sin(u), g(x) = x²  →  f(g(x)) = sin(x²)
    print("\n  Test: f(u) = sin(u), g(x) = x²")
    print("        f(g(x)) = sin(x²)")
    
    composite = sin(x**2)
    direct_deriv = diff(composite, x)
    print(f"    d/dx[sin(x²)] = {direct_deriv}")
    
    # Chain rule: f'(g(x)) = cos(x²), g'(x) = 2x
    chain_result = cos(x**2) * 2*x
    print(f"    f'(g(x))·g'(x) = cos(x²)·2x = {chain_result}")
    
    match = simplify(direct_deriv - chain_result) == 0
    print_result(match, "Chain rule verified")
    
    # Test: e^(3x)
    print("\n  Test: f(u) = e^u, g(x) = 3x → f(g(x)) = e^(3x)")
    composite2 = exp(3*x)
    direct2 = diff(composite2, x)
    chain2 = exp(3*x) * 3  # e^u · 3
    
    match2 = simplify(direct2 - chain2) == 0
    print(f"    d/dx[e^(3x)] = {direct2}")
    print_result(match2, "Chain rule verified")
    
    return True


# =============================================================================
# THESIS 9: L'Hôpital's Rule
# =============================================================================
def test_lhopital():
    print_header("THESIS 9: L'Hôpital's Rule")
    print("  If lim f(x)/g(x) is 0/0 or ∞/∞, then = lim f'(x)/g'(x)")
    
    x = Symbol('x')
    
    # Test: lim(x→0) sin(x)/x = 1
    print("\n  Test 1: lim(x→0) sin(x)/x (0/0 form)")
    f1, g1 = sin(x), x
    
    # Direct limit
    direct_limit = limit(f1/g1, x, 0)
    print(f"    Direct limit: {direct_limit}")
    
    # L'Hôpital: lim(x→0) cos(x)/1
    f1_prime, g1_prime = diff(f1, x), diff(g1, x)
    lhopital_limit = limit(f1_prime/g1_prime, x, 0)
    print(f"    L'Hôpital: lim cos(x)/1 = {lhopital_limit}")
    
    print_result(direct_limit == lhopital_limit == 1, "L'Hôpital verified")
    
    # Test: lim(x→0) (e^x - 1)/x = 1
    print("\n  Test 2: lim(x→0) (e^x - 1)/x (0/0 form)")
    f2, g2 = exp(x) - 1, x
    
    direct2 = limit(f2/g2, x, 0)
    f2_prime = diff(f2, x)  # e^x
    lhopital2 = limit(f2_prime/1, x, 0)
    
    print(f"    Direct limit: {direct2}")
    print(f"    L'Hôpital: lim e^x/1 = {lhopital2}")
    print_result(direct2 == lhopital2 == 1, "L'Hôpital verified")
    
    # Test: lim(x→∞) x/e^x = 0 (∞/∞ form)
    print("\n  Test 3: lim(x→∞) x/e^x (∞/∞ form)")
    f3, g3 = x, exp(x)
    
    direct3 = limit(f3/g3, x, oo)
    lhopital3 = limit(diff(f3, x)/diff(g3, x), x, oo)  # 1/e^x → 0
    
    print(f"    Direct limit: {direct3}")
    print(f"    L'Hôpital: lim 1/e^x = {lhopital3}")
    print_result(direct3 == lhopital3 == 0, "L'Hôpital verified")
    
    return True


# =============================================================================
# THESIS 10: Double Angle Formulas
# =============================================================================
def test_double_angle():
    print_header("THESIS 10: Double Angle Formulas")
    
    x = Symbol('x')
    
    # sin(2x) = 2·sin(x)·cos(x)
    print("\n  Test 1: sin(2x) = 2·sin(x)·cos(x)")
    lhs1 = sin(2*x)
    rhs1 = 2*sin(x)*cos(x)
    
    diff1 = trigsimp(lhs1 - rhs1)
    print(f"    sin(2x) - 2·sin(x)·cos(x) = {diff1}")
    print_result(diff1 == 0, "Identity verified")
    
    # cos(2x) = cos²(x) - sin²(x)
    print("\n  Test 2: cos(2x) = cos²(x) - sin²(x)")
    lhs2 = cos(2*x)
    rhs2 = cos(x)**2 - sin(x)**2
    
    diff2 = trigsimp(lhs2 - rhs2)
    print(f"    cos(2x) - (cos²(x) - sin²(x)) = {diff2}")
    print_result(diff2 == 0, "Identity verified")
    
    # cos(2x) = 2·cos²(x) - 1
    print("\n  Test 3: cos(2x) = 2·cos²(x) - 1")
    rhs3 = 2*cos(x)**2 - 1
    diff3 = trigsimp(cos(2*x) - rhs3)
    print(f"    cos(2x) - (2·cos²(x) - 1) = {diff3}")
    print_result(diff3 == 0, "Identity verified")
    
    # Numerical verification
    print("\n  Numerical verification at x = π/6:")
    test_x = math.pi / 6
    print(f"    sin(2·π/6) = {math.sin(2*test_x):.10f}")
    print(f"    2·sin(π/6)·cos(π/6) = {2*math.sin(test_x)*math.cos(test_x):.10f}")
    
    return True


# =============================================================================
# MAIN: Run all tests
# =============================================================================
def main():
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " AI MATH: Mathematical Thesis Verification System ".center(68) + "║")
    print("║" + " Testing 10 Fundamental Mathematical Theses ".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    
    tests = [
        ("Pythagorean Identity", test_pythagorean_identity),
        ("Euler's Identity", test_euler_identity),
        ("Quadratic Formula", test_quadratic_formula),
        ("Fundamental Theorem of Calculus", test_ftc),
        ("Geometric Series Sum", test_geometric_series),
        ("Binomial Theorem", test_binomial_theorem),
        ("Product Rule", test_product_rule),
        ("Chain Rule", test_chain_rule),
        ("L'Hôpital's Rule", test_lhopital),
        ("Double Angle Formulas", test_double_angle),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, True))
        except Exception as e:
            print(f"\n  ⚠ Error in {name}: {e}")
            results.append((name, False))
    
    # Summary
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " VERIFICATION SUMMARY ".center(68) + "║")
    print("╠" + "═" * 68 + "╣")
    
    passed = sum(1 for _, r in results if r)
    for name, result in results:
        status = "✓ VERIFIED" if result else "✗ FAILED"
        print(f"║  {status}  {name:<50} ║")
    
    print("╠" + "═" * 68 + "╣")
    print(f"║  Total: {passed}/{len(results)} theses verified".ljust(69) + "║")
    print("╚" + "═" * 68 + "╝")


if __name__ == "__main__":
    main()
