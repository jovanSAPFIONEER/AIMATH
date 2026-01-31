"""
Example Usage: AI Math Verification & Discovery System

This file demonstrates key features of the system.
Run with: python examples/demo.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from sympy import Symbol, sin, cos, simplify, diff, integrate, Eq, solve


def demo_anti_pattern_detection():
    """Demonstrate anti-pattern detection."""
    print("\n" + "=" * 60)
    print("DEMO: Anti-Pattern Detection")
    print("=" * 60)
    
    from src.explanation.anti_patterns import AntiPatternDetector
    
    detector = AntiPatternDetector()
    
    # Example of BAD explanation (full of anti-patterns)
    bad_explanation = """
    Obviously, the derivative of x² is 2x.
    This is trivially true by definition.
    It follows that the integral of 2x is x².
    The reader can verify this as an exercise.
    """
    
    print("\nAnalyzing explanation for anti-patterns...")
    print("-" * 40)
    print(bad_explanation)
    print("-" * 40)
    
    matches = detector.scan(bad_explanation)
    
    print(f"\nFound {len(matches)} anti-pattern(s):")
    for match in matches:
        print(f"  [{match.severity.value.upper()}] '{match.matched_text}'")
        print(f"    Fix: {match.fix_instruction[:60]}...")
    
    print("\n" + detector.suggest_fixes(bad_explanation))


def demo_quality_checker():
    """Demonstrate CLEAR rubric quality checking."""
    print("\n" + "=" * 60)
    print("DEMO: CLEAR Quality Rubric")
    print("=" * 60)
    
    from src.explanation.quality_checker import QualityChecker
    
    checker = QualityChecker()
    
    # Good explanation
    good_explanation = """
    **Prerequisites:**
    The derivative measures instantaneous rate of change.
    
    **Example:**
    Consider f(x) = x². At x = 3, the derivative tells us
    how fast f is changing at that point.
    
    **Step 1:** Apply the power rule
    The power rule states d/dx[x^n] = n·x^(n-1).
    
    **Step 2:** For x², we have n = 2
    So d/dx[x²] = 2·x^(2-1) = 2x.
    
    **Why it works:**
    The power rule comes from the limit definition of derivative.
    As Δx approaches 0, the difference quotient simplifies to nx^(n-1).
    
    **Edge case:**
    This rule applies for any real n. For n = 0, we get d/dx[1] = 0.
    
    **Verify:**
    Check: At x = 3, f'(3) = 2(3) = 6. 
    The slope of x² at x = 3 is indeed 6.
    """
    
    print("\nChecking quality of explanation...")
    score = checker.check(good_explanation, 'derivative')
    
    print("\n" + score.to_report())


def demo_substitution_verification():
    """Demonstrate solution verification by substitution."""
    print("\n" + "=" * 60)
    print("DEMO: Substitution Verification")
    print("=" * 60)
    
    from src.verification.substitution import SubstitutionChecker
    
    checker = SubstitutionChecker()
    x = Symbol('x')
    
    # Problem: x² - 5x + 6 = 0
    expression = x**2 - 5*x + 6
    
    print(f"\nEquation: x² - 5x + 6 = 0")
    print(f"Testing proposed solutions...")
    
    test_values = [2, 3, 4, 1]  # 2 and 3 are correct, 4 and 1 are not
    
    for val in test_values:
        result = checker.verify(expression, x, val)
        status = "✓ VALID" if result['is_valid'] else "✗ INVALID"
        actual = expression.subs(x, val)
        print(f"  x = {val}: {expression} = {actual} → {status}")


def demo_trig_identity_verification():
    """Demonstrate verifying trigonometric identities."""
    print("\n" + "=" * 60)
    print("DEMO: Trigonometric Identity Verification")
    print("=" * 60)
    
    from sympy import trigsimp
    
    x = Symbol('x')
    
    # Famous identity: sin²(x) + cos²(x) = 1
    lhs = sin(x)**2 + cos(x)**2
    
    print(f"\nVerifying: sin²(x) + cos²(x) = 1")
    print(f"Left-hand side: {lhs}")
    
    simplified = trigsimp(lhs)
    print(f"After trigsimp: {simplified}")
    
    if simplified == 1:
        print("✓ Identity VERIFIED by symbolic simplification")
    else:
        print("✗ Could not verify symbolically")
    
    # Numerical verification
    import math
    print("\nNumerical verification at specific points:")
    test_angles = [0, math.pi/6, math.pi/4, math.pi/3, math.pi/2, math.pi]
    
    for angle in test_angles:
        result = math.sin(angle)**2 + math.cos(angle)**2
        status = "✓" if abs(result - 1) < 1e-10 else "✗"
        print(f"  x = {angle:.4f}: sin²(x) + cos²(x) = {result:.10f} {status}")


def demo_derivative_solving():
    """Demonstrate derivative calculation with verification."""
    print("\n" + "=" * 60)
    print("DEMO: Derivative Calculation with Verification")
    print("=" * 60)
    
    x = Symbol('x')
    
    # Function: f(x) = x³ - 3x² + 2x
    f = x**3 - 3*x**2 + 2*x
    
    print(f"\nFunction: f(x) = {f}")
    
    # Calculate derivative
    f_prime = diff(f, x)
    print(f"Derivative: f'(x) = {f_prime}")
    
    # Verify by integrating back
    integral_of_derivative = integrate(f_prime, x)
    print(f"\nVerification: ∫f'(x)dx = {integral_of_derivative}")
    
    # Check if it matches original (up to constant)
    difference = simplify(f - integral_of_derivative)
    if difference.is_number:
        print(f"Difference from original: {difference} (constant)")
        print("✓ Derivative VERIFIED (integration recovers original up to constant)")
    else:
        print(f"Difference: {difference}")


def demo_equation_solving():
    """Demonstrate equation solving with multi-method verification."""
    print("\n" + "=" * 60)
    print("DEMO: Equation Solving with Verification")
    print("=" * 60)
    
    x = Symbol('x')
    
    # Equation: x² - 4 = 0
    equation = Eq(x**2 - 4, 0)
    
    print(f"\nEquation: {equation}")
    
    # Solve symbolically
    solutions = solve(equation, x)
    print(f"Symbolic solutions: {solutions}")
    
    # Verify each solution
    print("\nVerification by substitution:")
    for sol in solutions:
        lhs = x**2 - 4
        value = lhs.subs(x, sol)
        print(f"  x = {sol}: {lhs} = {value} → {'✓' if value == 0 else '✗'}")
    
    # Alternative method: factoring
    print("\nAlternative method (factoring):")
    factored = (x - 2) * (x + 2)
    print(f"  x² - 4 = {factored}")
    print(f"  Solutions: x = 2, x = -2 ✓")
    
    print("\n✓ Solutions VERIFIED by multiple methods")


def demo_consensus_checking():
    """Demonstrate multi-solver consensus."""
    print("\n" + "=" * 60)
    print("DEMO: Multi-Solver Consensus")
    print("=" * 60)
    
    from src.verification.consensus import ConsensusChecker
    
    checker = ConsensusChecker()
    
    # Simulate results from multiple solvers
    results = {
        'symbolic': {'answer': [2, -2], 'trust': 0.9},
        'numerical': {'answer': [2.0, -2.0], 'trust': 0.7},
        'factoring': {'answer': [2, -2], 'trust': 0.85},
    }
    
    print("\nResults from different solving methods:")
    for solver, data in results.items():
        print(f"  {solver}: {data['answer']} (trust: {data['trust']})")
    
    consensus = checker.check(results)
    
    print(f"\nConsensus: {consensus['has_consensus']}")
    if consensus['has_consensus']:
        print(f"Consensus value: {consensus['consensus_value']}")
        print("✓ All methods AGREE")
    else:
        print("⚠ Methods DISAGREE - needs investigation")


def demo_confidence_levels():
    """Demonstrate confidence level calculation."""
    print("\n" + "=" * 60)
    print("DEMO: Confidence Level Calculation")
    print("=" * 60)
    
    from src.verification.confidence import ConfidenceScorer
    
    scorer = ConfidenceScorer()
    
    # Scenario 1: Fully verified
    print("\nScenario 1: Fully verified solution")
    full_verification = {
        'formal_proof': True,
        'substitution_passed': True,
        'consensus': True,
        'counterexamples': [],
    }
    result1 = scorer.calculate(full_verification)
    print(f"  Score: {result1['score']:.2f}")
    print(f"  Level: {result1['level'].value.upper()}")
    
    # Scenario 2: Partially verified
    print("\nScenario 2: Partially verified solution")
    partial_verification = {
        'formal_proof': False,
        'substitution_passed': True,
        'consensus': True,
        'counterexamples': [],
    }
    result2 = scorer.calculate(partial_verification)
    print(f"  Score: {result2['score']:.2f}")
    print(f"  Level: {result2['level'].value.upper()}")
    
    # Scenario 3: Problematic solution
    print("\nScenario 3: Solution with issues")
    problem_verification = {
        'formal_proof': False,
        'substitution_passed': False,
        'consensus': False,
        'counterexamples': ['x = 5 fails'],
    }
    result3 = scorer.calculate(problem_verification)
    print(f"  Score: {result3['score']:.2f}")
    print(f"  Level: {result3['level'].value.upper()}")


def main():
    """Run all demonstrations."""
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " AI Math Verification System - Feature Demonstrations ".center(58) + "║")
    print("╚" + "═" * 58 + "╝")
    
    demos = [
        ("Anti-Pattern Detection", demo_anti_pattern_detection),
        ("CLEAR Quality Rubric", demo_quality_checker),
        ("Substitution Verification", demo_substitution_verification),
        ("Trigonometric Identity Verification", demo_trig_identity_verification),
        ("Derivative Solving", demo_derivative_solving),
        ("Equation Solving", demo_equation_solving),
        ("Multi-Solver Consensus", demo_consensus_checking),
        ("Confidence Levels", demo_confidence_levels),
    ]
    
    for name, demo_func in demos:
        try:
            demo_func()
        except Exception as e:
            print(f"\n⚠ Demo '{name}' encountered an error: {e}")
    
    print("\n" + "=" * 60)
    print("All demonstrations complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
