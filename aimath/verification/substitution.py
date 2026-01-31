"""
Substitution Checker - Verify by plugging answer back in.

The most fundamental verification: if x=2 is a solution to
x^2 - 4 = 0, then (2)^2 - 4 should equal 0.
"""

from typing import Any
import logging

from ..core.types import MathProblem, VerificationCheck, ProblemType

logger = logging.getLogger(__name__)


class SubstitutionChecker:
    """
    Verify solutions by substitution.
    
    This is the most reliable verification for equations:
    plug the answer back in and check if it satisfies the equation.
    
    Example:
        >>> checker = SubstitutionChecker()
        >>> check = checker.check(problem, answer=[2, 3])
        >>> print(check.passed)  # True if both satisfy equation
    """
    
    def __init__(self, tolerance: float = 1e-10):
        """
        Initialize checker.
        
        Args:
            tolerance: Numerical tolerance for equality
        """
        self.tolerance = tolerance
    
    def check(
        self, 
        problem: MathProblem, 
        answer: Any
    ) -> VerificationCheck:
        """
        Verify answer by substitution.
        
        For equations: substitute answer and verify = 0
        For derivatives: verify derivative rule application
        For integrals: verify by differentiating
        
        Args:
            problem: Original problem
            answer: Proposed answer
            
        Returns:
            VerificationCheck with pass/fail status
        """
        if answer is None:
            return VerificationCheck(
                check_type='substitution',
                passed=False,
                details="No answer to verify",
                error="Answer is None",
            )
        
        try:
            if problem.problem_type == ProblemType.EQUATION:
                return self._check_equation(problem, answer)
            elif problem.problem_type == ProblemType.DIFFERENTIATION:
                return self._check_derivative(problem, answer)
            elif problem.problem_type == ProblemType.INTEGRATION:
                return self._check_integral(problem, answer)
            else:
                return self._check_general(problem, answer)
                
        except Exception as e:
            logger.warning(f"Substitution check error: {e}")
            return VerificationCheck(
                check_type='substitution',
                passed=False,
                details=f"Substitution check failed with error",
                error=str(e),
            )
    
    def _check_equation(
        self, 
        problem: MathProblem, 
        answer: Any
    ) -> VerificationCheck:
        """
        Check equation solution by substitution.
        
        For f(x) = 0, verify f(answer) = 0.
        """
        from sympy import Symbol, Eq, simplify, N, Abs
        
        expr = problem.parsed_expression
        
        # Get variable
        if problem.variables:
            var = Symbol(problem.variables[0])
        elif hasattr(expr, 'free_symbols') and expr.free_symbols:
            var = list(expr.free_symbols)[0]
        else:
            var = Symbol('x')
        
        # Get equation expression (should equal 0)
        if isinstance(expr, Eq):
            check_expr = expr.lhs - expr.rhs
        else:
            check_expr = expr
        
        # Handle list of solutions
        answers = answer if isinstance(answer, (list, tuple, set)) else [answer]
        
        results = []
        for ans in answers:
            # Substitute answer into expression
            substituted = check_expr.subs(var, ans)
            
            # Simplify
            simplified = simplify(substituted)
            
            # Check if zero
            try:
                # Try numerical evaluation
                numerical = complex(N(simplified))
                is_zero = abs(numerical) < self.tolerance
            except (TypeError, ValueError):
                # Try symbolic check
                is_zero = simplified == 0 or simplify(simplified) == 0
            
            results.append({
                'value': ans,
                'substituted': str(simplified),
                'is_zero': is_zero,
            })
        
        all_passed = all(r['is_zero'] for r in results)
        
        return VerificationCheck(
            check_type='substitution',
            passed=all_passed,
            details=(
                f"All {len(results)} solution(s) verified" if all_passed
                else f"Some solutions failed verification"
            ),
            evidence=results,
            error=None if all_passed else "Not all solutions satisfy equation",
        )
    
    def _check_derivative(
        self, 
        problem: MathProblem, 
        answer: Any
    ) -> VerificationCheck:
        """
        Check derivative by verifying against SymPy's diff.
        
        Note: This checks that our answer matches what SymPy computes.
        """
        from sympy import diff, Symbol, simplify, Derivative
        
        expr = problem.parsed_expression
        
        # If expression is already a Derivative, get the inner expression
        if isinstance(expr, Derivative):
            inner_expr = expr.args[0]
            var = expr.args[1][0] if len(expr.args) > 1 else Symbol('x')
        else:
            inner_expr = expr
            if problem.variables:
                var = Symbol(problem.variables[0])
            elif hasattr(expr, 'free_symbols') and expr.free_symbols:
                var = list(expr.free_symbols)[0]
            else:
                var = Symbol('x')
        
        # Compute expected derivative
        expected = diff(inner_expr, var)
        
        # Compare with given answer
        difference = simplify(answer - expected)
        
        is_equal = difference == 0
        
        return VerificationCheck(
            check_type='substitution',
            passed=is_equal,
            details=(
                "Derivative verified by SymPy" if is_equal
                else f"Derivative mismatch: expected {expected}, got {answer}"
            ),
            evidence={
                'expected': str(expected),
                'got': str(answer),
                'difference': str(difference),
            },
        )
    
    def _check_integral(
        self, 
        problem: MathProblem, 
        answer: Any
    ) -> VerificationCheck:
        """
        Check integral by differentiating the answer.
        
        If ∫f(x)dx = F(x) + C, then F'(x) should equal f(x).
        """
        from sympy import diff, Symbol, simplify, Integral
        
        expr = problem.parsed_expression
        
        # Get the integrand
        if isinstance(expr, Integral):
            integrand = expr.args[0]
            var = expr.args[1][0] if len(expr.args) > 1 else Symbol('x')
        else:
            integrand = expr
            if problem.variables:
                var = Symbol(problem.variables[0])
            elif hasattr(expr, 'free_symbols') and expr.free_symbols:
                var = list(expr.free_symbols)[0]
            else:
                var = Symbol('x')
        
        # Differentiate the answer
        derivative_of_answer = diff(answer, var)
        
        # Compare with original integrand
        difference = simplify(derivative_of_answer - integrand)
        
        is_equal = difference == 0
        
        return VerificationCheck(
            check_type='substitution',
            passed=is_equal,
            details=(
                "Integral verified: derivative of answer equals integrand" 
                if is_equal else
                f"Integral mismatch: d/dx({answer}) = {derivative_of_answer} ≠ {integrand}"
            ),
            evidence={
                'integrand': str(integrand),
                'answer': str(answer),
                'derivative_of_answer': str(derivative_of_answer),
            },
        )
    
    def _check_general(
        self, 
        problem: MathProblem, 
        answer: Any
    ) -> VerificationCheck:
        """
        General verification for other problem types.
        
        When specific verification isn't available, we do basic
        sanity checks.
        """
        from sympy import simplify, N
        
        # Basic sanity checks
        checks_passed = []
        
        # Check 1: Answer is not None/undefined
        if answer is not None:
            checks_passed.append("answer_exists")
        
        # Check 2: Simplified answer equals itself
        try:
            simplified = simplify(answer)
            if simplified is not None:
                checks_passed.append("simplifiable")
        except Exception:
            pass
        
        # Check 3: Can evaluate numerically
        try:
            N(answer)
            checks_passed.append("evaluable")
        except Exception:
            pass
        
        passed = len(checks_passed) >= 2
        
        return VerificationCheck(
            check_type='substitution',
            passed=passed,
            details=f"Basic checks passed: {', '.join(checks_passed)}",
            evidence={'checks': checks_passed},
        )
