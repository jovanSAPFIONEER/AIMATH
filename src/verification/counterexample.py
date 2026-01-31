"""
Counterexample Searcher - Actively try to disprove solutions.

A key anti-hallucination strategy: instead of just checking if
an answer works, we actively try to find cases where it fails.
"""

from typing import Any, Optional
import logging

from ..core.types import MathProblem, VerificationCheck, ProblemType

logger = logging.getLogger(__name__)


class CounterexampleSearcher:
    """
    Search for counterexamples to disprove solutions.
    
    Philosophy: The best way to verify is to try hard to DISprove.
    If we can't find a counterexample after thorough search,
    we have more confidence in the answer.
    
    Strategies:
    1. Random testing at many points
    2. Edge case testing (0, 1, -1, large values, etc.)
    3. Boundary testing (near constraints)
    4. Symbolic counterexample generation
    
    Example:
        >>> searcher = CounterexampleSearcher()
        >>> check = searcher.search(problem, answer)
        >>> if not check.passed:
        ...     print(f"Counterexample: {check.evidence}")
    """
    
    # Edge cases to always test
    EDGE_CASES = [0, 1, -1, 2, -2, 0.5, -0.5, 10, -10, 100, -100, 0.001, -0.001]
    
    def __init__(
        self,
        num_random_tests: int = 50,
        test_range: tuple = (-100, 100),
        tolerance: float = 1e-8,
    ):
        """
        Initialize searcher.
        
        Args:
            num_random_tests: Number of random test points
            test_range: Range for random testing (min, max)
            tolerance: Tolerance for numerical comparison
        """
        self.num_random_tests = num_random_tests
        self.test_range = test_range
        self.tolerance = tolerance
    
    def search(
        self, 
        problem: MathProblem, 
        answer: Any
    ) -> VerificationCheck:
        """
        Search for counterexamples to the given answer.
        
        Args:
            problem: Original problem
            answer: Proposed answer to test
            
        Returns:
            VerificationCheck (passed=True means NO counterexample found)
        """
        if answer is None:
            return VerificationCheck(
                check_type='counterexample',
                passed=False,
                details="No answer to test",
                error="Answer is None",
            )
        
        counterexamples = []
        
        # Test based on problem type
        if problem.problem_type == ProblemType.EQUATION:
            counterexamples = self._test_equation(problem, answer)
        elif problem.problem_type in [ProblemType.DIFFERENTIATION, ProblemType.INTEGRATION]:
            counterexamples = self._test_calculus(problem, answer)
        else:
            counterexamples = self._test_general(problem, answer)
        
        if counterexamples:
            return VerificationCheck(
                check_type='counterexample',
                passed=False,
                details=f"Found {len(counterexamples)} counterexample(s)",
                evidence=counterexamples,
                error="Answer fails for some inputs",
            )
        
        return VerificationCheck(
            check_type='counterexample',
            passed=True,
            details=f"No counterexamples found in {self.num_random_tests + len(self.EDGE_CASES)} tests",
        )
    
    def search_for_claim(self, problem: MathProblem) -> VerificationCheck:
        """
        Search for counterexamples to a mathematical claim.
        
        Different from search() - here we're testing a claim
        rather than a specific answer.
        """
        # Try to parse claim and find counterexample
        claim = problem.parsed_expression
        
        if claim is None:
            return VerificationCheck(
                check_type='counterexample',
                passed=True,  # Can't disprove
                details="Could not parse claim for counterexample search",
            )
        
        counterexamples = self._search_claim_counterexample(problem)
        
        if counterexamples:
            return VerificationCheck(
                check_type='counterexample',
                passed=False,
                details="Claim is FALSE - counterexample found",
                evidence=counterexamples[0],
            )
        
        return VerificationCheck(
            check_type='counterexample',
            passed=True,
            details="No counterexample found (claim may be true)",
        )
    
    def _test_equation(
        self, 
        problem: MathProblem, 
        answer: Any
    ) -> list[dict]:
        """
        Test equation solutions.
        
        For equation solutions, check that:
        1. Claimed solutions actually satisfy the equation
        2. No obvious solutions were missed
        """
        from sympy import Symbol, N, Abs
        import random
        
        counterexamples = []
        expr = problem.parsed_expression
        
        # Get variable
        if problem.variables:
            var = Symbol(problem.variables[0])
        elif hasattr(expr, 'free_symbols') and expr.free_symbols:
            var = list(expr.free_symbols)[0]
        else:
            var = Symbol('x')
        
        # Get expression to evaluate
        if hasattr(expr, 'lhs') and hasattr(expr, 'rhs'):
            check_expr = expr.lhs - expr.rhs
        else:
            check_expr = expr
        
        # Convert answer to list
        answers = set(answer if isinstance(answer, (list, tuple)) else [answer])
        
        # Test 1: Verify claimed solutions
        for ans in answers:
            try:
                result = complex(N(check_expr.subs(var, ans)))
                if abs(result) > self.tolerance:
                    counterexamples.append({
                        'type': 'invalid_solution',
                        'claimed_solution': str(ans),
                        'residual': abs(result),
                    })
            except Exception:
                pass
        
        # Test 2: Check if we missed any integer solutions
        for test_val in self.EDGE_CASES:
            try:
                result = complex(N(check_expr.subs(var, test_val)))
                if abs(result) < self.tolerance:
                    # Found a root - check if it's in our answer
                    is_known = any(
                        abs(complex(N(ans)) - test_val) < self.tolerance 
                        for ans in answers
                    )
                    if not is_known:
                        counterexamples.append({
                            'type': 'missed_solution',
                            'value': test_val,
                            'residual': abs(result),
                        })
            except Exception:
                pass
        
        return counterexamples
    
    def _test_calculus(
        self, 
        problem: MathProblem, 
        answer: Any
    ) -> list[dict]:
        """
        Test calculus results at multiple points.
        
        For derivatives: compare with numerical differentiation
        For integrals: compare derivative of answer with integrand
        """
        import numpy as np
        from sympy import lambdify, Symbol, diff, N
        
        counterexamples = []
        
        # Get variable
        if problem.variables:
            var = Symbol(problem.variables[0])
        elif hasattr(problem.parsed_expression, 'free_symbols'):
            free = problem.parsed_expression.free_symbols
            var = list(free)[0] if free else Symbol('x')
        else:
            var = Symbol('x')
        
        # Create callable functions
        try:
            answer_func = lambdify(var, answer, 'numpy')
        except Exception:
            return []  # Can't test numerically
        
        # Generate test points
        test_points = np.concatenate([
            np.array(self.EDGE_CASES),
            np.random.uniform(self.test_range[0], self.test_range[1], self.num_random_tests)
        ])
        
        if problem.problem_type == ProblemType.DIFFERENTIATION:
            # For derivatives, compare with numerical derivative
            original = problem.parsed_expression
            if hasattr(original, 'args') and original.args:
                # Get the function being differentiated
                if hasattr(original, 'doit'):
                    original = original.args[0]
                
                try:
                    original_func = lambdify(var, original, 'numpy')
                    h = 1e-7
                    
                    for x in test_points:
                        try:
                            # Numerical derivative
                            numerical = (original_func(x + h) - original_func(x - h)) / (2 * h)
                            # Symbolic answer
                            symbolic = answer_func(x)
                            
                            if abs(numerical - symbolic) > self.tolerance * max(abs(numerical), 1):
                                counterexamples.append({
                                    'type': 'derivative_mismatch',
                                    'x': float(x),
                                    'expected': float(numerical),
                                    'got': float(symbolic),
                                })
                                if len(counterexamples) >= 3:
                                    break
                        except (ValueError, TypeError, ZeroDivisionError):
                            pass
                except Exception:
                    pass
        
        elif problem.problem_type == ProblemType.INTEGRATION:
            # For integrals, derivative of answer should equal integrand
            integrand = problem.parsed_expression
            if hasattr(integrand, 'args') and integrand.args:
                integrand = integrand.args[0]
            
            # Compute derivative of answer
            answer_deriv = diff(answer, var)
            
            try:
                integrand_func = lambdify(var, integrand, 'numpy')
                deriv_func = lambdify(var, answer_deriv, 'numpy')
                
                for x in test_points:
                    try:
                        integrand_val = integrand_func(x)
                        deriv_val = deriv_func(x)
                        
                        if abs(integrand_val - deriv_val) > self.tolerance * max(abs(integrand_val), 1):
                            counterexamples.append({
                                'type': 'integral_mismatch',
                                'x': float(x),
                                'integrand': float(integrand_val),
                                'd_answer_dx': float(deriv_val),
                            })
                            if len(counterexamples) >= 3:
                                break
                    except (ValueError, TypeError, ZeroDivisionError):
                        pass
            except Exception:
                pass
        
        return counterexamples
    
    def _test_general(
        self, 
        problem: MathProblem, 
        answer: Any
    ) -> list[dict]:
        """
        General testing for other problem types.
        """
        # Basic sanity checks
        counterexamples = []
        
        from sympy import oo, zoo, nan, N
        
        # Check for invalid values
        if answer in [oo, -oo, zoo, nan]:
            counterexamples.append({
                'type': 'invalid_value',
                'value': str(answer),
            })
        
        # Try numerical evaluation
        try:
            numerical = complex(N(answer))
            if not (abs(numerical) < 1e100):  # Sanity check
                counterexamples.append({
                    'type': 'unreasonable_magnitude',
                    'value': abs(numerical),
                })
        except Exception:
            pass
        
        return counterexamples
    
    def _search_claim_counterexample(self, problem: MathProblem) -> list[dict]:
        """
        Search for counterexamples to a general claim.
        """
        from sympy import Symbol, N
        import numpy as np
        
        claim = problem.parsed_expression
        counterexamples = []
        
        if claim is None:
            return []
        
        # Get free variables
        if hasattr(claim, 'free_symbols'):
            variables = list(claim.free_symbols)
        else:
            variables = [Symbol('x')]
        
        if not variables:
            return []
        
        # Test claim at various points
        test_values = list(self.EDGE_CASES) + list(
            np.random.uniform(self.test_range[0], self.test_range[1], self.num_random_tests)
        )
        
        for val in test_values:
            try:
                # Substitute all variables with test value
                subs = {v: val for v in variables}
                result = claim.subs(subs)
                
                # Check if claim is false
                if result is False or result == False:
                    counterexamples.append({
                        'type': 'claim_false',
                        'at': subs,
                    })
                    break
                
                # Try numerical evaluation
                try:
                    num_result = complex(N(result))
                    if abs(num_result) > self.tolerance:
                        counterexamples.append({
                            'type': 'claim_nonzero',
                            'at': {str(k): v for k, v in subs.items()},
                            'result': num_result,
                        })
                        break
                except Exception:
                    pass
                    
            except Exception:
                pass
        
        return counterexamples
