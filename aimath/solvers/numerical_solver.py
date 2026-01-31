"""
Numerical Solver - NumPy/SciPy-based numerical computation.

Trust Level: MEDIUM (0.7)

Used for verification and problems that don't have closed-form solutions.
Results are approximate due to floating-point representation.
"""

import time
from typing import Any, Optional, Callable
import logging

from .solver_base import BaseSolver, SolverResult, SolverStatus
from ..core.types import MathProblem, ProblemType, SolutionStep

logger = logging.getLogger(__name__)


class NumericalSolver(BaseSolver):
    """
    Numerical solver using NumPy and SciPy.
    
    Trust Level: MEDIUM (0.7)
    
    Provides numerical solutions when symbolic methods fail or
    for verification of symbolic results. Results are approximate.
    
    Supported problem types:
    - Numerical root finding
    - Numerical integration
    - Numerical differentiation
    - Optimization
    
    Example:
        >>> solver = NumericalSolver()
        >>> result = solver.solve(problem)
        >>> print(result.answer)  # Numerical approximation
    """
    
    name = "numerical"
    trust_level = 0.7  # Medium trust - floating point limitations
    
    def __init__(
        self,
        timeout: Optional[float] = 10.0,
        tolerance: float = 1e-10,
        max_iterations: int = 1000,
    ):
        """
        Initialize numerical solver.
        
        Args:
            timeout: Maximum solving time (seconds)
            tolerance: Numerical tolerance for convergence
            max_iterations: Maximum iterations for iterative methods
        """
        super().__init__(timeout)
        self.tolerance = tolerance
        self.max_iterations = max_iterations
    
    def can_solve(self, problem: MathProblem) -> bool:
        """Check if problem can be solved numerically."""
        supported_types = {
            ProblemType.EQUATION,
            ProblemType.INTEGRATION,
            ProblemType.DIFFERENTIATION,
            ProblemType.OPTIMIZATION,
            ProblemType.GENERAL,
        }
        return problem.problem_type in supported_types
    
    def solve(self, problem: MathProblem) -> SolverResult:
        """Solve problem numerically."""
        start_time = time.time()
        
        if not self.can_solve(problem):
            return self._create_unsupported_result(
                f"Problem type {problem.problem_type} not supported"
            )
        
        try:
            import numpy as np
            
            # Route to appropriate solver
            if problem.problem_type == ProblemType.EQUATION:
                answer = self._find_roots(problem)
            elif problem.problem_type == ProblemType.INTEGRATION:
                answer = self._numerical_integrate(problem)
            elif problem.problem_type == ProblemType.DIFFERENTIATION:
                answer = self._numerical_differentiate(problem)
            elif problem.problem_type == ProblemType.OPTIMIZATION:
                answer = self._optimize(problem)
            else:
                answer = self._evaluate(problem)
            
            elapsed = (time.time() - start_time) * 1000
            
            return SolverResult(
                status=SolverStatus.SUCCESS,
                answer=answer,
                method_name=self.name,
                confidence=self.trust_level,
                computation_time_ms=elapsed,
                metadata={
                    'tolerance': self.tolerance,
                    'method': 'numerical',
                }
            )
            
        except Exception as e:
            logger.error(f"Numerical solver error: {e}")
            return self._create_error_result(str(e))
    
    def _find_roots(self, problem: MathProblem) -> list[float]:
        """
        Find numerical roots of an equation.
        
        Uses multiple initial guesses to find all real roots.
        """
        import numpy as np
        from scipy.optimize import fsolve, brentq
        
        expr = problem.parsed_expression
        
        # Convert to callable function
        func = self._expr_to_callable(expr, problem)
        if func is None:
            raise ValueError("Could not convert expression to function")
        
        roots = []
        
        # Try multiple starting points
        initial_guesses = np.linspace(-10, 10, 21)
        
        for x0 in initial_guesses:
            try:
                root, info, ier, msg = fsolve(func, x0, full_output=True)
                if ier == 1:  # Converged
                    root_val = float(root[0])
                    # Check if this root is new (not already found)
                    is_new = all(
                        abs(root_val - r) > self.tolerance 
                        for r in roots
                    )
                    # Verify it's actually a root
                    if is_new and abs(func(root_val)) < self.tolerance:
                        roots.append(root_val)
            except Exception:
                continue
        
        # Sort roots
        roots.sort()
        
        # Round to remove floating point noise
        roots = [round(r, 10) for r in roots]
        
        return roots
    
    def _numerical_integrate(self, problem: MathProblem) -> float:
        """
        Compute definite integral numerically.
        
        For indefinite integrals, returns None (use symbolic).
        """
        import numpy as np
        from scipy.integrate import quad
        
        expr = problem.parsed_expression
        func = self._expr_to_callable(expr, problem)
        
        if func is None:
            raise ValueError("Could not convert expression to function")
        
        # Check for integration bounds in constraints
        lower, upper = -np.inf, np.inf
        for constraint in problem.constraints:
            if constraint.get('type') == 'lower_bound':
                lower = float(constraint.get('value', -np.inf))
            elif constraint.get('type') == 'upper_bound':
                upper = float(constraint.get('value', np.inf))
        
        # If no bounds specified, can't do numerical integration
        if np.isinf(lower) and np.isinf(upper):
            raise ValueError(
                "Numerical integration requires finite bounds. "
                "Use symbolic solver for indefinite integrals."
            )
        
        result, error = quad(func, lower, upper)
        
        return {
            'value': result,
            'error_estimate': error,
            'bounds': (lower, upper),
        }
    
    def _numerical_differentiate(self, problem: MathProblem) -> Any:
        """
        Compute derivative numerically at a point.
        
        Uses central difference method for better accuracy.
        """
        import numpy as np
        
        expr = problem.parsed_expression
        func = self._expr_to_callable(expr, problem)
        
        if func is None:
            raise ValueError("Could not convert expression to function")
        
        h = 1e-8  # Step size
        
        # Check for evaluation point in constraints
        point = 0.0  # Default
        for constraint in problem.constraints:
            if constraint.get('type') == 'eval_point':
                point = float(constraint.get('value', 0))
        
        # Central difference formula (more accurate)
        derivative = (func(point + h) - func(point - h)) / (2 * h)
        
        return {
            'value': derivative,
            'at_point': point,
            'method': 'central_difference',
        }
    
    def _optimize(self, problem: MathProblem) -> Any:
        """Find minimum/maximum of a function."""
        import numpy as np
        from scipy.optimize import minimize_scalar, minimize
        
        expr = problem.parsed_expression
        func = self._expr_to_callable(expr, problem)
        
        if func is None:
            raise ValueError("Could not convert expression to function")
        
        # Try to find minimum
        result_min = minimize_scalar(func)
        
        # Find maximum by minimizing negative
        def neg_func(x):
            return -func(x)
        result_max = minimize_scalar(neg_func)
        
        return {
            'minimum': {
                'x': result_min.x,
                'value': result_min.fun,
            },
            'maximum': {
                'x': result_max.x,
                'value': -result_max.fun,
            },
        }
    
    def _evaluate(self, problem: MathProblem) -> float:
        """Evaluate expression at specific values."""
        import numpy as np
        
        expr = problem.parsed_expression
        func = self._expr_to_callable(expr, problem)
        
        if func is None:
            # Try to evaluate directly with SymPy
            try:
                return float(expr.evalf())
            except Exception:
                raise ValueError("Could not evaluate expression")
        
        # Default evaluation at x=1
        return func(1.0)
    
    def _expr_to_callable(
        self, 
        expr: Any, 
        problem: MathProblem
    ) -> Optional[Callable]:
        """
        Convert SymPy expression to callable function.
        
        Uses lambdify for efficient numerical evaluation.
        """
        try:
            from sympy import lambdify, Symbol
            import numpy as np
            
            # Get variable
            if problem.variables:
                var = Symbol(problem.variables[0])
            elif hasattr(expr, 'free_symbols') and expr.free_symbols:
                var = list(expr.free_symbols)[0]
            else:
                var = Symbol('x')
            
            # Handle different expression types
            if hasattr(expr, 'lhs') and hasattr(expr, 'rhs'):
                # Equation: convert to f(x) = lhs - rhs = 0
                expr = expr.lhs - expr.rhs
            
            # Create callable
            func = lambdify(var, expr, modules=['numpy'])
            
            return func
            
        except Exception as e:
            logger.warning(f"Could not convert to callable: {e}")
            return None
    
    def verify_against(
        self,
        symbolic_answer: Any,
        problem: MathProblem,
        test_points: int = 10,
    ) -> dict:
        """
        Verify a symbolic answer numerically.
        
        Evaluates the symbolic answer at multiple points and
        compares with direct numerical computation.
        
        Args:
            symbolic_answer: The answer to verify
            problem: Original problem
            test_points: Number of test points
            
        Returns:
            Verification results with agreement percentage
        """
        import numpy as np
        
        results = {
            'agreement': 0.0,
            'test_points': test_points,
            'failures': [],
        }
        
        try:
            # Get functions
            original_func = self._expr_to_callable(
                problem.parsed_expression, problem
            )
            answer_func = self._expr_to_callable(symbolic_answer, problem)
            
            if original_func is None or answer_func is None:
                results['error'] = "Could not convert to functions"
                return results
            
            # Test at multiple points
            test_x = np.linspace(-5, 5, test_points)
            agreements = 0
            
            for x in test_x:
                try:
                    orig_val = original_func(x)
                    ans_val = answer_func(x)
                    
                    if np.isclose(orig_val, ans_val, rtol=self.tolerance):
                        agreements += 1
                    else:
                        results['failures'].append({
                            'x': x,
                            'expected': orig_val,
                            'got': ans_val,
                        })
                except Exception:
                    # Skip points where evaluation fails
                    test_points -= 1
            
            if test_points > 0:
                results['agreement'] = agreements / test_points
            
        except Exception as e:
            results['error'] = str(e)
        
        return results
