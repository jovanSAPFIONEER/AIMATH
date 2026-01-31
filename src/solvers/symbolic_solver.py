"""
Symbolic Solver - SymPy-based mathematical solving.

This is the primary solver with HIGH trust level.
Uses computer algebra for exact symbolic computation.
"""

import time
from typing import Any, Optional
import logging

from .solver_base import BaseSolver, SolverResult, SolverStatus
from ..core.types import MathProblem, ProblemType, SolutionStep

logger = logging.getLogger(__name__)


class SymbolicSolver(BaseSolver):
    """
    Symbolic mathematics solver using SymPy.
    
    Trust Level: HIGH (0.9)
    
    This solver provides exact symbolic solutions through
    computer algebra. It's the backbone of the verification
    system because symbolic results can be formally verified.
    
    Supported problem types:
    - Equations and systems
    - Derivatives and integrals
    - Limits
    - Simplification
    - Inequalities
    
    Example:
        >>> solver = SymbolicSolver()
        >>> result = solver.solve(problem)
        >>> print(result.answer)  # Exact symbolic solution
    """
    
    name = "symbolic"
    trust_level = 0.9  # High trust - exact computation
    
    def __init__(self, timeout: Optional[float] = 10.0):
        """Initialize symbolic solver."""
        super().__init__(timeout)
        self._sympy_imported = False
    
    def _ensure_sympy(self):
        """Ensure SymPy is imported."""
        if not self._sympy_imported:
            import sympy
            self._sympy_imported = True
    
    def can_solve(self, problem: MathProblem) -> bool:
        """
        Check if problem can be solved symbolically.
        
        Most algebraic problems can be handled.
        """
        supported_types = {
            ProblemType.EQUATION,
            ProblemType.INEQUALITY,
            ProblemType.SIMPLIFICATION,
            ProblemType.DIFFERENTIATION,
            ProblemType.INTEGRATION,
            ProblemType.LIMIT,
            ProblemType.SYSTEM,
            ProblemType.SERIES,
            ProblemType.GENERAL,
        }
        return problem.problem_type in supported_types
    
    def solve(self, problem: MathProblem) -> SolverResult:
        """
        Solve problem using SymPy.
        
        Routes to specialized solving methods based on problem type.
        """
        start_time = time.time()
        
        if not self.can_solve(problem):
            return self._create_unsupported_result(
                f"Problem type {problem.problem_type} not supported"
            )
        
        try:
            self._ensure_sympy()
            
            # Route to appropriate solver
            if problem.problem_type == ProblemType.EQUATION:
                answer = self._solve_equation(problem)
            elif problem.problem_type == ProblemType.DIFFERENTIATION:
                answer = self._differentiate(problem)
            elif problem.problem_type == ProblemType.INTEGRATION:
                answer = self._integrate(problem)
            elif problem.problem_type == ProblemType.LIMIT:
                answer = self._limit(problem)
            elif problem.problem_type == ProblemType.SIMPLIFICATION:
                answer = self._simplify(problem)
            elif problem.problem_type == ProblemType.INEQUALITY:
                answer = self._solve_inequality(problem)
            elif problem.problem_type == ProblemType.SYSTEM:
                answer = self._solve_system(problem)
            elif problem.problem_type == ProblemType.SERIES:
                answer = self._solve_series(problem)
            else:
                answer = self._solve_general(problem)
            
            elapsed = (time.time() - start_time) * 1000
            
            # Generate solution steps
            steps = self.get_steps(problem, answer)
            
            return SolverResult(
                status=SolverStatus.SUCCESS,
                answer=answer,
                steps=steps,
                method_name=self.name,
                confidence=self.trust_level,
                computation_time_ms=elapsed,
            )
            
        except Exception as e:
            logger.error(f"Symbolic solver error: {e}")
            return self._create_error_result(str(e))
    
    def _solve_equation(self, problem: MathProblem) -> Any:
        """Solve an equation."""
        from sympy import solve, Eq, Symbol, symbols
        
        expr = problem.parsed_expression
        
        # Determine variable to solve for
        if problem.variables:
            var = Symbol(problem.variables[0])
        elif hasattr(expr, 'free_symbols') and expr.free_symbols:
            var = list(expr.free_symbols)[0]
        else:
            var = Symbol('x')
        
        # Handle different expression types
        if isinstance(expr, Eq):
            # Already an equation
            solutions = solve(expr, var)
        elif hasattr(expr, 'lhs') and hasattr(expr, 'rhs'):
            # Equation-like object
            solutions = solve(Eq(expr.lhs, expr.rhs), var)
        else:
            # Assume expr = 0
            solutions = solve(expr, var)
        
        return solutions
    
    def _differentiate(self, problem: MathProblem) -> Any:
        """Compute derivative."""
        from sympy import diff, Derivative, Symbol
        
        expr = problem.parsed_expression
        
        # If already a Derivative object, evaluate it
        if isinstance(expr, Derivative):
            return expr.doit()
        
        # Determine differentiation variable
        if problem.variables:
            var = Symbol(problem.variables[0])
        elif hasattr(expr, 'free_symbols') and expr.free_symbols:
            var = list(expr.free_symbols)[0]
        else:
            var = Symbol('x')
        
        return diff(expr, var)
    
    def _integrate(self, problem: MathProblem) -> Any:
        """Compute integral."""
        from sympy import integrate, Integral, Symbol
        
        expr = problem.parsed_expression
        
        # If already an Integral object, evaluate it
        if isinstance(expr, Integral):
            return expr.doit()
        
        # Determine integration variable
        if problem.variables:
            var = Symbol(problem.variables[0])
        elif hasattr(expr, 'free_symbols') and expr.free_symbols:
            var = list(expr.free_symbols)[0]
        else:
            var = Symbol('x')
        
        return integrate(expr, var)
    
    def _limit(self, problem: MathProblem) -> Any:
        """Evaluate limit."""
        from sympy import limit, Limit, Symbol, oo
        
        expr = problem.parsed_expression
        
        # If already a Limit object, evaluate it
        if isinstance(expr, Limit):
            return expr.doit()
        
        # Determine variable and point
        if problem.variables:
            var = Symbol(problem.variables[0])
        elif hasattr(expr, 'free_symbols') and expr.free_symbols:
            var = list(expr.free_symbols)[0]
        else:
            var = Symbol('x')
        
        # Default limit at 0 (should be specified in problem)
        point = 0
        for constraint in problem.constraints:
            if constraint.get('type') == 'limit_point':
                point = constraint.get('value', 0)
        
        return limit(expr, var, point)
    
    def _simplify(self, problem: MathProblem) -> Any:
        """Simplify expression."""
        from sympy import simplify, expand, factor, trigsimp, cancel
        
        expr = problem.parsed_expression
        
        # Try multiple simplification strategies
        simplified = simplify(expr)
        
        # Also try factoring if result seems complex
        try:
            factored = factor(expr)
            if len(str(factored)) < len(str(simplified)):
                return factored
        except Exception:
            pass
        
        return simplified
    
    def _solve_inequality(self, problem: MathProblem) -> Any:
        """Solve inequality."""
        from sympy import solve, Symbol
        from sympy.solvers.inequalities import solve_univariate_inequality
        
        expr = problem.parsed_expression
        
        if problem.variables:
            var = Symbol(problem.variables[0])
        elif hasattr(expr, 'free_symbols') and expr.free_symbols:
            var = list(expr.free_symbols)[0]
        else:
            var = Symbol('x')
        
        try:
            return solve_univariate_inequality(expr, var)
        except Exception:
            return solve(expr, var)
    
    def _solve_system(self, problem: MathProblem) -> Any:
        """Solve system of equations."""
        from sympy import solve, symbols
        
        expr = problem.parsed_expression
        
        # For systems, expression should be a list/tuple of equations
        if isinstance(expr, (list, tuple)):
            equations = expr
        else:
            equations = [expr]
        
        # Get all variables
        all_vars = set()
        for eq in equations:
            if hasattr(eq, 'free_symbols'):
                all_vars.update(eq.free_symbols)
        
        return solve(equations, list(all_vars))
    
    def _solve_series(self, problem: MathProblem) -> Any:
        """Evaluate series/sequence."""
        from sympy import summation, Sum, Symbol, oo
        
        expr = problem.parsed_expression
        
        if isinstance(expr, Sum):
            return expr.doit()
        
        # Need more context for series problems
        return expr
    
    def _solve_general(self, problem: MathProblem) -> Any:
        """General solver - try common operations."""
        from sympy import simplify, solve, Symbol
        
        expr = problem.parsed_expression
        
        # If it's an equation (has =), try to solve
        if hasattr(expr, 'lhs'):
            return solve(expr)
        
        # Otherwise simplify
        return simplify(expr)
    
    def get_steps(self, problem: MathProblem, answer: Any) -> list[SolutionStep]:
        """
        Generate solution steps for the problem.
        
        This is crucial for explanations - we trace how the
        solution was obtained.
        """
        steps = []
        
        # Step 1: State the problem
        steps.append(SolutionStep(
            action="State the problem",
            expression=problem.parsed_expression,
            justification="We begin by clearly stating what we need to solve.",
        ))
        
        # Problem-type specific steps
        if problem.problem_type == ProblemType.EQUATION:
            steps.extend(self._equation_steps(problem, answer))
        elif problem.problem_type == ProblemType.DIFFERENTIATION:
            steps.extend(self._derivative_steps(problem, answer))
        elif problem.problem_type == ProblemType.INTEGRATION:
            steps.extend(self._integral_steps(problem, answer))
        
        # Final step: State the answer
        steps.append(SolutionStep(
            action="State the solution",
            expression=answer,
            justification="This is our final verified answer.",
        ))
        
        return steps
    
    def _equation_steps(self, problem: MathProblem, answer: Any) -> list[SolutionStep]:
        """Generate steps for equation solving."""
        from sympy import factor, expand
        
        steps = []
        expr = problem.parsed_expression
        
        # Try to show factoring if applicable
        try:
            factored = factor(expr)
            if factored != expr:
                steps.append(SolutionStep(
                    action="Factor the expression",
                    expression=factored,
                    justification=(
                        "Factoring helps us find roots by identifying "
                        "factors that equal zero."
                    ),
                ))
        except Exception:
            pass
        
        # Show each solution
        if isinstance(answer, list):
            for i, sol in enumerate(answer):
                steps.append(SolutionStep(
                    action=f"Find solution {i+1}",
                    expression=sol,
                    justification=f"Setting factor {i+1} to zero gives us this value.",
                ))
        
        return steps
    
    def _derivative_steps(self, problem: MathProblem, answer: Any) -> list[SolutionStep]:
        """Generate steps for differentiation."""
        steps = []
        
        steps.append(SolutionStep(
            action="Apply differentiation rules",
            expression="d/dx",
            justification=(
                "We apply the power rule (d/dx[x^n] = n*x^(n-1)), "
                "chain rule, and other relevant differentiation rules."
            ),
        ))
        
        return steps
    
    def _integral_steps(self, problem: MathProblem, answer: Any) -> list[SolutionStep]:
        """Generate steps for integration."""
        steps = []
        
        steps.append(SolutionStep(
            action="Apply integration rules",
            expression="∫",
            justification=(
                "We apply the power rule for integration "
                "(∫x^n dx = x^(n+1)/(n+1) + C for n ≠ -1) "
                "and other relevant integration techniques."
            ),
        ))
        
        steps.append(SolutionStep(
            action="Don't forget the constant",
            expression="+ C",
            justification=(
                "For indefinite integrals, we add an arbitrary "
                "constant C because the derivative of any constant is 0."
            ),
            warnings=["Remember to add +C for indefinite integrals!"],
        ))
        
        return steps
