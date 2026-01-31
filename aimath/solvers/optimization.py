"""
Optimization Module

Combines SymPy's symbolic differentiation with SciPy's numerical optimization.
Supports constrained and unconstrained optimization.
"""

from sympy import (
    Symbol, symbols, diff, solve, simplify, hessian, Matrix,
    oo, S, sqrt, Abs, sin, cos, exp, log, latex, sympify,
    lambdify, Function
)
from dataclasses import dataclass
from typing import List, Any, Optional, Dict, Union, Tuple, Callable
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class OptimizationType(Enum):
    """Types of optimization problems."""
    UNCONSTRAINED = "unconstrained"
    CONSTRAINED = "constrained"
    LINEAR_PROGRAMMING = "linear_programming"
    QUADRATIC = "quadratic"
    CONVEX = "convex"


class OptimizationResult(Enum):
    """Possible optimization outcomes."""
    MINIMUM = "minimum"
    MAXIMUM = "maximum"
    SADDLE = "saddle_point"
    INCONCLUSIVE = "inconclusive"


@dataclass
class CriticalPoint:
    """A critical point with classification."""
    point: Dict[Symbol, Any]
    value: Any
    classification: str
    hessian_eigenvalues: Optional[List] = None


@dataclass
class OptimizationSolution:
    """Result of an optimization problem."""
    objective: Any
    variables: List[Symbol]
    critical_points: List[CriticalPoint]
    global_min: Optional[CriticalPoint] = None
    global_max: Optional[CriticalPoint] = None
    steps: List[str] = None
    numerical_solution: Optional[Dict] = None
    
    def to_dict(self) -> dict:
        return {
            "objective": str(self.objective),
            "variables": [str(v) for v in self.variables],
            "critical_points": [
                {
                    "point": {str(k): str(v) for k, v in cp.point.items()},
                    "value": str(cp.value),
                    "classification": cp.classification
                }
                for cp in self.critical_points
            ],
            "global_min": str(self.global_min.point) if self.global_min else None,
            "global_max": str(self.global_max.point) if self.global_max else None,
            "steps": self.steps,
        }


class Optimizer:
    """
    Mathematical optimization solver.
    
    Features:
    - Symbolic optimization using calculus
    - Numerical optimization using SciPy
    - Constrained optimization (Lagrange multipliers)
    
    Usage:
        opt = Optimizer()
        result = opt.minimize("x**2 + y**2")
    """
    
    def __init__(self):
        self.x, self.y, self.z = symbols('x y z')
        
    def find_critical_points(self, f, variables: List[Symbol] = None) -> OptimizationSolution:
        """
        Find critical points of function f using calculus.
        
        Args:
            f: Objective function (SymPy expression or string)
            variables: List of variables (auto-detected if None)
            
        Returns:
            OptimizationSolution with all critical points classified
        """
        f = sympify(f)
        steps = []
        
        # Auto-detect variables if not provided
        if variables is None:
            variables = list(f.free_symbols)
            variables.sort(key=str)
        
        n = len(variables)
        steps.append(f"Objective function: f({', '.join(str(v) for v in variables)}) = {f}")
        
        # Step 1: Compute gradient
        gradient = [diff(f, var) for var in variables]
        steps.append(f"Step 1: Compute gradient")
        for i, (var, g) in enumerate(zip(variables, gradient)):
            steps.append(f"  ∂f/∂{var} = {g}")
        
        # Step 2: Set gradient = 0 and solve
        steps.append(f"Step 2: Set gradient = 0")
        try:
            critical_pts = solve(gradient, variables, dict=True)
            if not critical_pts:
                steps.append("  No critical points found (system may have no solution)")
                return OptimizationSolution(
                    objective=f,
                    variables=variables,
                    critical_points=[],
                    steps=steps
                )
            steps.append(f"  Found {len(critical_pts)} critical point(s)")
        except Exception as e:
            steps.append(f"  Could not solve symbolically: {e}")
            return OptimizationSolution(
                objective=f,
                variables=variables,
                critical_points=[],
                steps=steps
            )
        
        # Step 3: Compute Hessian
        steps.append(f"Step 3: Compute Hessian matrix for classification")
        H = hessian(f, variables)
        steps.append(f"  H = {H}")
        
        # Step 4: Classify each critical point
        steps.append(f"Step 4: Classify critical points using second derivative test")
        classified_points = []
        
        for cp in critical_pts:
            # Evaluate function at critical point
            try:
                f_val = f.subs(cp)
                f_val = simplify(f_val)
            except:
                f_val = "undefined"
            
            # Evaluate Hessian at critical point
            H_at_cp = H.subs(cp)
            
            # Classify using eigenvalues
            try:
                eigenvals = list(H_at_cp.eigenvals().keys())
                eigenvals = [simplify(ev) for ev in eigenvals]
                
                # Determine nature
                if all(ev > 0 for ev in eigenvals if ev.is_real):
                    classification = OptimizationResult.MINIMUM.value
                    steps.append(f"  At {cp}: all eigenvalues > 0 → LOCAL MINIMUM, f = {f_val}")
                elif all(ev < 0 for ev in eigenvals if ev.is_real):
                    classification = OptimizationResult.MAXIMUM.value
                    steps.append(f"  At {cp}: all eigenvalues < 0 → LOCAL MAXIMUM, f = {f_val}")
                elif any(ev > 0 for ev in eigenvals if ev.is_real) and any(ev < 0 for ev in eigenvals if ev.is_real):
                    classification = OptimizationResult.SADDLE.value
                    steps.append(f"  At {cp}: mixed eigenvalues → SADDLE POINT, f = {f_val}")
                else:
                    classification = OptimizationResult.INCONCLUSIVE.value
                    steps.append(f"  At {cp}: inconclusive (eigenvalues: {eigenvals})")
            except Exception as e:
                classification = OptimizationResult.INCONCLUSIVE.value
                eigenvals = None
                steps.append(f"  At {cp}: could not classify ({e})")
            
            classified_points.append(CriticalPoint(
                point=cp,
                value=f_val,
                classification=classification,
                hessian_eigenvalues=eigenvals
            ))
        
        # Find global extrema among critical points
        global_min = None
        global_max = None
        
        min_val = oo
        max_val = -oo
        
        for cp in classified_points:
            try:
                val = float(cp.value.evalf()) if hasattr(cp.value, 'evalf') else float(cp.value)
                if val < min_val:
                    min_val = val
                    global_min = cp
                if val > max_val:
                    max_val = val
                    global_max = cp
            except:
                pass
        
        return OptimizationSolution(
            objective=f,
            variables=variables,
            critical_points=classified_points,
            global_min=global_min if global_min and global_min.classification == OptimizationResult.MINIMUM.value else None,
            global_max=global_max if global_max and global_max.classification == OptimizationResult.MAXIMUM.value else None,
            steps=steps
        )
    
    def minimize(self, f, variables: List[Symbol] = None, method: str = 'symbolic') -> OptimizationSolution:
        """
        Minimize a function.
        
        Args:
            f: Objective function
            variables: Variables to optimize over
            method: 'symbolic' or 'numeric'
        """
        result = self.find_critical_points(f, variables)
        
        if method == 'numeric' and not result.global_min:
            # Try numerical optimization
            num_result = self._optimize_numeric(sympify(f), result.variables, minimize=True)
            result.numerical_solution = num_result
        
        return result
    
    def maximize(self, f, variables: List[Symbol] = None, method: str = 'symbolic') -> OptimizationSolution:
        """
        Maximize a function.
        """
        result = self.find_critical_points(f, variables)
        
        if method == 'numeric' and not result.global_max:
            # Try numerical optimization (minimize negative)
            num_result = self._optimize_numeric(sympify(f), result.variables, minimize=False)
            result.numerical_solution = num_result
        
        return result
    
    def _optimize_numeric(self, f, variables: List[Symbol], minimize: bool = True, 
                         initial_guess: List[float] = None, bounds: List[Tuple] = None) -> Dict:
        """
        Use SciPy for numerical optimization.
        """
        try:
            import numpy as np
            from scipy.optimize import minimize as scipy_minimize, minimize_scalar
        except ImportError:
            return {"error": "SciPy not installed. Run: pip install scipy"}
        
        n = len(variables)
        
        # Default initial guess
        if initial_guess is None:
            initial_guess = [0.0] * n
        
        # Compile function
        if minimize:
            f_numeric = lambdify(variables, f, 'numpy')
        else:
            f_numeric = lambdify(variables, -f, 'numpy')  # Minimize negative for max
        
        # Compile gradient
        gradient = [diff(f, var) for var in variables]
        if minimize:
            grad_numeric = lambdify(variables, gradient, 'numpy')
        else:
            grad_numeric = lambdify(variables, [-g for g in gradient], 'numpy')
        
        def func_wrapper(x):
            return float(f_numeric(*x))
        
        def grad_wrapper(x):
            g = grad_numeric(*x)
            return np.array([float(gi) for gi in g])
        
        # Optimize
        try:
            result = scipy_minimize(
                func_wrapper, 
                initial_guess, 
                jac=grad_wrapper,
                method='BFGS',
                bounds=bounds
            )
            
            optimal_point = {var: float(val) for var, val in zip(variables, result.x)}
            optimal_value = float(f.subs(optimal_point).evalf())
            if not minimize:
                optimal_value = -result.fun  # Undo negation
            
            return {
                "success": result.success,
                "optimal_point": optimal_point,
                "optimal_value": optimal_value,
                "iterations": result.nit,
                "message": result.message
            }
        except Exception as e:
            return {"error": str(e)}
    
    def lagrange_multipliers(self, f, constraints: List, variables: List[Symbol] = None) -> OptimizationSolution:
        """
        Solve constrained optimization using Lagrange multipliers.
        
        Args:
            f: Objective function to optimize
            constraints: List of constraint equations (= 0 form)
            variables: Variables
            
        Example:
            opt.lagrange_multipliers("x**2 + y**2", ["x + y - 1"], [x, y])
            # Minimize x² + y² subject to x + y = 1
        """
        f = sympify(f)
        constraints = [sympify(c) for c in constraints]
        
        if variables is None:
            variables = list(f.free_symbols)
            variables.sort(key=str)
        
        steps = []
        steps.append(f"Objective: minimize/maximize f = {f}")
        steps.append(f"Subject to constraints: {constraints}")
        
        # Create Lagrange multipliers
        lambdas = [Symbol(f'lambda_{i}') for i in range(len(constraints))]
        
        # Form Lagrangian: L = f - Σ λᵢgᵢ
        L = f
        for lam, g in zip(lambdas, constraints):
            L = L - lam * g
        
        steps.append(f"Step 1: Form Lagrangian")
        steps.append(f"  L = {L}")
        
        # Compute partials
        all_vars = variables + lambdas
        partials = [diff(L, var) for var in all_vars]
        
        steps.append(f"Step 2: Set all partial derivatives to zero")
        for var, partial in zip(all_vars, partials):
            steps.append(f"  ∂L/∂{var} = {partial} = 0")
        
        # Solve system
        steps.append(f"Step 3: Solve the system")
        try:
            solutions = solve(partials, all_vars, dict=True)
            steps.append(f"  Found {len(solutions)} solution(s)")
        except Exception as e:
            steps.append(f"  Could not solve: {e}")
            return OptimizationSolution(
                objective=f,
                variables=variables,
                critical_points=[],
                steps=steps
            )
        
        # Evaluate and classify
        critical_points = []
        for sol in solutions:
            # Extract only original variables
            point = {var: sol.get(var, var) for var in variables}
            
            try:
                f_val = f.subs(point)
                f_val = simplify(f_val)
            except:
                f_val = "undefined"
            
            critical_points.append(CriticalPoint(
                point=point,
                value=f_val,
                classification="constrained_extremum",
                hessian_eigenvalues=None
            ))
            
            steps.append(f"  At {point}: f = {f_val}")
        
        return OptimizationSolution(
            objective=f,
            variables=variables,
            critical_points=critical_points,
            steps=steps
        )
    
    def linear_program(self, c: List, A_ub: List = None, b_ub: List = None,
                       A_eq: List = None, b_eq: List = None, bounds: List = None) -> Dict:
        """
        Solve linear programming problem using SciPy.
        
        Minimize: c^T x
        Subject to: A_ub @ x <= b_ub
                    A_eq @ x == b_eq
                    bounds[i][0] <= x[i] <= bounds[i][1]
        """
        try:
            from scipy.optimize import linprog
            import numpy as np
        except ImportError:
            return {"error": "SciPy not installed"}
        
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
        
        return {
            "success": result.success,
            "optimal_point": list(result.x) if result.success else None,
            "optimal_value": float(result.fun) if result.success else None,
            "message": result.message
        }


# Convenience functions
def find_extrema(f, variables=None):
    """
    Find all extrema (minima, maxima, saddle points) of a function.
    
    Example:
        >>> find_extrema("x**3 - 3*x")
        # Returns critical points at x = ±1
    """
    opt = Optimizer()
    return opt.find_critical_points(f, variables)


def minimize(f, variables=None, method='symbolic'):
    """
    Minimize a function.
    
    Example:
        >>> minimize("x**2 + y**2 - 2*x - 4*y")
        # Returns minimum at (1, 2)
    """
    opt = Optimizer()
    return opt.minimize(f, variables, method)


def maximize(f, variables=None, method='symbolic'):
    """
    Maximize a function.
    """
    opt = Optimizer()
    return opt.maximize(f, variables, method)


def constrained_optimize(f, constraints, variables=None):
    """
    Optimize with equality constraints using Lagrange multipliers.
    
    Example:
        >>> constrained_optimize("x**2 + y**2", ["x + y - 1"])
        # Minimize x² + y² subject to x + y = 1
    """
    opt = Optimizer()
    return opt.lagrange_multipliers(f, constraints, variables)
