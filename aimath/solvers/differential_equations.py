"""
Differential Equations Solver

Solves ODEs and some PDEs with classification and step-by-step solutions.
"""

from sympy import (
    Symbol, symbols, Function, Eq, dsolve, classify_ode, checkodesol,
    sin, cos, exp, log, sqrt, diff, integrate, simplify, expand,
    Derivative, pprint, latex, oo, S
)
from sympy.solvers.ode import constantsimp
from dataclasses import dataclass
from typing import List, Any, Optional, Dict, Union, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ODEType(Enum):
    """Classification of ODE types."""
    SEPARABLE = "separable"
    FIRST_ORDER_LINEAR = "1st_linear"
    BERNOULLI = "Bernoulli"
    RICCATI = "Riccati"
    EXACT = "exact"
    HOMOGENEOUS = "1st_homogeneous"
    SECOND_ORDER_LINEAR_CONST = "2nd_linear_constant"
    SECOND_ORDER_LINEAR_VAR = "2nd_linear_variable"
    NTH_LINEAR_CONST = "nth_linear_constant_coeff"
    REDUCIBLE = "reducible"
    VARIATION_OF_PARAMS = "variation_of_parameters"


@dataclass
class ODESolution:
    """Result of solving an ODE."""
    equation: Any
    solution: Any
    ode_type: str
    classification: List[str]
    steps: List[str]
    is_verified: bool
    initial_conditions: Optional[Dict] = None
    particular_solution: Optional[Any] = None
    
    def to_dict(self) -> dict:
        return {
            "equation": str(self.equation),
            "solution": str(self.solution),
            "ode_type": self.ode_type,
            "classification": self.classification,
            "steps": self.steps,
            "is_verified": self.is_verified,
            "initial_conditions": str(self.initial_conditions) if self.initial_conditions else None,
            "particular_solution": str(self.particular_solution) if self.particular_solution else None,
        }


class ODESolver:
    """
    Ordinary Differential Equation Solver with classification and steps.
    
    Usage:
        solver = ODESolver()
        result = solver.solve("y' + 2*y = x")
    """
    
    def __init__(self):
        self.x = Symbol('x')
        self.y = Function('y')
        self.t = Symbol('t')
        
    def parse_ode(self, ode_input: Union[str, Eq]) -> Tuple[Eq, Function, Symbol]:
        """
        Parse ODE from string or equation.
        
        Supports formats:
            - "y' + 2*y = x"
            - "y'' - 3*y' + 2*y = 0"
            - "dy/dx = y*x"
            - SymPy Eq object
        """
        if isinstance(ode_input, Eq):
            # Already a SymPy equation
            eq = ode_input
        elif isinstance(ode_input, str):
            # Parse string format
            eq = self._parse_string_ode(ode_input)
        else:
            raise ValueError(f"Cannot parse ODE: {ode_input}")
        
        # Identify function and variable
        funcs = eq.atoms(Function)
        derivs = eq.atoms(Derivative)
        
        # Get function from derivatives
        if derivs:
            deriv = list(derivs)[0]
            func_expr = deriv.args[0]  # e.g., y(x)
            if hasattr(func_expr, 'func'):
                func = func_expr.func  # The Function class, e.g., y
                var = func_expr.args[0] if func_expr.args else self.x
            else:
                func = self.y
                var = self.x
        elif funcs:
            # Get from applied functions like y(x)
            for f in funcs:
                if hasattr(f, 'func') and f.func != Derivative:
                    func = f.func
                    var = f.args[0] if f.args else self.x
                    break
            else:
                func = self.y
                var = self.x
        else:
            func = self.y
            var = self.x
        
        return eq, func, var
    
    def _parse_string_ode(self, ode_str: str) -> Eq:
        """Parse string representation of ODE."""
        import re
        
        x = self.x
        y = self.y
        
        # Standardize notation
        ode_str = ode_str.replace("y''''", "Derivative(y(x), x, x, x, x)")
        ode_str = ode_str.replace("y'''", "Derivative(y(x), x, x, x)")
        ode_str = ode_str.replace("y''", "Derivative(y(x), x, x)")
        ode_str = ode_str.replace("y'", "Derivative(y(x), x)")
        
        # Handle dy/dx notation
        ode_str = re.sub(r'd(\d?)y/dx\1', 
                        lambda m: f"Derivative(y(x), x, {m.group(1) or 1})" if m.group(1) else "Derivative(y(x), x)", 
                        ode_str)
        
        # Replace standalone y with y(x) - but not if already y(x)
        ode_str = re.sub(r'\by\b(?!\()', 'y(x)', ode_str)
        
        # Create local context for parsing
        local_dict = {'x': x, 'y': y, 'Derivative': Derivative}
        
        # Split on = if present
        if '=' in ode_str:
            lhs, rhs = ode_str.split('=', 1)
            from sympy import sympify
            return Eq(sympify(lhs.strip(), locals=local_dict), 
                     sympify(rhs.strip(), locals=local_dict))
        else:
            from sympy import sympify
            return Eq(sympify(ode_str, locals=local_dict), 0)
    
    def classify(self, ode_input) -> List[str]:
        """
        Classify the ODE and return list of applicable solution methods.
        """
        eq, func, var = self.parse_ode(ode_input)
        
        try:
            classification = classify_ode(eq, func(var))
            return list(classification)
        except Exception as e:
            logger.warning(f"Classification failed: {e}")
            return ["unknown"]
    
    def solve(self, ode_input, ics: Optional[Dict] = None, hint: Optional[str] = None) -> ODESolution:
        """
        Solve an ODE with step-by-step explanation.
        
        Args:
            ode_input: ODE as string or SymPy equation
            ics: Initial conditions as dict, e.g., {y(0): 1, y'(0): 0}
            hint: Specific method to use (from classification)
            
        Returns:
            ODESolution with full solution details
        """
        eq, func, var = self.parse_ode(ode_input)
        steps = []
        
        # Step 1: State the ODE
        steps.append(f"Given ODE: {eq}")
        
        # Step 2: Classify
        try:
            classification = classify_ode(eq, func(var))
            ode_type = classification[0] if classification else "unknown"
            steps.append(f"ODE Classification: {ode_type}")
            steps.append(f"Applicable methods: {list(classification)[:5]}")  # Show top 5
        except Exception as e:
            classification = ["unknown"]
            ode_type = "unknown"
            steps.append(f"Could not classify ODE: {e}")
        
        # Step 3: Describe the method
        method_description = self._get_method_description(ode_type)
        if method_description:
            steps.append(f"Method: {method_description}")
        
        # Step 4: Solve
        try:
            if hint:
                solution = dsolve(eq, func(var), hint=hint)
            else:
                solution = dsolve(eq, func(var))
            
            if isinstance(solution, list):
                solution = solution[0]  # Take first solution
            
            steps.append(f"General solution: {solution}")
        except Exception as e:
            logger.error(f"ODE solve failed: {e}")
            return ODESolution(
                equation=eq,
                solution=None,
                ode_type=ode_type,
                classification=list(classification) if classification else [],
                steps=steps + [f"Error solving ODE: {e}"],
                is_verified=False
            )
        
        # Step 5: Apply initial conditions if provided
        particular_solution = None
        if ics:
            steps.append(f"Applying initial conditions: {ics}")
            try:
                particular_solution = dsolve(eq, func(var), ics=ics)
                steps.append(f"Particular solution: {particular_solution}")
            except Exception as e:
                steps.append(f"Could not apply ICs: {e}")
        
        # Step 6: Verify
        try:
            check = checkodesol(eq, solution, func(var))
            is_verified = check[0]
            if is_verified:
                steps.append("✓ Solution verified by substitution")
            else:
                steps.append(f"⚠ Verification inconclusive: {check[1]}")
        except Exception as e:
            is_verified = False
            steps.append(f"Could not verify solution: {e}")
        
        return ODESolution(
            equation=eq,
            solution=solution,
            ode_type=ode_type,
            classification=list(classification) if classification else [],
            steps=steps,
            is_verified=is_verified,
            initial_conditions=ics,
            particular_solution=particular_solution
        )
    
    def _get_method_description(self, ode_type: str) -> str:
        """Get human-readable description of solution method."""
        descriptions = {
            'separable': (
                "Separable ODE: Rewrite as g(y)dy = f(x)dx, then integrate both sides."
            ),
            '1st_linear': (
                "First-order linear ODE (y' + P(x)y = Q(x)): "
                "Use integrating factor μ(x) = e^(∫P(x)dx). "
                "Solution: y = (1/μ)∫μQ dx"
            ),
            'Bernoulli': (
                "Bernoulli ODE (y' + P(x)y = Q(x)y^n): "
                "Substitute v = y^(1-n) to get linear ODE."
            ),
            'exact': (
                "Exact ODE (M dx + N dy = 0 where ∂M/∂y = ∂N/∂x): "
                "Find F(x,y) where ∂F/∂x = M and ∂F/∂y = N."
            ),
            '1st_homogeneous_coeff': (
                "Homogeneous ODE: Substitute y = vx, dy = v dx + x dv. "
                "Results in separable ODE in v and x."
            ),
            'nth_linear_constant_coeff_homogeneous': (
                "Linear ODE with constant coefficients: "
                "Find roots r₁, r₂, ... of characteristic equation. "
                "General solution: y = C₁e^(r₁x) + C₂e^(r₂x) + ..."
            ),
            'nth_linear_constant_coeff_undetermined_coefficients': (
                "Non-homogeneous linear ODE: y = y_h + y_p. "
                "Find homogeneous solution, then guess particular solution form."
            ),
            'nth_linear_constant_coeff_variation_of_parameters': (
                "Variation of parameters: y_p = u₁y₁ + u₂y₂ "
                "where y₁, y₂ are homogeneous solutions."
            ),
        }
        return descriptions.get(ode_type, "")
    
    def solve_ivp(self, ode_input, initial_conditions: Dict) -> ODESolution:
        """
        Solve Initial Value Problem (IVP).
        
        Args:
            ode_input: The ODE
            initial_conditions: Dict like {y(0): 1} or {func(x0): y0}
            
        Returns:
            ODESolution with particular solution
        """
        return self.solve(ode_input, ics=initial_conditions)
    
    def solve_bvp(self, ode_input, boundary_conditions: Dict) -> ODESolution:
        """
        Solve Boundary Value Problem (BVP).
        
        Note: SymPy has limited BVP support. This uses general solution
        with boundary conditions as constraints.
        """
        eq, func, var = self.parse_ode(ode_input)
        steps = []
        
        steps.append(f"Given BVP: {eq}")
        steps.append(f"Boundary conditions: {boundary_conditions}")
        
        # Get general solution
        try:
            gen_solution = dsolve(eq, func(var))
            if isinstance(gen_solution, list):
                gen_solution = gen_solution[0]
            steps.append(f"General solution: {gen_solution}")
        except Exception as e:
            return ODESolution(
                equation=eq,
                solution=None,
                ode_type="bvp",
                classification=[],
                steps=steps + [f"Could not solve: {e}"],
                is_verified=False
            )
        
        # Apply boundary conditions
        steps.append("Applying boundary conditions to find constants...")
        
        # This is a simplified approach - full BVP solving is complex
        try:
            from sympy import solve
            rhs = gen_solution.rhs
            
            # Create equations from boundary conditions
            bc_eqs = []
            for condition, value in boundary_conditions.items():
                bc_eqs.append(Eq(rhs.subs(condition.args[0], condition), value))
            
            # Solve for constants
            constants = rhs.free_symbols - {var}
            if constants and bc_eqs:
                const_vals = solve(bc_eqs, list(constants))
                if const_vals:
                    particular = rhs.subs(const_vals)
                    steps.append(f"Constants: {const_vals}")
                    steps.append(f"Particular solution: {particular}")
                    
                    return ODESolution(
                        equation=eq,
                        solution=gen_solution,
                        ode_type="bvp",
                        classification=list(classify_ode(eq, func(var))),
                        steps=steps,
                        is_verified=True,
                        particular_solution=Eq(func(var), particular)
                    )
        except Exception as e:
            steps.append(f"Could not apply boundary conditions: {e}")
        
        return ODESolution(
            equation=eq,
            solution=gen_solution,
            ode_type="bvp",
            classification=[],
            steps=steps,
            is_verified=False
        )
    
    def phase_portrait_info(self, ode_input) -> Dict:
        """
        Get information for phase portrait analysis (2D systems).
        Returns equilibrium points and their classification.
        """
        # This is for autonomous systems dy/dx = f(y) or systems
        # Full implementation would require numerical analysis
        return {
            "note": "Phase portrait analysis requires numerical computation",
            "suggestion": "Use scipy.integrate.odeint for numerical phase portraits"
        }


# Convenience functions
def solve_ode(ode, ics=None, hint=None) -> ODESolution:
    """
    Solve an ODE.
    
    Examples:
        >>> solve_ode("y' + 2*y = x")
        >>> solve_ode("y'' + y = 0", ics={y(0): 1, y'(0): 0})
    """
    solver = ODESolver()
    return solver.solve(ode, ics=ics, hint=hint)


def classify_differential_equation(ode) -> List[str]:
    """
    Classify an ODE and return applicable solution methods.
    """
    solver = ODESolver()
    return solver.classify(ode)


# Common ODE examples for testing/learning
EXAMPLE_ODES = {
    "separable": "y' = x*y",
    "linear_first": "y' + 2*y = x",
    "bernoulli": "y' + y = y**2",
    "exact": "2*x*y + x**2*y' = 0",
    "homogeneous": "y' = y/x + x/y",
    "constant_coeff_2nd": "y'' + 3*y' + 2*y = 0",
    "undetermined_coeff": "y'' + y = sin(x)",
    "harmonic": "y'' + y = 0",
    "damped": "y'' + 2*y' + y = 0",
}
