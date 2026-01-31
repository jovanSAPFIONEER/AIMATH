"""
Calculus Step Generator

Uses SymPy's manualintegrate for integration steps and custom logic for derivatives.
Provides human-readable step-by-step solutions like a textbook.
"""

from sympy import (
    Symbol, symbols, diff, integrate, simplify, expand, factor,
    sin, cos, tan, exp, log, sqrt, Abs, pi, E,
    Derivative, Integral, Add, Mul, Pow, Function,
    sympify, latex
)
from sympy.integrals.manualintegrate import manualintegrate, integral_steps
from sympy.core.function import AppliedUndef
from dataclasses import dataclass, field
from typing import List, Any, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class RuleType(Enum):
    """Types of differentiation/integration rules."""
    POWER = "Power Rule"
    CONSTANT = "Constant Rule"
    CONSTANT_MULTIPLE = "Constant Multiple Rule"
    SUM = "Sum Rule"
    PRODUCT = "Product Rule"
    QUOTIENT = "Quotient Rule"
    CHAIN = "Chain Rule"
    TRIG = "Trigonometric Rule"
    INVERSE_TRIG = "Inverse Trig Rule"
    EXPONENTIAL = "Exponential Rule"
    LOGARITHM = "Logarithm Rule"
    SUBSTITUTION = "U-Substitution"
    PARTS = "Integration by Parts"
    PARTIAL_FRACTIONS = "Partial Fractions"
    TRIG_SUBSTITUTION = "Trig Substitution"
    SPECIAL = "Special Form"


@dataclass
class MathStep:
    """A single step in a mathematical solution."""
    rule: str
    description: str
    before: Any
    after: Any
    substeps: List['MathStep'] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "rule": self.rule,
            "description": self.description,
            "before": str(self.before),
            "after": str(self.after),
            "before_latex": latex(self.before) if hasattr(self.before, 'atoms') else str(self.before),
            "after_latex": latex(self.after) if hasattr(self.after, 'atoms') else str(self.after),
            "substeps": [s.to_dict() for s in self.substeps]
        }


def get_derivative_steps(expr, var=None) -> List[MathStep]:
    """
    Generate step-by-step derivative solution.
    
    Args:
        expr: SymPy expression to differentiate
        var: Variable (default: x)
        
    Returns:
        List of MathStep objects
    """
    if var is None:
        var = Symbol('x')
    
    steps = []
    expr = sympify(expr)
    
    # Handle Derivative objects
    if isinstance(expr, Derivative):
        func = expr.args[0]
        var = expr.args[1][0] if len(expr.args) > 1 else var
    else:
        func = expr
    
    result = diff(func, var)
    
    # Analyze the expression structure
    steps.extend(_analyze_derivative(func, var, result))
    
    return steps


def _analyze_derivative(expr, var, result) -> List[MathStep]:
    """Analyze expression and generate derivative steps."""
    steps = []
    
    # Constants
    if not expr.has(var):
        steps.append(MathStep(
            rule=RuleType.CONSTANT.value,
            description=f"The derivative of a constant is 0",
            before=f"d/d{var}[{expr}]",
            after=0
        ))
        return steps
    
    # Sum Rule: d/dx[f + g] = f' + g'
    if isinstance(expr, Add):
        terms = expr.as_ordered_terms()
        steps.append(MathStep(
            rule=RuleType.SUM.value,
            description="Differentiate each term separately",
            before=f"d/d{var}[{expr}]",
            after=" + ".join([f"d/d{var}[{t}]" for t in terms])
        ))
        for term in terms:
            term_result = diff(term, var)
            substeps = _analyze_derivative_term(term, var, term_result)
            if substeps:
                steps.extend(substeps)
        steps.append(MathStep(
            rule="Combine",
            description="Combine the derivatives",
            before=" + ".join([f"d/d{var}[{t}]" for t in terms]),
            after=result
        ))
        return steps
    
    # Single term
    steps.extend(_analyze_derivative_term(expr, var, result))
    return steps


def _analyze_derivative_term(term, var, result) -> List[MathStep]:
    """Analyze a single term for derivative."""
    steps = []
    
    # Constant multiple: d/dx[c*f] = c*f'
    coeff = term.as_coeff_Mul()[0]
    if coeff != 1 and coeff.is_number:
        inner = term / coeff
        steps.append(MathStep(
            rule=RuleType.CONSTANT_MULTIPLE.value,
            description=f"Factor out the constant {coeff}",
            before=f"d/d{var}[{term}]",
            after=f"{coeff} · d/d{var}[{inner}]"
        ))
        term = inner
    
    # Power Rule: d/dx[x^n] = n*x^(n-1)
    if isinstance(term, Pow):
        base, exponent = term.as_base_exp()
        if base == var and exponent.is_number:
            steps.append(MathStep(
                rule=RuleType.POWER.value,
                description=f"d/d{var}[{var}^n] = n·{var}^(n-1)",
                before=f"d/d{var}[{term}]",
                after=f"{exponent} · {var}^{exponent-1} = {diff(term, var)}"
            ))
        elif base == var:
            # Variable exponent - use logarithmic differentiation
            steps.append(MathStep(
                rule="Logarithmic Differentiation",
                description=f"For {var}^(f({var})), use d/d{var}[{var}^f] = {var}^f · (f' · ln({var}) + f/{var})",
                before=f"d/d{var}[{term}]",
                after=diff(term, var)
            ))
        elif exponent.has(var):
            # Chain rule needed
            steps.append(MathStep(
                rule=RuleType.CHAIN.value,
                description=f"Apply chain rule: d/d{var}[f^n] = n·f^(n-1)·f'",
                before=f"d/d{var}[{term}]",
                after=diff(term, var)
            ))
    
    # Variable alone
    elif term == var:
        steps.append(MathStep(
            rule=RuleType.POWER.value,
            description=f"d/d{var}[{var}] = 1 (since {var}^1, power rule gives 1·{var}^0 = 1)",
            before=f"d/d{var}[{var}]",
            after=1
        ))
    
    # Trigonometric functions
    elif term.func == sin:
        arg = term.args[0]
        if arg == var:
            steps.append(MathStep(
                rule=RuleType.TRIG.value,
                description=f"d/d{var}[sin({var})] = cos({var})",
                before=f"d/d{var}[sin({arg})]",
                after=f"cos({arg})"
            ))
        else:
            steps.append(MathStep(
                rule=f"{RuleType.TRIG.value} + {RuleType.CHAIN.value}",
                description=f"d/d{var}[sin(u)] = cos(u)·u'",
                before=f"d/d{var}[{term}]",
                after=diff(term, var)
            ))
    
    elif term.func == cos:
        arg = term.args[0]
        if arg == var:
            steps.append(MathStep(
                rule=RuleType.TRIG.value,
                description=f"d/d{var}[cos({var})] = -sin({var})",
                before=f"d/d{var}[cos({arg})]",
                after=f"-sin({arg})"
            ))
        else:
            steps.append(MathStep(
                rule=f"{RuleType.TRIG.value} + {RuleType.CHAIN.value}",
                description=f"d/d{var}[cos(u)] = -sin(u)·u'",
                before=f"d/d{var}[{term}]",
                after=diff(term, var)
            ))
    
    elif term.func == tan:
        steps.append(MathStep(
            rule=RuleType.TRIG.value,
            description=f"d/d{var}[tan(u)] = sec²(u)·u'",
            before=f"d/d{var}[{term}]",
            after=diff(term, var)
        ))
    
    # Exponential
    elif term.func == exp:
        arg = term.args[0]
        if arg == var:
            steps.append(MathStep(
                rule=RuleType.EXPONENTIAL.value,
                description=f"d/d{var}[e^{var}] = e^{var}",
                before=f"d/d{var}[{term}]",
                after=term
            ))
        else:
            steps.append(MathStep(
                rule=f"{RuleType.EXPONENTIAL.value} + {RuleType.CHAIN.value}",
                description=f"d/d{var}[e^u] = e^u · u'",
                before=f"d/d{var}[{term}]",
                after=diff(term, var)
            ))
    
    # Logarithm
    elif term.func == log:
        arg = term.args[0]
        if arg == var:
            steps.append(MathStep(
                rule=RuleType.LOGARITHM.value,
                description=f"d/d{var}[ln({var})] = 1/{var}",
                before=f"d/d{var}[{term}]",
                after=f"1/{var}"
            ))
        else:
            steps.append(MathStep(
                rule=f"{RuleType.LOGARITHM.value} + {RuleType.CHAIN.value}",
                description=f"d/d{var}[ln(u)] = u'/u",
                before=f"d/d{var}[{term}]",
                after=diff(term, var)
            ))
    
    # Product Rule: d/dx[f*g] = f'g + fg'
    elif isinstance(term, Mul):
        factors = [f for f in term.as_ordered_factors() if f.has(var)]
        if len(factors) >= 2:
            steps.append(MathStep(
                rule=RuleType.PRODUCT.value,
                description="d/d{var}[u·v] = u'·v + u·v'",
                before=f"d/d{var}[{term}]",
                after=diff(term, var)
            ))
    
    # Generic fallback
    if not steps:
        steps.append(MathStep(
            rule="Differentiation",
            description=f"Apply standard differentiation rules",
            before=f"d/d{var}[{term}]",
            after=diff(term, var)
        ))
    
    return steps


def get_integral_steps(expr, var=None, limits=None) -> List[MathStep]:
    """
    Generate step-by-step integration solution using SymPy's manualintegrate.
    
    Args:
        expr: SymPy expression to integrate
        var: Variable (default: x)
        limits: Optional tuple (lower, upper) for definite integrals
        
    Returns:
        List of MathStep objects
    """
    if var is None:
        var = Symbol('x')
    
    steps = []
    expr = sympify(expr)
    
    # Handle Integral objects
    if isinstance(expr, Integral):
        integrand = expr.args[0]
        if len(expr.args) > 1:
            int_var = expr.args[1]
            if isinstance(int_var, tuple):
                var = int_var[0]
                if len(int_var) == 3:
                    limits = (int_var[1], int_var[2])
    else:
        integrand = expr
    
    # Use SymPy's integral_steps for the heavy lifting
    try:
        sympy_steps = integral_steps(integrand, var)
        steps = _convert_sympy_steps(sympy_steps, var)
    except Exception as e:
        logger.warning(f"integral_steps failed: {e}")
        # Fallback to basic analysis
        steps = _analyze_integral_basic(integrand, var)
    
    # Add definite integral evaluation if limits provided
    if limits:
        antiderivative = manualintegrate(integrand, var)
        lower, upper = limits
        
        steps.append(MathStep(
            rule="Fundamental Theorem of Calculus",
            description=f"Evaluate F({upper}) - F({lower}) where F(x) = {antiderivative}",
            before=f"[{antiderivative}] from {lower} to {upper}",
            after=f"F({upper}) - F({lower})"
        ))
        
        result = antiderivative.subs(var, upper) - antiderivative.subs(var, lower)
        steps.append(MathStep(
            rule="Evaluate",
            description="Substitute limits and compute",
            before=f"({antiderivative.subs(var, upper)}) - ({antiderivative.subs(var, lower)})",
            after=simplify(result)
        ))
    
    return steps


def _convert_sympy_steps(sympy_step, var) -> List[MathStep]:
    """Convert SymPy's integral steps to our MathStep format."""
    steps = []
    
    rule_name = sympy_step.__class__.__name__
    
    # Map SymPy rule names to friendly names
    rule_map = {
        'ConstantRule': RuleType.CONSTANT.value,
        'ConstantTimesRule': RuleType.CONSTANT_MULTIPLE.value,
        'PowerRule': RuleType.POWER.value,
        'AddRule': RuleType.SUM.value,
        'ExpRule': RuleType.EXPONENTIAL.value,
        'TrigRule': RuleType.TRIG.value,
        'PartsRule': RuleType.PARTS.value,
        'URule': RuleType.SUBSTITUTION.value,
        'ReciprocalRule': RuleType.LOGARITHM.value,
        'ArctanRule': RuleType.INVERSE_TRIG.value,
        'ArcsinRule': RuleType.INVERSE_TRIG.value,
    }
    
    friendly_name = rule_map.get(rule_name, rule_name)
    
    # Get description based on rule
    descriptions = {
        RuleType.CONSTANT.value: f"∫c dx = c·x + C",
        RuleType.CONSTANT_MULTIPLE.value: f"∫c·f(x) dx = c·∫f(x) dx",
        RuleType.POWER.value: f"∫x^n dx = x^(n+1)/(n+1) + C (n≠-1)",
        RuleType.SUM.value: "∫[f(x) + g(x)] dx = ∫f(x)dx + ∫g(x)dx",
        RuleType.EXPONENTIAL.value: "∫e^x dx = e^x + C",
        RuleType.TRIG.value: "Apply trigonometric integration rules",
        RuleType.PARTS.value: "∫u dv = uv - ∫v du",
        RuleType.SUBSTITUTION.value: "Let u = g(x), then dx = du/g'(x)",
        RuleType.LOGARITHM.value: "∫1/x dx = ln|x| + C",
        RuleType.INVERSE_TRIG.value: "Apply inverse trig integration rules",
    }
    
    integrand = getattr(sympy_step, 'integrand', sympy_step)
    result = manualintegrate(integrand, var) if hasattr(sympy_step, 'integrand') else integrate(integrand, var)
    
    steps.append(MathStep(
        rule=friendly_name,
        description=descriptions.get(friendly_name, f"Apply {friendly_name}"),
        before=f"∫{integrand} d{var}",
        after=result
    ))
    
    # Process substeps if any
    if hasattr(sympy_step, 'substeps'):
        for substep in sympy_step.substeps:
            steps.extend(_convert_sympy_steps(substep, var))
    elif hasattr(sympy_step, 'substep'):
        steps.extend(_convert_sympy_steps(sympy_step.substep, var))
    
    return steps


def _analyze_integral_basic(expr, var) -> List[MathStep]:
    """Basic integral analysis when manualintegrate fails."""
    steps = []
    result = integrate(expr, var)
    
    # Constant
    if not expr.has(var):
        steps.append(MathStep(
            rule=RuleType.CONSTANT.value,
            description="∫c dx = c·x + C",
            before=f"∫{expr} d{var}",
            after=f"{expr}·{var}"
        ))
    
    # Power
    elif isinstance(expr, Pow) or expr == var:
        steps.append(MathStep(
            rule=RuleType.POWER.value,
            description="∫x^n dx = x^(n+1)/(n+1) + C",
            before=f"∫{expr} d{var}",
            after=result
        ))
    
    # Sum
    elif isinstance(expr, Add):
        steps.append(MathStep(
            rule=RuleType.SUM.value,
            description="Integrate each term separately",
            before=f"∫{expr} d{var}",
            after=result
        ))
    
    else:
        steps.append(MathStep(
            rule="Integration",
            description="Apply integration techniques",
            before=f"∫{expr} d{var}",
            after=result
        ))
    
    return steps


def format_steps_text(steps: List[MathStep], indent: int = 0) -> str:
    """Format steps as readable text."""
    lines = []
    prefix = "  " * indent
    
    for i, step in enumerate(steps, 1):
        lines.append(f"{prefix}Step {i}: {step.rule}")
        lines.append(f"{prefix}  {step.description}")
        lines.append(f"{prefix}  {step.before} → {step.after}")
        
        if step.substeps:
            lines.append(f"{prefix}  Substeps:")
            lines.append(format_steps_text(step.substeps, indent + 2))
        
        lines.append("")
    
    return "\n".join(lines)


def format_steps_latex(steps: List[MathStep]) -> str:
    """Format steps as LaTeX."""
    lines = ["\\begin{align*}"]
    
    for step in steps:
        before_latex = latex(sympify(step.before)) if not isinstance(step.before, str) else step.before
        after_latex = latex(sympify(step.after)) if not isinstance(step.after, str) else step.after
        lines.append(f"  & \\text{{{step.rule}:}} \\\\")
        lines.append(f"  & {before_latex} \\rightarrow {after_latex} \\\\")
    
    lines.append("\\end{align*}")
    return "\n".join(lines)


# Convenience functions
def show_derivative_steps(expr, var=None) -> str:
    """Show derivative steps as formatted text."""
    steps = get_derivative_steps(expr, var)
    return format_steps_text(steps)


def show_integral_steps(expr, var=None, limits=None) -> str:
    """Show integral steps as formatted text."""
    steps = get_integral_steps(expr, var, limits)
    return format_steps_text(steps)
