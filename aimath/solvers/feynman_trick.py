"""
Feynman's Trick Helper (Differentiation Under the Integral Sign)

Implementation of Richard Feynman's favorite integration technique:
"I had learned to do integrals by various methods shown in a book...
Then I had another book that was more advanced... 
differentiation under the integral sign."

This module helps users apply Feynman's technique to hard integrals.

The key identity:
    d/da ∫ f(x, a) dx = ∫ ∂f/∂a dx

This allows transforming a hard integral into an easier one by:
1. Introducing a parameter 'a' into the integrand
2. Differentiating with respect to 'a'
3. Solving the easier integral
4. Integrating back with respect to 'a'
"""

import sympy
from sympy import (
    Symbol, symbols, sqrt, log, atan, sin, cos, exp,
    pi, oo, Integral, sympify, latex, N, simplify,
    diff, integrate, Function, Derivative, Eq, dsolve
)
from typing import Optional, Union, List, Any, Tuple
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class FeynmanStep:
    """A step in applying Feynman's technique."""
    step_number: int
    description: str
    expression: Any
    latex_expr: str


@dataclass
class FeynmanResult:
    """Result of applying Feynman's technique."""
    original_integral: Any
    parameterized_integral: Any
    parameter: Symbol
    derivative_wrt_param: Any
    inner_integral_solved: Any
    final_result: Any
    steps: List[FeynmanStep]
    success: bool
    notes: List[str] = field(default_factory=list)


def introduce_parameter(
    integrand: Any,
    var: Symbol,
    param: Symbol = None,
    strategy: str = 'exponential'
) -> Tuple[Any, Symbol]:
    """
    Introduce a parameter into an integrand.
    
    Common strategies:
    - 'exponential': Multiply by e^(-a*x) or similar
    - 'power': Add parameter to exponent
    - 'coefficient': Add parameter as coefficient
    
    Args:
        integrand: Original integrand
        var: Integration variable
        param: Parameter symbol (default: creates 'a')
        strategy: How to introduce the parameter
        
    Returns:
        (parameterized_integrand, parameter)
    """
    if param is None:
        param = Symbol('a', positive=True)
    
    if strategy == 'exponential':
        # f(x) → f(x) * e^(-a*x)
        return integrand * exp(-param * var), param
    
    elif strategy == 'power':
        # x^n → x^(n+a) where applicable
        # This is more complex - look for power terms
        return integrand, param  # Placeholder
    
    elif strategy == 'coefficient':
        # f(x) → f(a*x) or similar
        return integrand.subs(var, param * var), param
    
    else:
        return integrand, param


def differentiate_under_integral(
    integrand: Any,
    var: Symbol,
    param: Symbol,
    lower: Any = None,
    upper: Any = None
) -> Any:
    """
    Apply d/da to an integral: d/da ∫ f(x,a) dx = ∫ ∂f/∂a dx
    
    (Assuming limits don't depend on 'a')
    """
    # Differentiate integrand with respect to parameter
    partial_derivative = diff(integrand, param)
    
    return partial_derivative


def apply_feynman_trick(
    integrand: Any,
    var: Symbol,
    param: Symbol,
    lower: Any,
    upper: Any,
    target_value: Any = None
) -> FeynmanResult:
    """
    Apply Feynman's technique to evaluate a definite integral.
    
    This is a guided approach - it shows the steps but may need
    human insight for complex cases.
    
    Args:
        integrand: The integrand f(x, a) already parameterized
        var: Integration variable (e.g., x)
        param: Parameter variable (e.g., a)
        lower: Lower limit
        upper: Upper limit
        target_value: The value of 'a' we want (e.g., a=1)
        
    Returns:
        FeynmanResult with steps and solution
    """
    steps = []
    notes = []
    
    # Step 1: Define I(a) = ∫ f(x, a) dx
    I_a = Integral(integrand, (var, lower, upper))
    
    steps.append(FeynmanStep(
        step_number=1,
        description=f"Define I({param}) = ∫ f({var}, {param}) d{var}",
        expression=I_a,
        latex_expr=latex(I_a)
    ))
    
    # Step 2: Compute dI/da = ∫ ∂f/∂a dx
    partial_f = diff(integrand, param)
    dI_da = Integral(partial_f, (var, lower, upper))
    
    steps.append(FeynmanStep(
        step_number=2,
        description=f"Differentiate under the integral: dI/d{param} = ∫ ∂f/∂{param} d{var}",
        expression=Eq(Symbol(f"dI/d{param}"), dI_da),
        latex_expr=f"\\frac{{dI}}{{d{param}}} = {latex(dI_da)}"
    ))
    
    steps.append(FeynmanStep(
        step_number=3,
        description=f"The partial derivative ∂f/∂{param}",
        expression=partial_f,
        latex_expr=f"\\frac{{\\partial f}}{{\\partial {param}}} = {latex(partial_f)}"
    ))
    
    # Step 3: Try to evaluate the inner integral
    try:
        inner_result = integrate(partial_f, (var, lower, upper))
        
        steps.append(FeynmanStep(
            step_number=4,
            description="Evaluate the inner integral (hopefully easier!)",
            expression=inner_result,
            latex_expr=latex(inner_result)
        ))
        
        # Check if we got a closed form
        if not inner_result.has(Integral):
            notes.append("✅ Inner integral evaluated successfully!")
            
            # Step 4: Now we have dI/da = g(a), so I(a) = ∫ g(a) da
            steps.append(FeynmanStep(
                step_number=5,
                description=f"Now dI/d{param} = {inner_result}, so integrate with respect to {param}",
                expression=Eq(Symbol(f"dI/d{param}"), inner_result),
                latex_expr=f"\\frac{{dI}}{{d{param}}} = {latex(inner_result)}"
            ))
            
            # Integrate with respect to parameter
            I_of_a = integrate(inner_result, param)
            
            steps.append(FeynmanStep(
                step_number=6,
                description=f"I({param}) = ∫ g({param}) d{param}",
                expression=I_of_a,
                latex_expr=f"I({param}) = {latex(I_of_a)} + C"
            ))
            
            notes.append(f"Need to find constant C using boundary condition")
            
            if target_value is not None:
                # Substitute target value
                final = I_of_a.subs(param, target_value)
                steps.append(FeynmanStep(
                    step_number=7,
                    description=f"Substitute {param} = {target_value}",
                    expression=final,
                    latex_expr=f"I({target_value}) = {latex(final)}"
                ))
                
                return FeynmanResult(
                    original_integral=I_a.subs(param, target_value) if target_value else I_a,
                    parameterized_integral=I_a,
                    parameter=param,
                    derivative_wrt_param=dI_da,
                    inner_integral_solved=inner_result,
                    final_result=final,
                    steps=steps,
                    success=True,
                    notes=notes
                )
            
            return FeynmanResult(
                original_integral=I_a,
                parameterized_integral=I_a,
                parameter=param,
                derivative_wrt_param=dI_da,
                inner_integral_solved=inner_result,
                final_result=I_of_a,
                steps=steps,
                success=True,
                notes=notes
            )
        else:
            notes.append("⚠️ Inner integral still not elementary - may need different parameterization")
            
    except Exception as e:
        notes.append(f"❌ Could not evaluate inner integral: {e}")
    
    return FeynmanResult(
        original_integral=I_a,
        parameterized_integral=I_a,
        parameter=param,
        derivative_wrt_param=dI_da,
        inner_integral_solved=None,
        final_result=None,
        steps=steps,
        success=False,
        notes=notes
    )


def diff_under_integral(
    expr: Any,
    integration_var: Symbol,
    param: Symbol,
    lower: Any = None,
    upper: Any = None,
    simplify_result: bool = True
) -> dict:
    """
    CLI-friendly function for differentiation under the integral sign.
    
    Computes: d/d(param) ∫[lower to upper] expr d(integration_var)
            = ∫[lower to upper] ∂expr/∂param d(integration_var)
    
    Args:
        expr: The integrand f(x, a)
        integration_var: Variable of integration (x)
        param: Parameter to differentiate with respect to (a)
        lower: Lower limit (optional)
        upper: Upper limit (optional)
        simplify_result: Whether to simplify
        
    Returns:
        Dictionary with original, derivative, and result
    """
    # Compute partial derivative
    partial = diff(expr, param)
    
    if simplify_result:
        partial = simplify(partial)
    
    result = {
        'original_integrand': expr,
        'integration_variable': integration_var,
        'parameter': param,
        'partial_derivative': partial,
        'partial_derivative_latex': latex(partial),
    }
    
    # If limits provided, try to evaluate
    if lower is not None and upper is not None:
        integral_of_partial = Integral(partial, (integration_var, lower, upper))
        result['new_integral'] = integral_of_partial
        result['new_integral_latex'] = latex(integral_of_partial)
        
        # Try to evaluate
        try:
            evaluated = integrate(partial, (integration_var, lower, upper))
            if not evaluated.has(Integral):
                result['evaluated'] = evaluated
                result['evaluated_latex'] = latex(evaluated)
        except:
            pass
    
    return result


def suggest_parameterization(integrand: Any, var: Symbol) -> List[dict]:
    """
    Suggest possible parameterizations for an integrand.
    
    This is heuristic-based and provides ideas for the user.
    """
    suggestions = []
    a = Symbol('a', positive=True)
    
    # Strategy 1: Exponential damping (good for oscillatory integrands)
    if integrand.has(sin) or integrand.has(cos):
        suggestions.append({
            'strategy': 'Exponential damping',
            'parameterized': integrand * exp(-a * var),
            'description': f'Multiply by e^(-a·{var}), then let a→0⁺',
            'when_useful': 'Oscillatory integrands like sin(x)/x'
        })
    
    # Strategy 2: Power parameter (good for logarithms)
    if integrand.has(log):
        suggestions.append({
            'strategy': 'Power differentiation',
            'parameterized': var**a * integrand.subs(log(var), log(var)),
            'description': f'Use {var}^a and differentiate w.r.t. a',
            'when_useful': 'Integrands with logarithms'
        })
    
    # Strategy 3: Coefficient parameter
    suggestions.append({
        'strategy': 'Coefficient scaling',
        'parameterized': integrand.subs(var, a * var),
        'description': f'Replace {var} with a·{var}',
        'when_useful': 'General technique'
    })
    
    return suggestions


def print_feynman_result(result: FeynmanResult):
    """Pretty print a Feynman technique result."""
    print("\n" + "═" * 60)
    print("FEYNMAN'S TECHNIQUE")
    print("═" * 60)
    
    for step in result.steps:
        print(f"\nStep {step.step_number}: {step.description}")
        print(f"  {step.expression}")
    
    print("\n" + "─" * 60)
    
    if result.success:
        print(f"✅ SUCCESS!")
        print(f"Result: {result.final_result}")
    else:
        print("⚠️ Technique applied but may need further work")
    
    if result.notes:
        print("\nNotes:")
        for note in result.notes:
            print(f"  • {note}")
    
    print("═" * 60)


# Quick test / demo
if __name__ == "__main__":
    x, a = symbols('x a', positive=True)
    
    print("="*70)
    print("FEYNMAN'S TRICK DEMONSTRATION")
    print("="*70)
    
    # Classic example: ∫₀^∞ sin(x)/x dx = π/2
    # Parameterize as ∫₀^∞ sin(x)/x · e^(-ax) dx
    print("\nExample 1: Dirichlet Integral via Feynman's Trick")
    print("Original: ∫₀^∞ sin(x)/x dx")
    print("Parameterized: ∫₀^∞ sin(x)/x · e^(-ax) dx")
    
    parameterized = sin(x) / x * exp(-a * x)
    result = diff_under_integral(parameterized, x, a, 0, oo)
    
    print(f"\n∂/∂a [sin(x)/x · e^(-ax)] = {result['partial_derivative']}")
    if 'evaluated' in result:
        print(f"∫₀^∞ ... dx = {result['evaluated']}")
        print(f"This gives dI/da = {result['evaluated']}")
        print(f"Integrating: I(a) = -arctan(a) + C")
        print(f"As a→∞, I(a)→0, so C = π/2")
        print(f"Therefore I(0) = π/2 ✓")
    
    print("\n" + "="*70)
    print("Example 2: Parameterization suggestions")
    print("="*70)
    
    test_integrand = sin(x) / x
    suggestions = suggest_parameterization(test_integrand, x)
    
    print(f"\nFor integrand: {test_integrand}")
    print("\nSuggested parameterizations:")
    for s in suggestions:
        print(f"\n  Strategy: {s['strategy']}")
        print(f"  Parameterized: {s['parameterized']}")
        print(f"  Description: {s['description']}")
        print(f"  When useful: {s['when_useful']}")
