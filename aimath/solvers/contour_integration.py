"""
Complex Analysis Module (Contour Integration)

Implements the Residue Theorem for evaluating contour integrals:
    [closed integral] f(z) dz = 2*pi*i * Sum of Res(f, z_k)

Standard symbolic engines struggle with contours because they don't know
"where" the poles are relative to the path. This module handles that.

Common applications:
- Evaluating real integrals via complex analysis
- Physics: quantum field theory, signal processing
- Engineering: control theory, circuit analysis
"""

import sympy
from sympy import (
    Symbol, symbols, I, pi, oo, exp, sin, cos, sqrt,
    residue, solve, denom, numer, simplify, nsimplify,
    im, re, Abs, factorial, binomial, gamma, zoo,
    limit, series, O, latex, Rational, Integer
)
from sympy.core.numbers import ComplexInfinity
from typing import List, Tuple, Any, Optional, Dict
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class Pole:
    """Information about a pole (singularity)."""
    location: Any  # Complex number
    order: int = 1  # Order of the pole
    residue: Any = None  # Residue at this pole
    is_inside_contour: bool = False
    

@dataclass
class ContourResult:
    """Result of a contour integration."""
    original_integrand: Any
    variable: Symbol
    contour_type: str  # 'upper_half_plane', 'lower_half_plane', 'unit_circle', 'custom'
    poles: List[Pole]
    poles_inside: List[Pole]
    sum_of_residues: Any
    result: Any
    result_simplified: Any
    steps: List[str] = field(default_factory=list)
    real_integral_value: Any = None  # If computing a real integral


def find_poles(expr, z_var) -> List[Pole]:
    """
    Find all poles (singularities) of an expression.
    
    Args:
        expr: SymPy expression (typically a rational function)
        z_var: Complex variable
        
    Returns:
        List of Pole objects
    """
    poles = []
    
    # Get denominator
    d = denom(expr)
    
    if d == 1:
        # No poles in a polynomial
        return poles
    
    # Solve for zeros of denominator
    try:
        singularities = solve(d, z_var)
    except Exception as e:
        logger.warning(f"Could not find poles: {e}")
        return poles
    
    for s in singularities:
        # Determine order of pole
        order = 1
        test_denom = d
        
        # Count multiplicity
        while True:
            test_denom = simplify(test_denom / (z_var - s))
            if test_denom.subs(z_var, s) == 0:
                order += 1
            else:
                break
            if order > 10:  # Safety limit
                break
        
        # Calculate residue
        try:
            res = residue(expr, z_var, s)
        except Exception:
            res = None
        
        poles.append(Pole(
            location=s,
            order=order,
            residue=res
        ))
    
    return poles


def filter_poles_in_contour(
    poles: List[Pole], 
    contour_type: str = 'upper_half_plane',
    radius: float = None
) -> List[Pole]:
    """
    Filter poles that lie inside the given contour.
    
    Contour types:
    - 'upper_half_plane': Im(z) > 0 (for integrals from -oo to +oo)
    - 'lower_half_plane': Im(z) < 0
    - 'unit_circle': |z| < 1
    - 'circle_radius_R': |z| < R
    """
    inside = []
    
    for pole in poles:
        z = pole.location
        
        # Evaluate imaginary and real parts
        try:
            im_part = complex(z).imag if z.is_number else float(im(z).evalf())
            re_part = complex(z).real if z.is_number else float(re(z).evalf())
            abs_val = abs(complex(z)) if z.is_number else float(Abs(z).evalf())
        except (TypeError, ValueError):
            # Symbolic pole - try to determine
            im_part = im(z)
            re_part = re(z)
            abs_val = Abs(z)
        
        is_inside = False
        
        if contour_type == 'upper_half_plane':
            # Pole is inside if Im(z) > 0
            try:
                is_inside = im_part > 0
            except TypeError:
                # Symbolic comparison
                is_inside = sympy.simplify(im(z) > 0)
                
        elif contour_type == 'lower_half_plane':
            try:
                is_inside = im_part < 0
            except TypeError:
                is_inside = sympy.simplify(im(z) < 0)
                
        elif contour_type == 'unit_circle':
            try:
                is_inside = abs_val < 1
            except TypeError:
                is_inside = sympy.simplify(Abs(z) < 1)
                
        elif contour_type.startswith('circle_radius_'):
            R = float(contour_type.split('_')[-1])
            try:
                is_inside = abs_val < R
            except TypeError:
                is_inside = sympy.simplify(Abs(z) < R)
        
        if is_inside:
            pole.is_inside_contour = True
            inside.append(pole)
    
    return inside


def solve_contour_integral(
    expr,
    z_var,
    contour_type: str = 'upper_half_plane',
    compute_real_integral: bool = True
) -> ContourResult:
    """
    Evaluates a contour integral using the Residue Theorem.
    
    For real integrals Int_{-oo}^{oo} f(x) dx:
    1. Replace x with z
    2. Close contour in upper half plane
    3. Apply: Int = 2*pi*i * Sum of Res(f, poles in UHP)
    
    Args:
        expr: The integrand f(z)
        z_var: The complex variable z
        contour_type: Type of contour
        compute_real_integral: Whether to return the real integral value
        
    Returns:
        ContourResult with all computation details
    """
    steps = []
    
    steps.append(f"Given: [closed integral] {expr} d{z_var}")
    steps.append(f"Contour: {contour_type.replace('_', ' ').title()}")
    
    # Step 1: Find all poles
    steps.append("\nStep 1: Find poles (zeros of denominator)")
    all_poles = find_poles(expr, z_var)
    
    for p in all_poles:
        steps.append(f"  * Pole at z = {p.location} (order {p.order})")
        if p.residue is not None:
            steps.append(f"    Residue: {p.residue}")
    
    if not all_poles:
        steps.append("  No poles found - integrand is entire")
        return ContourResult(
            original_integrand=expr,
            variable=z_var,
            contour_type=contour_type,
            poles=[],
            poles_inside=[],
            sum_of_residues=0,
            result=0,
            result_simplified=0,
            steps=steps
        )
    
    # Step 2: Filter poles inside contour
    steps.append(f"\nStep 2: Identify poles inside the contour")
    inside_poles = filter_poles_in_contour(all_poles, contour_type)
    
    if inside_poles:
        for p in inside_poles:
            steps.append(f"  [OK] z = {p.location} is INSIDE the contour")
    else:
        steps.append("  No poles inside the contour")
    
    # Step 3: Sum residues
    steps.append("\nStep 3: Sum residues of poles inside contour")
    
    residue_sum = 0
    for p in inside_poles:
        if p.residue is None:
            # Calculate if not already done
            try:
                p.residue = residue(expr, z_var, p.location)
            except Exception as e:
                steps.append(f"  Could not compute residue at {p.location}: {e}")
                continue
        
        steps.append(f"  Res(f, {p.location}) = {p.residue}")
        residue_sum += p.residue
    
    residue_sum = simplify(residue_sum)
    steps.append(f"  Sum of Residues = {residue_sum}")
    
    # Step 4: Apply Residue Theorem
    steps.append("\nStep 4: Apply Residue Theorem")
    steps.append("  [closed integral] f(z) dz = 2*pi*i * Sum of Res(f, z_k)")
    
    contour_value = 2 * pi * I * residue_sum
    contour_simplified = simplify(contour_value)
    
    steps.append(f"  = 2*pi*i * ({residue_sum})")
    steps.append(f"  = {contour_simplified}")
    
    # Step 5: Extract real integral if applicable
    real_value = None
    if compute_real_integral and contour_type in ['upper_half_plane', 'lower_half_plane']:
        steps.append("\nStep 5: Extract real integral")
        steps.append("  For Int_{-oo}^{oo} f(x) dx, the contour integral equals the real integral")
        steps.append("  (when the semicircular arc contribution -> 0)")
        
        # The real integral is the real part of the contour integral
        # For standard cases, the contour integral is purely real
        real_value = simplify(contour_simplified)
        
        # If result is purely imaginary times i, extract real part
        if real_value.has(I):
            # Try to simplify
            real_value = simplify(real_value / I) * I
            real_value = simplify(re(contour_simplified))
            if real_value == 0:
                real_value = simplify(contour_simplified / I)
        
        steps.append(f"  Int_{{-oo}}^{{oo}} f(x) dx = {real_value}")
    
    return ContourResult(
        original_integrand=expr,
        variable=z_var,
        contour_type=contour_type,
        poles=all_poles,
        poles_inside=inside_poles,
        sum_of_residues=residue_sum,
        result=contour_value,
        result_simplified=contour_simplified,
        steps=steps,
        real_integral_value=real_value
    )


def real_integral_via_contour(expr, x_var) -> ContourResult:
    """
    Compute a real integral Int_{-oo}^{oo} f(x) dx using contour integration.
    
    This is the most common use case - evaluating a real integral by
    extending to the complex plane.
    
    Args:
        expr: Real integrand f(x)
        x_var: Real variable
        
    Returns:
        ContourResult with the real integral value
    """
    # Create complex variable
    z = Symbol('z')
    
    # Replace x with z
    complex_expr = expr.subs(x_var, z)
    
    # Solve using upper half plane contour
    result = solve_contour_integral(complex_expr, z, 'upper_half_plane', True)
    
    return result


def evaluate_trigonometric_integral(expr, theta_var) -> Any:
    """
    Evaluate integrals of the form Int_0^{2*pi} R(cos t, sin t) dt
    using the substitution z = e^{i*t}.
    
    The substitution gives:
    - cos t = (z + 1/z) / 2
    - sin t = (z - 1/z) / (2i)
    - dt = dz / (iz)
    
    The integral becomes a contour integral over the unit circle.
    """
    z = Symbol('z')
    
    # Substitutions
    cos_sub = (z + 1/z) / 2
    sin_sub = (z - 1/z) / (2*I)
    
    # Transform integrand
    transformed = expr.subs(sympy.cos(theta_var), cos_sub)
    transformed = transformed.subs(sympy.sin(theta_var), sin_sub)
    
    # Include the dt = dz/(iz) factor
    transformed = transformed / (I * z)
    transformed = simplify(transformed)
    
    # Solve over unit circle
    result = solve_contour_integral(transformed, z, 'unit_circle')
    
    return result


def print_contour_result(result: ContourResult):
    """Pretty print a contour integration result."""
    print("\n" + "=" * 60)
    print("CONTOUR INTEGRATION (Residue Theorem)")
    print("=" * 60)
    
    for step in result.steps:
        print(step)
    
    print("\n" + "-" * 60)
    print(f"CONTOUR INTEGRAL: {result.result_simplified}")
    
    if result.real_integral_value is not None:
        print(f"REAL INTEGRAL:    {result.real_integral_value}")
    
    print("=" * 60)


# Convenience function for CLI
def contour_integrate(expr_str: str, var: str = 'x') -> ContourResult:
    """
    User-friendly contour integration.
    
    Example:
        contour_integrate("1/(x**2 + 1)", "x")
    """
    x = Symbol(var)
    expr = sympy.sympify(expr_str)
    
    return real_integral_via_contour(expr, x)


def plot_poles(result: ContourResult, contour_radius: float = 5.0, save_path: str = None):
    """
    Visualize poles in the complex plane with the integration contour.
    
    This helps users understand WHY the Residue Theorem works:
    - Shows all poles (singularities) of the function
    - Highlights which poles are INSIDE the contour (contribute to integral)
    - Draws the semicircular contour path
    
    Args:
        result: ContourResult from solve_contour_integral
        contour_radius: Radius of the semicircular contour
        save_path: If provided, save figure to this path instead of displaying
        
    Returns:
        matplotlib figure object
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("Visualization requires matplotlib: pip install matplotlib")
        return None
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.axhline(y=0, color='k', linewidth=0.8)
    ax.axvline(x=0, color='k', linewidth=0.8)
    
    # Draw semicircular contour (upper half plane)
    theta = np.linspace(0, np.pi, 200)
    x_contour = contour_radius * np.cos(theta)
    y_contour = contour_radius * np.sin(theta)
    
    # Contour path with arrow
    ax.plot(x_contour, y_contour, 'b-', linewidth=2.5, label='Contour C (Upper Half Plane)')
    ax.plot([-contour_radius, contour_radius], [0, 0], 'b-', linewidth=2.5)
    
    # Add direction arrows
    arrow_idx = len(theta) // 4
    ax.annotate('', 
                xy=(x_contour[arrow_idx+5], y_contour[arrow_idx+5]),
                xytext=(x_contour[arrow_idx], y_contour[arrow_idx]),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    ax.annotate('',
                xy=(0, 0),
                xytext=(-contour_radius/2, 0),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    
    # Shade the interior of contour
    theta_fill = np.linspace(0, np.pi, 100)
    x_fill = np.concatenate([contour_radius * np.cos(theta_fill), [-contour_radius, contour_radius]])
    y_fill = np.concatenate([contour_radius * np.sin(theta_fill), [0, 0]])
    ax.fill(x_fill, y_fill, alpha=0.1, color='blue')
    
    # Get pole locations
    inside_locs = [p.location for p in result.poles_inside]
    
    # Plot all poles
    for pole in result.poles:
        try:
            # Convert to complex number
            c_val = complex(pole.location.evalf())
            is_inside = pole.location in inside_locs or pole.is_inside_contour
            
            # Style based on inside/outside
            if is_inside:
                color = 'red'
                marker = 'X'
                size = 200
                edge_color = 'darkred'
            else:
                color = 'gray'
                marker = 'o'
                size = 120
                edge_color = 'black'
            
            ax.scatter(c_val.real, c_val.imag, 
                      c=color, s=size, marker=marker,
                      edgecolors=edge_color, linewidths=2,
                      zorder=10)
            
            # Label the pole
            label_text = f'z = {pole.location}'
            if pole.residue is not None:
                label_text += f'\nRes = {pole.residue}'
            
            # Offset label to avoid overlap
            offset_x = 0.3 if c_val.real >= 0 else -0.3
            offset_y = 0.3
            ha = 'left' if c_val.real >= 0 else 'right'
            
            ax.annotate(label_text, 
                       (c_val.real, c_val.imag),
                       xytext=(c_val.real + offset_x, c_val.imag + offset_y),
                       fontsize=10,
                       ha=ha,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
                       
        except (TypeError, ValueError) as e:
            # Symbolic pole that can't be converted
            print(f"Could not plot pole {pole.location}: {e}")
    
    # Labels and title
    ax.set_xlabel('Re(z)', fontsize=12)
    ax.set_ylabel('Im(z)', fontsize=12)
    
    title = 'Residue Theorem Visualization\n'
    title += f'Integrand: {result.original_integrand}'
    ax.set_title(title, fontsize=13, fontweight='bold')
    
    # Legend with explanation
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    
    legend_elements = [
        Line2D([0], [0], color='blue', linewidth=2.5, label='Contour path C'),
        Patch(facecolor='blue', alpha=0.1, label='Interior of C'),
        Line2D([0], [0], marker='X', color='w', markerfacecolor='red', 
               markersize=12, markeredgecolor='darkred', markeredgewidth=2, 
               label='Poles INSIDE (contribute)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
               markersize=10, markeredgecolor='black', markeredgewidth=2,
               label='Poles OUTSIDE (ignore)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # Add result annotation
    result_text = f'Result: {result.result_simplified}'
    if result.real_integral_value is not None:
        result_text += f'\nReal integral: {result.real_integral_value}'
    
    ax.text(0.02, 0.02, result_text,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    # Set limits
    max_pole = max([abs(complex(p.location.evalf())) for p in result.poles] + [1])
    limit = max(contour_radius, max_pole) + 1
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit * 0.3, limit)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    else:
        plt.show()
    
    return fig


# Quick test
if __name__ == "__main__":
    z = Symbol('z')
    x = Symbol('x')
    
    print("="*70)
    print("TEST 1: Classic integral Int 1/(x^2 + 1) dx from -oo to oo")
    print("        Expected: pi")
    print("="*70)
    
    result = real_integral_via_contour(1/(x**2 + 1), x)
    print_contour_result(result)
    
    print("\n" + "="*70)
    print("TEST 2: Int 1/(x^2 + 1)^2 dx from -oo to oo")
    print("        Expected: pi/2")
    print("="*70)
    
    result = real_integral_via_contour(1/(x**2 + 1)**2, x)
    print_contour_result(result)
    
    print("\n" + "="*70)
    print("TEST 3: Int x^2/(x^4 + 1) dx from -oo to oo")
    print("        Expected: pi/sqrt(2)")
    print("="*70)
    
    result = real_integral_via_contour(x**2/(x**4 + 1), x)
    print_contour_result(result)
