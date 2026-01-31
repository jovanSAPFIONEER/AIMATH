#!/usr/bin/env python3
"""
AIMATH: Research-Grade Neuro-Symbolic Math Engine
=================================================

A unified command-line interface for advanced mathematical computation.

Features:
- Interactive step-by-step problem builder (wizard)
- Contour integration via Residue Theorem
- PDE solving (Heat, Wave, Transport equations)
- Conjecture fuzz testing (counterexample finder)
- Hybrid integration (database + symbolic + numeric)
- Constant recognition (inverse symbolic calculator)

Usage:
    python main.py wizard              # Interactive mode
    python main.py contour "1/(z**2+1)"   # Contour integral
    python main.py pde heat            # Solve heat equation
    python main.py verify "(a+b)**2" "a**2+b**2"  # Fuzz test
    python main.py integrate "atan(sqrt(x**2+2))/((x**2+1)*sqrt(x**2+2))" --bounds 0 1
    python main.py recognize 0.5148668...

Author: AIMATH Team
"""

import click
import sympy
from sympy import symbols, sympify, latex, Symbol, Function, Eq, Derivative, pi, I
from sympy.abc import x, t, z
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ═══════════════════════════════════════════════════════════════════════════════
# CLI GROUP
# ═══════════════════════════════════════════════════════════════════════════════

@click.group()
@click.version_option(version='1.0.0', prog_name='AIMATH')
def cli():
    """
    AIMATH: Research-Grade Neuro-Symbolic Math Engine
    
    A powerful toolkit for symbolic computation, complex analysis,
    PDEs, and mathematical verification.
    
    \b
    Examples:
      aimath wizard                    - Interactive problem builder
      aimath contour "1/(z**2+1)"      - Contour integral via residues
      aimath pde heat                  - Solve heat equation
      aimath verify "(a+b)**2" "a**2+b**2" - Fuzz test identity
    """
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# 1. INTERACTIVE WIZARD MODE
# ═══════════════════════════════════════════════════════════════════════════════

@cli.command()
def wizard():
    """Launch the interactive step-by-step math builder."""
    click.echo(click.style("\n🧙 AIMATH Interactive Wizard", fg='cyan', bold=True))
    click.echo("=" * 50)
    
    try:
        from aimath.cli import MathCLI
        cli_instance = MathCLI()
        cli_instance.run()
    except ImportError as e:
        click.echo(click.style(f"Error loading wizard: {e}", fg='red'))
        click.echo("Falling back to simple REPL mode...")
        _simple_repl()


def _simple_repl():
    """Simple fallback REPL if full CLI not available."""
    from aimath.core.engine import MathEngine
    engine = MathEngine()
    
    click.echo("\nType 'quit' to exit, or enter a math expression:")
    while True:
        try:
            expr = click.prompt(click.style(">>> ", fg='green'), prompt_suffix='')
            if expr.lower() in ('quit', 'exit', 'q'):
                break
            result = engine.solve(expr)
            click.echo(f"Result: {result.solution}")
        except (KeyboardInterrupt, EOFError):
            break
        except Exception as e:
            click.echo(click.style(f"Error: {e}", fg='red'))


# ═══════════════════════════════════════════════════════════════════════════════
# 2. CONTOUR INTEGRATION (RESIDUE THEOREM)
# ═══════════════════════════════════════════════════════════════════════════════

@cli.command()
@click.argument('expression')
@click.option('--var', default='x', help='Variable of integration (default: x)')
@click.option('--plot', is_flag=True, help='Visualize poles in complex plane')
def contour(expression, var, plot):
    """
    Evaluate contour integrals using the Residue Theorem.
    
    \b
    Examples:
      aimath contour "1/(x**2+1)"        - Classic integral = pi
      aimath contour "1/(x**2+1)**2"     - Result = pi/2
      aimath contour "x**2/(x**4+1)"     - Result = pi/sqrt(2)
      aimath contour "1/(z**2+1)" --plot - With visualization
    """
    click.echo(click.style(f"\n🌀 Contour Integration (Residue Theorem)", fg='cyan', bold=True))
    click.echo("=" * 50)
    click.echo(f"Integrand: {expression}")
    click.echo(f"Variable:  {var}")
    click.echo("-" * 50)
    
    try:
        from aimath.solvers.contour_integration import contour_integrate, print_contour_result
        
        result = contour_integrate(expression, var)
        
        # Display steps
        for step in result.steps:
            click.echo(step)
        
        click.echo("-" * 50)
        click.echo(click.style(f"CONTOUR INTEGRAL: {result.result_simplified}", fg='green', bold=True))
        
        if result.real_integral_value is not None:
            click.echo(click.style(f"REAL INTEGRAL:    {result.real_integral_value}", fg='green', bold=True))
            click.echo(f"LaTeX: ${latex(result.real_integral_value)}$")
        
        # Visualization
        if plot and result.poles:
            click.echo("\n📊 Generating pole visualization...")
            _plot_poles([p.location for p in result.poles], 
                       [p.location for p in result.poles_inside])
            
    except ImportError as e:
        click.echo(click.style(f"Module error: {e}", fg='red'))
    except Exception as e:
        click.echo(click.style(f"Error: {e}", fg='red'))


def _plot_poles(all_poles, inside_poles, contour_radius=5):
    """Visualize poles in the complex plane with contour."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5)
        
        # Draw semicircular contour (upper half plane)
        theta = np.linspace(0, np.pi, 100)
        x_c = contour_radius * np.cos(theta)
        y_c = contour_radius * np.sin(theta)
        ax.plot(x_c, y_c, 'b--', linewidth=2, label='Contour (UHP)')
        ax.plot([-contour_radius, contour_radius], [0, 0], 'b--', linewidth=2)
        
        # Add arrow to show direction
        ax.annotate('', xy=(0, contour_radius), xytext=(contour_radius*0.7, contour_radius*0.7),
                    arrowprops=dict(arrowstyle='->', color='blue', lw=2))
        
        # Plot all poles
        for p in all_poles:
            try:
                c_val = complex(p.evalf())
                is_inside = p in inside_poles
                color = 'red' if is_inside else 'gray'
                marker = 'X' if is_inside else 'o'
                size = 150 if is_inside else 100
                label = 'Pole (inside)' if is_inside else 'Pole (outside)'
                
                ax.scatter(c_val.real, c_val.imag, c=color, s=size, marker=marker, 
                          zorder=5, label=label if p == all_poles[0] else '')
                ax.annotate(f'  z={p}', (c_val.real, c_val.imag), fontsize=10)
            except:
                pass
        
        ax.set_xlim(-contour_radius-1, contour_radius+1)
        ax.set_ylim(-2, contour_radius+1)
        ax.set_xlabel('Re(z)', fontsize=12)
        ax.set_ylabel('Im(z)', fontsize=12)
        ax.set_title('Residue Theorem: Poles & Contour Path', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        click.echo(click.style("matplotlib not installed. Run: pip install matplotlib", fg='yellow'))


# ═══════════════════════════════════════════════════════════════════════════════
# 3. PDE SOLVER
# ═══════════════════════════════════════════════════════════════════════════════

@cli.command()
@click.argument('pde_type', type=click.Choice(['heat', 'wave', 'transport', 'laplace']))
@click.option('--plot', is_flag=True, help='Visualize solution (heat map)')
def pde(pde_type, plot):
    """
    Solve standard PDEs via Separation of Variables.
    
    \b
    Supported equations:
      heat      - du/dt = alpha * d²u/dx²  (diffusion)
      wave      - d²u/dt² = c² * d²u/dx²   (vibrating string)
      transport - du/dt + c*du/dx = 0      (advection)
      laplace   - d²u/dx² + d²u/dy² = 0    (steady-state)
    
    \b
    Examples:
      aimath pde heat
      aimath pde wave
      aimath pde heat --plot
    """
    click.echo(click.style(f"\n🔥 PDE Solver (Separation of Variables)", fg='cyan', bold=True))
    click.echo("=" * 50)
    
    x_sym, t_sym = symbols('x t')
    u = Function('u')(x_sym, t_sym)
    
    equations = {
        'heat': Eq(Derivative(u, t_sym), Derivative(u, x_sym, 2)),
        'wave': Eq(Derivative(u, t_sym, 2), symbols('c')**2 * Derivative(u, x_sym, 2)),
        'transport': Eq(Derivative(u, t_sym) + symbols('c') * Derivative(u, x_sym), 0),
        'laplace': Eq(Derivative(u, x_sym, 2) + Derivative(u, symbols('y'), 2), 0)
    }
    
    descriptions = {
        'heat': 'Heat Equation: du/dt = alpha * d^2u/dx^2',
        'wave': 'Wave Equation: d^2u/dt^2 = c^2 * d^2u/dx^2',
        'transport': 'Transport Equation: du/dt + c*du/dx = 0',
        'laplace': 'Laplace Equation: d^2u/dx^2 + d^2u/dy^2 = 0'
    }
    
    click.echo(f"Equation: {descriptions[pde_type]}")
    click.echo("-" * 50)
    
    try:
        from aimath.solvers.pde_solver import solve_pde_separation, print_pde_solution
        
        eq = equations[pde_type]
        result = solve_pde_separation(eq, u, x_sym, t_sym)
        print_pde_solution(result)
        
        if plot and pde_type == 'heat':
            click.echo("\n📊 Generating heat map visualization...")
            _plot_heat_solution()
            
    except ImportError as e:
        click.echo(click.style(f"Module error: {e}", fg='red'))
    except Exception as e:
        click.echo(click.style(f"Error: {e}", fg='red'))


def _plot_heat_solution():
    """Visualize heat equation solution as a heat map."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Numerical solution of heat equation with initial condition sin(pi*x)
        L = 1.0  # Domain length
        alpha = 0.01  # Diffusivity
        nx = 50
        nt = 100
        dx = L / (nx - 1)
        dt = 0.5 * dx**2 / alpha  # Stability condition
        
        x = np.linspace(0, L, nx)
        t_vals = np.linspace(0, nt*dt, nt)
        
        # Initial condition: sin(pi*x)
        u = np.zeros((nt, nx))
        u[0, :] = np.sin(np.pi * x)
        
        # Finite difference time stepping
        for n in range(0, nt-1):
            for i in range(1, nx-1):
                u[n+1, i] = u[n, i] + alpha * dt / dx**2 * (u[n, i+1] - 2*u[n, i] + u[n, i-1])
            # Boundary conditions
            u[n+1, 0] = 0
            u[n+1, -1] = 0
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Heat map
        im = ax1.imshow(u, aspect='auto', origin='lower', 
                       extent=[0, L, 0, t_vals[-1]], cmap='hot')
        ax1.set_xlabel('Position x')
        ax1.set_ylabel('Time t')
        ax1.set_title('Heat Equation: u(x,t) Evolution')
        plt.colorbar(im, ax=ax1, label='Temperature u')
        
        # Solution at different times
        for i, t_idx in enumerate([0, nt//4, nt//2, 3*nt//4, nt-1]):
            ax2.plot(x, u[t_idx, :], label=f't = {t_vals[t_idx]:.3f}')
        ax2.set_xlabel('Position x')
        ax2.set_ylabel('Temperature u')
        ax2.set_title('Temperature Profiles at Different Times')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        click.echo(click.style("matplotlib not installed. Run: pip install matplotlib", fg='yellow'))


# ═══════════════════════════════════════════════════════════════════════════════
# 4. CONJECTURE FUZZER (VERIFY)
# ═══════════════════════════════════════════════════════════════════════════════

@cli.command()
@click.argument('lhs')
@click.argument('rhs')
@click.option('--trials', default=10000, help='Number of random trials (default: 10000)')
@click.option('--domain', default=10.0, help='Test domain: [-domain, domain] (default: 10)')
def verify(lhs, rhs, trials, domain):
    """
    Fuzz test a mathematical identity (LHS = RHS).
    
    \b
    Examples:
      aimath verify "sin(x)**2+cos(x)**2" "1"           - True identity
      aimath verify "(a+b)**2" "a**2+b**2"              - Find counterexample!
      aimath verify "(a+b)**2" "a**2+2*a*b+b**2"        - True identity
      aimath verify "exp(a+b)" "exp(a)*exp(b)"          - True identity
    """
    click.echo(click.style(f"\n🕵️  Conjecture Fuzzer", fg='cyan', bold=True))
    click.echo("=" * 50)
    click.echo(f"Testing: {lhs} == {rhs}")
    click.echo(f"Trials:  {trials}")
    click.echo(f"Domain:  [-{domain}, {domain}]")
    click.echo("-" * 50)
    
    try:
        from aimath.solvers.conjecture_tester import test_conjecture, print_conjecture_result
        
        result = test_conjecture(lhs, rhs, trials=trials, domain=(-domain, domain))
        print_conjecture_result(result)
        
    except ImportError as e:
        click.echo(click.style(f"Module error: {e}", fg='red'))
        # Fallback basic test
        _basic_fuzz_test(lhs, rhs, trials, domain)
    except Exception as e:
        click.echo(click.style(f"Error: {e}", fg='red'))


def _basic_fuzz_test(lhs, rhs, trials, domain):
    """Basic fallback fuzz tester."""
    import random
    
    lhs_expr = sympify(lhs)
    rhs_expr = sympify(rhs)
    
    # Get variables
    all_vars = list(lhs_expr.free_symbols | rhs_expr.free_symbols)
    
    if not all_vars:
        # No variables - direct comparison
        if sympy.simplify(lhs_expr - rhs_expr) == 0:
            click.echo(click.style("✅ Trivially TRUE (identical expressions)", fg='green', bold=True))
        else:
            click.echo(click.style("❌ FALSE (expressions differ)", fg='red', bold=True))
        return
    
    # Fuzz test
    counterexamples = []
    for _ in range(min(trials, 1000)):
        vals = {v: random.uniform(-domain, domain) for v in all_vars}
        try:
            lhs_val = float(lhs_expr.subs(vals).evalf())
            rhs_val = float(rhs_expr.subs(vals).evalf())
            if abs(lhs_val - rhs_val) > 1e-9:
                counterexamples.append((vals, lhs_val, rhs_val))
                if len(counterexamples) >= 3:
                    break
        except:
            pass
    
    if counterexamples:
        click.echo(click.style("❌ DISPROVEN - Counterexample found!", fg='red', bold=True))
        for vals, lhs_v, rhs_v in counterexamples[:3]:
            vals_str = ', '.join(f"{k}={v:.2f}" for k, v in vals.items())
            click.echo(f"   {vals_str} -> LHS={lhs_v:.4f}, RHS={rhs_v:.4f}")
    else:
        click.echo(click.style(f"✅ Conjecture holds for {min(trials, 1000)} trials", fg='green', bold=True))


# ═══════════════════════════════════════════════════════════════════════════════
# 5. HYBRID INTEGRATOR
# ═══════════════════════════════════════════════════════════════════════════════

@cli.command()
@click.argument('expression')
@click.option('--bounds', nargs=2, type=float, default=None, help='Integration bounds: --bounds 0 1')
@click.option('--var', default='x', help='Variable of integration (default: x)')
def integrate(expression, bounds, var):
    """
    Hybrid integration: Database -> Symbolic -> Numeric + Recognition.
    
    \b
    Examples:
      aimath integrate "x**2"                    - Indefinite integral
      aimath integrate "1/x" --bounds 1 2        - Definite integral
      aimath integrate "atan(sqrt(x**2+2))/((x**2+1)*sqrt(x**2+2))" --bounds 0 1
    """
    click.echo(click.style(f"\n∫ Hybrid Integrator", fg='cyan', bold=True))
    click.echo("=" * 50)
    click.echo(f"Integrand: {expression}")
    if bounds:
        click.echo(f"Bounds:    [{bounds[0]}, {bounds[1]}]")
    click.echo("-" * 50)
    
    try:
        from aimath.solvers.hybrid_integrator import hybrid_integrate
        
        result = hybrid_integrate(expression, var, bounds[0] if bounds else None, 
                                  bounds[1] if bounds else None)
        
        click.echo(f"Method:    {result.method}")
        click.echo(f"Result:    {result.result}")
        
        if result.numerical_value is not None:
            click.echo(f"Numeric:   {result.numerical_value}")
        
        click.echo(click.style(f"EXACT:     {result.result}", fg='green', bold=True))
        click.echo(f"LaTeX:     ${result.result_latex}$")
        click.echo(f"Confidence: {result.confidence}")
        
    except ImportError as e:
        click.echo(click.style(f"Module error: {e}", fg='red'))
    except Exception as e:
        click.echo(click.style(f"Error: {e}", fg='red'))


# ═══════════════════════════════════════════════════════════════════════════════
# 6. CONSTANT RECOGNIZER
# ═══════════════════════════════════════════════════════════════════════════════

@cli.command()
@click.argument('value', type=float)
@click.option('--tolerance', default=1e-10, help='Recognition tolerance')
def recognize(value, tolerance):
    """
    Inverse symbolic calculator: decimal -> exact form.
    
    \b
    Examples:
      aimath recognize 3.14159265358979    - Recognizes pi
      aimath recognize 0.5148668            - Recognizes 5*pi^2/96
      aimath recognize 1.6449340668         - Recognizes pi^2/6 (Basel)
    """
    click.echo(click.style(f"\n🔍 Constant Recognizer", fg='cyan', bold=True))
    click.echo("=" * 50)
    click.echo(f"Input:     {value}")
    click.echo("-" * 50)
    
    try:
        from aimath.solvers.constant_recognizer import recognize_constant
        
        result = recognize_constant(value, tolerance)
        
        if result is not None:
            click.echo(click.style(f"RECOGNIZED: {result.recognized_form}", fg='green', bold=True))
            click.echo(f"LaTeX:      ${result.latex}$")
            click.echo(f"Numeric:    {float(result.recognized_form.evalf())}")
            click.echo(f"Error:      {result.error:.2e}")
            click.echo(f"Confidence: {result.confidence}")
            click.echo(f"Method:     {result.method}")
        else:
            click.echo(click.style("No exact form recognized", fg='yellow'))
            click.echo(f"The value {value} could not be matched to a known constant.")
            
    except ImportError as e:
        click.echo(click.style(f"Module error: {e}", fg='red'))
    except Exception as e:
        click.echo(click.style(f"Error: {e}", fg='red'))


# ═══════════════════════════════════════════════════════════════════════════════
# 7. QUICK SOLVE
# ═══════════════════════════════════════════════════════════════════════════════

@cli.command()
@click.argument('problem')
def solve(problem):
    """
    Solve a mathematical problem (equations, simplification, etc.)
    
    \b
    Examples:
      aimath solve "x**2 - 4 = 0"
      aimath solve "diff(sin(x), x)"
      aimath solve "simplify((x**2-1)/(x-1))"
    """
    click.echo(click.style(f"\n📐 Math Solver", fg='cyan', bold=True))
    click.echo("=" * 50)
    click.echo(f"Problem: {problem}")
    click.echo("-" * 50)
    
    try:
        from aimath.core.engine import MathEngine
        engine = MathEngine()
        result = engine.solve(problem)
        
        click.echo(click.style(f"SOLUTION: {result.solution}", fg='green', bold=True))
        
        if hasattr(result, 'latex') and result.latex:
            click.echo(f"LaTeX:    ${result.latex}$")
            
    except Exception as e:
        click.echo(click.style(f"Error: {e}", fg='red'))


# ═══════════════════════════════════════════════════════════════════════════════
# 8. RIEMANN ZETA VISUALIZER
# ═══════════════════════════════════════════════════════════════════════════════

@cli.command()
@click.option('--t-max', default=50, help='Height up the critical line (default: 50)')
@click.option('--resolution', default=1000, help='Plot resolution (default: 1000)')
@click.option('--save', default=None, help='Save plot to file instead of displaying')
def riemann(t_max, resolution, save):
    """
    Visualize the Riemann Zeta Function along the Critical Line.
    
    Plots |zeta(0.5 + it)|. The Riemann Hypothesis states that ALL 
    non-trivial zeros lie on this line (Re(s) = 1/2).
    
    Zeros appear where the graph touches the x-axis!
    
    \b
    Examples:
      aimath riemann                   - Default view (t=0 to 50)
      aimath riemann --t-max 100       - See more zeros
      aimath riemann --resolution 2000 - Higher detail
      aimath riemann --save zeta.png   - Save to file
    """
    click.echo(click.style(f"\n🌌 Riemann Zeta Critical Line Explorer", fg='cyan', bold=True))
    click.echo("=" * 60)
    click.echo(f"Exploring the Critical Line: s = 0.5 + it, where 0 < t < {t_max}")
    click.echo(f"Resolution: {resolution} points")
    click.echo("-" * 60)
    
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.special import zeta
    except ImportError as e:
        click.echo(click.style(f"Missing dependency: {e}", fg='red'))
        click.echo("Install with: pip install numpy matplotlib scipy")
        return
    
    click.echo("Computing zeta values along critical line...")
    
    # 1. Define the Critical Line: s = 0.5 + i*t
    t_values = np.linspace(0.1, t_max, resolution)  # Start at 0.1 to avoid pole
    s_values = 0.5 + 1j * t_values
    
    # 2. Compute Zeta (Vectorized for speed)
    zeta_values = zeta(s_values)
    zeta_mag = np.abs(zeta_values)
    zeta_real = np.real(zeta_values)
    zeta_imag = np.imag(zeta_values)
    
    # 3. Find zeros (local minima where magnitude < threshold)
    zero_threshold = 0.5
    zeros_indices = []
    for i in range(1, len(zeta_mag) - 1):
        if zeta_mag[i] < zeta_mag[i-1] and zeta_mag[i] < zeta_mag[i+1] and zeta_mag[i] < zero_threshold:
            zeros_indices.append(i)
    
    zeros_t = t_values[zeros_indices]
    zeros_mag = zeta_mag[zeros_indices]
    
    click.echo(f"Found {len(zeros_t)} zeros in range [0, {t_max}]:")
    for i, zt in enumerate(zeros_t[:10]):  # Show first 10
        click.echo(f"  Zero #{i+1}: t ≈ {zt:.6f}  (s = 0.5 + {zt:.6f}i)")
    if len(zeros_t) > 10:
        click.echo(f"  ... and {len(zeros_t) - 10} more")
    
    # 4. Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(r'Riemann Zeta Function: $\zeta(0.5 + it)$ on the Critical Line', 
                 fontsize=14, fontweight='bold')
    
    # Plot 1: Magnitude |zeta(s)|
    ax1 = axes[0, 0]
    ax1.plot(t_values, zeta_mag, 'b-', linewidth=0.8, label=r'$|\zeta(0.5 + it)|$')
    ax1.scatter(zeros_t, zeros_mag, color='red', s=50, zorder=5, label=f'Zeros ({len(zeros_t)} found)')
    ax1.axhline(0, color='black', linewidth=1)
    ax1.set_xlabel('t (Imaginary part)')
    ax1.set_ylabel('Magnitude')
    ax1.set_title(r'Magnitude $|\zeta(s)|$ - Zeros touch the axis!')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, t_max)
    ax1.set_ylim(0, min(5, max(zeta_mag)))
    
    # Plot 2: Real and Imaginary parts
    ax2 = axes[0, 1]
    ax2.plot(t_values, zeta_real, 'b-', linewidth=0.8, label=r'Re($\zeta$)', alpha=0.8)
    ax2.plot(t_values, zeta_imag, 'r-', linewidth=0.8, label=r'Im($\zeta$)', alpha=0.8)
    ax2.axhline(0, color='black', linewidth=1)
    ax2.scatter(zeros_t, np.zeros_like(zeros_t), color='green', s=50, zorder=5, marker='x')
    ax2.set_xlabel('t (Imaginary part)')
    ax2.set_ylabel('Value')
    ax2.set_title(r'Real & Imaginary Parts of $\zeta(0.5 + it)$')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, t_max)
    
    # Plot 3: Complex plane trajectory (Argand diagram)
    ax3 = axes[1, 0]
    # Color by t value
    points = ax3.scatter(zeta_real, zeta_imag, c=t_values, cmap='viridis', s=1, alpha=0.6)
    ax3.scatter([0], [0], color='red', s=100, marker='*', zorder=5, label='Origin (zeros pass here)')
    ax3.axhline(0, color='black', linewidth=0.5)
    ax3.axvline(0, color='black', linewidth=0.5)
    ax3.set_xlabel(r'Re($\zeta$)')
    ax3.set_ylabel(r'Im($\zeta$)')
    ax3.set_title(r'Complex Plane: $\zeta(0.5 + it)$ trajectory')
    ax3.set_aspect('equal')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    plt.colorbar(points, ax=ax3, label='t value')
    
    # Plot 4: Zoomed view near first zero
    ax4 = axes[1, 1]
    if len(zeros_t) > 0:
        first_zero = zeros_t[0]
        zoom_range = 3
        mask = (t_values > first_zero - zoom_range) & (t_values < first_zero + zoom_range)
        ax4.plot(t_values[mask], zeta_mag[mask], 'b-', linewidth=1.5)
        ax4.scatter([first_zero], [zeta_mag[zeros_indices[0]]], color='red', s=100, zorder=5)
        ax4.axhline(0, color='black', linewidth=1)
        ax4.axvline(first_zero, color='red', linestyle='--', alpha=0.5, label=f't ≈ {first_zero:.4f}')
        ax4.set_xlabel('t')
        ax4.set_ylabel('Magnitude')
        ax4.set_title(f'Zoomed: First Zero at t ≈ {first_zero:.6f}')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.text(first_zero, 0.3, f'ζ(0.5 + {first_zero:.2f}i) ≈ 0', ha='center', fontsize=10)
    else:
        ax4.text(0.5, 0.5, 'No zeros found in range', ha='center', va='center', transform=ax4.transAxes)
    
    plt.tight_layout()
    
    if save:
        plt.savefig(save, dpi=150, bbox_inches='tight')
        click.echo(click.style(f"✅ Plot saved to: {save}", fg='green'))
    else:
        click.echo(click.style("✅ Rendering plot... Look for red dots where the curve touches zero!", fg='green'))
        plt.show()
    
    # Summary
    click.echo("\n" + "-" * 60)
    click.echo("The Riemann Hypothesis (UNSOLVED $1M problem):")
    click.echo("  'All non-trivial zeros have real part exactly 1/2'")
    click.echo(f"  We found {len(zeros_t)} zeros on this line in [0, {t_max}]")
    click.echo("  Every zero you see confirms the hypothesis... but doesn't prove it!")
    click.echo("=" * 60)


# ═══════════════════════════════════════════════════════════════════════════════
# 8b. RIEMANN ZETA 3D LANDSCAPE
# ═══════════════════════════════════════════════════════════════════════════════

@cli.command()
@click.option('--t-max', default=50, help='Height up the critical strip (default: 50)')
@click.option('--sigma-min', default=0.0, help='Min real part (default: 0.0)')
@click.option('--sigma-max', default=1.0, help='Max real part (default: 1.0)')
@click.option('--resolution', default=150, help='Grid resolution (default: 150)')
@click.option('--save', default=None, help='Save plot to file instead of displaying')
def riemann3d(t_max, sigma_min, sigma_max, resolution, save):
    """
    Interactive 3D visualization of the Riemann Zeta function.
    
    Shows |zeta(sigma + it)| as a 3D surface with full keyboard controls!
    
    \b
    CONTROLS:
      Arrow Keys  - Rotate the 3D view (smooth, no lag!)
      R           - Reset to default view
      1-5         - Preset camera positions
      +/-         - Zoom in/out
      Mouse       - Still works for fine control
      Q/Escape    - Quit
    
    \b
    Examples:
      aimath riemann3d                  - Interactive 3D view
      aimath riemann3d --t-max 100      - See more zeros
      aimath riemann3d --resolution 200 - Higher detail
    """
    click.echo(click.style(f"\n🏔️  Riemann Zeta 3D Landscape (Interactive)", fg='cyan', bold=True))
    click.echo("=" * 60)
    click.echo(f"Critical Strip: {sigma_min} < Re(s) < {sigma_max}")
    click.echo(f"Imaginary range: 0 < Im(s) < {t_max}")
    click.echo(f"Resolution: {resolution} x {resolution} grid")
    click.echo("-" * 60)
    
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from scipy.special import zeta
        from matplotlib import cm
        from matplotlib.colors import LightSource
        from matplotlib.widgets import Button
    except ImportError as e:
        click.echo(click.style(f"Missing dependency: {e}", fg='red'))
        click.echo("Install with: pip install numpy matplotlib scipy")
        return
    
    click.echo("Computing zeta values across the critical strip...")
    
    # Create grid
    sigma_values = np.linspace(sigma_min + 0.02, sigma_max - 0.02, resolution)
    t_values = np.linspace(1, t_max, resolution)
    
    SIGMA, T = np.meshgrid(sigma_values, t_values)
    S = SIGMA + 1j * T
    
    # Compute zeta magnitude
    Z = np.abs(zeta(S))
    Z = np.clip(Z, 0, 5)
    
    # Find zeros on critical line
    critical_idx = np.argmin(np.abs(sigma_values - 0.5))
    critical_profile = Z[:, critical_idx]
    
    zeros_indices = []
    for i in range(1, len(critical_profile) - 1):
        if critical_profile[i] < critical_profile[i-1] and critical_profile[i] < critical_profile[i+1] and critical_profile[i] < 0.5:
            zeros_indices.append(i)
    
    zeros_t = t_values[zeros_indices]
    zeros_mag = critical_profile[zeros_indices]
    
    click.echo(click.style(f"✓ Found {len(zeros_t)} zeros on the critical line", fg='green'))
    for i, zt in enumerate(zeros_t[:5]):
        click.echo(f"  Zero #{i+1}: t ≈ {zt:.4f}")
    if len(zeros_t) > 5:
        click.echo(f"  ... and {len(zeros_t) - 5} more")
    
    # ═══════════════════════════════════════════════════════════════
    # Create the interactive figure
    # ═══════════════════════════════════════════════════════════════
    plt.style.use('default')
    fig = plt.figure(figsize=(14, 10), facecolor='#f5f5f5')
    
    # Main 3D plot (larger area)
    ax = fig.add_axes([0.05, 0.15, 0.65, 0.75], projection='3d')
    
    # Create surface with lighting
    ls = LightSource(270, 45)
    rgb = ls.shade(Z, cmap=cm.coolwarm, vert_exag=0.1, blend_mode='soft')
    
    surf = ax.plot_surface(SIGMA, T, Z, facecolors=rgb, 
                           linewidth=0, antialiased=True, alpha=0.9,
                           rstride=2, cstride=2)
    
    # Draw critical line
    critical_line_z = Z[:, critical_idx]
    ax.plot(np.full_like(t_values, 0.5), t_values, critical_line_z, 
            color='#FF0000', linewidth=4, label='Critical Line (σ = 0.5)', zorder=10)
    
    # Mark zeros
    for zt, zm in zip(zeros_t, zeros_mag):
        ax.scatter([0.5], [zt], [0], color='red', s=150, marker='o', 
                  edgecolors='darkred', linewidths=2, zorder=15)
    
    # Floor projection
    ax.plot(np.full_like(t_values, 0.5), t_values, np.zeros_like(t_values), 
            'r--', linewidth=2, alpha=0.5)
    
    ax.set_xlabel(r'Re(s) = σ', fontsize=12, labelpad=10)
    ax.set_ylabel(r'Im(s) = t', fontsize=12, labelpad=10)
    ax.set_zlabel(r'$|\zeta(s)|$', fontsize=12, labelpad=10)
    ax.set_xlim(sigma_min, sigma_max)
    ax.set_ylim(0, t_max)
    ax.set_zlim(0, 5)
    
    # Clean 3D appearance
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('lightgray')
    ax.yaxis.pane.set_edgecolor('lightgray')
    ax.zaxis.pane.set_edgecolor('lightgray')
    ax.grid(True, alpha=0.3)
    
    # ═══════════════════════════════════════════════════════════════
    # View presets and state
    # ═══════════════════════════════════════════════════════════════
    views = {
        'default': (25, -50),
        'top': (90, 0),
        'front': (0, 0),
        'side': (0, -90),
        'birds_eye': (60, -45),
        'dramatic': (15, -70)
    }
    
    current_view = {'elev': 25, 'azim': -50, 'dist': 10}
    ax.view_init(elev=current_view['elev'], azim=current_view['azim'])
    ax.dist = current_view['dist']
    
    # ═══════════════════════════════════════════════════════════════
    # Keyboard controls
    # ═══════════════════════════════════════════════════════════════
    def on_key(event):
        step = 5  # Rotation step in degrees
        
        if event.key == 'left':
            current_view['azim'] -= step
        elif event.key == 'right':
            current_view['azim'] += step
        elif event.key == 'up':
            current_view['elev'] = min(90, current_view['elev'] + step)
        elif event.key == 'down':
            current_view['elev'] = max(-90, current_view['elev'] - step)
        elif event.key in ['+', '=']:
            current_view['dist'] = max(5, current_view['dist'] - 1)
        elif event.key == '-':
            current_view['dist'] = min(20, current_view['dist'] + 1)
        elif event.key == 'r':
            current_view['elev'], current_view['azim'] = views['default']
            current_view['dist'] = 10
        elif event.key == '1':
            current_view['elev'], current_view['azim'] = views['default']
        elif event.key == '2':
            current_view['elev'], current_view['azim'] = views['top']
        elif event.key == '3':
            current_view['elev'], current_view['azim'] = views['front']
        elif event.key == '4':
            current_view['elev'], current_view['azim'] = views['side']
        elif event.key == '5':
            current_view['elev'], current_view['azim'] = views['birds_eye']
        elif event.key == '6':
            current_view['elev'], current_view['azim'] = views['dramatic']
        elif event.key in ['q', 'escape']:
            plt.close(fig)
            return
        
        ax.view_init(elev=current_view['elev'], azim=current_view['azim'])
        ax.dist = current_view['dist']
        update_info()
        fig.canvas.draw_idle()
    
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    # ═══════════════════════════════════════════════════════════════
    # Button panel on the right
    # ═══════════════════════════════════════════════════════════════
    button_color = '#4a90d9'
    button_hover = '#357abd'
    
    # View preset buttons
    btn_reset = Button(plt.axes([0.75, 0.82, 0.12, 0.05]), 'RESET (R)', 
                      color='#e74c3c', hovercolor='#c0392b')
    btn_default = Button(plt.axes([0.75, 0.74, 0.12, 0.05]), '1: Default', 
                        color=button_color, hovercolor=button_hover)
    btn_top = Button(plt.axes([0.75, 0.67, 0.12, 0.05]), '2: Top Down', 
                    color=button_color, hovercolor=button_hover)
    btn_front = Button(plt.axes([0.75, 0.60, 0.12, 0.05]), '3: Front', 
                      color=button_color, hovercolor=button_hover)
    btn_side = Button(plt.axes([0.75, 0.53, 0.12, 0.05]), '4: Side', 
                     color=button_color, hovercolor=button_hover)
    btn_birds = Button(plt.axes([0.75, 0.46, 0.12, 0.05]), '5: Bird\'s Eye', 
                      color=button_color, hovercolor=button_hover)
    btn_dramatic = Button(plt.axes([0.75, 0.39, 0.12, 0.05]), '6: Dramatic', 
                         color=button_color, hovercolor=button_hover)
    
    # Zoom buttons
    btn_zoom_in = Button(plt.axes([0.75, 0.28, 0.055, 0.05]), '+ Zoom', 
                        color='#27ae60', hovercolor='#1e8449')
    btn_zoom_out = Button(plt.axes([0.815, 0.28, 0.055, 0.05]), '- Zoom', 
                         color='#27ae60', hovercolor='#1e8449')
    
    def set_view(elev, azim):
        current_view['elev'] = elev
        current_view['azim'] = azim
        ax.view_init(elev=elev, azim=azim)
        update_info()
        fig.canvas.draw_idle()
    
    def reset_view(event):
        current_view['elev'], current_view['azim'] = views['default']
        current_view['dist'] = 10
        ax.view_init(elev=current_view['elev'], azim=current_view['azim'])
        ax.dist = current_view['dist']
        update_info()
        fig.canvas.draw_idle()
    
    def zoom_in(event):
        current_view['dist'] = max(5, current_view['dist'] - 1)
        ax.dist = current_view['dist']
        update_info()
        fig.canvas.draw_idle()
    
    def zoom_out(event):
        current_view['dist'] = min(20, current_view['dist'] + 1)
        ax.dist = current_view['dist']
        update_info()
        fig.canvas.draw_idle()
    
    btn_reset.on_clicked(reset_view)
    btn_default.on_clicked(lambda e: set_view(*views['default']))
    btn_top.on_clicked(lambda e: set_view(*views['top']))
    btn_front.on_clicked(lambda e: set_view(*views['front']))
    btn_side.on_clicked(lambda e: set_view(*views['side']))
    btn_birds.on_clicked(lambda e: set_view(*views['birds_eye']))
    btn_dramatic.on_clicked(lambda e: set_view(*views['dramatic']))
    btn_zoom_in.on_clicked(zoom_in)
    btn_zoom_out.on_clicked(zoom_out)
    
    # Style buttons
    for btn in [btn_reset, btn_default, btn_top, btn_front, btn_side, btn_birds, btn_dramatic, btn_zoom_in, btn_zoom_out]:
        btn.label.set_fontsize(9)
        btn.label.set_color('white')
        btn.label.set_fontweight('bold')
    
    # ═══════════════════════════════════════════════════════════════
    # Info panel
    # ═══════════════════════════════════════════════════════════════
    info_ax = fig.add_axes([0.72, 0.02, 0.26, 0.18], facecolor='#2c3e50')
    info_ax.set_xticks([])
    info_ax.set_yticks([])
    
    info_text = info_ax.text(0.5, 0.5, '', transform=info_ax.transAxes, 
                             fontsize=9, color='white', ha='center', va='center',
                             family='monospace')
    
    def update_info():
        info_str = (f"╔══════════════════════╗\n"
                   f"║   VIEW CONTROLS      ║\n"
                   f"╠══════════════════════╣\n"
                   f"║ Elev: {current_view['elev']:+4d}°          ║\n"
                   f"║ Azim: {current_view['azim']:+4d}°          ║\n"
                   f"║ Zoom: {current_view['dist']:4d}           ║\n"
                   f"╠══════════════════════╣\n"
                   f"║ Zeros: {len(zeros_t):2d}            ║\n"
                   f"╚══════════════════════╝")
        info_text.set_text(info_str)
    
    update_info()
    
    # ═══════════════════════════════════════════════════════════════
    # Title and instructions
    # ═══════════════════════════════════════════════════════════════
    fig.suptitle(r'Riemann Zeta Function $|\zeta(s)|$ — Interactive 3D', 
                fontsize=16, fontweight='bold', y=0.97)
    
    # Instructions at bottom
    fig.text(0.35, 0.02, '⬆⬇⬅➡ Rotate  |  +/- Zoom  |  R Reset  |  1-6 Presets  |  Q Quit', 
            ha='center', fontsize=10, color='#555',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    if save:
        plt.savefig(save, dpi=200, bbox_inches='tight', facecolor='#f5f5f5')
        click.echo(click.style(f"✅ Saved to: {save}", fg='green'))
    else:
        click.echo(click.style("\n✅ Interactive 3D view ready!", fg='green'))
        click.echo("-" * 60)
        click.echo("🎮 CONTROLS:")
        click.echo("   Arrow Keys  →  Rotate view (smooth!)")
        click.echo("   +/-         →  Zoom in/out")
        click.echo("   R           →  Reset to default")
        click.echo("   1-6         →  Preset camera angles")
        click.echo("   Q / Escape  →  Close window")
        click.echo("   Mouse       →  Fine control (drag to rotate)")
        click.echo("-" * 60)
        plt.show()
    
    # Summary
    click.echo("\n" + "═" * 60)
    click.echo("📊 SUMMARY")
    click.echo("═" * 60)
    click.echo(f"  Zeros found:     {len(zeros_t)}")
    click.echo(f"  Range explored:  t ∈ [0, {t_max}]")
    click.echo(f"  Critical line:   σ = 0.5 (Re(s) = 1/2)")
    click.echo("")
    click.echo("  🔴 Red dots = zeros of ζ(s)")
    click.echo("  All zeros lie exactly on σ = 0.5")
    click.echo("  This is the Riemann Hypothesis in action!")
    click.echo("═" * 60)


# ═══════════════════════════════════════════════════════════════════════════════
# 9. INFO / STATUS
# ═══════════════════════════════════════════════════════════════════════════════

@cli.command()
def info():
    """Display AIMATH capabilities and module status."""
    click.echo(click.style("\n" + "=" * 60, fg='cyan'))
    click.echo(click.style("  AIMATH: Research-Grade Neuro-Symbolic Math Engine", fg='cyan', bold=True))
    click.echo(click.style("=" * 60, fg='cyan'))
    
    click.echo("\n📦 Module Status:")
    
    modules = [
        ('aimath.core.engine', 'Core Engine', 'Symbolic computation'),
        ('aimath.solvers.hybrid_integrator', 'Hybrid Integrator', 'Database + Symbolic + Numeric'),
        ('aimath.solvers.contour_integration', 'Contour Integration', 'Residue Theorem'),
        ('aimath.solvers.pde_solver', 'PDE Solver', 'Heat, Wave, Transport'),
        ('aimath.solvers.conjecture_tester', 'Conjecture Tester', 'Fuzz verification'),
        ('aimath.solvers.constant_recognizer', 'Constant Recognizer', 'Inverse symbolic'),
        ('aimath.solvers.integral_database', 'Integral Database', 'Famous integrals'),
        ('aimath.solvers.feynman_trick', 'Feynman Trick', 'Differentiation under integral'),
    ]
    
    for module, name, desc in modules:
        try:
            __import__(module)
            status = click.style("✅", fg='green')
        except ImportError:
            status = click.style("❌", fg='red')
        click.echo(f"  {status} {name:<22} - {desc}")
    
    click.echo("\n🛠️  Commands:")
    commands = [
        ('wizard', 'Interactive problem builder'),
        ('solve', 'General math solver'),
        ('integrate', 'Hybrid integration'),
        ('contour', 'Contour integrals (Residue Theorem)'),
        ('pde', 'Partial differential equations'),
        ('verify', 'Fuzz test conjectures'),
        ('recognize', 'Decimal to exact form'),
        ('riemann', 'Riemann Zeta 2D plots (critical line)'),
        ('riemann3d', 'Riemann Zeta 3D landscape'),
        ('info', 'This help screen'),
    ]
    
    for cmd, desc in commands:
        click.echo(f"  {click.style(cmd, fg='yellow'):<15} - {desc}")
    
    click.echo(f"\n💡 Run 'python main.py <command> --help' for command details")
    click.echo(click.style("=" * 60 + "\n", fg='cyan'))


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    cli()
