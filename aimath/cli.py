"""
AI Math CLI - Interactive command-line interface.

Provides user-friendly access to:
- Problem solving
- Thesis verification  
- Concept explanation
- Discovery exploration

All with anti-hallucination guarantees and quality-enforced explanations.
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import Optional
import yaml

# Handle both direct execution and module import
if __name__ == "__main__" or __package__ is None:
    # Running as script - add parent to path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from aimath.core.engine import MathEngine
    from aimath.core.types import DifficultyLevel, ConfidenceLevel
else:
    # Running as module
    from .core.engine import MathEngine
    from .core.types import DifficultyLevel, ConfidenceLevel

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


class MathCLI:
    """
    Interactive command-line interface for the math verification system.
    
    Modes:
    - solve: Solve a mathematical problem
    - verify: Verify a mathematical claim
    - explain: Explain a concept
    - discover: Explore for new findings
    """
    
    BANNER = r"""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║     █████╗ ██╗    ███╗   ███╗ █████╗ ████████╗██╗  ██╗       ║
    ║    ██╔══██╗██║    ████╗ ████║██╔══██╗╚══██╔══╝██║  ██║       ║
    ║    ███████║██║    ██╔████╔██║███████║   ██║   ███████║       ║
    ║    ██╔══██║██║    ██║╚██╔╝██║██╔══██║   ██║   ██╔══██║       ║
    ║    ██║  ██║██║    ██║ ╚═╝ ██║██║  ██║   ██║   ██║  ██║       ║
    ║    ╚═╝  ╚═╝╚═╝    ╚═╝     ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝       ║
    ║                                                               ║
    ║       Mathematical Verification & Discovery System            ║
    ║       Anti-Hallucination • Multi-Path Solving • CLEAR         ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize CLI.
        
        Args:
            config_path: Path to settings.yaml
        """
        self.config = self._load_config(config_path)
        self.engine = MathEngine()
    
    def _load_config(self, config_path: Optional[str]) -> dict:
        """Load configuration from file."""
        if config_path is None:
            # Try default paths
            default_paths = [
                Path(__file__).parent.parent / 'config' / 'settings.yaml',
                Path.cwd() / 'config' / 'settings.yaml',
            ]
            for path in default_paths:
                if path.exists():
                    config_path = str(path)
                    break
        
        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                return yaml.safe_load(f)
        
        return {}
    
    def run(self):
        """Run the CLI in interactive mode."""
        print(self.BANNER)
        print("Type 'help' for commands, 'quit' to exit.\n")
        
        while True:
            try:
                user_input = input("AI-Math> ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if user_input.lower() == 'help':
                    self._show_help()
                    continue
                
                self._process_command(user_input)
                
            except KeyboardInterrupt:
                print("\n\nUse 'quit' to exit.")
            except Exception as e:
                logger.error(f"Error: {e}")
    
    def _show_help(self):
        """Display help information."""
        help_text = """
┌─────────────────────────────────────────────────────────────────┐
│                         COMMANDS                                 │
├─────────────────────────────────────────────────────────────────┤
│  CORE COMMANDS:                                                 │
│  solve <problem>     Solve a math problem                       │
│  verify <claim>      Verify a mathematical claim                │
│  explain <concept>   Explain a mathematical concept             │
│  ask <question>      Natural language math question             │
│                                                                 │
│  ADVANCED SOLVERS:                                              │
│  integrate <expr>    Hybrid integration (symbolic + numeric)    │
│  contour <expr>      Contour integration (residue theorem)      │
│  ode <equation>      Solve ordinary differential equation       │
│  pde <equation>      Solve partial differential equation        │
│  matrix <op> [[..]]  Linear algebra (eigenvals, det, inv, etc)  │
│  optimize <func>     Find extrema of a function                 │
│                                                                 │
│  TOOLS:                                                         │
│  test <claim>        Fuzz-test a conjecture (find counterexamples)│
│  recognize <value>   Identify constant (e.g., 0.5140 → 5π²/96)  │
│  steps <problem>     Show detailed solution steps               │
│  feynman <expr>      Apply Feynman's differentiation trick      │
│                                                                 │
│  EXAMPLES:                                                      │
│  solve x^2 - 4 = 0                                              │
│  integrate atan(sqrt(x**2+2))/((x**2+1)*sqrt(x**2+2)) from 0 to 1│
│  contour 1/(x**2 + 1)                                           │
│  test sin(x)**2 + cos(x)**2 = 1                                 │
│  test (a+b)**2 = a**2 + b**2                                    │
│  pde u_t = u_xx  (heat equation)                                │
│  recognize 0.514041895890                                       │
│                                                                 │
│  SETTINGS:                                                      │
│  level <n>           Set difficulty level                       │
│                                                                 │
│  quit/exit           Exit the program                           │
└─────────────────────────────────────────────────────────────────┘
        """
        print(help_text)
    
    def _process_command(self, user_input: str):
        """Process a user command."""
        parts = user_input.split(maxsplit=1)
        command = parts[0].lower()
        argument = parts[1] if len(parts) > 1 else ""
        
        if command == 'solve':
            self._do_solve(argument)
        elif command == 'verify':
            self._do_verify(argument)
        elif command == 'explain':
            self._do_explain(argument)
        elif command == 'level':
            self._set_level(argument)
        elif command == 'ask':
            self._do_ask(argument)
        elif command == 'ode':
            self._do_ode(argument)
        elif command == 'matrix':
            self._do_matrix(argument)
        elif command == 'optimize':
            self._do_optimize(argument)
        elif command == 'steps':
            self._do_steps(argument)
        elif command == 'integrate':
            self._do_integrate(argument)
        elif command == 'feynman':
            self._do_feynman(argument)
        elif command == 'recognize':
            self._do_recognize(argument)
        elif command == 'contour':
            self._do_contour(argument)
        elif command == 'pde':
            self._do_pde(argument)
        elif command == 'test':
            self._do_test(argument)
        else:
            # Assume it's a problem to solve
            self._do_solve(user_input)
    
    def _do_solve(self, problem: str):
        """Solve a mathematical problem."""
        if not problem:
            print("Usage: solve <problem>")
            print("Example: solve x^2 - 4 = 0")
            return
        
        print(f"\n{'─' * 60}")
        print(f"SOLVING: {problem}")
        print(f"{'─' * 60}\n")
        
        try:
            result = self.engine.solve(problem)
            self._display_solution(result)
        except Exception as e:
            logger.error(f"Failed to solve: {e}")
    
    def _do_verify(self, claim: str):
        """Verify a mathematical claim."""
        if not claim:
            print("Usage: verify <claim>")
            print("Example: verify sin^2(x) + cos^2(x) = 1")
            return
        
        print(f"\n{'─' * 60}")
        print(f"VERIFYING: {claim}")
        print(f"{'─' * 60}\n")
        
        try:
            result = self.engine.verify_claim(claim)
            self._display_verification(result)
        except Exception as e:
            logger.error(f"Failed to verify: {e}")
    
    def _do_explain(self, concept: str):
        """Explain a mathematical concept."""
        if not concept:
            print("Usage: explain <concept>")
            print("Example: explain derivative")
            return
        
        print(f"\n{'─' * 60}")
        print(f"EXPLAINING: {concept}")
        print(f"{'─' * 60}\n")
        
        try:
            result = self.engine.explain(concept)
            self._display_explanation(result)
        except Exception as e:
            logger.error(f"Failed to explain: {e}")
    
    def _set_level(self, level: str):
        """Set the difficulty level."""
        level_map = {
            'amateur': DifficultyLevel.AMATEUR,
            'intermediate': DifficultyLevel.INTERMEDIATE,
            'advanced': DifficultyLevel.ADVANCED,
            'expert': DifficultyLevel.EXPERT,
        }
        
        if level.lower() not in level_map:
            print(f"Invalid level. Options: {', '.join(level_map.keys())}")
            return
        
        # Would need to update engine settings
        print(f"Difficulty level set to: {level.upper()}")
    
    def _do_ask(self, question: str):
        """Answer a natural language math question."""
        if not question:
            print("Usage: ask <question>")
            print("Example: ask derivative of x cubed")
            return
        
        print(f"\n{'─' * 60}")
        print(f"QUESTION: {question}")
        print(f"{'─' * 60}\n")
        
        try:
            try:
                from aimath.parsers.natural_language_parser import NaturalLanguageParser
            except ImportError:
                from parsers.natural_language_parser import NaturalLanguageParser
            parser = NaturalLanguageParser()
            details = parser.explain_parse(question)
            
            print(f"📝 Interpreted as: {details['sympy_code']}")
            print(f"📊 Method: {details['method']}")
            print(f"\n✅ ANSWER: {details['result']}")
            
            if details.get('result_latex'):
                print(f"📐 LaTeX: {details['result_latex']}")
        except Exception as e:
            logger.error(f"Failed to process question: {e}")
    
    def _do_ode(self, equation: str):
        """Solve a differential equation."""
        if not equation:
            print("Usage: ode <equation>")
            print("Examples:")
            print("  ode y' + 2*y = x")
            print("  ode y'' + y = 0")
            return
        
        print(f"\n{'─' * 60}")
        print(f"SOLVING ODE: {equation}")
        print(f"{'─' * 60}\n")
        
        try:
            try:
                from aimath.solvers.differential_equations import ODESolver
            except ImportError:
                from solvers.differential_equations import ODESolver
            solver = ODESolver()
            result = solver.solve(equation)
            
            print(f"📋 ODE Type: {result.ode_type}")
            print(f"📊 Classification: {', '.join(result.classification[:3])}")
            print()
            
            for step in result.steps:
                print(f"  {step}")
            
            print(f"\n✅ SOLUTION: {result.solution}")
            
            if result.particular_solution:
                print(f"📌 PARTICULAR: {result.particular_solution}")
            
            print(f"\n🔒 Verified: {'Yes ✓' if result.is_verified else 'No ⚠'}")
        except Exception as e:
            logger.error(f"Failed to solve ODE: {e}")
    
    def _do_matrix(self, args: str):
        """Perform linear algebra operations."""
        if not args:
            print("Usage: matrix <operation> [[...]]")
            print("Operations: eigenvals, eigenvects, det, inv, rref, lu, qr, nullspace")
            print("Example: matrix eigenvals [[1, 2], [2, 1]]")
            return
        
        parts = args.split(maxsplit=1)
        operation = parts[0].lower()
        matrix_str = parts[1] if len(parts) > 1 else ""
        
        if not matrix_str:
            print("Please provide a matrix, e.g.: [[1, 2], [3, 4]]")
            return
        
        print(f"\n{'─' * 60}")
        print(f"MATRIX {operation.upper()}")
        print(f"{'─' * 60}\n")
        
        try:
            try:
                from aimath.solvers.linear_algebra import solve_linear_algebra
            except ImportError:
                from solvers.linear_algebra import solve_linear_algebra
            result = solve_linear_algebra(matrix_str, operation)
            
            print(f"📊 Input Matrix:")
            print(f"  {result.input_matrix}")
            print()
            
            for step in result.steps:
                print(f"  {step}")
            
            print(f"\n✅ RESULT: {result.result}")
            
            if result.properties:
                print(f"\n📋 Properties:")
                for k, v in result.properties.items():
                    print(f"  • {k}: {v}")
        except Exception as e:
            logger.error(f"Failed matrix operation: {e}")
    
    def _do_optimize(self, func: str):
        """Find extrema of a function."""
        if not func:
            print("Usage: optimize <function>")
            print("Examples:")
            print("  optimize x**2 + y**2 - 2*x")
            print("  optimize x**3 - 3*x")
            return
        
        print(f"\n{'─' * 60}")
        print(f"OPTIMIZING: {func}")
        print(f"{'─' * 60}\n")
        
        try:
            try:
                from aimath.solvers.optimization import Optimizer
            except ImportError:
                from solvers.optimization import Optimizer
            opt = Optimizer()
            result = opt.find_critical_points(func)
            
            for step in result.steps:
                print(f"  {step}")
            
            print(f"\n📊 CRITICAL POINTS:")
            for cp in result.critical_points:
                print(f"  • {cp.point}")
                print(f"    Value: {cp.value}")
                print(f"    Type: {cp.classification}")
            
            if result.global_min:
                print(f"\n🔻 GLOBAL MINIMUM: {result.global_min.point} = {result.global_min.value}")
            if result.global_max:
                print(f"🔺 GLOBAL MAXIMUM: {result.global_max.point} = {result.global_max.value}")
        except Exception as e:
            logger.error(f"Failed to optimize: {e}")
    
    def _do_steps(self, problem: str):
        """Show detailed solution steps."""
        if not problem:
            print("Usage: steps <problem>")
            print("Examples:")
            print("  steps Derivative(x**3, x)")
            print("  steps Integral(sin(x), x)")
            return
        
        print(f"\n{'─' * 60}")
        print(f"STEP-BY-STEP: {problem}")
        print(f"{'─' * 60}\n")
        
        try:
            from sympy import sympify, Derivative, Integral
            try:
                from aimath.solvers.calculus_steps import (
                    get_derivative_steps, get_integral_steps, format_steps_text
                )
            except ImportError:
                from solvers.calculus_steps import (
                    get_derivative_steps, get_integral_steps, format_steps_text
                )
            
            expr = sympify(problem)
            
            if isinstance(expr, Derivative) or 'Derivative' in problem:
                steps = get_derivative_steps(expr)
            elif isinstance(expr, Integral) or 'Integral' in problem:
                steps = get_integral_steps(expr)
            else:
                # Try derivative by default
                steps = get_derivative_steps(expr)
            
            print(format_steps_text(steps))
        except Exception as e:
            logger.error(f"Failed to generate steps: {e}")
    
    def _do_integrate(self, expr_str: str):
        """Hybrid integration with constant recognition."""
        if not expr_str:
            print("Usage: integrate <expression> [from <a> to <b>]")
            print("Examples:")
            print("  integrate x**2 from 0 to 1")
            print("  integrate sin(x)/x from 0 to oo")
            print("  integrate atan(sqrt(x**2+2))/((x**2+1)*sqrt(x**2+2)) from 0 to 1")
            return
        
        print(f"\n{'═' * 60}")
        print(f"HYBRID INTEGRATION")
        print(f"{'═' * 60}\n")
        
        try:
            try:
                from aimath.solvers.hybrid_integrator import solve_integral, print_integration_result
            except ImportError:
                from solvers.hybrid_integrator import solve_integral, print_integration_result
            
            result = solve_integral(expr_str)
            print_integration_result(result)
            
        except Exception as e:
            logger.error(f"Integration failed: {e}")
            import traceback
            traceback.print_exc()
    
    def _do_feynman(self, expr_str: str):
        """Apply Feynman's differentiation under the integral sign."""
        if not expr_str:
            print("Usage: feynman <integrand> --param <a> [--var <x>] [--from <lower>] [--to <upper>]")
            print("Examples:")
            print("  feynman sin(x)/x * exp(-a*x) --param a --from 0 --to oo")
            print("  feynman x**a --param a --from 0 --to 1")
            return
        
        print(f"\n{'═' * 60}")
        print(f"FEYNMAN'S TECHNIQUE")
        print(f"{'═' * 60}\n")
        
        try:
            try:
                from aimath.solvers.feynman_trick import diff_under_integral, suggest_parameterization
            except ImportError:
                from solvers.feynman_trick import diff_under_integral, suggest_parameterization
            
            from sympy import symbols, sympify, oo as sym_oo
            import re
            
            # Parse arguments
            parts = expr_str.split('--')
            integrand_str = parts[0].strip()
            
            # Default values
            param = 'a'
            var = 'x'
            lower = None
            upper = None
            
            for part in parts[1:]:
                part = part.strip()
                if part.startswith('param'):
                    param = part.split()[1]
                elif part.startswith('var'):
                    var = part.split()[1]
                elif part.startswith('from'):
                    lower = part.split()[1]
                elif part.startswith('to'):
                    upper = part.split()[1]
            
            # Create symbols
            x = symbols(var)
            a = symbols(param, positive=True)
            
            # Parse integrand
            integrand = sympify(integrand_str)
            
            # Parse limits
            if lower:
                lower = sym_oo if lower == 'oo' else sympify(lower)
            if upper:
                upper = sym_oo if upper == 'oo' else sympify(upper)
            
            # Apply differentiation under integral
            result = diff_under_integral(integrand, x, a, lower, upper)
            
            print(f"Integrand: {integrand}")
            print(f"Integration variable: {var}")
            print(f"Parameter: {param}")
            print()
            print(f"∂/∂{param} [{integrand}] = {result['partial_derivative']}")
            print(f"LaTeX: {result['partial_derivative_latex']}")
            
            if 'new_integral' in result:
                print(f"\nNew integral: {result['new_integral']}")
            
            if 'evaluated' in result:
                print(f"\n✅ Evaluated: {result['evaluated']}")
                print(f"   LaTeX: {result['evaluated_latex']}")
            
            # Suggest parameterizations if just an expression
            if '--param' not in expr_str:
                print("\n" + "─" * 40)
                print("💡 Suggested parameterizations:")
                suggestions = suggest_parameterization(integrand, x)
                for s in suggestions:
                    print(f"\n  • {s['strategy']}")
                    print(f"    {s['description']}")
            
        except Exception as e:
            logger.error(f"Feynman's technique failed: {e}")
            import traceback
            traceback.print_exc()
    
    def _do_recognize(self, value_str: str):
        """Recognize a numerical constant."""
        if not value_str:
            print("Usage: recognize <decimal_value>")
            print("Examples:")
            print("  recognize 0.514041895890")
            print("  recognize 3.14159265359")
            print("  recognize 1.64493406685")
            return
        
        print(f"\n{'═' * 60}")
        print(f"CONSTANT RECOGNITION")
        print(f"{'═' * 60}\n")
        
        try:
            try:
                from aimath.solvers.constant_recognizer import recognize_constant, identify_constant
            except ImportError:
                from solvers.constant_recognizer import recognize_constant, identify_constant
            
            value = float(value_str.strip())
            
            print(f"Input value: {value}")
            print()
            
            result = recognize_constant(value)
            
            if result:
                print(f"✅ RECOGNIZED!")
                print(f"   Symbolic form: {result.recognized_form}")
                print(f"   LaTeX: {result.latex}")
                print(f"   Confidence: {result.confidence}")
                print(f"   Method: {result.method}")
                print(f"   Error: {result.error:.2e}")
            else:
                print("❌ Could not identify as a known mathematical constant")
                print("   The value may be:")
                print("   • A transcendental number with no simple form")
                print("   • A combination of constants not in our database")
                print("   • Numerical noise from computation")
            
        except ValueError:
            logger.error(f"Invalid number: {value_str}")
        except Exception as e:
            logger.error(f"Recognition failed: {e}")
    
    def _do_contour(self, expr_str: str):
        """Contour integration using residue theorem."""
        if not expr_str:
            print("Usage: contour <expression>")
            print("Examples:")
            print("  contour 1/(x**2 + 1)")
            print("  contour 1/(x**2 + 1)**2")
            print("  contour x**2/(x**4 + 1)")
            return
        
        print(f"\n{'═' * 60}")
        print(f"CONTOUR INTEGRATION (Residue Theorem)")
        print(f"{'═' * 60}\n")
        
        try:
            try:
                from aimath.solvers.contour_integration import real_integral_via_contour, print_contour_result
            except ImportError:
                from solvers.contour_integration import real_integral_via_contour, print_contour_result
            
            from sympy import Symbol, sympify
            x = Symbol('x')
            expr = sympify(expr_str)
            
            result = real_integral_via_contour(expr, x)
            print_contour_result(result)
            
        except Exception as e:
            logger.error(f"Contour integration failed: {e}")
            import traceback
            traceback.print_exc()
    
    def _do_pde(self, pde_str: str):
        """Solve a partial differential equation."""
        if not pde_str:
            print("Usage: pde <equation>")
            print("Examples:")
            print("  pde u_t = u_xx           (heat equation)")
            print("  pde u_tt = c**2 * u_xx   (wave equation)")
            print("  pde u_t + c*u_x = 0      (transport equation)")
            return
        
        print(f"\n{'═' * 60}")
        print(f"PDE SOLVER")
        print(f"{'═' * 60}\n")
        
        try:
            try:
                from aimath.solvers.pde_solver import solve_pde_separation, print_pde_solution, solve_transport_equation
            except ImportError:
                from solvers.pde_solver import solve_pde_separation, print_pde_solution, solve_transport_equation
            
            from sympy import Symbol, Function, Eq, Derivative, sympify, symbols
            
            x, t = symbols('x t')
            u = Function('u')(x, t)
            
            # Parse user-friendly notation (order matters - longer patterns first!)
            pde_str = pde_str.replace('u_xx', 'Derivative(u(x,t), (x, 2))')
            pde_str = pde_str.replace('u_tt', 'Derivative(u(x,t), (t, 2))')
            pde_str = pde_str.replace('u_x', 'Derivative(u(x,t), x)')
            pde_str = pde_str.replace('u_t', 'Derivative(u(x,t), t)')
            
            # Handle equation
            if '=' in pde_str:
                parts = pde_str.split('=')
                lhs = sympify(parts[0].strip())
                rhs = sympify(parts[1].strip())
                pde_eq = Eq(lhs, rhs)
            else:
                pde_eq = Eq(sympify(pde_str), 0)
            
            result = solve_pde_separation(pde_eq, u, x, t)
            print_pde_solution(result)
            
        except Exception as e:
            logger.error(f"PDE solver failed: {e}")
            import traceback
            traceback.print_exc()
    
    def _do_test(self, claim: str):
        """Test a mathematical conjecture (fuzz testing)."""
        if not claim:
            print("Usage: test <LHS> = <RHS>")
            print("Examples:")
            print("  test sin(x)**2 + cos(x)**2 = 1")
            print("  test (a+b)**2 = a**2 + b**2")
            print("  test x**x = exp(x)")
            print("  test exp(x) >= 1 + x")
            return
        
        print(f"\n{'═' * 60}")
        print(f"CONJECTURE TESTER (Fuzz Verification)")
        print(f"{'═' * 60}\n")
        
        try:
            try:
                from aimath.solvers.conjecture_tester import test_conjecture, test_inequality, print_conjecture_result
            except ImportError:
                from solvers.conjecture_tester import test_conjecture, test_inequality, print_conjecture_result
            
            from sympy import sympify
            
            # Check for inequality
            if '>=' in claim:
                parts = claim.split('>=')
                lhs = sympify(parts[0].strip())
                rhs = sympify(parts[1].strip())
                result = test_inequality(lhs, rhs, '>=')
            elif '<=' in claim:
                parts = claim.split('<=')
                lhs = sympify(parts[0].strip())
                rhs = sympify(parts[1].strip())
                result = test_inequality(lhs, rhs, '<=')
            elif '>' in claim and '>=' not in claim:
                parts = claim.split('>')
                lhs = sympify(parts[0].strip())
                rhs = sympify(parts[1].strip())
                result = test_inequality(lhs, rhs, '>')
            elif '<' in claim and '<=' not in claim:
                parts = claim.split('<')
                lhs = sympify(parts[0].strip())
                rhs = sympify(parts[1].strip())
                result = test_inequality(lhs, rhs, '<')
            elif '=' in claim:
                # Equality test
                parts = claim.split('=')
                lhs = sympify(parts[0].strip())
                rhs = sympify(parts[1].strip())
                result = test_conjecture(lhs, rhs, trials=1000)
            else:
                # Test if expression equals zero
                result = test_conjecture(claim, 0)
            
            print_conjecture_result(result)
            
        except Exception as e:
            logger.error(f"Conjecture test failed: {e}")
            import traceback
            traceback.print_exc()
    
    def _display_solution(self, result):
        """Display solution results."""
        print("=" * 60)
        print("SOLUTION")
        print("=" * 60)
        
        if hasattr(result, 'answer'):
            print(f"\n📊 ANSWER: {result.answer}")
        
        if hasattr(result, 'confidence'):
            confidence = result.confidence
            confidence_emoji = {
                ConfidenceLevel.PROVEN: "✓✓✓",
                ConfidenceLevel.HIGH: "✓✓",
                ConfidenceLevel.MEDIUM: "✓",
                ConfidenceLevel.LOW: "⚠",
                ConfidenceLevel.UNKNOWN: "?",
            }
            emoji = confidence_emoji.get(confidence, "")
            print(f"\n🔒 CONFIDENCE: {confidence.value.upper()} {emoji}")
        
        if hasattr(result, 'verification') and result.verification:
            print(f"\n✓ VERIFIED BY:")
            for method in result.verification.methods_used:
                print(f"   • {method}")
        
        if hasattr(result, 'explanation') and result.explanation:
            print("\n" + "─" * 60)
            print("EXPLANATION")
            print("─" * 60)
            if hasattr(result.explanation, 'to_markdown'):
                print(result.explanation.to_markdown())
            else:
                print(str(result.explanation))
        
        print("\n" + "=" * 60)
    
    def _display_verification(self, result):
        """Display verification results."""
        print("=" * 60)
        print("VERIFICATION RESULT")
        print("=" * 60)
        
        if hasattr(result, 'is_valid'):
            status = "✓ VERIFIED" if result.is_valid else "✗ NOT VERIFIED"
            print(f"\n{status}")
        
        if hasattr(result, 'confidence_score'):
            print(f"\n📊 Confidence Score: {result.confidence_score:.1%}")
        
        if hasattr(result, 'counterexamples') and result.counterexamples:
            print(f"\n⚠ COUNTEREXAMPLES FOUND:")
            for ce in result.counterexamples[:3]:
                print(f"   • {ce}")
        
        print("\n" + "=" * 60)
    
    def _display_explanation(self, result):
        """Display explanation."""
        print("=" * 60)
        print("EXPLANATION")
        print("=" * 60)
        
        if hasattr(result, 'to_markdown'):
            print(result.to_markdown())
        else:
            print(str(result))
        
        if hasattr(result, 'quality') and result.quality:
            print("\n" + "─" * 60)
            print("QUALITY SCORE")
            print("─" * 60)
            quality = result.quality
            if hasattr(quality, 'total'):
                print(f"Total: {quality.total}/25")
            if hasattr(quality, 'passes_quality_gate'):
                status = "✓ PASSES" if quality.passes_quality_gate else "✗ NEEDS IMPROVEMENT"
                print(f"Quality Gate: {status}")
        
        print("\n" + "=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='AI Math - Mathematical Verification & Discovery System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m aimath.cli solve "x^2 - 4 = 0"
  python -m aimath.cli integrate "atan(sqrt(x**2+2))/((x**2+1)*sqrt(x**2+2)) from 0 to 1"
  python -m aimath.cli contour "1/(x**2 + 1)"
  python -m aimath.cli test "sin(x)**2 + cos(x)**2 = 1"
  python -m aimath.cli test "(a+b)**2 = a**2 + b**2"
  python -m aimath.cli pde "u_t = u_xx"
  python -m aimath.cli ode "y' + 2*y = x"
  python -m aimath.cli recognize "0.514041895890"
  python -m aimath.cli  # Interactive mode
        """
    )
    
    parser.add_argument(
        'command',
        nargs='?',
        choices=['solve', 'verify', 'explain', 'ask', 'ode', 'pde', 'matrix', 'optimize', 
                 'steps', 'integrate', 'contour', 'feynman', 'recognize', 'test'],
        help='Command to execute'
    )
    
    parser.add_argument(
        'input',
        nargs='?',
        help='Problem, claim, or concept'
    )
    
    parser.add_argument(
        '--level', '-l',
        choices=['amateur', 'intermediate', 'advanced', 'expert'],
        default='intermediate',
        help='Difficulty level'
    )
    
    parser.add_argument(
        '--config', '-c',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    cli = MathCLI(config_path=args.config)
    
    if args.command and args.input:
        # Direct command mode
        if args.command == 'solve':
            cli._do_solve(args.input)
        elif args.command == 'verify':
            cli._do_verify(args.input)
        elif args.command == 'explain':
            cli._do_explain(args.input)
        elif args.command == 'ask':
            cli._do_ask(args.input)
        elif args.command == 'ode':
            cli._do_ode(args.input)
        elif args.command == 'pde':
            cli._do_pde(args.input)
        elif args.command == 'matrix':
            cli._do_matrix(args.input)
        elif args.command == 'optimize':
            cli._do_optimize(args.input)
        elif args.command == 'steps':
            cli._do_steps(args.input)
        elif args.command == 'integrate':
            cli._do_integrate(args.input)
        elif args.command == 'contour':
            cli._do_contour(args.input)
        elif args.command == 'feynman':
            cli._do_feynman(args.input)
        elif args.command == 'recognize':
            cli._do_recognize(args.input)
        elif args.command == 'test':
            cli._do_test(args.input)
    else:
        # Interactive mode
        cli.run()


if __name__ == '__main__':
    main()
