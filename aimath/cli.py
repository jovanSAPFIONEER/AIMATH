#!/usr/bin/env python3
"""
AIMATH Command Line Interface

A simple, accessible CLI for mathematical verification and problem solving.

Usage:
    aimath solve "x^2 - 5x + 6 = 0"
    aimath verify "sqrt(2) is irrational"
    aimath prove "For all n, n + 0 = n"
    aimath explain "What is the quadratic formula?"
    aimath interactive
"""

import argparse
import sys
from typing import Optional


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for AIMATH CLI."""
    parser = argparse.ArgumentParser(
        prog="aimath",
        description="🧮 AIMATH - AI Math Verification & Discovery Tool",
        epilog="For more help: https://github.com/jovanSAPFIONEER/AIMATH",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--version", "-v",
        action="version",
        version="%(prog)s 1.0.0"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Solve command
    solve_parser = subparsers.add_parser(
        "solve",
        help="Solve a mathematical equation or problem",
        description="Solve equations with multi-path verification"
    )
    solve_parser.add_argument(
        "problem",
        type=str,
        help="The mathematical problem to solve (e.g., 'x^2 - 5x + 6 = 0')"
    )
    solve_parser.add_argument(
        "--explain", "-e",
        action="store_true",
        help="Include step-by-step explanation"
    )
    solve_parser.add_argument(
        "--verify", "-V",
        action="store_true",
        default=True,
        help="Verify the solution (default: True)"
    )
    
    # Verify command
    verify_parser = subparsers.add_parser(
        "verify",
        help="Verify a mathematical claim or statement",
        description="Check if a mathematical statement is true with proof"
    )
    verify_parser.add_argument(
        "claim",
        type=str,
        help="The claim to verify (e.g., 'sqrt(2) is irrational')"
    )
    
    # Prove command
    prove_parser = subparsers.add_parser(
        "prove",
        help="Construct a formal proof",
        description="Use the proof assistant to construct formal proofs"
    )
    prove_parser.add_argument(
        "statement",
        type=str,
        help="The statement to prove"
    )
    prove_parser.add_argument(
        "--tactic", "-t",
        choices=["direct", "contradiction", "induction", "cases"],
        default="direct",
        help="Proof tactic to use"
    )
    
    # Explain command
    explain_parser = subparsers.add_parser(
        "explain",
        help="Get an explanation of a mathematical concept",
        description="Get a quality-enforced explanation with no hand-waving"
    )
    explain_parser.add_argument(
        "topic",
        type=str,
        help="The topic to explain"
    )
    explain_parser.add_argument(
        "--level", "-l",
        choices=["beginner", "intermediate", "advanced"],
        default="intermediate",
        help="Explanation level"
    )
    
    # Interactive mode
    interactive_parser = subparsers.add_parser(
        "interactive",
        help="Start interactive mode",
        description="Launch an interactive session for exploring math"
    )
    
    # Quick examples
    examples_parser = subparsers.add_parser(
        "examples",
        help="Show usage examples",
        description="Display example commands and use cases"
    )
    
    return parser


def cmd_solve(args) -> int:
    """Handle the solve command."""
    print(f"\n🔍 Solving: {args.problem}")
    print("-" * 50)
    
    try:
        from .core.engine import MathEngine
        engine = MathEngine()
        result = engine.solve(args.problem)
        
        print(f"\n✅ Solution(s): {result.solutions}")
        print(f"📊 Confidence: {result.confidence.value}")
        print(f"🔬 Method: {result.method}")
        
        if result.verified:
            print("✓ Solution verified by substitution")
        
        if args.explain and result.explanation:
            print(f"\n📖 Explanation:\n{result.explanation}")
            
        return 0
        
    except ImportError:
        # Fallback to simple SymPy if full engine not available
        print("(Using simplified solver)")
        try:
            import sympy as sp
            from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
            
            # Parse the problem
            problem = args.problem.replace("^", "**")
            if "=" in problem:
                lhs, rhs = problem.split("=")
                lhs = lhs.strip()
                rhs = rhs.strip()
                
                transformations = standard_transformations + (implicit_multiplication_application,)
                x = sp.Symbol('x')
                
                try:
                    lhs_expr = parse_expr(lhs, local_dict={'x': x}, transformations=transformations)
                    rhs_expr = parse_expr(rhs, local_dict={'x': x}, transformations=transformations)
                    equation = sp.Eq(lhs_expr, rhs_expr)
                    solutions = sp.solve(equation, x)
                    
                    print(f"\n✅ Solution(s): {solutions}")
                    
                    # Verify by substitution
                    print("\n🔬 Verification:")
                    for sol in solutions:
                        check = lhs_expr.subs(x, sol) - rhs_expr.subs(x, sol)
                        if sp.simplify(check) == 0:
                            print(f"  ✓ x = {sol} verified (substitution gives 0 = 0)")
                    
                    return 0
                except Exception as e:
                    print(f"❌ Could not parse: {e}")
                    return 1
            else:
                print("Please provide an equation with '='")
                return 1
                
        except ImportError:
            print("❌ SymPy not installed. Run: pip install sympy")
            return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


def cmd_verify(args) -> int:
    """Handle the verify command."""
    print(f"\n🔍 Verifying: {args.claim}")
    print("-" * 50)
    
    try:
        from .core.engine import MathEngine
        engine = MathEngine()
        result = engine.verify_claim(args.claim)
        
        if result.is_valid:
            print(f"\n✅ VERIFIED: The claim is TRUE")
        else:
            print(f"\n❌ NOT VERIFIED: The claim could not be proven")
            
        if result.proof:
            print(f"\n📜 Proof:\n{result.proof}")
            
        return 0
        
    except ImportError:
        print("(Verification engine not fully loaded)")
        print("Try: pip install aimath[all]")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


def cmd_prove(args) -> int:
    """Handle the prove command."""
    print(f"\n📜 Proving: {args.statement}")
    print(f"   Tactic: {args.tactic}")
    print("-" * 50)
    
    try:
        from .proof_assistant import ProofAssistant, Proposition
        
        prover = ProofAssistant()
        theorem = prover.state_theorem(
            name="user_theorem",
            statement=args.statement,
            description=f"User requested proof of: {args.statement}"
        )
        
        print(f"\n📋 Theorem stated: {theorem.name}")
        print(f"   Status: {theorem.status}")
        print(f"\nUse the Python API for full proof construction.")
        print("\nExample:")
        print("  from aimath import ProofAssistant, Proposition")
        print("  prover = ProofAssistant()")
        print(f'  theorem = prover.state_theorem("my_theorem", "{args.statement}")')
        
        return 0
        
    except ImportError as e:
        print(f"(Proof assistant not fully loaded: {e})")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


def cmd_explain(args) -> int:
    """Handle the explain command."""
    print(f"\n📖 Explaining: {args.topic}")
    print(f"   Level: {args.level}")
    print("-" * 50)
    
    # Built-in explanations for common topics
    explanations = {
        "quadratic formula": """
The Quadratic Formula
=====================

For any quadratic equation ax² + bx + c = 0 where a ≠ 0:

         -b ± √(b² - 4ac)
    x = ─────────────────
              2a

WHY does this work?
-------------------
1. Start with ax² + bx + c = 0
2. Divide by a: x² + (b/a)x + (c/a) = 0
3. Complete the square: (x + b/2a)² = (b² - 4ac)/4a²
4. Take square root: x + b/2a = ±√(b² - 4ac)/2a
5. Solve for x: x = (-b ± √(b² - 4ac))/2a

The discriminant (b² - 4ac) tells us:
• > 0: Two distinct real solutions
• = 0: One repeated real solution  
• < 0: Two complex conjugate solutions

Example:
--------
Solve x² - 5x + 6 = 0
• a=1, b=-5, c=6
• x = (5 ± √(25-24))/2 = (5 ± 1)/2
• x = 3 or x = 2 ✓
""",
        "pythagorean theorem": """
The Pythagorean Theorem
=======================

For a right triangle with legs a and b, and hypotenuse c:

    a² + b² = c²

WHY does this work?
-------------------
Visual Proof (Rearrangement):
1. Create a large square with side (a+b)
2. Arrange four copies of the right triangle inside
3. The remaining area can be shown to equal c²
4. By counting: (a+b)² - 4(½ab) = a² + b² = c²

Example:
--------
• If a = 3 and b = 4, then c² = 9 + 16 = 25, so c = 5
• The 3-4-5 triangle is a classic right triangle

Limitations:
------------
• Only works for RIGHT triangles (90° angle)
• For other triangles, use the Law of Cosines: c² = a² + b² - 2ab·cos(C)
""",
        "derivative": """
The Derivative
==============

The derivative f'(x) measures the instantaneous rate of change of f(x).

Definition:
                  f(x + h) - f(x)
    f'(x) = lim  ─────────────────
           h→0          h

WHY this definition?
--------------------
1. (f(x+h) - f(x))/h is the average rate of change over interval h
2. As h → 0, this approaches the instantaneous rate
3. Geometrically: slope of tangent line at point x

Common Derivatives:
-------------------
• d/dx[xⁿ] = n·xⁿ⁻¹  (Power Rule)
• d/dx[eˣ] = eˣ
• d/dx[ln(x)] = 1/x
• d/dx[sin(x)] = cos(x)
• d/dx[cos(x)] = -sin(x)

Example:
--------
f(x) = x³
f'(x) = 3x²

At x = 2: f'(2) = 3(4) = 12
This means the function is increasing at rate 12 units per unit x at that point.
""",
    }
    
    topic_lower = args.topic.lower()
    
    # Check for matching explanation
    for key, explanation in explanations.items():
        if key in topic_lower or topic_lower in key:
            print(explanation)
            return 0
    
    # Default response
    print(f"""
No built-in explanation for "{args.topic}" yet.

Try these topics:
• "quadratic formula"
• "pythagorean theorem"  
• "derivative"

Or use the Python API for custom explanations:

    from aimath import ExplanationEngine
    engine = ExplanationEngine()
    explanation = engine.explain("{args.topic}", level="{args.level}")
""")
    return 0


def cmd_interactive(args) -> int:
    """Handle interactive mode."""
    print("""
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   🧮  AIMATH Interactive Mode                                 ║
║   ─────────────────────────────                               ║
║   AI Math Verification & Discovery Tool                       ║
║                                                               ║
║   Commands:                                                   ║
║     solve <equation>     - Solve an equation                  ║
║     verify <claim>       - Verify a mathematical claim        ║
║     explain <topic>      - Get an explanation                 ║
║     help                 - Show this help                     ║
║     quit / exit          - Exit interactive mode              ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
""")
    
    while True:
        try:
            user_input = input("\n🧮 aimath> ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ("quit", "exit", "q"):
                print("\n👋 Goodbye! Keep exploring math!")
                break
                
            if user_input.lower() == "help":
                print("""
Commands:
  solve <equation>     Solve an equation (e.g., solve x^2 - 4 = 0)
  verify <claim>       Verify a claim (e.g., verify sqrt(2) is irrational)
  explain <topic>      Explain a topic (e.g., explain quadratic formula)
  help                 Show this help
  quit                 Exit
""")
                continue
            
            parts = user_input.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""
            
            if cmd == "solve" and arg:
                class Args:
                    problem = arg
                    explain = True
                    verify = True
                cmd_solve(Args())
                
            elif cmd == "verify" and arg:
                class Args:
                    claim = arg
                cmd_verify(Args())
                
            elif cmd == "explain" and arg:
                class Args:
                    topic = arg
                    level = "intermediate"
                cmd_explain(Args())
                
            else:
                print(f"Unknown command: {user_input}")
                print("Type 'help' for available commands")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except EOFError:
            break
            
    return 0


def cmd_examples(args) -> int:
    """Show usage examples."""
    print("""
╔═══════════════════════════════════════════════════════════════╗
║                    AIMATH Usage Examples                      ║
╚═══════════════════════════════════════════════════════════════╝

📐 SOLVING EQUATIONS
────────────────────
  aimath solve "x^2 - 5x + 6 = 0"
  aimath solve "2x + 3 = 7" --explain
  aimath solve "sin(x) = 0.5"

✓ VERIFYING CLAIMS  
───────────────────
  aimath verify "sqrt(2) is irrational"
  aimath verify "the sum of angles in a triangle is 180 degrees"
  aimath verify "e^(i*pi) + 1 = 0"

📜 CONSTRUCTING PROOFS
──────────────────────
  aimath prove "For all n, n + 0 = n" --tactic direct
  aimath prove "There exists no largest prime" --tactic contradiction
  aimath prove "Sum of first n integers is n(n+1)/2" --tactic induction

📖 GETTING EXPLANATIONS
───────────────────────
  aimath explain "quadratic formula"
  aimath explain "pythagorean theorem" --level beginner
  aimath explain "derivative" --level advanced

🖥️ INTERACTIVE MODE
───────────────────
  aimath interactive

📦 PYTHON API
─────────────
  from aimath import MathEngine, ProofAssistant
  
  # Solve equations
  engine = MathEngine()
  result = engine.solve("x^2 - 4 = 0")
  print(result.solutions)  # [-2, 2]
  
  # Formal proofs
  prover = ProofAssistant()
  theorem = prover.state_theorem("my_theorem", "P → Q")

For more: https://github.com/jovanSAPFIONEER/AIMATH
""")
    return 0


def main(argv: Optional[list] = None) -> int:
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args(argv)
    
    if args.command is None:
        # No command given, show welcome and help
        print("""
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   🧮  AIMATH - AI Math Verification & Discovery Tool          ║
║                                                               ║
║   Rigorous mathematical verification for everyone             ║
║   From amateurs to professionals                              ║
║                                                               ║
║   • Multi-path verification (never trust, always verify)      ║
║   • Anti-hallucination protection                             ║
║   • Quality-enforced explanations                             ║
║   • Formal proof assistant                                    ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝

Quick Start:
  aimath solve "x^2 - 5x + 6 = 0"     Solve an equation
  aimath explain "quadratic formula"  Get an explanation
  aimath interactive                  Start interactive mode
  aimath examples                     See more examples

For help: aimath --help
""")
        return 0
    
    # Dispatch to command handlers
    handlers = {
        "solve": cmd_solve,
        "verify": cmd_verify,
        "prove": cmd_prove,
        "explain": cmd_explain,
        "interactive": cmd_interactive,
        "examples": cmd_examples,
    }
    
    handler = handlers.get(args.command)
    if handler:
        return handler(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
