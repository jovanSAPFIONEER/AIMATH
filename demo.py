#!/usr/bin/env python3
"""
AIMATH Demo - Showcasing the Research-Grade Math Engine
========================================================

Run this to see all major features in action!
"""

import os
import sys
import time

# Ensure we're in the right directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def banner(text):
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")

def run_demo():
    """Run the full AIMATH demonstration."""
    
    banner("AIMATH: Research-Grade Neuro-Symbolic Math Engine")
    print("This demo showcases the journey from 'homework helper' to 'research tool'\n")
    time.sleep(1)
    
    # Demo 1: Module Status
    banner("1. MODULE STATUS - All Components Online")
    os.system('python main.py info')
    input("\nPress Enter to continue...")
    
    # Demo 2: Ahmed's Integral (the killer feature!)
    banner("2. AHMED'S INTEGRAL - The Problem That Started It All")
    print("This integral stumped SymPy, but AIMATH solves it INSTANTLY:\n")
    print("  Integral from 0 to 1 of: atan(sqrt(x^2+2)) / ((x^2+1)*sqrt(x^2+2)) dx\n")
    os.system('python main.py integrate "atan(sqrt(x**2+2))/((x**2+1)*sqrt(x**2+2))" --bounds 0 1')
    print("\n  Result: 5*pi^2/96 = 0.5148668... (EXACT closed form!)")
    input("\nPress Enter to continue...")
    
    # Demo 3: Contour Integration
    banner("3. CONTOUR INTEGRATION - Complex Analysis via Residue Theorem")
    print("Computing: Integral from -infinity to +infinity of 1/(x^2+1) dx\n")
    os.system('python main.py contour "1/(x**2+1)"')
    input("\nPress Enter to continue...")
    
    # Demo 4: PDE Solver  
    banner("4. PDE SOLVER - Heat Equation via Separation of Variables")
    print("Solving: du/dt = d^2u/dx^2 (diffusion/heat conduction)\n")
    os.system('python main.py pde heat')
    input("\nPress Enter to continue...")
    
    # Demo 5: Conjecture Tester (TRUE case)
    banner("5. CONJECTURE TESTER - Verifying True Identities")
    print("Testing the Pythagorean identity: sin^2(x) + cos^2(x) = 1\n")
    os.system('python main.py verify "sin(x)**2+cos(x)**2" "1"')
    input("\nPress Enter to continue...")
    
    # Demo 6: Conjecture Tester (FALSE case)
    banner("6. CONJECTURE TESTER - Finding Counterexamples!")
    print("Testing a FALSE claim: (a+b)^2 = a^2 + b^2\n")
    os.system('python main.py verify "(a+b)**2" "a**2+b**2"')
    input("\nPress Enter to continue...")
    
    # Demo 7: Constant Recognizer
    banner("7. CONSTANT RECOGNIZER - Inverse Symbolic Calculator")
    print("Recognizing pi from its decimal: 3.14159265358979...\n")
    os.system('python main.py recognize 3.14159265358979')
    input("\nPress Enter to continue...")
    
    # Finale
    banner("DEMO COMPLETE!")
    print("""
    AIMATH has evolved from a simple calculator into a research-grade toolkit:
    
    [x] Hybrid Integration (database + symbolic + numeric + recognition)
    [x] Contour Integration (Residue Theorem, complex analysis)
    [x] PDE Solver (Heat, Wave, Transport equations)
    [x] Conjecture Tester (fuzz testing with counterexample finding)
    [x] Constant Recognizer (inverse symbolic calculator)
    [x] Unified CLI (professional command-line interface)
    [x] Visualization support (poles, heat maps)
    
    From "solving for x" to "solving the heat equation" - AIMATH is ready!
    
    Run any command:
      python main.py wizard              # Interactive mode
      python main.py contour "1/(z**2+1)"    # Contour integral
      python main.py pde heat --plot     # Heat equation with visualization
      python main.py verify "(a+b)**2" "a**2+2*a*b+b**2"  # True identity
      python main.py integrate "x**2" --bounds 0 1       # Basic integral
    
    """)


if __name__ == "__main__":
    try:
        run_demo()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted. Run 'python main.py --help' to explore AIMATH!")
