# 🧮 AIMATH - AI Math Verification & Discovery Tool

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**A rigorous mathematical verification system that helps everyone—from amateurs to professionals—solve, verify, and discover mathematical concepts with built-in anti-hallucination mechanisms.**

> "Never trust, always verify"

---

## 🆕 What's New: MathClaw Autonomous Discovery

**MathClaw** is an autonomous mathematical discovery engine that continuously generates and proves theorems - forever! It uses LLMs to generate conjectures and AIMATH's verification layer to prove them.

```python
from mathclaw import MathClaw

claw = MathClaw(openai_api_key="sk-...")
claw.start()  # Discovers math autonomously forever!
```

See [MathClaw Documentation](mathclaw/README.md) for full details.

---

## 🚀 Quick Install

```bash
pip install aimath
```

Or install from source:
```bash
git clone https://github.com/jovanSAPFIONEER/AIMATH.git
cd AIMATH
pip install -e .
```

---

## 🎯 Quick Start

### Command Line (Easiest!)

```bash
# Solve an equation
aimath solve "x^2 - 5x + 6 = 0"

# Get an explanation
aimath explain "quadratic formula"

# Interactive mode
aimath interactive

# See all examples
aimath examples
```

### Python API

```python
from aimath import MathEngine, ProofAssistant

# Solve equations with verification
engine = MathEngine()
result = engine.solve("x^2 - 5x + 6 = 0")
print(result.solutions)      # [2, 3]
print(result.confidence)     # 100% (verified)

# Formal proof construction
prover = ProofAssistant()
theorem = prover.state_theorem(
    name="commutativity",
    statement="For all a, b: a + b = b + a"
)
```

---

## ✨ Key Features

### 🔬 Multi-Path Verification
Every problem is solved by 2+ independent methods with consensus required before returning results.

```
HIGHEST TRUST:  Formal theorem provers (Z3, Lean)
     ↓
HIGH TRUST:     Symbolic computation (SymPy + verification)
     ↓
MEDIUM TRUST:   Numerical computation (with error bounds)
     ↓
LOWEST TRUST:   LLM output (ALWAYS requires verification)
```

### 🛡️ Anti-Hallucination Protection
- **Substitution tests**: Plug answers back into original equations
- **Counterexample search**: Actively try to disprove claims
- **Formal proof verification**: Using Z3 theorem prover
- **Domain constraint checking**: Ensure solutions are valid

### 📚 Quality-Enforced Explanations
No hand-waving allowed! Every explanation must:
- Define all terms before use
- Provide concrete examples first
- Show step-by-step reasoning with **WHY** for each step
- Include edge cases and limitations

**Banned patterns** (auto-expanded):
- ❌ "Obviously..." → Must prove it's obvious
- ❌ "Clearly..." → Must show clearly  
- ❌ "It follows that..." → Must show the inference chain
- ❌ "The reader can verify..." → We verify it ourselves

### 📜 Formal Proof Assistant
Construct rigorous proofs with:
- Propositional & first-order logic
- Peano arithmetic axioms
- Multiple proof tactics (direct, contradiction, induction)
- Automated proof verification

---

## 📖 Usage Examples

### Solving Equations

```python
from aimath import MathEngine

engine = MathEngine()

# Polynomial equations
result = engine.solve("x^3 - 6x^2 + 11x - 6 = 0")
# Solutions: [1, 2, 3]

# Trigonometric equations
result = engine.solve("sin(x) = 0.5")
# Solutions: [π/6, 5π/6, ...]

# Systems of equations
result = engine.solve(["x + y = 10", "x - y = 4"])
# Solutions: {x: 7, y: 3}
```

### Verifying Claims

```python
from aimath import MathEngine

engine = MathEngine()

# Verify mathematical claims
result = engine.verify_claim("√2 is irrational")
print(result.is_valid)  # True
print(result.proof)     # Proof by contradiction...

result = engine.verify_claim("e^(iπ) + 1 = 0")
print(result.is_valid)  # True (Euler's identity)
```

### Formal Proofs

```python
from aimath import ProofAssistant, Proposition

prover = ProofAssistant()

# State a theorem
theorem = prover.state_theorem(
    name="modus_ponens_example",
    statement="(P → Q) ∧ P → Q"
)

# The proof assistant guides you through construction
# with verification at each step
```

### Getting Explanations

```bash
# From command line
aimath explain "derivative" --level beginner
aimath explain "pythagorean theorem"
aimath explain "quadratic formula" --level advanced
```

---

## 🏗️ Project Structure

```
AIMATH/
├── aimath/                 # Core verification package
│   ├── core/              # Math engine and types
│   ├── proof_assistant/   # Formal proof system
│   ├── solvers/           # Symbolic & numerical solvers
│   │   ├── hybrid_integrator.py    # Database + Symbolic + Numeric
│   │   ├── contour_integration.py  # Residue Theorem
│   │   ├── pde_solver.py           # Heat, Wave, Transport PDEs
│   │   ├── conjecture_tester.py    # Fuzz testing
│   │   └── constant_recognizer.py  # Inverse symbolic calculator
│   ├── verification/      # Anti-hallucination checks
│   └── explanation/       # Quality-enforced explanations
├── mathclaw/              # 🔮 Autonomous Discovery Engine (NEW!)
│   ├── security/          # Input validation, sandboxing
│   ├── protection/        # Code integrity, rollback
│   ├── evolution/         # Strategy evolution (TEXT only)
│   ├── discovery/         # Conjecture generation & verification
│   ├── api/              # LLM providers (OpenAI, Anthropic, etc.)
│   └── cli/              # MathClaw CLI
├── main.py               # Unified CLI interface
├── tests/                # Test suites
└── examples/             # Usage examples
```

---

## 🛠️ Unified CLI

```bash
# Core AIMATH commands
python main.py wizard              # Interactive mode
python main.py solve "x^2 - 4 = 0" # Solve equations
python main.py integrate "x**2" --bounds 0 1
python main.py contour "1/(z**2+1)"  # Residue theorem
python main.py pde heat --plot       # Solve heat equation
python main.py verify "(a+b)**2" "a**2+b**2"  # Fuzz test
python main.py recognize 3.14159265  # Constant recognition
python main.py riemann --t-max 100   # Riemann Zeta 2D
python main.py riemann3d             # Riemann Zeta 3D explorer

# MathClaw autonomous discovery
python -m mathclaw start --provider openai
python -m mathclaw discover --count 5
python -m mathclaw theorems --limit 20
python -m mathclaw export --format markdown
```

---

## 🧪 Testing

```bash
# Run all tests
pytest

# Run specific test suites
python tests/test_theses.py              # 10 mathematical theses
python tests/test_proof_assistant.py     # Formal proof system
python tests/test_discover_orion_formal_proofs.py  # GWT theory verification
```

---

## 📊 Verified Against

AIMATH has been tested against:
- **10 fundamental mathematical theses** (quadratic formula, Pythagorean theorem, etc.)
- **DISCOVER/Orion consciousness research** (Global Workspace Theory)
- **Statistical formulas** (Wilson CI, Newcombe CI, Cohen's h, etc.)

---

## 🤝 Who Is This For?

| User | Use Case |
|------|----------|
| 🎓 **Students** | Homework help with verified solutions and real explanations |
| 👨‍🏫 **Teachers** | Generate quality problem sets and explanations |
| 🔬 **Researchers** | Verify mathematical claims in papers |
| 💻 **Developers** | Integrate verified math into applications |
| 🤖 **AI Systems** | Ground LLM outputs with rigorous verification |
| 🔮 **Explorers** | Autonomous mathematical discovery with MathClaw |

---

## 🔧 Requirements

- Python 3.9+
- SymPy (symbolic computation)
- NumPy, SciPy (numerical computation)
- Z3-solver (formal verification)
- Click (CLI framework)

For MathClaw autonomous discovery:
- OpenAI API key, OR
- Anthropic API key, OR
- Google API key, OR
- Ollama (local, no key needed)

Install all dependencies:
```bash
pip install -e ".[all]"
```

---

## 📜 License

MIT License - Use freely for any purpose.

---

## 🙏 Contributing

Contributions welcome! Please ensure all code passes verification tests.

```bash
# Before submitting
pytest
python -m aimath.cli solve "x^2 - 4 = 0"  # Quick sanity check
```

---

## 📬 Links

- **GitHub**: https://github.com/jovanSAPFIONEER/AIMATH
- **Issues**: https://github.com/jovanSAPFIONEER/AIMATH/issues

---

<p align="center">
  <b>Made with ❤️ for the math community</b><br>
  <i>"Because everyone deserves verified mathematics"</i>
</p>
