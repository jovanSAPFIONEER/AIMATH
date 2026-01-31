# AI Math Verification & Discovery Tool

A rigorous mathematical verification system that helps amateurs to professionals solve, verify, and discover mathematical concepts—with built-in anti-hallucination mechanisms and genuine explanation quality enforcement.

## Core Philosophy

**"Never trust, always verify"**

```
HIGHEST TRUST:  Formal theorem provers (Lean, Z3, Coq)
                     ↓
HIGH TRUST:     Symbolic computation with verification (SymPy + checks)
                     ↓
MEDIUM TRUST:   Numerical computation (floating point limits)
                     ↓
LOWEST TRUST:   LLM output (ALWAYS requires verification)
```

## Features

### 🔬 Multi-Path Verification
- Every problem solved by 2+ independent methods
- Consensus required before returning results
- Confidence scores (100% proven → <70% flagged for review)

### 🛡️ Anti-Hallucination Core
- Substitution tests (plug answers back in)
- Counterexample search (actively try to disprove)
- Formal proof verification via Z3/Lean
- Domain constraint checking

### 📚 Genuine Explanation Engine
- **No hand-waving**: Auto-detects and expands "obviously", "clearly", "simply"
- **No skipped steps**: Every logical gap explicitly bridged
- **Concrete first**: Examples before abstraction
- **Why, not just how**: Motivation accompanies every procedure
- **Failure cases required**: Shows when methods break down

### 📊 Quality Gates
- CLEAR rubric scoring (Completeness, Logic, Explicit terms, Accessibility, Reasoning)
- Explanations must score ≥20/25 to pass
- Teach-back simulation test
- Superficiality detection

## Installation

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Install Z3 theorem prover
pip install z3-solver
```

## Quick Start

```python
from src.core.engine import MathEngine

engine = MathEngine()

# Solve with verification
result = engine.solve("x^2 - 5x + 6 = 0")
print(result.solution)        # [2, 3]
print(result.confidence)      # 100% (verified)
print(result.explanation)     # Step-by-step with WHY

# Verify a claim
verification = engine.verify_claim("√2 is irrational")
print(verification.is_valid)  # True
print(verification.proof)     # Formal proof
```

## Project Structure

```
AI MATH/
├── src/
│   ├── core/           # Main engine and types
│   ├── parsers/        # LaTeX, natural language parsing
│   ├── solvers/        # Symbolic, numerical, LLM solvers
│   ├── verification/   # Anti-hallucination verification
│   └── explanation/    # Quality-enforced explanations
├── tests/              # Test suites
├── examples/           # Usage examples
└── config/             # Configuration files
```

## Explanation Quality Standards

Every explanation must:

1. **Define all terms before use**
2. **Provide concrete example first**
3. **Show step-by-step with WHY for each step**
4. **Include edge cases and limitations**
5. **Pass adversarial "Skeptical Student" test**

### Banned Patterns (Auto-Expanded)
- "Obviously..." → Must prove it's obvious
- "Clearly..." → Must show clearly
- "It follows that..." → Must show the inference chain
- "By definition..." → Must state the definition
- "The reader can verify..." → Must verify it ourselves

## License

MIT License - See LICENSE file

## Contributing

Contributions welcome! Please ensure all code passes verification tests.
