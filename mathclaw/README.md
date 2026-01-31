# 🔮 MathClaw

**Autonomous Mathematical Discovery Engine**

MathClaw is an AI system that autonomously discovers and proves mathematical theorems, with built-in protections against hallucinations and self-corruption.

## ✨ Features

- **Autonomous Discovery**: Continuously generates and verifies mathematical conjectures
- **Anti-Hallucination**: Only verified results enter the knowledge base
- **Self-Protection**: Core code cannot be modified by the AI
- **Safe Evolution**: Strategies evolve through text mutations, never code changes
- **Multi-Provider**: Works with OpenAI, Anthropic, Gemini, or local Ollama

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     MathClaw Core                        │
├─────────────────────────────────────────────────────────┤
│  Security Layer      │  Protection Layer                │
│  ├─ InputValidator   │  ├─ FrozenRegistry               │
│  ├─ SafeParser       │  ├─ ChecksumGuardian             │
│  ├─ Sandbox          │  ├─ RollbackManager              │
│  └─ RateLimiter      │  └─ HealthChecker                │
├─────────────────────────────────────────────────────────┤
│  Evolution Layer     │  Discovery Layer                 │
│  ├─ StrategyStore    │  ├─ ConjectureGenerator          │
│  ├─ PromptMutator    │  ├─ VerificationBridge           │
│  ├─ DomainSelector   │  ├─ TheoremStore                 │
│  └─ SuccessTracker   │  └─ DiscoveryEngine              │
├─────────────────────────────────────────────────────────┤
│  API Layer                                               │
│  ├─ LLM Providers (OpenAI, Anthropic, Gemini, Ollama)   │
│  └─ CLI Interface                                        │
└─────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Installation

```bash
cd mathclaw
pip install -e ".[all]"
```

### Set Your API Key

```bash
export OPENAI_API_KEY="sk-..."
# or
export ANTHROPIC_API_KEY="sk-ant-..."
# or
export GOOGLE_API_KEY="..."
```

### Python API

```python
from mathclaw import MathClaw

# Create engine
claw = MathClaw(openai_api_key="sk-...")

# Run a single discovery
result = claw.discover_one()
print(result)

# Start autonomous discovery
claw.start()

# Check status
print(claw.status())

# Export discoveries
claw.export("theorems.md")
```

### CLI

```bash
# Start autonomous discovery
mathclaw start --provider openai

# Run 5 discovery attempts
mathclaw discover --count 5

# View discovered theorems
mathclaw theorems --limit 20

# Export to markdown
mathclaw export --format markdown --output discoveries.md

# Check system health
mathclaw health
```

## 🛡️ Safety Guarantees

### 1. No Hallucinations
Every conjecture passes through AIMATH's verification layer. Only symbolically proven or extensively tested results enter the theorem database.

### 2. No Self-Corruption  
The security and protection layers are **frozen** - they cannot be modified by the AI. SHA-256 checksums protect all critical files.

### 3. Safe Evolution
The system can only modify:
- Prompt templates (text)
- Strategy weights (numbers)
- Domain selection (configuration)

It **cannot** modify:
- Python code
- Security layer
- Verification layer

### 4. Rate Limiting
Built-in cost controls prevent runaway API usage:
- Token budgets
- Request limits
- Automatic backoff

## 📊 Mathematical Domains

MathClaw explores:
- **Algebra**: Polynomial identities, factorizations
- **Calculus**: Integrals, derivatives, limits
- **Trigonometry**: Trig identities, angle formulas
- **Number Theory**: Prime patterns, divisibility
- **Analysis**: Series, sequences, convergence
- **Combinatorics**: Counting, binomials

## 🔧 Configuration

Create a `.env` file:

```env
OPENAI_API_KEY=sk-your-key-here
ANTHROPIC_API_KEY=sk-ant-your-key-here
GOOGLE_API_KEY=your-key-here

# Optional settings
MATHCLAW_MAX_ATTEMPTS=100
MATHCLAW_COST_BUDGET=5.0
```

Or use `mathclaw.yaml`:

```yaml
llm_provider: openai
llm_model: gpt-4o-mini
min_interval_seconds: 5
max_attempts_per_hour: 100
exploration_rate: 0.15
```

## 📁 Project Structure

```
mathclaw/
├── security/           # Input validation, safe parsing, sandboxing
│   ├── input_validator.py
│   ├── safe_parser.py
│   ├── sandbox.py
│   └── rate_limiter.py
├── protection/         # Code integrity, rollback, health checks
│   ├── frozen_registry.py
│   ├── checksum_guardian.py
│   ├── rollback_manager.py
│   └── health_checker.py
├── evolution/          # Strategy evolution (TEXT ONLY)
│   ├── strategy_store.py
│   ├── prompt_mutator.py
│   ├── domain_selector.py
│   └── success_tracker.py
├── discovery/          # Core discovery loop
│   ├── conjecture_generator.py
│   ├── verification_bridge.py
│   ├── theorem_store.py
│   ├── knowledge_exporter.py
│   └── discovery_engine.py
├── api/               # LLM providers and config
│   ├── providers.py
│   └── config.py
└── cli/               # Command-line interface
    └── mathclaw_cli.py
```

## 🎯 How It Works

1. **Select Domain & Strategy**: Uses epsilon-greedy selection to balance exploration vs. exploitation
2. **Generate Conjecture**: LLM creates mathematical statements based on prompt templates
3. **Verify**: AIMATH's verification layer attempts to prove/disprove
4. **Store**: Only proven results enter the theorem database
5. **Learn**: Strategy weights update based on success/failure
6. **Repeat**: Forever, or until stopped

## 📈 Success Metrics

MathClaw tracks:
- Proof success rate per strategy
- Domain performance
- Verification time
- Total theorems discovered

View with: `mathclaw status`

## ⚠️ Limitations

- Cannot prove theorems requiring advanced reasoning
- Relies on SymPy's simplification capabilities  
- May rediscover known theorems
- LLM quality affects conjecture quality

## 🤝 Contributing

Contributions welcome! Please ensure:
1. Security/protection code remains immutable
2. New strategies are TEXT ONLY (no code generation)
3. All conjectures go through verification

## 📄 License

MIT License - See LICENSE file

---

*MathClaw: Where AI discovers mathematics, safely.*
