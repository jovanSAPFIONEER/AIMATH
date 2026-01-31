"""
Core type definitions for the AI Math Verification System.

These types form the foundation of all data flowing through the system.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Union
from datetime import datetime


class ConfidenceLevel(Enum):
    """
    Confidence levels for mathematical results.
    
    PROVEN (100%): Formally verified by theorem prover
    HIGH (95%+): Multiple methods agree + substitution passes
    MEDIUM (70-95%): Single method, passes basic checks
    LOW (<70%): Methods disagree, needs human review
    UNKNOWN: Problem type not verifiable automatically
    """
    PROVEN = "proven"      # 100% - Formally verified
    HIGH = "high"          # 95%+ - Multi-method consensus
    MEDIUM = "medium"      # 70-95% - Single method verified
    LOW = "low"            # <70% - Needs review
    UNKNOWN = "unknown"    # Cannot determine


class DifficultyLevel(Enum):
    """
    Difficulty levels for explanation adaptation.
    
    Each level assumes different mathematical background:
    - AMATEUR: High school algebra, basic geometry
    - INTERMEDIATE: Calculus, linear algebra basics
    - ADVANCED: Real analysis, abstract algebra
    - EXPERT: Research-level, assumes domain expertise
    """
    AMATEUR = "amateur"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class ProblemType(Enum):
    """Types of mathematical problems the system can handle."""
    EQUATION = "equation"           # Solve for x
    INEQUALITY = "inequality"       # Solve inequality
    SIMPLIFICATION = "simplify"     # Simplify expression
    DIFFERENTIATION = "derivative"  # Find derivative
    INTEGRATION = "integral"        # Find integral
    LIMIT = "limit"                 # Evaluate limit
    PROOF = "proof"                 # Prove statement
    VERIFICATION = "verify"         # Verify claim
    OPTIMIZATION = "optimize"       # Find min/max
    SYSTEM = "system"               # System of equations
    SERIES = "series"               # Series/sequence
    MATRIX = "matrix"               # Matrix operations
    GENERAL = "general"             # General problem


@dataclass
class MathProblem:
    """
    Represents a mathematical problem to be solved.
    
    Attributes:
        raw_input: Original input string (LaTeX, natural language, etc.)
        parsed_expression: SymPy expression after parsing
        problem_type: Categorized problem type
        variables: List of variables in the problem
        constraints: Domain constraints (e.g., x > 0)
        context: Additional context for the problem
        difficulty_hint: Suggested difficulty level
    """
    raw_input: str
    parsed_expression: Optional[Any] = None  # SymPy expression
    problem_type: ProblemType = ProblemType.GENERAL
    variables: list[str] = field(default_factory=list)
    constraints: list[Any] = field(default_factory=list)
    context: Optional[str] = None
    difficulty_hint: Optional[DifficultyLevel] = None
    
    def __post_init__(self):
        """Validate problem after initialization."""
        if not self.raw_input or not self.raw_input.strip():
            raise ValueError("Problem input cannot be empty")


@dataclass 
class SolutionStep:
    """
    A single step in a solution.
    
    Attributes:
        action: What is being done (e.g., "Factor the quadratic")
        expression: The mathematical expression at this step
        justification: WHY this step is valid (not just what)
        from_step: Reference to previous step (for tracing)
        warnings: Any caveats or assumptions made
    """
    action: str
    expression: Any  # SymPy expression or string
    justification: str
    from_step: Optional[int] = None
    warnings: list[str] = field(default_factory=list)


@dataclass
class VerificationCheck:
    """
    Result of a single verification check.
    
    Attributes:
        check_type: Type of verification performed
        passed: Whether the check passed
        details: Explanation of what was checked
        evidence: Supporting evidence (e.g., substitution result)
        error: Error message if check failed
    """
    check_type: str  # e.g., "substitution", "counterexample", "formal_proof"
    passed: bool
    details: str
    evidence: Optional[Any] = None
    error: Optional[str] = None


@dataclass
class VerificationResult:
    """
    Complete verification result for a solution.
    
    Attributes:
        is_verified: Overall verification status
        confidence: Confidence level of the result
        confidence_score: Numerical confidence (0-100)
        checks: List of all verification checks performed
        methods_used: Names of solving methods that agreed
        counterexamples: Any counterexamples found (if invalid)
        formal_proof: Formal proof if available
        warnings: Any warnings or caveats
    """
    is_verified: bool
    confidence: ConfidenceLevel
    confidence_score: float  # 0-100
    checks: list[VerificationCheck] = field(default_factory=list)
    methods_used: list[str] = field(default_factory=list)
    counterexamples: list[Any] = field(default_factory=list)
    formal_proof: Optional[str] = None
    warnings: list[str] = field(default_factory=list)
    
    @property
    def summary(self) -> str:
        """Generate human-readable verification summary."""
        status = "✓ VERIFIED" if self.is_verified else "✗ NOT VERIFIED"
        return (
            f"{status} (Confidence: {self.confidence.value}, "
            f"Score: {self.confidence_score:.1f}%)\n"
            f"Methods: {', '.join(self.methods_used)}\n"
            f"Checks passed: {sum(1 for c in self.checks if c.passed)}/{len(self.checks)}"
        )


@dataclass
class ExplanationQuality:
    """
    Quality assessment of an explanation using CLEAR rubric.
    
    C - Completeness (0-5): All necessary steps present
    L - Logical Flow (0-5): Each step follows from previous
    E - Explicit Terms (0-5): All terms defined before use
    A - Accessibility (0-5): Matched to learner's level
    R - Reasoning (0-5): "Why" provided, not just "how"
    """
    completeness: int  # 0-5
    logical_flow: int  # 0-5
    explicit_terms: int  # 0-5
    accessibility: int  # 0-5
    reasoning: int  # 0-5
    
    anti_patterns_found: list[str] = field(default_factory=list)
    missing_definitions: list[str] = field(default_factory=list)
    skipped_steps: list[str] = field(default_factory=list)
    
    @property
    def total_score(self) -> int:
        """Total CLEAR score out of 25."""
        return (
            self.completeness + self.logical_flow + 
            self.explicit_terms + self.accessibility + self.reasoning
        )
    
    @property
    def passes_quality_gate(self) -> bool:
        """Check if explanation passes minimum quality requirements."""
        min_total = 20
        min_category = 3
        return (
            self.total_score >= min_total and
            all(score >= min_category for score in [
                self.completeness, self.logical_flow,
                self.explicit_terms, self.accessibility, self.reasoning
            ])
        )
    
    @property
    def report(self) -> str:
        """Generate quality report."""
        status = "✓ PASSES" if self.passes_quality_gate else "✗ FAILS"
        return (
            f"CLEAR Quality Assessment: {status}\n"
            f"  Completeness:   {self.completeness}/5\n"
            f"  Logical Flow:   {self.logical_flow}/5\n"
            f"  Explicit Terms: {self.explicit_terms}/5\n"
            f"  Accessibility:  {self.accessibility}/5\n"
            f"  Reasoning:      {self.reasoning}/5\n"
            f"  TOTAL:          {self.total_score}/25\n"
            f"\nIssues found:\n"
            f"  Anti-patterns: {len(self.anti_patterns_found)}\n"
            f"  Missing definitions: {len(self.missing_definitions)}\n"
            f"  Skipped steps: {len(self.skipped_steps)}"
        )


@dataclass
class Explanation:
    """
    A complete, quality-verified explanation.
    
    Attributes:
        prerequisites: Terms/concepts defined before use
        intuition: Concrete example before abstraction
        core_idea: One-sentence essential insight
        steps: Step-by-step solution with WHY for each
        why_it_works: Underlying principle explanation
        edge_cases: When method fails, limitations
        verify_understanding: Follow-up questions/problems
        difficulty_level: Target audience level
        quality: Quality assessment scores
    """
    prerequisites: dict[str, str]  # term -> definition
    intuition: str  # Concrete example first
    core_idea: str  # Essential insight
    steps: list[SolutionStep]
    why_it_works: str
    edge_cases: list[str]
    verify_understanding: list[str]  # Follow-up problems
    difficulty_level: DifficultyLevel
    quality: Optional[ExplanationQuality] = None
    
    def to_markdown(self) -> str:
        """Render explanation as Markdown."""
        md = []
        
        # Prerequisites
        if self.prerequisites:
            md.append("## Prerequisites\n")
            for term, definition in self.prerequisites.items():
                md.append(f"- **{term}**: {definition}\n")
            md.append("")
        
        # Intuition
        md.append("## Intuition (Concrete Example)\n")
        md.append(self.intuition + "\n")
        
        # Core idea
        md.append("## Core Idea\n")
        md.append(f"**{self.core_idea}**\n")
        
        # Steps
        md.append("## Step-by-Step Solution\n")
        for i, step in enumerate(self.steps, 1):
            md.append(f"### Step {i}: {step.action}\n")
            md.append(f"**Expression:** {step.expression}\n")
            md.append(f"**Why:** {step.justification}\n")
            if step.warnings:
                md.append(f"**⚠ Watch out:** {', '.join(step.warnings)}\n")
            md.append("")
        
        # Why it works
        md.append("## Why This Works\n")
        md.append(self.why_it_works + "\n")
        
        # Edge cases
        md.append("## Edge Cases & Limitations\n")
        for edge in self.edge_cases:
            md.append(f"- {edge}\n")
        md.append("")
        
        # Verify understanding
        md.append("## Verify Your Understanding\n")
        for q in self.verify_understanding:
            md.append(f"- {q}\n")
        
        return "".join(md)


@dataclass
class Solution:
    """
    Complete solution with verification and explanation.
    
    This is the main output type combining:
    - The actual answer
    - Verification that it's correct
    - Quality explanation of how/why
    """
    problem: MathProblem
    answer: Any  # The solution value(s)
    answer_latex: str  # LaTeX representation
    verification: VerificationResult
    explanation: Explanation
    
    # Metadata
    solve_time_ms: float = 0
    timestamp: datetime = field(default_factory=datetime.now)
    solver_methods: list[str] = field(default_factory=list)
    
    @property
    def confidence(self) -> ConfidenceLevel:
        """Shortcut to verification confidence."""
        return self.verification.confidence
    
    @property
    def is_verified(self) -> bool:
        """Shortcut to verification status."""
        return self.verification.is_verified
    
    def summary(self) -> str:
        """Generate concise solution summary."""
        return (
            f"Problem: {self.problem.raw_input}\n"
            f"Answer: {self.answer_latex}\n"
            f"Confidence: {self.confidence.value} "
            f"({self.verification.confidence_score:.1f}%)\n"
            f"Verified: {'Yes' if self.is_verified else 'No'}"
        )
