"""
Main Proof Assistant Interface.

This module provides the high-level interface for:
- Constructing formal proofs
- Verifying proofs step-by-step
- Explaining proof structure
- Managing theorems and lemmas
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Union, Callable
from enum import Enum, auto
import logging
import json
from datetime import datetime

from .logic import (
    LogicalFormula, Proposition, Predicate, Negation, BinaryConnective,
    QuantifiedFormula, Variable, Constant, Function, Term,
    Connective, Quantifier,
    And, Or, Implies, Iff, Not, ForAll, Exists, Equals,
    is_tautology, is_contradiction, is_satisfiable
)
from .axioms import AxiomSystem, Axiom, StandardMathAxioms, PeanoAxioms
from .inference import (
    InferenceRule, ProofStep, Justification, RuleRegistry,
    ModusPonens, ModusTollens, ConjunctionIntro, ConjunctionElim,
)
from .verifier import (
    ProofVerifier, VerificationResult, VerificationStatus, ProofGap
)
from .tactics import (
    ProofTactic, TacticResult, TacticStatus, TacticRegistry,
    DirectProof, AutoTactic
)

logger = logging.getLogger(__name__)


class TheoremStatus(Enum):
    """Status of a theorem."""
    UNPROVEN = auto()     # Statement without proof
    PROVEN = auto()       # Fully verified proof
    CONJECTURED = auto()  # Believed true, no proof
    AXIOM = auto()        # Taken as axiom


@dataclass
class Theorem:
    """
    A mathematical theorem with its proof.
    """
    name: str
    statement: LogicalFormula
    premises: List[LogicalFormula] = field(default_factory=list)
    proof: List[ProofStep] = field(default_factory=list)
    status: TheoremStatus = TheoremStatus.UNPROVEN
    
    # Metadata
    description: str = ""
    tags: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)  # Names of theorems used
    created: str = field(default_factory=lambda: datetime.now().isoformat())
    verified: bool = False
    
    def __str__(self) -> str:
        status_symbol = {
            TheoremStatus.UNPROVEN: "?",
            TheoremStatus.PROVEN: "✓",
            TheoremStatus.CONJECTURED: "~",
            TheoremStatus.AXIOM: "⊢"
        }
        return f"[{status_symbol[self.status]}] {self.name}: {self.statement}"
    
    def full_statement(self) -> str:
        """Get full theorem statement with premises."""
        if not self.premises:
            return str(self.statement)
        
        premises_str = ", ".join(str(p) for p in self.premises)
        return f"{premises_str} ⊢ {self.statement}"


@dataclass
class Proof:
    """
    A formal proof under construction.
    """
    theorem: Theorem
    steps: List[ProofStep] = field(default_factory=list)
    current_goals: List[LogicalFormula] = field(default_factory=list)
    available: List[LogicalFormula] = field(default_factory=list)
    
    # Subproof management
    assumption_stack: List[Tuple[int, LogicalFormula]] = field(default_factory=list)
    
    def __post_init__(self):
        # Initialize with premises and main goal
        self.available = list(self.theorem.premises)
        self.current_goals = [self.theorem.statement]
    
    @property
    def is_complete(self) -> bool:
        """Check if proof is complete (all goals discharged)."""
        return len(self.current_goals) == 0
    
    @property
    def next_step_index(self) -> int:
        """Get the index for the next step."""
        return len(self.steps) + 1
    
    def add_step(
        self,
        formula: LogicalFormula,
        justification: Justification,
        depth: int = 0
    ) -> ProofStep:
        """Add a step to the proof."""
        step = ProofStep(
            index=self.next_step_index,
            formula=formula,
            justification=justification,
            depth=depth
        )
        self.steps.append(step)
        self.available.append(formula)
        
        # Check if this discharges a goal
        if formula in self.current_goals:
            self.current_goals.remove(formula)
        
        return step
    
    def assume(self, formula: LogicalFormula) -> ProofStep:
        """Begin a subproof with an assumption."""
        step = self.add_step(
            formula,
            Justification("Assumption", ()),
            depth=len(self.assumption_stack) + 1
        )
        self.assumption_stack.append((step.index, formula))
        return step
    
    def discharge(self, conclusion: LogicalFormula) -> Optional[ProofStep]:
        """Discharge an assumption and conclude implication."""
        if not self.assumption_stack:
            return None
        
        assumption_idx, assumption = self.assumption_stack.pop()
        implication = Implies(assumption, conclusion)
        
        step = self.add_step(
            implication,
            Justification(
                "Implication Introduction",
                (assumption_idx, self.steps[-1].index if self.steps else 1)
            ),
            depth=len(self.assumption_stack)
        )
        
        return step


class ProofAssistant:
    """
    Interactive proof assistant.
    
    Provides a high-level interface for:
    - Creating and verifying proofs
    - Applying proof tactics
    - Managing theorem library
    - Explaining proof structure
    """
    
    def __init__(
        self,
        axiom_system: Optional[AxiomSystem] = None,
        auto_verify: bool = True
    ):
        """
        Initialize the proof assistant.
        
        Args:
            axiom_system: The axiom system to use
            auto_verify: Automatically verify steps as they're added
        """
        self.axiom_system = axiom_system or StandardMathAxioms()
        self.verifier = ProofVerifier(axiom_system=self.axiom_system)
        self.auto_verify = auto_verify
        
        # Theorem library
        self.theorems: Dict[str, Theorem] = {}
        self.current_proof: Optional[Proof] = None
    
    # =========================================================================
    # THEOREM MANAGEMENT
    # =========================================================================
    
    def state_theorem(
        self,
        name: str,
        statement: LogicalFormula,
        premises: Optional[List[LogicalFormula]] = None,
        description: str = ""
    ) -> Theorem:
        """
        State a new theorem to be proven.
        
        Args:
            name: Name of the theorem
            statement: What to prove
            premises: Assumptions (if any)
            description: Human-readable description
            
        Returns:
            The new Theorem object
        """
        theorem = Theorem(
            name=name,
            statement=statement,
            premises=premises or [],
            description=description,
            status=TheoremStatus.UNPROVEN
        )
        self.theorems[name] = theorem
        return theorem
    
    def add_axiom(
        self,
        name: str,
        statement: LogicalFormula,
        description: str = ""
    ) -> Theorem:
        """Add an axiom to the system."""
        theorem = Theorem(
            name=name,
            statement=statement,
            description=description,
            status=TheoremStatus.AXIOM,
            verified=True
        )
        self.theorems[name] = theorem
        return theorem
    
    def get_theorem(self, name: str) -> Optional[Theorem]:
        """Get a theorem by name."""
        return self.theorems.get(name)
    
    def list_theorems(
        self,
        status: Optional[TheoremStatus] = None
    ) -> List[Theorem]:
        """List theorems, optionally filtered by status."""
        theorems = list(self.theorems.values())
        if status is not None:
            theorems = [t for t in theorems if t.status == status]
        return theorems
    
    # =========================================================================
    # PROOF CONSTRUCTION
    # =========================================================================
    
    def begin_proof(self, theorem: Union[str, Theorem]) -> Proof:
        """
        Begin proving a theorem.
        
        Args:
            theorem: Theorem name or Theorem object
            
        Returns:
            A new Proof object
        """
        if isinstance(theorem, str):
            theorem = self.theorems.get(theorem)
            if theorem is None:
                raise ValueError(f"Unknown theorem: {theorem}")
        
        self.current_proof = Proof(theorem=theorem)
        return self.current_proof
    
    def apply_rule(
        self,
        rule_name: str,
        premise_indices: List[int],
        **kwargs
    ) -> Optional[ProofStep]:
        """
        Apply an inference rule.
        
        Args:
            rule_name: Name of the rule to apply
            premise_indices: Indices of steps to use as premises
            **kwargs: Additional arguments for the rule
            
        Returns:
            The new proof step, or None if rule doesn't apply
        """
        if self.current_proof is None:
            raise ValueError("No active proof")
        
        proof = self.current_proof
        rule = RuleRegistry.get(rule_name)
        
        if rule is None:
            logger.warning(f"Unknown rule: {rule_name}")
            return None
        
        # Get premise formulas
        premises = []
        for idx in premise_indices:
            if 1 <= idx <= len(proof.steps):
                premises.append(proof.steps[idx - 1].formula)
            else:
                logger.warning(f"Invalid premise index: {idx}")
                return None
        
        # Apply the rule
        result = rule.apply(premises, **kwargs)
        
        if result is None:
            logger.warning(f"Rule {rule_name} does not apply to given premises")
            return None
        
        # Add the step
        step = proof.add_step(
            result,
            Justification(rule_name, tuple(premise_indices))
        )
        
        return step
    
    def apply_tactic(
        self,
        tactic_name: str,
        **kwargs
    ) -> TacticResult:
        """
        Apply a proof tactic.
        
        Args:
            tactic_name: Name of the tactic
            **kwargs: Additional arguments
            
        Returns:
            TacticResult with steps and subgoals
        """
        if self.current_proof is None:
            raise ValueError("No active proof")
        
        proof = self.current_proof
        tactic = TacticRegistry.get(tactic_name)
        
        if tactic is None:
            return TacticResult(
                status=TacticStatus.FAILURE,
                message=f"Unknown tactic: {tactic_name}"
            )
        
        if not proof.current_goals:
            return TacticResult(
                status=TacticStatus.SUCCESS,
                message="No goals remaining"
            )
        
        # Apply tactic to first goal
        goal = proof.current_goals[0]
        result = tactic.apply(goal, proof.available, proof.next_step_index)
        
        # Add generated steps
        for step in result.steps:
            proof.steps.append(step)
            proof.available.append(step.formula)
        
        # Update goals
        if result.status == TacticStatus.SUCCESS:
            if goal in proof.current_goals:
                proof.current_goals.remove(goal)
        elif result.status == TacticStatus.SUBGOALS:
            if goal in proof.current_goals:
                proof.current_goals.remove(goal)
            proof.current_goals = result.subgoals + proof.current_goals
        
        return result
    
    def auto_prove(self, max_depth: int = 10) -> TacticResult:
        """
        Attempt automatic proof.
        
        Args:
            max_depth: Maximum search depth
            
        Returns:
            TacticResult indicating success/failure
        """
        auto = AutoTactic(max_depth=max_depth)
        return self.apply_tactic("Auto")
    
    def end_proof(self) -> VerificationResult:
        """
        Complete and verify the current proof.
        
        Returns:
            VerificationResult with detailed analysis
        """
        if self.current_proof is None:
            raise ValueError("No active proof")
        
        proof = self.current_proof
        theorem = proof.theorem
        
        # Verify the proof
        result = self.verifier.verify(
            proof.steps,
            theorem.premises,
            theorem.statement
        )
        
        if result.is_valid:
            theorem.proof = proof.steps
            theorem.status = TheoremStatus.PROVEN
            theorem.verified = True
        
        return result
    
    def qed(self) -> VerificationResult:
        """Alias for end_proof (Q.E.D. - quod erat demonstrandum)."""
        return self.end_proof()
    
    # =========================================================================
    # PROOF VERIFICATION
    # =========================================================================
    
    def verify(
        self,
        theorem: Union[str, Theorem]
    ) -> VerificationResult:
        """
        Verify a theorem's proof.
        
        Args:
            theorem: Theorem name or object
            
        Returns:
            VerificationResult
        """
        if isinstance(theorem, str):
            theorem = self.theorems.get(theorem)
            if theorem is None:
                raise ValueError(f"Unknown theorem: {theorem}")
        
        return self.verifier.verify(
            theorem.proof,
            theorem.premises,
            theorem.statement
        )
    
    def check_step(
        self,
        step_index: int
    ) -> VerificationResult:
        """
        Verify a single proof step.
        
        Args:
            step_index: Index of the step to check
            
        Returns:
            VerificationResult for that step
        """
        if self.current_proof is None:
            raise ValueError("No active proof")
        
        proof = self.current_proof
        
        if not (1 <= step_index <= len(proof.steps)):
            raise ValueError(f"Invalid step index: {step_index}")
        
        steps_so_far = proof.steps[:step_index]
        step = proof.steps[step_index - 1]
        
        result = self.verifier.verify(
            steps_so_far,
            proof.theorem.premises,
            step.formula
        )
        
        return result
    
    # =========================================================================
    # PROOF EXPLANATION
    # =========================================================================
    
    def explain_proof(
        self,
        theorem: Union[str, Theorem],
        detail_level: str = "medium"
    ) -> str:
        """
        Generate human-readable explanation of a proof.
        
        Args:
            theorem: Theorem to explain
            detail_level: "brief", "medium", or "detailed"
            
        Returns:
            Explanation string
        """
        if isinstance(theorem, str):
            theorem = self.theorems.get(theorem)
            if theorem is None:
                return f"Unknown theorem: {theorem}"
        
        lines = []
        lines.append("=" * 60)
        lines.append(f"THEOREM: {theorem.name}")
        lines.append("=" * 60)
        lines.append("")
        
        # Statement
        lines.append("STATEMENT:")
        if theorem.premises:
            premises_str = "\n    ".join(str(p) for p in theorem.premises)
            lines.append(f"  Given:\n    {premises_str}")
        lines.append(f"  Prove: {theorem.statement}")
        lines.append("")
        
        if theorem.description:
            lines.append(f"DESCRIPTION: {theorem.description}")
            lines.append("")
        
        # Proof
        if theorem.status == TheoremStatus.PROVEN and theorem.proof:
            lines.append("PROOF:")
            lines.append("-" * 40)
            
            for step in theorem.proof:
                indent = "  " * step.depth
                
                if detail_level == "brief":
                    lines.append(f"{indent}{step.index}. {step.formula}")
                else:
                    lines.append(f"{indent}{step.index}. {step.formula}")
                    lines.append(f"{indent}   [{step.justification}]")
                    
                    if detail_level == "detailed":
                        explanation = self._explain_step(step)
                        if explanation:
                            lines.append(f"{indent}   → {explanation}")
                    lines.append("")
            
            lines.append("-" * 40)
            lines.append("∎ Q.E.D.")
        
        elif theorem.status == TheoremStatus.AXIOM:
            lines.append("(Axiom - taken without proof)")
        
        else:
            lines.append("(Not yet proven)")
        
        lines.append("")
        return "\n".join(lines)
    
    def _explain_step(self, step: ProofStep) -> str:
        """Generate explanation for a single step."""
        rule_name = step.justification.rule_name.lower()
        
        explanations = {
            "premise": "Given as a premise",
            "assumption": "Assumed for the sake of argument",
            "modus ponens": "If we know P and P→Q, we can conclude Q",
            "modus tollens": "If we know ¬Q and P→Q, we can conclude ¬P",
            "conjunction introduction": "Combining two facts into one",
            "conjunction elimination": "Extracting one fact from a conjunction",
            "disjunction introduction": "Weakening: if P then P∨Q",
            "disjunctive syllogism": "If P∨Q and ¬P, then Q",
            "implication introduction": "If assuming P leads to Q, then P→Q",
            "double negation elimination": "¬¬P is equivalent to P",
            "universal instantiation": "What's true for all is true for any specific case",
            "existential generalization": "A specific example proves existence",
            "tautology": "A logical truth",
            "axiom": "Taken as a fundamental truth",
        }
        
        for key, explanation in explanations.items():
            if key in rule_name:
                return explanation
        
        return ""
    
    def show_goals(self) -> str:
        """Show current proof goals."""
        if self.current_proof is None:
            return "No active proof"
        
        proof = self.current_proof
        
        if not proof.current_goals:
            return "No goals remaining! Use qed() to complete the proof."
        
        lines = ["Current goals:"]
        for i, goal in enumerate(proof.current_goals, 1):
            lines.append(f"  {i}. {goal}")
        
        return "\n".join(lines)
    
    def show_available(self) -> str:
        """Show available formulas."""
        if self.current_proof is None:
            return "No active proof"
        
        proof = self.current_proof
        
        lines = ["Available formulas:"]
        
        # Premises
        for i, p in enumerate(proof.theorem.premises, 1):
            lines.append(f"  P{i}. {p}  [Premise]")
        
        # Derived steps
        for step in proof.steps:
            lines.append(f"  {step.index}. {step.formula}  [{step.justification}]")
        
        return "\n".join(lines)
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def formula(self, text: str) -> LogicalFormula:
        """
        Parse a formula from text.
        
        Convenience method for interactive use.
        """
        from .verifier import ProofChecker
        checker = ProofChecker()
        return checker._parse_formula(text)
    
    def save_library(self, filepath: str) -> None:
        """Save theorem library to file."""
        data = {
            name: {
                "name": t.name,
                "statement": str(t.statement),
                "premises": [str(p) for p in t.premises],
                "status": t.status.name,
                "description": t.description,
                "verified": t.verified
            }
            for name, t in self.theorems.items()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_library(self, filepath: str) -> None:
        """Load theorem library from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        for name, t_data in data.items():
            self.theorems[name] = Theorem(
                name=t_data["name"],
                statement=self.formula(t_data["statement"]),
                premises=[self.formula(p) for p in t_data["premises"]],
                status=TheoremStatus[t_data["status"]],
                description=t_data.get("description", ""),
                verified=t_data.get("verified", False)
            )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_proof_assistant() -> ProofAssistant:
    """Create a new proof assistant with default settings."""
    return ProofAssistant()


def quick_verify(
    premises: List[str],
    conclusion: str,
    proof_steps: List[str]
) -> VerificationResult:
    """
    Quickly verify a proof given as text.
    
    Args:
        premises: List of premise strings
        conclusion: Conclusion string
        proof_steps: List of proof step strings
        
    Returns:
        VerificationResult
    """
    from .verifier import ProofChecker
    checker = ProofChecker()
    
    proof_text = "\n".join(proof_steps)
    return checker.check(proof_text, premises, conclusion)
