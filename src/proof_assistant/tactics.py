"""
Proof Tactics - Automated proof strategies.

This module provides tactics (proof strategies) that can automatically
construct proofs or guide proof search:

- Direct proof
- Proof by contradiction
- Mathematical induction
- Case analysis
- Forward/backward reasoning
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Callable, Generator
from enum import Enum, auto
import logging

from .logic import (
    LogicalFormula, Proposition, Predicate, Negation, BinaryConnective,
    QuantifiedFormula, Variable, Constant, Function, Term,
    Connective, Quantifier,
    And, Or, Implies, Iff, Not, ForAll, Exists, Equals,
    is_tautology, is_satisfiable, to_nnf
)
from .inference import (
    InferenceRule, ProofStep, Justification, RuleRegistry,
    ModusPonens, ConjunctionIntro, ConjunctionElim,
    ImplicationIntro, NegationIntro
)

logger = logging.getLogger(__name__)


class TacticStatus(Enum):
    """Result status of applying a tactic."""
    SUCCESS = auto()      # Tactic completed the proof
    PROGRESS = auto()     # Tactic made progress but proof not complete
    FAILURE = auto()      # Tactic could not be applied
    SUBGOALS = auto()     # Tactic generated subgoals to prove


@dataclass
class TacticResult:
    """Result of applying a proof tactic."""
    status: TacticStatus
    steps: List[ProofStep] = field(default_factory=list)
    subgoals: List[LogicalFormula] = field(default_factory=list)
    message: str = ""
    
    @property
    def success(self) -> bool:
        return self.status in (TacticStatus.SUCCESS, TacticStatus.PROGRESS)


class ProofTactic(ABC):
    """Base class for proof tactics."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the tactic."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Description of when to use this tactic."""
        pass
    
    @abstractmethod
    def applicable(
        self,
        goal: LogicalFormula,
        available: List[LogicalFormula]
    ) -> bool:
        """Check if this tactic can be applied."""
        pass
    
    @abstractmethod
    def apply(
        self,
        goal: LogicalFormula,
        available: List[LogicalFormula],
        step_counter: int = 1
    ) -> TacticResult:
        """Apply the tactic to prove the goal."""
        pass


# =============================================================================
# DIRECT PROOF TACTICS
# =============================================================================

class DirectProof(ProofTactic):
    """
    Direct proof strategy.
    
    Attempts to prove the goal by forward reasoning from
    available premises, using basic inference rules.
    """
    
    @property
    def name(self) -> str:
        return "Direct Proof"
    
    @property
    def description(self) -> str:
        return "Prove by direct application of inference rules"
    
    def applicable(
        self,
        goal: LogicalFormula,
        available: List[LogicalFormula]
    ) -> bool:
        return True  # Always try direct proof
    
    def apply(
        self,
        goal: LogicalFormula,
        available: List[LogicalFormula],
        step_counter: int = 1
    ) -> TacticResult:
        """Try to prove goal directly from available formulas."""
        
        # Check if goal is already available
        if goal in available:
            return TacticResult(
                status=TacticStatus.SUCCESS,
                message="Goal already available"
            )
        
        # Check if goal is a tautology
        try:
            if is_tautology(goal):
                step = ProofStep(
                    index=step_counter,
                    formula=goal,
                    justification=Justification("Tautology", ())
                )
                return TacticResult(
                    status=TacticStatus.SUCCESS,
                    steps=[step],
                    message="Goal is a tautology"
                )
        except Exception:
            pass
        
        # Try to derive goal using inference rules
        steps: List[ProofStep] = []
        derived: List[LogicalFormula] = list(available)
        
        # Keep trying rules until we can't make progress
        max_iterations = 20
        for _ in range(max_iterations):
            made_progress = False
            
            for rule in RuleRegistry.all_rules():
                new_formulas = self._try_rule(rule, derived, goal)
                
                for formula in new_formulas:
                    if formula not in derived:
                        derived.append(formula)
                        step = ProofStep(
                            index=step_counter + len(steps),
                            formula=formula,
                            justification=Justification(rule.name, ())
                        )
                        steps.append(step)
                        made_progress = True
                        
                        if formula == goal:
                            return TacticResult(
                                status=TacticStatus.SUCCESS,
                                steps=steps,
                                message="Goal derived by direct proof"
                            )
            
            if not made_progress:
                break
        
        if steps:
            return TacticResult(
                status=TacticStatus.PROGRESS,
                steps=steps,
                message="Made progress but did not reach goal"
            )
        
        return TacticResult(
            status=TacticStatus.FAILURE,
            message="Could not make progress with direct proof"
        )
    
    def _try_rule(
        self,
        rule: InferenceRule,
        available: List[LogicalFormula],
        goal: LogicalFormula
    ) -> List[LogicalFormula]:
        """Try applying a rule to available formulas."""
        results: List[LogicalFormula] = []
        n = rule.premises_required
        
        if n < 0:
            return results
        
        from itertools import permutations
        
        if n == 0:
            result = rule.apply([])
            if result is not None:
                results.append(result)
        
        elif n == 1:
            for f in available:
                try:
                    result = rule.apply([f])
                    if result is not None:
                        results.append(result)
                except Exception:
                    continue
        
        elif n == 2:
            for combo in permutations(available, 2):
                try:
                    result = rule.apply(list(combo))
                    if result is not None:
                        results.append(result)
                except Exception:
                    continue
        
        return results


class Contradiction(ProofTactic):
    """
    Proof by contradiction (reductio ad absurdum).
    
    To prove φ:
    1. Assume ¬φ
    2. Derive a contradiction (ψ ∧ ¬ψ)
    3. Conclude φ
    """
    
    @property
    def name(self) -> str:
        return "Proof by Contradiction"
    
    @property
    def description(self) -> str:
        return "Assume the negation and derive a contradiction"
    
    def applicable(
        self,
        goal: LogicalFormula,
        available: List[LogicalFormula]
    ) -> bool:
        # Don't use contradiction if goal is already a negation
        return not isinstance(goal, Negation)
    
    def apply(
        self,
        goal: LogicalFormula,
        available: List[LogicalFormula],
        step_counter: int = 1
    ) -> TacticResult:
        """Try proof by contradiction."""
        steps: List[ProofStep] = []
        
        # Step 1: Assume ¬goal
        negated_goal = Not(goal)
        assumption_step = ProofStep(
            index=step_counter,
            formula=negated_goal,
            justification=Justification("Assumption (for contradiction)", ()),
            depth=1
        )
        steps.append(assumption_step)
        
        # Step 2: Try to derive a contradiction
        extended_available = available + [negated_goal]
        
        # Look for pairs that could form a contradiction
        for f in extended_available:
            neg_f = Not(f) if not isinstance(f, Negation) else f.formula
            
            if neg_f in extended_available or Not(f) in extended_available:
                # Found contradiction: f and ¬f
                
                # Add conjunction step
                if isinstance(f, Negation):
                    contradiction = And(f.formula, f)
                else:
                    contradiction = And(f, Not(f))
                
                conj_step = ProofStep(
                    index=step_counter + 1,
                    formula=contradiction,
                    justification=Justification("Conjunction Introduction", ()),
                    depth=1
                )
                steps.append(conj_step)
                
                # Conclude by contradiction
                conclude_step = ProofStep(
                    index=step_counter + 2,
                    formula=goal,
                    justification=Justification(
                        "Negation Elimination (Contradiction)",
                        (step_counter, step_counter + 1)
                    ),
                    depth=0
                )
                steps.append(conclude_step)
                
                return TacticResult(
                    status=TacticStatus.SUCCESS,
                    steps=steps,
                    message="Proved by contradiction"
                )
        
        # Couldn't find immediate contradiction
        # Return subgoal: need to derive a contradiction from ¬goal
        return TacticResult(
            status=TacticStatus.SUBGOALS,
            steps=steps,
            subgoals=[And(Proposition("⊥"), Proposition("⊥"))],  # Placeholder for contradiction
            message="Need to derive contradiction from assumption"
        )


class Induction(ProofTactic):
    """
    Mathematical induction for proving ∀n.P(n).
    
    To prove ∀n.P(n):
    1. Base case: Prove P(0)
    2. Inductive step: Prove P(k) → P(k+1)
    3. Conclude ∀n.P(n)
    """
    
    @property
    def name(self) -> str:
        return "Mathematical Induction"
    
    @property
    def description(self) -> str:
        return "Prove ∀n.P(n) by base case and inductive step"
    
    def applicable(
        self,
        goal: LogicalFormula,
        available: List[LogicalFormula]
    ) -> bool:
        # Applicable when goal is universally quantified over naturals
        return isinstance(goal, QuantifiedFormula) and goal.quantifier == Quantifier.FORALL
    
    def apply(
        self,
        goal: LogicalFormula,
        available: List[LogicalFormula],
        step_counter: int = 1
    ) -> TacticResult:
        """Set up induction proof."""
        if not isinstance(goal, QuantifiedFormula):
            return TacticResult(
                status=TacticStatus.FAILURE,
                message="Induction requires universally quantified goal"
            )
        
        var = goal.variable
        formula = goal.formula
        zero = Constant("0")
        k = Variable("k")
        succ_k = Function("S", (k,))
        
        # Base case: P(0)
        base_case = formula.substitute(var, zero)
        
        # Inductive hypothesis: P(k)
        ind_hyp = formula.substitute(var, k)
        
        # Inductive conclusion: P(S(k))
        ind_concl = formula.substitute(var, succ_k)
        
        # Inductive step: P(k) → P(S(k))
        ind_step = Implies(ind_hyp, ind_concl)
        
        return TacticResult(
            status=TacticStatus.SUBGOALS,
            subgoals=[base_case, ind_step],
            message=f"Prove by induction:\n  Base: {base_case}\n  Step: {ind_step}"
        )


class CaseAnalysis(ProofTactic):
    """
    Proof by case analysis.
    
    If we have φ ∨ ψ and want to prove χ:
    1. Show φ → χ
    2. Show ψ → χ
    3. Conclude χ by disjunction elimination
    """
    
    @property
    def name(self) -> str:
        return "Case Analysis"
    
    @property
    def description(self) -> str:
        return "Split on disjunction and prove each case"
    
    def applicable(
        self,
        goal: LogicalFormula,
        available: List[LogicalFormula]
    ) -> bool:
        # Look for a disjunction to split on
        for f in available:
            if isinstance(f, BinaryConnective) and f.connective == Connective.OR:
                return True
        return False
    
    def apply(
        self,
        goal: LogicalFormula,
        available: List[LogicalFormula],
        step_counter: int = 1
    ) -> TacticResult:
        """Apply case analysis on a disjunction."""
        
        # Find a disjunction to split on
        for f in available:
            if isinstance(f, BinaryConnective) and f.connective == Connective.OR:
                left = f.left
                right = f.right
                
                # Subgoals: prove goal from each disjunct
                case1 = Implies(left, goal)
                case2 = Implies(right, goal)
                
                return TacticResult(
                    status=TacticStatus.SUBGOALS,
                    subgoals=[case1, case2],
                    message=f"Case analysis on {f}:\n  Case 1: {left} → {goal}\n  Case 2: {right} → {goal}"
                )
        
        return TacticResult(
            status=TacticStatus.FAILURE,
            message="No disjunction available for case analysis"
        )


class ConditionalProof(ProofTactic):
    """
    Conditional proof for implications.
    
    To prove φ → ψ:
    1. Assume φ
    2. Derive ψ (possibly using other tactics)
    3. Conclude φ → ψ
    """
    
    @property
    def name(self) -> str:
        return "Conditional Proof"
    
    @property
    def description(self) -> str:
        return "Assume antecedent and derive consequent"
    
    def applicable(
        self,
        goal: LogicalFormula,
        available: List[LogicalFormula]
    ) -> bool:
        return (
            isinstance(goal, BinaryConnective) and 
            goal.connective == Connective.IMPLIES
        )
    
    def apply(
        self,
        goal: LogicalFormula,
        available: List[LogicalFormula],
        step_counter: int = 1
    ) -> TacticResult:
        """Apply conditional proof."""
        if not isinstance(goal, BinaryConnective) or goal.connective != Connective.IMPLIES:
            return TacticResult(
                status=TacticStatus.FAILURE,
                message="Goal is not an implication"
            )
        
        antecedent = goal.left
        consequent = goal.right
        
        steps: List[ProofStep] = []
        
        # Step 1: Assume antecedent
        assumption_step = ProofStep(
            index=step_counter,
            formula=antecedent,
            justification=Justification("Assumption", ()),
            depth=1
        )
        steps.append(assumption_step)
        
        # Try to derive consequent
        extended_available = available + [antecedent]
        
        # Use direct proof to derive consequent
        direct = DirectProof()
        result = direct.apply(consequent, extended_available, step_counter + 1)
        
        if result.success:
            # Add the derivation steps
            for step in result.steps:
                step.depth = 1  # Mark as part of subproof
                steps.append(step)
            
            # Discharge assumption
            final_step = ProofStep(
                index=step_counter + len(steps),
                formula=goal,
                justification=Justification(
                    "Implication Introduction",
                    (step_counter, step_counter + len(steps) - 1)
                ),
                depth=0
            )
            steps.append(final_step)
            
            return TacticResult(
                status=TacticStatus.SUCCESS,
                steps=steps,
                message="Proved by conditional proof"
            )
        
        # Return subgoal
        return TacticResult(
            status=TacticStatus.SUBGOALS,
            steps=steps,
            subgoals=[consequent],
            message=f"Assumed {antecedent}, need to derive {consequent}"
        )


class BiconditionalProof(ProofTactic):
    """
    Prove biconditional φ ↔ ψ by proving both directions.
    """
    
    @property
    def name(self) -> str:
        return "Biconditional Proof"
    
    @property
    def description(self) -> str:
        return "Prove both directions of the biconditional"
    
    def applicable(
        self,
        goal: LogicalFormula,
        available: List[LogicalFormula]
    ) -> bool:
        return (
            isinstance(goal, BinaryConnective) and 
            goal.connective == Connective.IFF
        )
    
    def apply(
        self,
        goal: LogicalFormula,
        available: List[LogicalFormula],
        step_counter: int = 1
    ) -> TacticResult:
        """Split biconditional into two implications."""
        if not isinstance(goal, BinaryConnective) or goal.connective != Connective.IFF:
            return TacticResult(
                status=TacticStatus.FAILURE,
                message="Goal is not a biconditional"
            )
        
        left_to_right = Implies(goal.left, goal.right)
        right_to_left = Implies(goal.right, goal.left)
        
        return TacticResult(
            status=TacticStatus.SUBGOALS,
            subgoals=[left_to_right, right_to_left],
            message=f"To prove {goal}, show both:\n  1. {left_to_right}\n  2. {right_to_left}"
        )


class UniversalIntro(ProofTactic):
    """
    Introduce universal quantifier by proving for arbitrary element.
    
    To prove ∀x.P(x):
    1. Let a be an arbitrary element
    2. Prove P(a)
    3. Conclude ∀x.P(x) by generalization
    """
    
    @property
    def name(self) -> str:
        return "Universal Introduction"
    
    @property
    def description(self) -> str:
        return "Prove for arbitrary element, then generalize"
    
    def applicable(
        self,
        goal: LogicalFormula,
        available: List[LogicalFormula]
    ) -> bool:
        return (
            isinstance(goal, QuantifiedFormula) and 
            goal.quantifier == Quantifier.FORALL
        )
    
    def apply(
        self,
        goal: LogicalFormula,
        available: List[LogicalFormula],
        step_counter: int = 1
    ) -> TacticResult:
        """Apply universal introduction."""
        if not isinstance(goal, QuantifiedFormula):
            return TacticResult(
                status=TacticStatus.FAILURE,
                message="Goal is not quantified"
            )
        
        # Create arbitrary constant
        arbitrary = Constant(f"a_{goal.variable}")
        
        # Subgoal: prove P(a)
        instance = goal.formula.substitute(goal.variable, arbitrary)
        
        return TacticResult(
            status=TacticStatus.SUBGOALS,
            subgoals=[instance],
            message=f"Let {arbitrary} be arbitrary. Prove: {instance}"
        )


# =============================================================================
# TACTIC COMBINATORS
# =============================================================================

class TryTactic(ProofTactic):
    """
    Try a tactic, but don't fail if it doesn't work.
    """
    
    def __init__(self, tactic: ProofTactic):
        self.tactic = tactic
    
    @property
    def name(self) -> str:
        return f"Try({self.tactic.name})"
    
    @property
    def description(self) -> str:
        return f"Try {self.tactic.name}, continue if it fails"
    
    def applicable(self, goal: LogicalFormula, available: List[LogicalFormula]) -> bool:
        return True
    
    def apply(
        self,
        goal: LogicalFormula,
        available: List[LogicalFormula],
        step_counter: int = 1
    ) -> TacticResult:
        result = self.tactic.apply(goal, available, step_counter)
        if result.status == TacticStatus.FAILURE:
            return TacticResult(
                status=TacticStatus.PROGRESS,
                message=f"{self.tactic.name} did not apply"
            )
        return result


class SequenceTactic(ProofTactic):
    """
    Try tactics in sequence until one works.
    """
    
    def __init__(self, tactics: List[ProofTactic]):
        self.tactics = tactics
    
    @property
    def name(self) -> str:
        names = [t.name for t in self.tactics]
        return f"Sequence({', '.join(names)})"
    
    @property
    def description(self) -> str:
        return "Try tactics in sequence"
    
    def applicable(self, goal: LogicalFormula, available: List[LogicalFormula]) -> bool:
        return any(t.applicable(goal, available) for t in self.tactics)
    
    def apply(
        self,
        goal: LogicalFormula,
        available: List[LogicalFormula],
        step_counter: int = 1
    ) -> TacticResult:
        for tactic in self.tactics:
            if tactic.applicable(goal, available):
                result = tactic.apply(goal, available, step_counter)
                if result.success:
                    return result
        
        return TacticResult(
            status=TacticStatus.FAILURE,
            message="No tactic in sequence succeeded"
        )


class AutoTactic(ProofTactic):
    """
    Automatic proof search.
    
    Tries multiple tactics with backtracking.
    """
    
    def __init__(self, max_depth: int = 10):
        self.max_depth = max_depth
        self.tactics = [
            DirectProof(),
            ConditionalProof(),
            BiconditionalProof(),
            CaseAnalysis(),
            Contradiction(),
        ]
    
    @property
    def name(self) -> str:
        return "Auto"
    
    @property
    def description(self) -> str:
        return "Automatic proof search with backtracking"
    
    def applicable(self, goal: LogicalFormula, available: List[LogicalFormula]) -> bool:
        return True
    
    def apply(
        self,
        goal: LogicalFormula,
        available: List[LogicalFormula],
        step_counter: int = 1
    ) -> TacticResult:
        """Automatic proof search."""
        return self._search(goal, available, step_counter, depth=0)
    
    def _search(
        self,
        goal: LogicalFormula,
        available: List[LogicalFormula],
        step_counter: int,
        depth: int
    ) -> TacticResult:
        """Recursive proof search with backtracking."""
        if depth >= self.max_depth:
            return TacticResult(
                status=TacticStatus.FAILURE,
                message="Maximum search depth reached"
            )
        
        # Try each tactic
        for tactic in self.tactics:
            if tactic.applicable(goal, available):
                result = tactic.apply(goal, available, step_counter)
                
                if result.status == TacticStatus.SUCCESS:
                    return result
                
                elif result.status == TacticStatus.SUBGOALS:
                    # Try to prove all subgoals
                    all_steps = list(result.steps)
                    current_counter = step_counter + len(all_steps)
                    current_available = available + [s.formula for s in all_steps]
                    
                    all_proved = True
                    for subgoal in result.subgoals:
                        sub_result = self._search(
                            subgoal, current_available, 
                            current_counter, depth + 1
                        )
                        
                        if sub_result.success:
                            all_steps.extend(sub_result.steps)
                            current_counter += len(sub_result.steps)
                            current_available.extend([s.formula for s in sub_result.steps])
                        else:
                            all_proved = False
                            break
                    
                    if all_proved:
                        return TacticResult(
                            status=TacticStatus.SUCCESS,
                            steps=all_steps,
                            message=f"Proved using {tactic.name}"
                        )
        
        return TacticResult(
            status=TacticStatus.FAILURE,
            message="No tactic succeeded"
        )


# =============================================================================
# TACTIC REGISTRY
# =============================================================================

class TacticRegistry:
    """Registry of available proof tactics."""
    
    _tactics: Dict[str, ProofTactic] = {}
    
    @classmethod
    def register(cls, tactic: ProofTactic) -> None:
        cls._tactics[tactic.name] = tactic
    
    @classmethod
    def get(cls, name: str) -> Optional[ProofTactic]:
        return cls._tactics.get(name)
    
    @classmethod
    def all_tactics(cls) -> List[ProofTactic]:
        return list(cls._tactics.values())


# Register standard tactics
_standard_tactics = [
    DirectProof(),
    Contradiction(),
    Induction(),
    CaseAnalysis(),
    ConditionalProof(),
    BiconditionalProof(),
    UniversalIntro(),
    AutoTactic(),
]

for tactic in _standard_tactics:
    TacticRegistry.register(tactic)
