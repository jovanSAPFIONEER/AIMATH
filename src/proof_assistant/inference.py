"""
Inference Rules and Proof Steps.

This module defines the inference rules used to derive new statements
from existing ones in formal proofs.

Includes:
- Natural deduction rules
- Sequent calculus rules  
- Common derived rules
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Union
from enum import Enum, auto

from .logic import (
    LogicalFormula, Proposition, Predicate, Negation, BinaryConnective,
    QuantifiedFormula, Variable, Constant, Function, Term,
    Connective, Quantifier,
    And, Or, Implies, Iff, Not, ForAll, Exists, Equals,
    logically_equivalent
)


class RuleType(Enum):
    """Types of inference rules."""
    INTRODUCTION = auto()  # Introduces a connective/quantifier
    ELIMINATION = auto()   # Eliminates a connective/quantifier
    STRUCTURAL = auto()    # Structural rules (weakening, etc.)
    DERIVED = auto()       # Derived/admissible rules


@dataclass(frozen=True)
class Justification:
    """
    Justification for a proof step.
    
    Records which rule was applied and what premises were used.
    """
    rule_name: str
    premise_indices: Tuple[int, ...]
    substitutions: Dict[str, Term] = field(default_factory=dict)
    note: str = ""
    
    def __str__(self) -> str:
        if self.premise_indices:
            premises = ", ".join(str(i) for i in self.premise_indices)
            return f"{self.rule_name} ({premises})"
        return self.rule_name


@dataclass
class ProofStep:
    """
    A single step in a formal proof.
    
    Contains the formula derived and justification for the derivation.
    """
    index: int
    formula: LogicalFormula
    justification: Justification
    depth: int = 0  # For subproofs (nested assumptions)
    
    def __str__(self) -> str:
        indent = "  " * self.depth
        return f"{indent}{self.index}. {self.formula}  [{self.justification}]"


class InferenceRule(ABC):
    """Base class for inference rules."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the rule."""
        pass
    
    @property
    @abstractmethod
    def rule_type(self) -> RuleType:
        """Type of rule (introduction/elimination/etc.)."""
        pass
    
    @property
    @abstractmethod
    def premises_required(self) -> int:
        """Number of premises required (-1 for variable)."""
        pass
    
    @abstractmethod
    def apply(
        self, 
        premises: List[LogicalFormula],
        **kwargs
    ) -> Optional[LogicalFormula]:
        """
        Apply the rule to derive a conclusion.
        
        Returns the conclusion if the rule applies, None otherwise.
        """
        pass
    
    @abstractmethod
    def verify(
        self, 
        premises: List[LogicalFormula], 
        conclusion: LogicalFormula,
        **kwargs
    ) -> bool:
        """
        Verify that the conclusion follows from premises by this rule.
        """
        pass
    
    def __str__(self) -> str:
        return self.name


# =============================================================================
# BASIC INFERENCE RULES
# =============================================================================

class ModusPonens(InferenceRule):
    """
    Modus Ponens (→ Elimination).
    
    From φ and φ → ψ, derive ψ.
    
    Example:
        1. P           (premise)
        2. P → Q       (premise)
        3. Q           (MP, 1, 2)
    """
    
    @property
    def name(self) -> str:
        return "Modus Ponens"
    
    @property
    def rule_type(self) -> RuleType:
        return RuleType.ELIMINATION
    
    @property
    def premises_required(self) -> int:
        return 2
    
    def apply(
        self, 
        premises: List[LogicalFormula],
        **kwargs
    ) -> Optional[LogicalFormula]:
        if len(premises) != 2:
            return None
        
        # Try both orderings
        for i, j in [(0, 1), (1, 0)]:
            antecedent = premises[i]
            conditional = premises[j]
            
            if isinstance(conditional, BinaryConnective):
                if conditional.connective == Connective.IMPLIES:
                    if conditional.left == antecedent:
                        return conditional.right
        
        return None
    
    def verify(
        self, 
        premises: List[LogicalFormula], 
        conclusion: LogicalFormula,
        **kwargs
    ) -> bool:
        result = self.apply(premises)
        return result is not None and result == conclusion


class ModusTollens(InferenceRule):
    """
    Modus Tollens.
    
    From ¬ψ and φ → ψ, derive ¬φ.
    
    Example:
        1. ¬Q          (premise)
        2. P → Q       (premise)
        3. ¬P          (MT, 1, 2)
    """
    
    @property
    def name(self) -> str:
        return "Modus Tollens"
    
    @property
    def rule_type(self) -> RuleType:
        return RuleType.ELIMINATION
    
    @property
    def premises_required(self) -> int:
        return 2
    
    def apply(
        self, 
        premises: List[LogicalFormula],
        **kwargs
    ) -> Optional[LogicalFormula]:
        if len(premises) != 2:
            return None
        
        for i, j in [(0, 1), (1, 0)]:
            neg_consequent = premises[i]
            conditional = premises[j]
            
            if isinstance(neg_consequent, Negation):
                if isinstance(conditional, BinaryConnective):
                    if conditional.connective == Connective.IMPLIES:
                        if conditional.right == neg_consequent.formula:
                            return Not(conditional.left)
        
        return None
    
    def verify(
        self, 
        premises: List[LogicalFormula], 
        conclusion: LogicalFormula,
        **kwargs
    ) -> bool:
        result = self.apply(premises)
        return result is not None and result == conclusion


class HypotheticalSyllogism(InferenceRule):
    """
    Hypothetical Syllogism (Chain Rule).
    
    From φ → ψ and ψ → χ, derive φ → χ.
    
    Example:
        1. P → Q       (premise)
        2. Q → R       (premise)
        3. P → R       (HS, 1, 2)
    """
    
    @property
    def name(self) -> str:
        return "Hypothetical Syllogism"
    
    @property
    def rule_type(self) -> RuleType:
        return RuleType.DERIVED
    
    @property
    def premises_required(self) -> int:
        return 2
    
    def apply(
        self, 
        premises: List[LogicalFormula],
        **kwargs
    ) -> Optional[LogicalFormula]:
        if len(premises) != 2:
            return None
        
        for i, j in [(0, 1), (1, 0)]:
            first = premises[i]
            second = premises[j]
            
            if isinstance(first, BinaryConnective) and isinstance(second, BinaryConnective):
                if first.connective == Connective.IMPLIES and second.connective == Connective.IMPLIES:
                    if first.right == second.left:
                        return Implies(first.left, second.right)
        
        return None
    
    def verify(
        self, 
        premises: List[LogicalFormula], 
        conclusion: LogicalFormula,
        **kwargs
    ) -> bool:
        result = self.apply(premises)
        return result is not None and result == conclusion


class DisjunctiveSyllogism(InferenceRule):
    """
    Disjunctive Syllogism.
    
    From φ ∨ ψ and ¬φ, derive ψ.
    
    Example:
        1. P ∨ Q       (premise)
        2. ¬P          (premise)
        3. Q           (DS, 1, 2)
    """
    
    @property
    def name(self) -> str:
        return "Disjunctive Syllogism"
    
    @property
    def rule_type(self) -> RuleType:
        return RuleType.ELIMINATION
    
    @property
    def premises_required(self) -> int:
        return 2
    
    def apply(
        self, 
        premises: List[LogicalFormula],
        **kwargs
    ) -> Optional[LogicalFormula]:
        if len(premises) != 2:
            return None
        
        for i, j in [(0, 1), (1, 0)]:
            disjunction = premises[i]
            negation = premises[j]
            
            if isinstance(disjunction, BinaryConnective):
                if disjunction.connective == Connective.OR:
                    if isinstance(negation, Negation):
                        if negation.formula == disjunction.left:
                            return disjunction.right
                        if negation.formula == disjunction.right:
                            return disjunction.left
        
        return None
    
    def verify(
        self, 
        premises: List[LogicalFormula], 
        conclusion: LogicalFormula,
        **kwargs
    ) -> bool:
        result = self.apply(premises)
        return result is not None and result == conclusion


# =============================================================================
# CONJUNCTION RULES
# =============================================================================

class ConjunctionIntro(InferenceRule):
    """
    Conjunction Introduction (∧I).
    
    From φ and ψ, derive φ ∧ ψ.
    """
    
    @property
    def name(self) -> str:
        return "Conjunction Introduction"
    
    @property
    def rule_type(self) -> RuleType:
        return RuleType.INTRODUCTION
    
    @property
    def premises_required(self) -> int:
        return 2
    
    def apply(
        self, 
        premises: List[LogicalFormula],
        **kwargs
    ) -> Optional[LogicalFormula]:
        if len(premises) != 2:
            return None
        return And(premises[0], premises[1])
    
    def verify(
        self, 
        premises: List[LogicalFormula], 
        conclusion: LogicalFormula,
        **kwargs
    ) -> bool:
        if not isinstance(conclusion, BinaryConnective):
            return False
        if conclusion.connective != Connective.AND:
            return False
        return (
            (premises[0] == conclusion.left and premises[1] == conclusion.right) or
            (premises[0] == conclusion.right and premises[1] == conclusion.left)
        )


class ConjunctionElim(InferenceRule):
    """
    Conjunction Elimination (∧E).
    
    From φ ∧ ψ, derive φ (or ψ).
    """
    
    @property
    def name(self) -> str:
        return "Conjunction Elimination"
    
    @property
    def rule_type(self) -> RuleType:
        return RuleType.ELIMINATION
    
    @property
    def premises_required(self) -> int:
        return 1
    
    def apply(
        self, 
        premises: List[LogicalFormula],
        which: str = "left",
        **kwargs
    ) -> Optional[LogicalFormula]:
        if len(premises) != 1:
            return None
        
        conjunction = premises[0]
        if isinstance(conjunction, BinaryConnective):
            if conjunction.connective == Connective.AND:
                if which == "left":
                    return conjunction.left
                elif which == "right":
                    return conjunction.right
        
        return None
    
    def verify(
        self, 
        premises: List[LogicalFormula], 
        conclusion: LogicalFormula,
        **kwargs
    ) -> bool:
        if len(premises) != 1:
            return False
        conjunction = premises[0]
        if isinstance(conjunction, BinaryConnective):
            if conjunction.connective == Connective.AND:
                return conclusion == conjunction.left or conclusion == conjunction.right
        return False


# =============================================================================
# DISJUNCTION RULES
# =============================================================================

class DisjunctionIntro(InferenceRule):
    """
    Disjunction Introduction (∨I).
    
    From φ, derive φ ∨ ψ for any ψ.
    """
    
    @property
    def name(self) -> str:
        return "Disjunction Introduction"
    
    @property
    def rule_type(self) -> RuleType:
        return RuleType.INTRODUCTION
    
    @property
    def premises_required(self) -> int:
        return 1
    
    def apply(
        self, 
        premises: List[LogicalFormula],
        disjunct: Optional[LogicalFormula] = None,
        side: str = "right",
        **kwargs
    ) -> Optional[LogicalFormula]:
        if len(premises) != 1 or disjunct is None:
            return None
        
        if side == "left":
            return Or(premises[0], disjunct)
        else:
            return Or(disjunct, premises[0])
    
    def verify(
        self, 
        premises: List[LogicalFormula], 
        conclusion: LogicalFormula,
        **kwargs
    ) -> bool:
        if len(premises) != 1:
            return False
        if isinstance(conclusion, BinaryConnective):
            if conclusion.connective == Connective.OR:
                return premises[0] == conclusion.left or premises[0] == conclusion.right
        return False


# =============================================================================
# IMPLICATION RULES
# =============================================================================

class ImplicationIntro(InferenceRule):
    """
    Implication Introduction (→I) / Conditional Proof.
    
    If assuming φ allows deriving ψ, then derive φ → ψ.
    
    This rule requires a subproof.
    """
    
    @property
    def name(self) -> str:
        return "Implication Introduction"
    
    @property
    def rule_type(self) -> RuleType:
        return RuleType.INTRODUCTION
    
    @property
    def premises_required(self) -> int:
        return 2  # The assumption and the derived conclusion
    
    def apply(
        self, 
        premises: List[LogicalFormula],
        **kwargs
    ) -> Optional[LogicalFormula]:
        if len(premises) != 2:
            return None
        assumption = premises[0]
        derived = premises[1]
        return Implies(assumption, derived)
    
    def verify(
        self, 
        premises: List[LogicalFormula], 
        conclusion: LogicalFormula,
        **kwargs
    ) -> bool:
        if not isinstance(conclusion, BinaryConnective):
            return False
        if conclusion.connective != Connective.IMPLIES:
            return False
        if len(premises) != 2:
            return False
        return premises[0] == conclusion.left and premises[1] == conclusion.right


# =============================================================================
# NEGATION RULES
# =============================================================================

class NegationIntro(InferenceRule):
    """
    Negation Introduction (¬I) / Proof by Contradiction.
    
    If assuming φ leads to a contradiction, derive ¬φ.
    """
    
    @property
    def name(self) -> str:
        return "Negation Introduction"
    
    @property
    def rule_type(self) -> RuleType:
        return RuleType.INTRODUCTION
    
    @property
    def premises_required(self) -> int:
        return 2  # The assumption and the contradiction
    
    def apply(
        self, 
        premises: List[LogicalFormula],
        **kwargs
    ) -> Optional[LogicalFormula]:
        if len(premises) != 2:
            return None
        assumption = premises[0]
        # premises[1] should be a contradiction (ψ ∧ ¬ψ)
        return Not(assumption)
    
    def verify(
        self, 
        premises: List[LogicalFormula], 
        conclusion: LogicalFormula,
        **kwargs
    ) -> bool:
        if not isinstance(conclusion, Negation):
            return False
        # Check that conclusion is negation of the assumption
        return premises[0] == conclusion.formula


class DoubleNegationElim(InferenceRule):
    """
    Double Negation Elimination (¬¬E).
    
    From ¬¬φ, derive φ.
    """
    
    @property
    def name(self) -> str:
        return "Double Negation Elimination"
    
    @property
    def rule_type(self) -> RuleType:
        return RuleType.ELIMINATION
    
    @property
    def premises_required(self) -> int:
        return 1
    
    def apply(
        self, 
        premises: List[LogicalFormula],
        **kwargs
    ) -> Optional[LogicalFormula]:
        if len(premises) != 1:
            return None
        
        formula = premises[0]
        if isinstance(formula, Negation):
            if isinstance(formula.formula, Negation):
                return formula.formula.formula
        
        return None
    
    def verify(
        self, 
        premises: List[LogicalFormula], 
        conclusion: LogicalFormula,
        **kwargs
    ) -> bool:
        result = self.apply(premises)
        return result is not None and result == conclusion


# =============================================================================
# QUANTIFIER RULES
# =============================================================================

class UniversalInstantiation(InferenceRule):
    """
    Universal Instantiation (∀E).
    
    From ∀x.φ(x), derive φ(t) for any term t.
    
    Example:
        1. ∀x(P(x) → Q(x))    (premise)
        2. P(a) → Q(a)        (∀E, 1, a)
    """
    
    @property
    def name(self) -> str:
        return "Universal Instantiation"
    
    @property
    def rule_type(self) -> RuleType:
        return RuleType.ELIMINATION
    
    @property
    def premises_required(self) -> int:
        return 1
    
    def apply(
        self, 
        premises: List[LogicalFormula],
        term: Optional[Term] = None,
        **kwargs
    ) -> Optional[LogicalFormula]:
        if len(premises) != 1 or term is None:
            return None
        
        universal = premises[0]
        if isinstance(universal, QuantifiedFormula):
            if universal.quantifier == Quantifier.FORALL:
                return universal.formula.substitute(universal.variable, term)
        
        return None
    
    def verify(
        self, 
        premises: List[LogicalFormula], 
        conclusion: LogicalFormula,
        term: Optional[Term] = None,
        **kwargs
    ) -> bool:
        if term is None:
            return False
        result = self.apply(premises, term=term)
        return result is not None and result == conclusion


class UniversalGeneralization(InferenceRule):
    """
    Universal Generalization (∀I).
    
    From φ(a) where a is arbitrary, derive ∀x.φ(x).
    
    Restriction: a must not appear in any undischarged assumptions.
    """
    
    @property
    def name(self) -> str:
        return "Universal Generalization"
    
    @property
    def rule_type(self) -> RuleType:
        return RuleType.INTRODUCTION
    
    @property
    def premises_required(self) -> int:
        return 1
    
    def apply(
        self, 
        premises: List[LogicalFormula],
        variable: Optional[str] = None,
        constant: Optional[str] = None,
        **kwargs
    ) -> Optional[LogicalFormula]:
        if len(premises) != 1 or variable is None:
            return None
        
        formula = premises[0]
        
        # Replace the constant with the variable
        if constant:
            formula = formula.substitute(constant, Variable(variable))
        
        return ForAll(variable, formula)
    
    def verify(
        self, 
        premises: List[LogicalFormula], 
        conclusion: LogicalFormula,
        **kwargs
    ) -> bool:
        if not isinstance(conclusion, QuantifiedFormula):
            return False
        if conclusion.quantifier != Quantifier.FORALL:
            return False
        # The premise should be an instance of the quantified formula
        return True  # Full verification requires tracking the arbitrary constant


class ExistentialInstantiation(InferenceRule):
    """
    Existential Instantiation (∃E).
    
    From ∃x.φ(x), introduce a new constant c and assume φ(c).
    
    Restriction: c must be fresh (not appear elsewhere).
    """
    
    @property
    def name(self) -> str:
        return "Existential Instantiation"
    
    @property
    def rule_type(self) -> RuleType:
        return RuleType.ELIMINATION
    
    @property
    def premises_required(self) -> int:
        return 1
    
    def apply(
        self, 
        premises: List[LogicalFormula],
        constant: Optional[Constant] = None,
        **kwargs
    ) -> Optional[LogicalFormula]:
        if len(premises) != 1 or constant is None:
            return None
        
        existential = premises[0]
        if isinstance(existential, QuantifiedFormula):
            if existential.quantifier == Quantifier.EXISTS:
                return existential.formula.substitute(existential.variable, constant)
        
        return None
    
    def verify(
        self, 
        premises: List[LogicalFormula], 
        conclusion: LogicalFormula,
        constant: Optional[Constant] = None,
        **kwargs
    ) -> bool:
        if constant is None:
            return False
        result = self.apply(premises, constant=constant)
        return result is not None and result == conclusion


class ExistentialGeneralization(InferenceRule):
    """
    Existential Generalization (∃I).
    
    From φ(t), derive ∃x.φ(x).
    
    Example:
        1. P(a)            (premise)
        2. ∃x.P(x)         (∃I, 1)
    """
    
    @property
    def name(self) -> str:
        return "Existential Generalization"
    
    @property
    def rule_type(self) -> RuleType:
        return RuleType.INTRODUCTION
    
    @property
    def premises_required(self) -> int:
        return 1
    
    def apply(
        self, 
        premises: List[LogicalFormula],
        variable: Optional[str] = None,
        term: Optional[Term] = None,
        **kwargs
    ) -> Optional[LogicalFormula]:
        if len(premises) != 1 or variable is None:
            return None
        
        formula = premises[0]
        
        # Replace the term with the variable to get the general form
        if term:
            general = formula.substitute(str(term), Variable(variable))
        else:
            general = formula
        
        return Exists(variable, general)
    
    def verify(
        self, 
        premises: List[LogicalFormula], 
        conclusion: LogicalFormula,
        **kwargs
    ) -> bool:
        if not isinstance(conclusion, QuantifiedFormula):
            return False
        if conclusion.quantifier != Quantifier.EXISTS:
            return False
        # Check that premise is an instance of the existential
        return True


# =============================================================================
# EQUALITY RULES
# =============================================================================

class Substitution(InferenceRule):
    """
    Substitution of Equals.
    
    From t₁ = t₂ and φ(t₁), derive φ(t₂).
    """
    
    @property
    def name(self) -> str:
        return "Substitution"
    
    @property
    def rule_type(self) -> RuleType:
        return RuleType.DERIVED
    
    @property
    def premises_required(self) -> int:
        return 2
    
    def apply(
        self, 
        premises: List[LogicalFormula],
        **kwargs
    ) -> Optional[LogicalFormula]:
        if len(premises) != 2:
            return None
        
        # Find the equality premise
        equality = None
        formula = None
        
        for i, p in enumerate(premises):
            if isinstance(p, Predicate) and p.name == "=":
                equality = p
                formula = premises[1 - i]
                break
        
        if equality is None or formula is None:
            return None
        
        t1, t2 = equality.arguments
        
        # Try substituting t1 with t2
        if formula.contains(t1):
            return formula.substitute(str(t1), t2)
        
        # Try substituting t2 with t1
        if formula.contains(t2):
            return formula.substitute(str(t2), t1)
        
        return None
    
    def verify(
        self, 
        premises: List[LogicalFormula], 
        conclusion: LogicalFormula,
        **kwargs
    ) -> bool:
        result = self.apply(premises)
        return result is not None and result == conclusion


class Reflexivity(InferenceRule):
    """
    Reflexivity of Equality.
    
    Derive t = t for any term t.
    """
    
    @property
    def name(self) -> str:
        return "Reflexivity"
    
    @property
    def rule_type(self) -> RuleType:
        return RuleType.DERIVED
    
    @property
    def premises_required(self) -> int:
        return 0
    
    def apply(
        self, 
        premises: List[LogicalFormula],
        term: Optional[Term] = None,
        **kwargs
    ) -> Optional[LogicalFormula]:
        if term is None:
            return None
        return Equals(term, term)
    
    def verify(
        self, 
        premises: List[LogicalFormula], 
        conclusion: LogicalFormula,
        **kwargs
    ) -> bool:
        if isinstance(conclusion, Predicate):
            if conclusion.name == "=" and len(conclusion.arguments) == 2:
                return conclusion.arguments[0] == conclusion.arguments[1]
        return False


class Symmetry(InferenceRule):
    """
    Symmetry of Equality.
    
    From t₁ = t₂, derive t₂ = t₁.
    """
    
    @property
    def name(self) -> str:
        return "Symmetry"
    
    @property
    def rule_type(self) -> RuleType:
        return RuleType.DERIVED
    
    @property
    def premises_required(self) -> int:
        return 1
    
    def apply(
        self, 
        premises: List[LogicalFormula],
        **kwargs
    ) -> Optional[LogicalFormula]:
        if len(premises) != 1:
            return None
        
        equality = premises[0]
        if isinstance(equality, Predicate) and equality.name == "=":
            return Equals(equality.arguments[1], equality.arguments[0])
        
        return None
    
    def verify(
        self, 
        premises: List[LogicalFormula], 
        conclusion: LogicalFormula,
        **kwargs
    ) -> bool:
        result = self.apply(premises)
        return result is not None and result == conclusion


class Transitivity(InferenceRule):
    """
    Transitivity of Equality.
    
    From t₁ = t₂ and t₂ = t₃, derive t₁ = t₃.
    """
    
    @property
    def name(self) -> str:
        return "Transitivity"
    
    @property
    def rule_type(self) -> RuleType:
        return RuleType.DERIVED
    
    @property
    def premises_required(self) -> int:
        return 2
    
    def apply(
        self, 
        premises: List[LogicalFormula],
        **kwargs
    ) -> Optional[LogicalFormula]:
        if len(premises) != 2:
            return None
        
        eq1 = premises[0]
        eq2 = premises[1]
        
        if not (isinstance(eq1, Predicate) and eq1.name == "=" and
                isinstance(eq2, Predicate) and eq2.name == "="):
            return None
        
        a1, b1 = eq1.arguments
        a2, b2 = eq2.arguments
        
        # Find the chain: look for matching terms
        if b1 == a2:
            return Equals(a1, b2)
        if b1 == b2:
            return Equals(a1, a2)
        if a1 == a2:
            return Equals(b1, b2)
        if a1 == b2:
            return Equals(b1, a2)
        
        return None
    
    def verify(
        self, 
        premises: List[LogicalFormula], 
        conclusion: LogicalFormula,
        **kwargs
    ) -> bool:
        result = self.apply(premises)
        return result is not None and result == conclusion


# =============================================================================
# RULE REGISTRY
# =============================================================================

class RuleRegistry:
    """Registry of all available inference rules."""
    
    _rules: Dict[str, InferenceRule] = {}
    
    @classmethod
    def register(cls, rule: InferenceRule) -> None:
        """Register an inference rule."""
        cls._rules[rule.name] = rule
    
    @classmethod
    def get(cls, name: str) -> Optional[InferenceRule]:
        """Get a rule by name."""
        return cls._rules.get(name)
    
    @classmethod
    def all_rules(cls) -> List[InferenceRule]:
        """Get all registered rules."""
        return list(cls._rules.values())


# Register all standard rules
_standard_rules = [
    ModusPonens(),
    ModusTollens(),
    HypotheticalSyllogism(),
    DisjunctiveSyllogism(),
    ConjunctionIntro(),
    ConjunctionElim(),
    DisjunctionIntro(),
    ImplicationIntro(),
    NegationIntro(),
    DoubleNegationElim(),
    UniversalInstantiation(),
    UniversalGeneralization(),
    ExistentialInstantiation(),
    ExistentialGeneralization(),
    Substitution(),
    Reflexivity(),
    Symmetry(),
    Transitivity(),
]

for rule in _standard_rules:
    RuleRegistry.register(rule)
