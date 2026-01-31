"""
Formal Logic Types and Structures.

This module defines the fundamental logical structures for formal proofs:
- Propositional logic (AND, OR, NOT, IMPLIES, IFF)
- First-order logic (quantifiers, predicates, functions)
- Terms and formulas

These form the foundation for rigorous mathematical proof.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any, Dict, List, Optional, Set, Tuple, Union, FrozenSet,
    Iterator, Callable
)
import hashlib


class Connective(Enum):
    """Logical connectives for propositional and first-order logic."""
    NOT = auto()        # ¬ (negation)
    AND = auto()        # ∧ (conjunction)
    OR = auto()         # ∨ (disjunction)
    IMPLIES = auto()    # → (implication)
    IFF = auto()        # ↔ (biconditional)
    XOR = auto()        # ⊕ (exclusive or)


class Quantifier(Enum):
    """Quantifiers for first-order logic."""
    FORALL = auto()     # ∀ (universal)
    EXISTS = auto()     # ∃ (existential)
    UNIQUE = auto()     # ∃! (unique existence)


# Unicode symbols for pretty printing
CONNECTIVE_SYMBOLS = {
    Connective.NOT: "¬",
    Connective.AND: "∧",
    Connective.OR: "∨",
    Connective.IMPLIES: "→",
    Connective.IFF: "↔",
    Connective.XOR: "⊕",
}

QUANTIFIER_SYMBOLS = {
    Quantifier.FORALL: "∀",
    Quantifier.EXISTS: "∃",
    Quantifier.UNIQUE: "∃!",
}


class LogicalExpression(ABC):
    """Base class for all logical expressions."""
    
    @abstractmethod
    def __str__(self) -> str:
        """Human-readable string representation."""
        pass
    
    @abstractmethod
    def __eq__(self, other: object) -> bool:
        """Structural equality."""
        pass
    
    @abstractmethod
    def __hash__(self) -> int:
        """Hash for use in sets and dicts."""
        pass
    
    @abstractmethod
    def free_variables(self) -> Set[str]:
        """Return set of free (unbound) variables."""
        pass
    
    @abstractmethod
    def substitute(self, var: str, term: 'Term') -> 'LogicalExpression':
        """Substitute a term for a variable."""
        pass
    
    @abstractmethod
    def contains(self, expr: 'LogicalExpression') -> bool:
        """Check if expression contains another expression."""
        pass


# =============================================================================
# TERMS (objects in the domain)
# =============================================================================

@dataclass(frozen=True)
class Variable(LogicalExpression):
    """
    A variable representing an arbitrary element of the domain.
    
    Example: x, y, z in ∀x(P(x) → Q(x))
    """
    name: str
    
    def __str__(self) -> str:
        return self.name
    
    def __eq__(self, other: object) -> bool:
        return isinstance(other, Variable) and self.name == other.name
    
    def __hash__(self) -> int:
        return hash(("Variable", self.name))
    
    def free_variables(self) -> Set[str]:
        return {self.name}
    
    def substitute(self, var: str, term: 'Term') -> 'Term':
        if self.name == var:
            return term
        return self
    
    def contains(self, expr: LogicalExpression) -> bool:
        return self == expr


@dataclass(frozen=True)
class Constant(LogicalExpression):
    """
    A constant representing a specific element of the domain.
    
    Example: 0, 1, π, e
    """
    name: str
    value: Optional[Any] = None
    
    def __str__(self) -> str:
        return self.name
    
    def __eq__(self, other: object) -> bool:
        return isinstance(other, Constant) and self.name == other.name
    
    def __hash__(self) -> int:
        return hash(("Constant", self.name))
    
    def free_variables(self) -> Set[str]:
        return set()
    
    def substitute(self, var: str, term: 'Term') -> 'Term':
        return self  # Constants don't change
    
    def contains(self, expr: LogicalExpression) -> bool:
        return self == expr


@dataclass(frozen=True)
class Function(LogicalExpression):
    """
    A function applied to terms.
    
    Example: f(x), g(x, y), successor(n), plus(a, b)
    """
    name: str
    arguments: Tuple['Term', ...]
    
    def __str__(self) -> str:
        if not self.arguments:
            return f"{self.name}()"
        args = ", ".join(str(arg) for arg in self.arguments)
        return f"{self.name}({args})"
    
    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Function) and 
            self.name == other.name and 
            self.arguments == other.arguments
        )
    
    def __hash__(self) -> int:
        return hash(("Function", self.name, self.arguments))
    
    def free_variables(self) -> Set[str]:
        result: Set[str] = set()
        for arg in self.arguments:
            result |= arg.free_variables()
        return result
    
    def substitute(self, var: str, term: 'Term') -> 'Function':
        new_args = tuple(arg.substitute(var, term) for arg in self.arguments)
        return Function(self.name, new_args)
    
    def contains(self, expr: LogicalExpression) -> bool:
        if self == expr:
            return True
        return any(arg.contains(expr) for arg in self.arguments)


# Term is either a Variable, Constant, or Function
Term = Union[Variable, Constant, Function]


# =============================================================================
# FORMULAS (logical statements)
# =============================================================================

@dataclass(frozen=True)
class Proposition(LogicalExpression):
    """
    An atomic proposition (0-ary predicate).
    
    Example: P, Q, "it is raining"
    """
    name: str
    
    def __str__(self) -> str:
        return self.name
    
    def __eq__(self, other: object) -> bool:
        return isinstance(other, Proposition) and self.name == other.name
    
    def __hash__(self) -> int:
        return hash(("Proposition", self.name))
    
    def free_variables(self) -> Set[str]:
        return set()
    
    def substitute(self, var: str, term: Term) -> 'Proposition':
        return self  # Propositions have no variables
    
    def contains(self, expr: LogicalExpression) -> bool:
        return self == expr


@dataclass(frozen=True)
class Predicate(LogicalExpression):
    """
    A predicate applied to terms.
    
    Example: P(x), Q(x, y), x < y, x ∈ S
    """
    name: str
    arguments: Tuple[Term, ...]
    
    def __str__(self) -> str:
        if not self.arguments:
            return self.name
        # Special infix notation for common predicates
        if self.name == "=" and len(self.arguments) == 2:
            return f"{self.arguments[0]} = {self.arguments[1]}"
        if self.name == "<" and len(self.arguments) == 2:
            return f"{self.arguments[0]} < {self.arguments[1]}"
        if self.name == ">" and len(self.arguments) == 2:
            return f"{self.arguments[0]} > {self.arguments[1]}"
        if self.name == "≤" and len(self.arguments) == 2:
            return f"{self.arguments[0]} ≤ {self.arguments[1]}"
        if self.name == "≥" and len(self.arguments) == 2:
            return f"{self.arguments[0]} ≥ {self.arguments[1]}"
        if self.name == "∈" and len(self.arguments) == 2:
            return f"{self.arguments[0]} ∈ {self.arguments[1]}"
        
        args = ", ".join(str(arg) for arg in self.arguments)
        return f"{self.name}({args})"
    
    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Predicate) and
            self.name == other.name and
            self.arguments == other.arguments
        )
    
    def __hash__(self) -> int:
        return hash(("Predicate", self.name, self.arguments))
    
    def free_variables(self) -> Set[str]:
        result: Set[str] = set()
        for arg in self.arguments:
            result |= arg.free_variables()
        return result
    
    def substitute(self, var: str, term: Term) -> 'Predicate':
        new_args = tuple(arg.substitute(var, term) for arg in self.arguments)
        return Predicate(self.name, new_args)
    
    def contains(self, expr: LogicalExpression) -> bool:
        if self == expr:
            return True
        return any(arg.contains(expr) for arg in self.arguments)


@dataclass(frozen=True)
class Negation(LogicalExpression):
    """
    Negation of a formula: ¬φ
    
    Example: ¬P, ¬(P ∧ Q)
    """
    formula: 'LogicalFormula'
    
    def __str__(self) -> str:
        if isinstance(self.formula, (Proposition, Predicate)):
            return f"¬{self.formula}"
        return f"¬({self.formula})"
    
    def __eq__(self, other: object) -> bool:
        return isinstance(other, Negation) and self.formula == other.formula
    
    def __hash__(self) -> int:
        return hash(("Negation", self.formula))
    
    def free_variables(self) -> Set[str]:
        return self.formula.free_variables()
    
    def substitute(self, var: str, term: Term) -> 'Negation':
        return Negation(self.formula.substitute(var, term))
    
    def contains(self, expr: LogicalExpression) -> bool:
        return self == expr or self.formula.contains(expr)


@dataclass(frozen=True)
class BinaryConnective(LogicalExpression):
    """
    Binary connective joining two formulas.
    
    Example: P ∧ Q, P → Q, P ↔ Q
    """
    connective: Connective
    left: 'LogicalFormula'
    right: 'LogicalFormula'
    
    def __str__(self) -> str:
        symbol = CONNECTIVE_SYMBOLS[self.connective]
        
        # Add parentheses for clarity
        left_str = str(self.left)
        right_str = str(self.right)
        
        if isinstance(self.left, BinaryConnective):
            left_str = f"({left_str})"
        if isinstance(self.right, BinaryConnective):
            right_str = f"({right_str})"
        
        return f"{left_str} {symbol} {right_str}"
    
    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, BinaryConnective) and
            self.connective == other.connective and
            self.left == other.left and
            self.right == other.right
        )
    
    def __hash__(self) -> int:
        return hash(("BinaryConnective", self.connective, self.left, self.right))
    
    def free_variables(self) -> Set[str]:
        return self.left.free_variables() | self.right.free_variables()
    
    def substitute(self, var: str, term: Term) -> 'BinaryConnective':
        return BinaryConnective(
            self.connective,
            self.left.substitute(var, term),
            self.right.substitute(var, term)
        )
    
    def contains(self, expr: LogicalExpression) -> bool:
        return (
            self == expr or 
            self.left.contains(expr) or 
            self.right.contains(expr)
        )


@dataclass(frozen=True)
class QuantifiedFormula(LogicalExpression):
    """
    A quantified formula: ∀x.φ or ∃x.φ
    
    Example: ∀x(P(x) → Q(x)), ∃x(x > 0)
    """
    quantifier: Quantifier
    variable: str
    formula: 'LogicalFormula'
    
    def __str__(self) -> str:
        symbol = QUANTIFIER_SYMBOLS[self.quantifier]
        return f"{symbol}{self.variable}({self.formula})"
    
    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, QuantifiedFormula) and
            self.quantifier == other.quantifier and
            self.variable == other.variable and
            self.formula == other.formula
        )
    
    def __hash__(self) -> int:
        return hash(("QuantifiedFormula", self.quantifier, self.variable, self.formula))
    
    def free_variables(self) -> Set[str]:
        return self.formula.free_variables() - {self.variable}
    
    def substitute(self, var: str, term: Term) -> 'QuantifiedFormula':
        # Don't substitute bound variables
        if var == self.variable:
            return self
        
        # Check for variable capture
        if self.variable in term.free_variables():
            # Need to rename bound variable (alpha conversion)
            new_var = self._fresh_variable(term.free_variables())
            renamed_formula = self.formula.substitute(
                self.variable, Variable(new_var)
            )
            return QuantifiedFormula(
                self.quantifier,
                new_var,
                renamed_formula.substitute(var, term)
            )
        
        return QuantifiedFormula(
            self.quantifier,
            self.variable,
            self.formula.substitute(var, term)
        )
    
    def _fresh_variable(self, avoid: Set[str]) -> str:
        """Generate a fresh variable name not in avoid set."""
        base = self.variable.rstrip("0123456789")
        i = 1
        while f"{base}{i}" in avoid:
            i += 1
        return f"{base}{i}"
    
    def contains(self, expr: LogicalExpression) -> bool:
        return self == expr or self.formula.contains(expr)


# LogicalFormula is any formula type
LogicalFormula = Union[
    Proposition, 
    Predicate, 
    Negation, 
    BinaryConnective, 
    QuantifiedFormula
]


# =============================================================================
# HELPER FUNCTIONS FOR BUILDING FORMULAS
# =============================================================================

def And(left: LogicalFormula, right: LogicalFormula) -> BinaryConnective:
    """Conjunction: φ ∧ ψ"""
    return BinaryConnective(Connective.AND, left, right)


def Or(left: LogicalFormula, right: LogicalFormula) -> BinaryConnective:
    """Disjunction: φ ∨ ψ"""
    return BinaryConnective(Connective.OR, left, right)


def Implies(left: LogicalFormula, right: LogicalFormula) -> BinaryConnective:
    """Implication: φ → ψ"""
    return BinaryConnective(Connective.IMPLIES, left, right)


def Iff(left: LogicalFormula, right: LogicalFormula) -> BinaryConnective:
    """Biconditional: φ ↔ ψ"""
    return BinaryConnective(Connective.IFF, left, right)


def Not(formula: LogicalFormula) -> Negation:
    """Negation: ¬φ"""
    return Negation(formula)


def ForAll(var: str, formula: LogicalFormula) -> QuantifiedFormula:
    """Universal quantification: ∀x.φ"""
    return QuantifiedFormula(Quantifier.FORALL, var, formula)


def Exists(var: str, formula: LogicalFormula) -> QuantifiedFormula:
    """Existential quantification: ∃x.φ"""
    return QuantifiedFormula(Quantifier.EXISTS, var, formula)


def ExistsUnique(var: str, formula: LogicalFormula) -> QuantifiedFormula:
    """Unique existence: ∃!x.φ"""
    return QuantifiedFormula(Quantifier.UNIQUE, var, formula)


def Equals(left: Term, right: Term) -> Predicate:
    """Equality predicate: t₁ = t₂"""
    return Predicate("=", (left, right))


def LessThan(left: Term, right: Term) -> Predicate:
    """Less than predicate: t₁ < t₂"""
    return Predicate("<", (left, right))


def ElementOf(element: Term, set_term: Term) -> Predicate:
    """Set membership: x ∈ S"""
    return Predicate("∈", (element, set_term))


# =============================================================================
# TRUTH VALUES AND EVALUATION
# =============================================================================

@dataclass
class TruthAssignment:
    """Assignment of truth values to propositions."""
    values: Dict[str, bool] = field(default_factory=dict)
    
    def __getitem__(self, prop: str) -> bool:
        return self.values.get(prop, False)
    
    def __setitem__(self, prop: str, value: bool) -> None:
        self.values[prop] = value


def evaluate_propositional(
    formula: LogicalFormula, 
    assignment: TruthAssignment
) -> bool:
    """
    Evaluate a propositional formula under a truth assignment.
    
    Args:
        formula: The formula to evaluate
        assignment: Truth values for atomic propositions
        
    Returns:
        True if formula is true under assignment, False otherwise
    """
    if isinstance(formula, Proposition):
        return assignment[formula.name]
    
    elif isinstance(formula, Negation):
        return not evaluate_propositional(formula.formula, assignment)
    
    elif isinstance(formula, BinaryConnective):
        left = evaluate_propositional(formula.left, assignment)
        right = evaluate_propositional(formula.right, assignment)
        
        if formula.connective == Connective.AND:
            return left and right
        elif formula.connective == Connective.OR:
            return left or right
        elif formula.connective == Connective.IMPLIES:
            return (not left) or right
        elif formula.connective == Connective.IFF:
            return left == right
        elif formula.connective == Connective.XOR:
            return left != right
    
    raise ValueError(f"Cannot evaluate {type(formula)} in propositional logic")


def is_tautology(formula: LogicalFormula) -> bool:
    """
    Check if a propositional formula is a tautology (always true).
    
    Uses truth table method - checks all possible assignments.
    """
    props = _get_propositions(formula)
    
    # Check all 2^n assignments
    for i in range(2 ** len(props)):
        assignment = TruthAssignment()
        for j, prop in enumerate(props):
            assignment[prop] = bool((i >> j) & 1)
        
        if not evaluate_propositional(formula, assignment):
            return False
    
    return True


def is_contradiction(formula: LogicalFormula) -> bool:
    """Check if a propositional formula is a contradiction (always false)."""
    return is_tautology(Not(formula))


def is_satisfiable(formula: LogicalFormula) -> bool:
    """Check if a propositional formula is satisfiable (true for some assignment)."""
    return not is_contradiction(formula)


def _get_propositions(formula: LogicalFormula) -> List[str]:
    """Get all proposition names in a formula."""
    props: Set[str] = set()
    
    def collect(f: LogicalFormula) -> None:
        if isinstance(f, Proposition):
            props.add(f.name)
        elif isinstance(f, Negation):
            collect(f.formula)
        elif isinstance(f, BinaryConnective):
            collect(f.left)
            collect(f.right)
        elif isinstance(f, QuantifiedFormula):
            collect(f.formula)
    
    collect(formula)
    return sorted(props)


# =============================================================================
# LOGICAL EQUIVALENCES
# =============================================================================

def logically_equivalent(f1: LogicalFormula, f2: LogicalFormula) -> bool:
    """
    Check if two propositional formulas are logically equivalent.
    
    Two formulas are equivalent if they have the same truth value
    under all possible assignments.
    """
    return is_tautology(Iff(f1, f2))


def to_nnf(formula: LogicalFormula) -> LogicalFormula:
    """
    Convert formula to Negation Normal Form (NNF).
    
    In NNF, negations only appear directly before atomic formulas.
    """
    if isinstance(formula, (Proposition, Predicate)):
        return formula
    
    elif isinstance(formula, Negation):
        inner = formula.formula
        
        if isinstance(inner, (Proposition, Predicate)):
            return formula  # Already in NNF
        
        elif isinstance(inner, Negation):
            # ¬¬φ → φ (double negation)
            return to_nnf(inner.formula)
        
        elif isinstance(inner, BinaryConnective):
            if inner.connective == Connective.AND:
                # ¬(φ ∧ ψ) → ¬φ ∨ ¬ψ (De Morgan)
                return Or(to_nnf(Not(inner.left)), to_nnf(Not(inner.right)))
            elif inner.connective == Connective.OR:
                # ¬(φ ∨ ψ) → ¬φ ∧ ¬ψ (De Morgan)
                return And(to_nnf(Not(inner.left)), to_nnf(Not(inner.right)))
            elif inner.connective == Connective.IMPLIES:
                # ¬(φ → ψ) → φ ∧ ¬ψ
                return And(to_nnf(inner.left), to_nnf(Not(inner.right)))
            elif inner.connective == Connective.IFF:
                # ¬(φ ↔ ψ) → (φ ∧ ¬ψ) ∨ (¬φ ∧ ψ)
                return Or(
                    And(to_nnf(inner.left), to_nnf(Not(inner.right))),
                    And(to_nnf(Not(inner.left)), to_nnf(inner.right))
                )
        
        elif isinstance(inner, QuantifiedFormula):
            if inner.quantifier == Quantifier.FORALL:
                # ¬∀x.φ → ∃x.¬φ
                return Exists(inner.variable, to_nnf(Not(inner.formula)))
            elif inner.quantifier == Quantifier.EXISTS:
                # ¬∃x.φ → ∀x.¬φ
                return ForAll(inner.variable, to_nnf(Not(inner.formula)))
    
    elif isinstance(formula, BinaryConnective):
        if formula.connective == Connective.IMPLIES:
            # φ → ψ → ¬φ ∨ ψ
            return Or(to_nnf(Not(formula.left)), to_nnf(formula.right))
        elif formula.connective == Connective.IFF:
            # φ ↔ ψ → (φ ∧ ψ) ∨ (¬φ ∧ ¬ψ)
            return Or(
                And(to_nnf(formula.left), to_nnf(formula.right)),
                And(to_nnf(Not(formula.left)), to_nnf(Not(formula.right)))
            )
        else:
            return BinaryConnective(
                formula.connective,
                to_nnf(formula.left),
                to_nnf(formula.right)
            )
    
    elif isinstance(formula, QuantifiedFormula):
        return QuantifiedFormula(
            formula.quantifier,
            formula.variable,
            to_nnf(formula.formula)
        )
    
    return formula
