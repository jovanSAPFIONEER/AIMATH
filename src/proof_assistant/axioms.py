"""
Axiom Systems for Formal Proofs.

This module defines standard axiom systems:
- Propositional logic axioms (Hilbert-style)
- First-order logic axioms
- Peano axioms for natural numbers
- Basic set theory axioms (ZF-style)

Axioms are the foundational truths from which theorems are derived.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Callable
from enum import Enum, auto

from .logic import (
    LogicalFormula, Proposition, Predicate, Negation, BinaryConnective,
    QuantifiedFormula, Variable, Constant, Function, Term,
    Connective, Quantifier,
    And, Or, Implies, Iff, Not, ForAll, Exists, Equals,
    is_tautology
)


class AxiomType(Enum):
    """Types of axioms."""
    LOGICAL = auto()      # Pure logical axioms
    MATHEMATICAL = auto()  # Mathematical axioms (Peano, etc.)
    SET_THEORETIC = auto() # Set theory axioms
    EQUALITY = auto()      # Axioms about equality
    CUSTOM = auto()        # User-defined axioms


@dataclass(frozen=True)
class Axiom:
    """
    An axiom - a statement assumed to be true without proof.
    
    Axioms are the foundation of mathematical reasoning.
    All theorems must ultimately be derived from axioms.
    """
    name: str
    formula: LogicalFormula
    axiom_type: AxiomType
    description: str = ""
    
    def __str__(self) -> str:
        return f"{self.name}: {self.formula}"


@dataclass
class AxiomSchema:
    """
    An axiom schema - a template that generates infinitely many axioms.
    
    Example: P → (Q → P) is a schema where P, Q can be any formulas.
    """
    name: str
    template: Callable[..., LogicalFormula]
    description: str
    axiom_type: AxiomType
    
    def instantiate(self, *args: LogicalFormula) -> Axiom:
        """Create a specific axiom from this schema."""
        formula = self.template(*args)
        return Axiom(
            name=f"{self.name}_instance",
            formula=formula,
            axiom_type=self.axiom_type,
            description=f"Instance of {self.name}"
        )


class AxiomSystem(ABC):
    """Base class for axiom systems."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of this axiom system."""
        pass
    
    @property
    @abstractmethod
    def axioms(self) -> List[Axiom]:
        """List of axioms in this system."""
        pass
    
    @property
    @abstractmethod
    def schemas(self) -> List[AxiomSchema]:
        """List of axiom schemas in this system."""
        pass
    
    def is_axiom(self, formula: LogicalFormula) -> bool:
        """Check if a formula is an axiom (or instance of a schema)."""
        # Check direct axioms
        for axiom in self.axioms:
            if axiom.formula == formula:
                return True
        
        # Check if it's an instance of any schema
        # This is more complex and depends on the specific system
        return False
    
    def get_axiom(self, name: str) -> Optional[Axiom]:
        """Get axiom by name."""
        for axiom in self.axioms:
            if axiom.name == name:
                return axiom
        return None


# =============================================================================
# PROPOSITIONAL LOGIC AXIOMS (Hilbert System)
# =============================================================================

class PropositionalAxioms(AxiomSystem):
    """
    Hilbert-style axiom system for propositional logic.
    
    Uses three axiom schemas and Modus Ponens as the only inference rule.
    This system is complete for propositional logic.
    """
    
    @property
    def name(self) -> str:
        return "Propositional Logic (Hilbert)"
    
    @property
    def axioms(self) -> List[Axiom]:
        return []  # All axioms come from schemas
    
    @property
    def schemas(self) -> List[AxiomSchema]:
        return [
            AxiomSchema(
                name="PROP1",
                template=lambda P, Q: Implies(P, Implies(Q, P)),
                description="φ → (ψ → φ) [Simplification]",
                axiom_type=AxiomType.LOGICAL
            ),
            AxiomSchema(
                name="PROP2",
                template=lambda P, Q, R: Implies(
                    Implies(P, Implies(Q, R)),
                    Implies(Implies(P, Q), Implies(P, R))
                ),
                description="(φ → (ψ → χ)) → ((φ → ψ) → (φ → χ)) [Distribution]",
                axiom_type=AxiomType.LOGICAL
            ),
            AxiomSchema(
                name="PROP3",
                template=lambda P, Q: Implies(
                    Implies(Not(P), Not(Q)),
                    Implies(Q, P)
                ),
                description="(¬φ → ¬ψ) → (ψ → φ) [Contraposition]",
                axiom_type=AxiomType.LOGICAL
            ),
        ]
    
    def is_axiom_instance(self, formula: LogicalFormula) -> Optional[str]:
        """
        Check if formula is an instance of a propositional axiom schema.
        Returns the schema name if so, None otherwise.
        """
        # Check PROP1: φ → (ψ → φ)
        if isinstance(formula, BinaryConnective):
            if formula.connective == Connective.IMPLIES:
                if isinstance(formula.right, BinaryConnective):
                    if formula.right.connective == Connective.IMPLIES:
                        if formula.left == formula.right.right:
                            return "PROP1"
        
        # Check PROP3: (¬φ → ¬ψ) → (ψ → φ)
        if isinstance(formula, BinaryConnective):
            if formula.connective == Connective.IMPLIES:
                left = formula.left
                right = formula.right
                if (isinstance(left, BinaryConnective) and 
                    left.connective == Connective.IMPLIES and
                    isinstance(left.left, Negation) and
                    isinstance(left.right, Negation) and
                    isinstance(right, BinaryConnective) and
                    right.connective == Connective.IMPLIES):
                    # Check: left.left = ¬φ, left.right = ¬ψ
                    # Check: right.left = ψ, right.right = φ
                    phi = left.left.formula
                    psi = left.right.formula
                    if right.left == psi and right.right == phi:
                        return "PROP3"
        
        # PROP2 is more complex to check pattern-match
        # For now, verify using tautology checking
        if is_tautology(formula):
            return "TAUTOLOGY"
        
        return None


# =============================================================================
# FIRST-ORDER LOGIC AXIOMS
# =============================================================================

class FirstOrderAxioms(AxiomSystem):
    """
    Axiom system for first-order predicate logic.
    
    Extends propositional logic with quantifier axioms.
    """
    
    def __init__(self):
        self._prop_axioms = PropositionalAxioms()
    
    @property
    def name(self) -> str:
        return "First-Order Logic"
    
    @property
    def axioms(self) -> List[Axiom]:
        return []
    
    @property
    def schemas(self) -> List[AxiomSchema]:
        prop_schemas = self._prop_axioms.schemas
        fol_schemas = [
            AxiomSchema(
                name="FOL1",
                template=self._fol1_template,
                description="∀x.φ(x) → φ(t) [Universal Instantiation]",
                axiom_type=AxiomType.LOGICAL
            ),
            AxiomSchema(
                name="FOL2",
                template=self._fol2_template,
                description="φ(t) → ∃x.φ(x) [Existential Generalization]",
                axiom_type=AxiomType.LOGICAL
            ),
            AxiomSchema(
                name="FOL3",
                template=self._fol3_template,
                description="∀x.(φ → ψ) → (∀x.φ → ∀x.ψ) [Universal Distribution]",
                axiom_type=AxiomType.LOGICAL
            ),
            AxiomSchema(
                name="FOL4",
                template=self._fol4_template,
                description="φ → ∀x.φ [Generalization, x not free in φ]",
                axiom_type=AxiomType.LOGICAL
            ),
        ]
        return prop_schemas + fol_schemas
    
    def _fol1_template(
        self, 
        var: str, 
        formula: LogicalFormula, 
        term: Term
    ) -> LogicalFormula:
        """∀x.φ(x) → φ(t)"""
        universal = ForAll(var, formula)
        instantiated = formula.substitute(var, term)
        return Implies(universal, instantiated)
    
    def _fol2_template(
        self, 
        var: str, 
        formula: LogicalFormula, 
        term: Term
    ) -> LogicalFormula:
        """φ(t) → ∃x.φ(x)"""
        instantiated = formula.substitute(var, term)
        existential = Exists(var, formula)
        return Implies(instantiated, existential)
    
    def _fol3_template(
        self, 
        var: str, 
        phi: LogicalFormula, 
        psi: LogicalFormula
    ) -> LogicalFormula:
        """∀x.(φ → ψ) → (∀x.φ → ∀x.ψ)"""
        return Implies(
            ForAll(var, Implies(phi, psi)),
            Implies(ForAll(var, phi), ForAll(var, psi))
        )
    
    def _fol4_template(
        self, 
        var: str, 
        phi: LogicalFormula
    ) -> LogicalFormula:
        """φ → ∀x.φ (when x not free in φ)"""
        return Implies(phi, ForAll(var, phi))


# =============================================================================
# EQUALITY AXIOMS
# =============================================================================

class EqualityAxioms(AxiomSystem):
    """
    Axioms for equality.
    
    These define the behavior of the = predicate.
    """
    
    @property
    def name(self) -> str:
        return "Equality"
    
    @property
    def axioms(self) -> List[Axiom]:
        x = Variable("x")
        y = Variable("y")
        z = Variable("z")
        
        return [
            Axiom(
                name="EQ_REFL",
                formula=ForAll("x", Equals(x, x)),
                axiom_type=AxiomType.EQUALITY,
                description="Reflexivity: ∀x(x = x)"
            ),
            Axiom(
                name="EQ_SYM",
                formula=ForAll("x", ForAll("y", 
                    Implies(Equals(x, y), Equals(y, x))
                )),
                axiom_type=AxiomType.EQUALITY,
                description="Symmetry: ∀x∀y(x = y → y = x)"
            ),
            Axiom(
                name="EQ_TRANS",
                formula=ForAll("x", ForAll("y", ForAll("z",
                    Implies(
                        And(Equals(x, y), Equals(y, z)),
                        Equals(x, z)
                    )
                ))),
                axiom_type=AxiomType.EQUALITY,
                description="Transitivity: ∀x∀y∀z((x = y ∧ y = z) → x = z)"
            ),
        ]
    
    @property
    def schemas(self) -> List[AxiomSchema]:
        return [
            AxiomSchema(
                name="EQ_SUBST",
                template=self._substitution_template,
                description="Substitution: x = y → (φ(x) → φ(y))",
                axiom_type=AxiomType.EQUALITY
            ),
        ]
    
    def _substitution_template(
        self, 
        x: Variable, 
        y: Variable, 
        phi: LogicalFormula
    ) -> LogicalFormula:
        """x = y → (φ(x) → φ(y))"""
        phi_y = phi.substitute(x.name, y)
        return Implies(
            Equals(x, y),
            Implies(phi, phi_y)
        )


# =============================================================================
# PEANO AXIOMS FOR NATURAL NUMBERS
# =============================================================================

class PeanoAxioms(AxiomSystem):
    """
    Peano axioms for natural numbers.
    
    These axioms define the natural numbers (0, 1, 2, ...) and
    provide the foundation for arithmetic.
    """
    
    def __init__(self):
        # Define the signature
        self.zero = Constant("0")
        self.succ = lambda t: Function("S", (t,))  # Successor function
    
    @property
    def name(self) -> str:
        return "Peano Arithmetic"
    
    @property
    def axioms(self) -> List[Axiom]:
        x = Variable("x")
        y = Variable("y")
        zero = self.zero
        Sx = self.succ(x)
        Sy = self.succ(y)
        
        return [
            Axiom(
                name="PA1",
                formula=ForAll("x", Not(Equals(Sx, zero))),
                axiom_type=AxiomType.MATHEMATICAL,
                description="Zero is not a successor: ∀x(S(x) ≠ 0)"
            ),
            Axiom(
                name="PA2",
                formula=ForAll("x", ForAll("y",
                    Implies(Equals(Sx, Sy), Equals(x, y))
                )),
                axiom_type=AxiomType.MATHEMATICAL,
                description="Successor is injective: ∀x∀y(S(x) = S(y) → x = y)"
            ),
        ]
    
    @property
    def schemas(self) -> List[AxiomSchema]:
        return [
            AxiomSchema(
                name="PA_INDUCTION",
                template=self._induction_template,
                description="Induction: (φ(0) ∧ ∀x(φ(x) → φ(S(x)))) → ∀x.φ(x)",
                axiom_type=AxiomType.MATHEMATICAL
            ),
        ]
    
    def _induction_template(self, phi: LogicalFormula) -> LogicalFormula:
        """
        Mathematical induction schema.
        
        (φ(0) ∧ ∀x(φ(x) → φ(S(x)))) → ∀x.φ(x)
        """
        x = Variable("x")
        zero = self.zero
        Sx = self.succ(x)
        
        base_case = phi.substitute("x", zero)
        inductive_step = ForAll("x", Implies(phi, phi.substitute("x", Sx)))
        conclusion = ForAll("x", phi)
        
        return Implies(And(base_case, inductive_step), conclusion)
    
    def addition_axioms(self) -> List[Axiom]:
        """Axioms defining addition."""
        x = Variable("x")
        y = Variable("y")
        zero = self.zero
        
        # x + 0 = x
        plus_zero = Function("+", (x, zero))
        # x + S(y) = S(x + y)
        plus_succ = Function("+", (x, self.succ(y)))
        sum_xy = Function("+", (x, y))
        
        return [
            Axiom(
                name="ADD1",
                formula=ForAll("x", Equals(plus_zero, x)),
                axiom_type=AxiomType.MATHEMATICAL,
                description="x + 0 = x"
            ),
            Axiom(
                name="ADD2",
                formula=ForAll("x", ForAll("y",
                    Equals(plus_succ, self.succ(sum_xy))
                )),
                axiom_type=AxiomType.MATHEMATICAL,
                description="x + S(y) = S(x + y)"
            ),
        ]
    
    def multiplication_axioms(self) -> List[Axiom]:
        """Axioms defining multiplication."""
        x = Variable("x")
        y = Variable("y")
        zero = self.zero
        
        # x * 0 = 0
        times_zero = Function("*", (x, zero))
        # x * S(y) = x * y + x
        times_succ = Function("*", (x, self.succ(y)))
        prod_xy = Function("*", (x, y))
        sum_prod_x = Function("+", (prod_xy, x))
        
        return [
            Axiom(
                name="MUL1",
                formula=ForAll("x", Equals(times_zero, zero)),
                axiom_type=AxiomType.MATHEMATICAL,
                description="x * 0 = 0"
            ),
            Axiom(
                name="MUL2",
                formula=ForAll("x", ForAll("y",
                    Equals(times_succ, sum_prod_x)
                )),
                axiom_type=AxiomType.MATHEMATICAL,
                description="x * S(y) = x * y + x"
            ),
        ]


# =============================================================================
# SET THEORY AXIOMS (ZF-style)
# =============================================================================

class SetTheoryAxioms(AxiomSystem):
    """
    Zermelo-Fraenkel set theory axioms (without Choice).
    
    These axioms form the foundation of modern mathematics.
    """
    
    @property
    def name(self) -> str:
        return "Zermelo-Fraenkel Set Theory"
    
    @property
    def axioms(self) -> List[Axiom]:
        x = Variable("x")
        y = Variable("y")
        z = Variable("z")
        A = Variable("A")
        B = Variable("B")
        
        return [
            Axiom(
                name="ZF_EXT",
                formula=ForAll("A", ForAll("B",
                    Implies(
                        ForAll("x", Iff(
                            Predicate("∈", (x, A)),
                            Predicate("∈", (x, B))
                        )),
                        Equals(A, B)
                    )
                )),
                axiom_type=AxiomType.SET_THEORETIC,
                description="Extensionality: Sets with same elements are equal"
            ),
            Axiom(
                name="ZF_EMPTY",
                formula=Exists("A", ForAll("x", Not(Predicate("∈", (x, A))))),
                axiom_type=AxiomType.SET_THEORETIC,
                description="Empty Set: There exists a set with no elements"
            ),
            Axiom(
                name="ZF_PAIR",
                formula=ForAll("x", ForAll("y",
                    Exists("A", ForAll("z",
                        Iff(
                            Predicate("∈", (z, A)),
                            Or(Equals(z, x), Equals(z, y))
                        )
                    ))
                )),
                axiom_type=AxiomType.SET_THEORETIC,
                description="Pairing: For any x, y there exists {x, y}"
            ),
            Axiom(
                name="ZF_UNION",
                formula=ForAll("A",
                    Exists("B", ForAll("x",
                        Iff(
                            Predicate("∈", (x, B)),
                            Exists("C", And(
                                Predicate("∈", (x, C)),
                                Predicate("∈", (C, A))
                            ))
                        )
                    ))
                ),
                axiom_type=AxiomType.SET_THEORETIC,
                description="Union: The union of any set of sets exists"
            ),
            Axiom(
                name="ZF_POWER",
                formula=ForAll("A",
                    Exists("B", ForAll("x",
                        Iff(
                            Predicate("∈", (x, B)),
                            ForAll("y", Implies(
                                Predicate("∈", (y, x)),
                                Predicate("∈", (y, A))
                            ))
                        )
                    ))
                ),
                axiom_type=AxiomType.SET_THEORETIC,
                description="Power Set: The power set of any set exists"
            ),
            Axiom(
                name="ZF_INF",
                formula=Exists("A", And(
                    Exists("x", And(
                        Predicate("∈", (x, A)),
                        ForAll("y", Not(Predicate("∈", (y, x))))
                    )),
                    ForAll("x", Implies(
                        Predicate("∈", (x, A)),
                        Exists("y", And(
                            Predicate("∈", (y, A)),
                            ForAll("z", Iff(
                                Predicate("∈", (z, y)),
                                Or(Predicate("∈", (z, x)), Equals(z, x))
                            ))
                        ))
                    ))
                )),
                axiom_type=AxiomType.SET_THEORETIC,
                description="Infinity: An infinite set exists"
            ),
        ]
    
    @property
    def schemas(self) -> List[AxiomSchema]:
        return [
            AxiomSchema(
                name="ZF_SEP",
                template=self._separation_template,
                description="Separation: {x ∈ A : φ(x)} exists for any formula φ",
                axiom_type=AxiomType.SET_THEORETIC
            ),
            AxiomSchema(
                name="ZF_REP",
                template=self._replacement_template,
                description="Replacement: Image of a set under a function is a set",
                axiom_type=AxiomType.SET_THEORETIC
            ),
        ]
    
    def _separation_template(
        self, 
        phi: LogicalFormula
    ) -> LogicalFormula:
        """
        Separation (Comprehension) schema.
        
        ∀A ∃B ∀x (x ∈ B ↔ (x ∈ A ∧ φ(x)))
        """
        x = Variable("x")
        A = Variable("A")
        B = Variable("B")
        
        return ForAll("A", Exists("B", ForAll("x",
            Iff(
                Predicate("∈", (x, B)),
                And(Predicate("∈", (x, A)), phi)
            )
        )))
    
    def _replacement_template(
        self, 
        phi: LogicalFormula  # φ(x, y) - functional relation
    ) -> LogicalFormula:
        """
        Replacement schema.
        
        If φ defines a function, then the image of any set under φ exists.
        """
        x = Variable("x")
        y = Variable("y")
        z = Variable("z")
        A = Variable("A")
        B = Variable("B")
        
        # φ is functional: ∀x∀y∀z((φ(x,y) ∧ φ(x,z)) → y = z)
        functional = ForAll("x", ForAll("y", ForAll("z",
            Implies(
                And(phi, phi.substitute("y", z)),
                Equals(y, z)
            )
        )))
        
        # Image exists
        image_exists = ForAll("A", Exists("B", ForAll("y",
            Iff(
                Predicate("∈", (y, B)),
                Exists("x", And(
                    Predicate("∈", (x, A)),
                    phi
                ))
            )
        )))
        
        return Implies(functional, image_exists)


# =============================================================================
# COMBINED AXIOM SYSTEM
# =============================================================================

class StandardMathAxioms(AxiomSystem):
    """
    Combined axiom system for standard mathematics.
    
    Includes:
    - First-order logic
    - Equality axioms
    - Set theory (optional)
    - Peano arithmetic (optional)
    """
    
    def __init__(
        self, 
        include_sets: bool = True,
        include_peano: bool = True
    ):
        self._fol = FirstOrderAxioms()
        self._eq = EqualityAxioms()
        self._sets = SetTheoryAxioms() if include_sets else None
        self._peano = PeanoAxioms() if include_peano else None
    
    @property
    def name(self) -> str:
        return "Standard Mathematical Axioms"
    
    @property
    def axioms(self) -> List[Axiom]:
        result = list(self._eq.axioms)
        if self._sets:
            result.extend(self._sets.axioms)
        if self._peano:
            result.extend(self._peano.axioms)
            result.extend(self._peano.addition_axioms())
            result.extend(self._peano.multiplication_axioms())
        return result
    
    @property
    def schemas(self) -> List[AxiomSchema]:
        result = list(self._fol.schemas)
        result.extend(self._eq.schemas)
        if self._sets:
            result.extend(self._sets.schemas)
        if self._peano:
            result.extend(self._peano.schemas)
        return result
