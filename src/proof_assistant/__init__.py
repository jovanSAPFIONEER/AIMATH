"""
Formal Proof Assistant Module.

This module provides rigorous mathematical proof verification,
construction, and explanation capabilities.

Components:
- logic: Formal logic types (propositions, predicates, quantifiers)
- axioms: Axiom systems (propositional, Peano, set theory)
- inference: Inference rules and proof steps
- verifier: Proof verification engine
- tactics: Automated proof strategies
- prover: Main proof assistant interface
"""

from .logic import (
    LogicalFormula,
    Proposition,
    Predicate,
    Connective,
    Quantifier,
    Variable,
    Constant,
    Function,
    Term,
)

from .axioms import (
    AxiomSystem,
    PropositionalAxioms,
    PeanoAxioms,
    SetTheoryAxioms,
)

from .inference import (
    InferenceRule,
    ProofStep,
    ModusPonens,
    ModusTollens,
    UniversalInstantiation,
    ExistentialGeneralization,
    Substitution,
)

from .verifier import (
    ProofVerifier,
    VerificationResult,
    ProofGap,
)

from .tactics import (
    ProofTactic,
    DirectProof,
    Contradiction,
    Induction,
    CaseAnalysis,
)

from .prover import (
    ProofAssistant,
    Proof,
    Theorem,
)

__all__ = [
    # Logic
    "LogicalFormula",
    "Proposition",
    "Predicate",
    "Connective",
    "Quantifier",
    "Variable",
    "Constant",
    "Function",
    "Term",
    # Axioms
    "AxiomSystem",
    "PropositionalAxioms",
    "PeanoAxioms",
    "SetTheoryAxioms",
    # Inference
    "InferenceRule",
    "ProofStep",
    "ModusPonens",
    "ModusTollens",
    "UniversalInstantiation",
    "ExistentialGeneralization",
    "Substitution",
    # Verifier
    "ProofVerifier",
    "VerificationResult",
    "ProofGap",
    # Tactics
    "ProofTactic",
    "DirectProof",
    "Contradiction",
    "Induction",
    "CaseAnalysis",
    # Prover
    "ProofAssistant",
    "Proof",
    "Theorem",
]
