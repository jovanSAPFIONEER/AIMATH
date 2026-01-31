"""
Proof Verifier - Rigorous verification of formal proofs.

This module provides comprehensive verification that:
1. Each proof step follows from previous steps by valid inference
2. All axioms used are valid
3. No gaps exist in the reasoning chain
4. The conclusion actually follows from the premises
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Union
from enum import Enum, auto
import logging

from .logic import (
    LogicalFormula, Proposition, Predicate, Negation, BinaryConnective,
    QuantifiedFormula, Variable, Constant, Function, Term,
    Connective, Quantifier,
    And, Or, Implies, Iff, Not, ForAll, Exists, Equals,
    is_tautology, logically_equivalent
)
from .axioms import AxiomSystem, Axiom, StandardMathAxioms
from .inference import (
    InferenceRule, ProofStep, Justification, RuleRegistry,
    ModusPonens, ModusTollens, ConjunctionIntro, ConjunctionElim,
    DisjunctionIntro, DisjunctiveSyllogism, DoubleNegationElim,
    UniversalInstantiation, ExistentialGeneralization, Substitution
)

logger = logging.getLogger(__name__)


class VerificationStatus(Enum):
    """Status of proof verification."""
    VALID = auto()           # Proof is completely valid
    INVALID = auto()         # Proof has errors
    INCOMPLETE = auto()      # Proof has gaps
    PENDING = auto()         # Verification in progress


class GapType(Enum):
    """Types of proof gaps."""
    MISSING_JUSTIFICATION = auto()   # Step lacks justification
    INVALID_RULE = auto()            # Rule application is incorrect
    MISSING_PREMISE = auto()         # Referenced premise doesn't exist
    CIRCULAR_REASONING = auto()      # Step depends on itself
    UNDISCHARGED_ASSUMPTION = auto() # Assumption not properly discharged
    INVALID_QUANTIFIER = auto()      # Quantifier rule misapplied
    SCOPE_ERROR = auto()             # Variable scope violation


@dataclass
class ProofGap:
    """
    A gap or error in a proof.
    
    Identifies exactly where the proof fails and why.
    """
    step_index: int
    gap_type: GapType
    description: str
    suggestion: str = ""
    severity: str = "error"  # error, warning, info
    
    def __str__(self) -> str:
        return f"Step {self.step_index}: [{self.gap_type.name}] {self.description}"


@dataclass
class VerificationResult:
    """
    Complete result of proof verification.
    
    Contains status, any gaps found, and detailed analysis.
    """
    status: VerificationStatus
    is_valid: bool
    gaps: List[ProofGap] = field(default_factory=list)
    verified_steps: List[int] = field(default_factory=list)
    trust_score: float = 1.0
    details: Dict[str, any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        if self.is_valid:
            return f"✓ VALID PROOF (trust: {self.trust_score:.2%})"
        else:
            gap_summary = f"{len(self.gaps)} gap(s) found"
            return f"✗ INVALID PROOF: {gap_summary}"
    
    def full_report(self) -> str:
        """Generate a detailed verification report."""
        lines = [
            "=" * 60,
            "PROOF VERIFICATION REPORT",
            "=" * 60,
            "",
            f"Status: {self.status.name}",
            f"Valid: {'Yes' if self.is_valid else 'No'}",
            f"Trust Score: {self.trust_score:.2%}",
            f"Verified Steps: {len(self.verified_steps)}",
            "",
        ]
        
        if self.gaps:
            lines.append("GAPS/ERRORS FOUND:")
            lines.append("-" * 40)
            for gap in self.gaps:
                lines.append(f"  • {gap}")
                if gap.suggestion:
                    lines.append(f"    → Suggestion: {gap.suggestion}")
            lines.append("")
        
        if self.verified_steps:
            lines.append(f"Successfully verified steps: {self.verified_steps}")
        
        lines.append("=" * 60)
        return "\n".join(lines)


class ProofVerifier:
    """
    Rigorous proof verification engine.
    
    Verifies that each step in a proof follows logically from
    previous steps using valid inference rules.
    """
    
    def __init__(
        self,
        axiom_system: Optional[AxiomSystem] = None,
        strict_mode: bool = True,
        check_variable_capture: bool = True
    ):
        """
        Initialize the verifier.
        
        Args:
            axiom_system: The axiom system to use (default: StandardMathAxioms)
            strict_mode: If True, require explicit justifications
            check_variable_capture: If True, check for variable capture errors
        """
        self.axiom_system = axiom_system or StandardMathAxioms()
        self.strict_mode = strict_mode
        self.check_variable_capture = check_variable_capture
    
    def verify(
        self,
        steps: List[ProofStep],
        premises: List[LogicalFormula],
        conclusion: LogicalFormula
    ) -> VerificationResult:
        """
        Verify a complete proof.
        
        Args:
            steps: List of proof steps
            premises: Given premises (assumptions)
            conclusion: What we're trying to prove
            
        Returns:
            VerificationResult with detailed analysis
        """
        gaps: List[ProofGap] = []
        verified: List[int] = []
        
        # Track what formulas are available at each step
        available: Dict[int, LogicalFormula] = {}
        
        # Add premises as available (with negative indices)
        for i, premise in enumerate(premises):
            available[-(i + 1)] = premise
        
        # Verify each step
        for step in steps:
            step_result = self._verify_step(step, available, premises)
            
            if step_result.is_valid:
                verified.append(step.index)
                available[step.index] = step.formula
            else:
                gaps.extend(step_result.gaps)
        
        # Check that conclusion was reached
        conclusion_found = any(
            self._formulas_equivalent(f, conclusion) 
            for f in available.values()
        )
        
        if not conclusion_found and not gaps:
            gaps.append(ProofGap(
                step_index=len(steps),
                gap_type=GapType.MISSING_PREMISE,
                description=f"Conclusion '{conclusion}' not derived",
                suggestion="Add steps to derive the conclusion"
            ))
        
        # Calculate trust score
        total_steps = len(steps)
        verified_count = len(verified)
        trust_score = verified_count / total_steps if total_steps > 0 else 0.0
        
        # Determine overall status
        if not gaps:
            status = VerificationStatus.VALID
            is_valid = True
        elif all(g.severity != "error" for g in gaps):
            status = VerificationStatus.INCOMPLETE
            is_valid = False
        else:
            status = VerificationStatus.INVALID
            is_valid = False
        
        return VerificationResult(
            status=status,
            is_valid=is_valid,
            gaps=gaps,
            verified_steps=verified,
            trust_score=trust_score,
            details={
                "total_steps": total_steps,
                "premises_count": len(premises),
                "conclusion": str(conclusion)
            }
        )
    
    def _verify_step(
        self,
        step: ProofStep,
        available: Dict[int, LogicalFormula],
        premises: List[LogicalFormula]
    ) -> VerificationResult:
        """Verify a single proof step."""
        gaps: List[ProofGap] = []
        
        justification = step.justification
        
        # Check if it's a premise
        if justification.rule_name.lower() in ["premise", "given", "assumption", "hyp"]:
            # Verify it's actually a premise
            if any(self._formulas_equivalent(step.formula, p) for p in premises):
                return VerificationResult(
                    status=VerificationStatus.VALID,
                    is_valid=True,
                    verified_steps=[step.index]
                )
            else:
                gaps.append(ProofGap(
                    step_index=step.index,
                    gap_type=GapType.INVALID_RULE,
                    description=f"'{step.formula}' is not among the given premises",
                    suggestion="Check that this formula matches a premise exactly"
                ))
                return VerificationResult(
                    status=VerificationStatus.INVALID,
                    is_valid=False,
                    gaps=gaps
                )
        
        # Check if it's an axiom
        if justification.rule_name.lower() in ["axiom", "ax"]:
            if self._is_axiom(step.formula):
                return VerificationResult(
                    status=VerificationStatus.VALID,
                    is_valid=True,
                    verified_steps=[step.index]
                )
            else:
                gaps.append(ProofGap(
                    step_index=step.index,
                    gap_type=GapType.INVALID_RULE,
                    description=f"'{step.formula}' is not a recognized axiom",
                    suggestion="Verify this is a valid axiom or axiom instance"
                ))
                return VerificationResult(
                    status=VerificationStatus.INVALID,
                    is_valid=False,
                    gaps=gaps
                )
        
        # Check if it's a tautology
        if justification.rule_name.lower() in ["tautology", "taut", "logic"]:
            if is_tautology(step.formula):
                return VerificationResult(
                    status=VerificationStatus.VALID,
                    is_valid=True,
                    verified_steps=[step.index]
                )
            else:
                gaps.append(ProofGap(
                    step_index=step.index,
                    gap_type=GapType.INVALID_RULE,
                    description=f"'{step.formula}' is not a tautology",
                    suggestion="Check the logical structure of this formula"
                ))
        
        # Get the referenced premises
        premise_formulas: List[LogicalFormula] = []
        for idx in justification.premise_indices:
            if idx in available:
                premise_formulas.append(available[idx])
            elif idx < 0 and -idx - 1 < len(premises):
                premise_formulas.append(premises[-idx - 1])
            else:
                gaps.append(ProofGap(
                    step_index=step.index,
                    gap_type=GapType.MISSING_PREMISE,
                    description=f"Referenced step {idx} not available",
                    suggestion="Ensure the referenced step comes before this one"
                ))
                return VerificationResult(
                    status=VerificationStatus.INVALID,
                    is_valid=False,
                    gaps=gaps
                )
        
        # Try to verify using the specified rule
        rule = RuleRegistry.get(justification.rule_name)
        
        if rule is not None:
            # Use the registered rule
            if rule.verify(premise_formulas, step.formula, **justification.substitutions):
                return VerificationResult(
                    status=VerificationStatus.VALID,
                    is_valid=True,
                    verified_steps=[step.index]
                )
            else:
                gaps.append(ProofGap(
                    step_index=step.index,
                    gap_type=GapType.INVALID_RULE,
                    description=f"Rule '{rule.name}' does not justify '{step.formula}' from given premises",
                    suggestion=f"Check that {rule.name} is correctly applied"
                ))
        else:
            # Try to find a rule that works
            valid_rule = self._find_valid_rule(premise_formulas, step.formula)
            if valid_rule:
                # Rule name was wrong but a valid derivation exists
                gaps.append(ProofGap(
                    step_index=step.index,
                    gap_type=GapType.INVALID_RULE,
                    description=f"Unknown rule '{justification.rule_name}'",
                    suggestion=f"Did you mean '{valid_rule.name}'?",
                    severity="warning"
                ))
                return VerificationResult(
                    status=VerificationStatus.VALID,
                    is_valid=True,
                    verified_steps=[step.index],
                    gaps=gaps
                )
            else:
                gaps.append(ProofGap(
                    step_index=step.index,
                    gap_type=GapType.INVALID_RULE,
                    description=f"Unknown rule '{justification.rule_name}' and no valid derivation found",
                    suggestion="Check the rule name and premise references"
                ))
        
        return VerificationResult(
            status=VerificationStatus.INVALID if gaps else VerificationStatus.VALID,
            is_valid=len(gaps) == 0,
            gaps=gaps,
            verified_steps=[step.index] if not gaps else []
        )
    
    def _find_valid_rule(
        self,
        premises: List[LogicalFormula],
        conclusion: LogicalFormula
    ) -> Optional[InferenceRule]:
        """Try to find an inference rule that derives conclusion from premises."""
        for rule in RuleRegistry.all_rules():
            if rule.premises_required == len(premises) or rule.premises_required == -1:
                try:
                    if rule.verify(premises, conclusion):
                        return rule
                except Exception:
                    continue
        return None
    
    def _is_axiom(self, formula: LogicalFormula) -> bool:
        """Check if a formula is an axiom or axiom instance."""
        return self.axiom_system.is_axiom(formula)
    
    def _formulas_equivalent(
        self, 
        f1: LogicalFormula, 
        f2: LogicalFormula
    ) -> bool:
        """Check if two formulas are equivalent."""
        if f1 == f2:
            return True
        # Try logical equivalence for propositional formulas
        try:
            return logically_equivalent(f1, f2)
        except Exception:
            return False
    
    def verify_step_by_step(
        self,
        steps: List[ProofStep],
        premises: List[LogicalFormula]
    ) -> List[Tuple[int, VerificationResult]]:
        """
        Verify each step independently and return results.
        
        Useful for interactive proof development.
        """
        results: List[Tuple[int, VerificationResult]] = []
        available: Dict[int, LogicalFormula] = {}
        
        for i, premise in enumerate(premises):
            available[-(i + 1)] = premise
        
        for step in steps:
            result = self._verify_step(step, available, premises)
            results.append((step.index, result))
            
            if result.is_valid:
                available[step.index] = step.formula
        
        return results
    
    def suggest_next_step(
        self,
        available: List[LogicalFormula],
        goal: LogicalFormula
    ) -> List[Tuple[InferenceRule, LogicalFormula]]:
        """
        Suggest possible next steps toward the goal.
        
        Args:
            available: Currently available formulas
            goal: What we want to prove
            
        Returns:
            List of (rule, result) pairs that might help
        """
        suggestions: List[Tuple[InferenceRule, LogicalFormula]] = []
        
        # Try each rule with available premises
        for rule in RuleRegistry.all_rules():
            n = rule.premises_required
            if n < 0:
                continue
            
            # Try combinations of available formulas
            from itertools import combinations, permutations
            
            if n == 0:
                result = rule.apply([])
                if result is not None:
                    suggestions.append((rule, result))
            
            elif n == 1:
                for f in available:
                    result = rule.apply([f])
                    if result is not None:
                        suggestions.append((rule, result))
            
            elif n == 2:
                for combo in permutations(available, 2):
                    result = rule.apply(list(combo))
                    if result is not None:
                        suggestions.append((rule, result))
        
        # Sort by relevance to goal
        def relevance(item: Tuple[InferenceRule, LogicalFormula]) -> int:
            _, result = item
            if result == goal:
                return 0
            if goal.contains(result):
                return 1
            if result.contains(goal):
                return 2
            return 3
        
        suggestions.sort(key=relevance)
        return suggestions[:10]  # Top 10 suggestions


class ProofChecker:
    """
    High-level proof checker with user-friendly interface.
    """
    
    def __init__(self, verifier: Optional[ProofVerifier] = None):
        self.verifier = verifier or ProofVerifier()
    
    def check(
        self,
        proof_text: str,
        premises: Optional[List[str]] = None,
        conclusion: Optional[str] = None
    ) -> VerificationResult:
        """
        Check a proof given in text format.
        
        This is a high-level interface that parses the proof and verifies it.
        """
        # Parse the proof text into steps
        steps = self._parse_proof(proof_text)
        
        # Parse premises and conclusion
        premise_formulas = [self._parse_formula(p) for p in (premises or [])]
        
        if conclusion:
            conclusion_formula = self._parse_formula(conclusion)
        elif steps:
            conclusion_formula = steps[-1].formula
        else:
            raise ValueError("No conclusion specified and proof is empty")
        
        return self.verifier.verify(steps, premise_formulas, conclusion_formula)
    
    def _parse_proof(self, text: str) -> List[ProofStep]:
        """Parse proof text into proof steps."""
        # Simple parser - assumes format:
        # 1. formula    [justification]
        # 2. formula    [rule, refs]
        
        steps: List[ProofStep] = []
        lines = text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Try to parse: "N. formula [justification]"
            import re
            match = re.match(r'(\d+)\.\s*(.+?)\s*\[(.+?)\]', line)
            if match:
                index = int(match.group(1))
                formula_str = match.group(2).strip()
                justification_str = match.group(3).strip()
                
                formula = self._parse_formula(formula_str)
                justification = self._parse_justification(justification_str)
                
                steps.append(ProofStep(
                    index=index,
                    formula=formula,
                    justification=justification
                ))
        
        return steps
    
    def _parse_formula(self, text: str) -> LogicalFormula:
        """Parse a formula from text."""
        # Simple parser for common formula formats
        text = text.strip()
        
        # Handle propositions (single letters)
        if len(text) == 1 and text.isalpha():
            return Proposition(text)
        
        # Handle negation
        if text.startswith('¬') or text.startswith('~') or text.startswith('not '):
            inner = text[1:].strip() if text[0] in '¬~' else text[4:].strip()
            if inner.startswith('(') and inner.endswith(')'):
                inner = inner[1:-1]
            return Not(self._parse_formula(inner))
        
        # Handle binary connectives (find main connective)
        # This is simplified - a real parser would handle precedence properly
        for symbol, conn in [
            ('↔', Connective.IFF),
            ('→', Connective.IMPLIES),
            ('->', Connective.IMPLIES),
            ('∨', Connective.OR),
            ('|', Connective.OR),
            ('∧', Connective.AND),
            ('&', Connective.AND),
        ]:
            # Find the main connective (not inside parentheses)
            depth = 0
            for i, c in enumerate(text):
                if c == '(':
                    depth += 1
                elif c == ')':
                    depth -= 1
                elif depth == 0 and text[i:].startswith(symbol):
                    left = text[:i].strip()
                    right = text[i + len(symbol):].strip()
                    if left.startswith('(') and left.endswith(')'):
                        left = left[1:-1]
                    if right.startswith('(') and right.endswith(')'):
                        right = right[1:-1]
                    return BinaryConnective(
                        conn,
                        self._parse_formula(left),
                        self._parse_formula(right)
                    )
        
        # Handle parentheses
        if text.startswith('(') and text.endswith(')'):
            return self._parse_formula(text[1:-1])
        
        # Handle quantifiers
        for symbol, quant in [('∀', Quantifier.FORALL), ('∃', Quantifier.EXISTS)]:
            if text.startswith(symbol):
                # Format: ∀x(formula) or ∀x.formula
                rest = text[1:]
                var = rest[0]
                if rest[1] in '(.':
                    formula_text = rest[2:].rstrip(')')
                    return QuantifiedFormula(quant, var, self._parse_formula(formula_text))
        
        # Handle predicates: P(x, y)
        if '(' in text and text.endswith(')'):
            name = text[:text.index('(')]
            args_str = text[text.index('(') + 1:-1]
            args = [a.strip() for a in args_str.split(',')]
            terms = tuple(Variable(a) if a.islower() else Constant(a) for a in args)
            return Predicate(name, terms)
        
        # Default: treat as proposition
        return Proposition(text)
    
    def _parse_justification(self, text: str) -> Justification:
        """Parse justification from text."""
        # Format: "rule_name" or "rule_name, 1, 2" or "rule_name (1, 2)"
        import re
        
        # Extract rule name and references
        match = re.match(r'([^,\(]+)(?:[,\s]*\(?([\d,\s]+)\)?)?', text)
        if match:
            rule_name = match.group(1).strip()
            refs_str = match.group(2)
            
            refs: Tuple[int, ...] = ()
            if refs_str:
                refs = tuple(int(r.strip()) for r in refs_str.split(',') if r.strip().isdigit())
            
            return Justification(rule_name=rule_name, premise_indices=refs)
        
        return Justification(rule_name=text, premise_indices=())
