"""
Consensus Checker - Verify multiple solvers agree.

A key anti-hallucination strategy: require multiple independent
solving methods to produce the same answer.
"""

from typing import Any
import logging

from ..core.types import MathProblem, VerificationCheck

logger = logging.getLogger(__name__)


class ConsensusChecker:
    """
    Check that multiple solvers agree on the answer.
    
    Philosophy: If symbolic and numerical solvers both get the
    same answer independently, it's much more likely to be correct
    than relying on any single method.
    
    Consensus Rules:
    - HIGH trust if formal prover agrees
    - MEDIUM trust if symbolic + numerical agree
    - LOW trust if only one method available
    - FAIL if methods disagree
    
    Example:
        >>> checker = ConsensusChecker(min_paths=2)
        >>> solutions = {'symbolic': [2, 3], 'numerical': [2.0, 3.0]}
        >>> check, best = checker.check(problem, solutions)
        >>> print(check.passed)  # True if they agree
    """
    
    # Trust levels for different solver types
    SOLVER_TRUST = {
        'formal': 1.0,
        'symbolic': 0.9,
        'numerical': 0.7,
        'llm': 0.3,
    }
    
    def __init__(
        self,
        min_consensus_paths: int = 2,
        tolerance: float = 1e-8,
    ):
        """
        Initialize consensus checker.
        
        Args:
            min_consensus_paths: Minimum number of methods that must agree
            tolerance: Tolerance for comparing numerical values
        """
        self.min_consensus_paths = min_consensus_paths
        self.tolerance = tolerance
    
    def check(
        self,
        problem: MathProblem,
        solutions: dict[str, Any],
    ) -> tuple[VerificationCheck, Any]:
        """
        Check consensus among solutions.
        
        Args:
            problem: Original problem
            solutions: Dict mapping solver name to answer
            
        Returns:
            Tuple of (VerificationCheck, best_answer)
        """
        if not solutions:
            return VerificationCheck(
                check_type='consensus',
                passed=False,
                details="No solutions to compare",
                error="Empty solutions dict",
            ), None
        
        if len(solutions) == 1:
            # Only one solver - can't check consensus
            solver_name = list(solutions.keys())[0]
            answer = list(solutions.values())[0]
            
            return VerificationCheck(
                check_type='consensus',
                passed=False,
                details=f"Only one solver ({solver_name}) - cannot verify consensus",
                evidence={'single_solver': solver_name},
            ), answer
        
        # Group equivalent answers
        groups = self._group_equivalent_answers(solutions)
        
        # Find largest consensus group
        largest_group = max(groups, key=lambda g: len(g['solvers']))
        
        consensus_count = len(largest_group['solvers'])
        total_solvers = len(solutions)
        
        # Calculate weighted consensus (trust-weighted)
        weighted_agreement = sum(
            self.SOLVER_TRUST.get(s.split('_')[0], 0.5)
            for s in largest_group['solvers']
        )
        total_weight = sum(
            self.SOLVER_TRUST.get(s.split('_')[0], 0.5)
            for s in solutions.keys()
        )
        
        agreement_ratio = weighted_agreement / total_weight if total_weight > 0 else 0
        
        # Determine if consensus is sufficient
        passed = (
            consensus_count >= self.min_consensus_paths or
            agreement_ratio >= 0.7
        )
        
        # Check for disagreements
        disagreements = []
        if len(groups) > 1:
            for group in groups:
                if group != largest_group:
                    disagreements.append({
                        'solvers': group['solvers'],
                        'answer': str(group['answer']),
                    })
        
        details = (
            f"{consensus_count}/{total_solvers} solvers agree "
            f"(weighted: {agreement_ratio:.0%})"
        )
        
        if disagreements:
            details += f"; {len(disagreements)} disagreeing group(s)"
        
        return VerificationCheck(
            check_type='consensus',
            passed=passed,
            details=details,
            evidence={
                'consensus_group': largest_group['solvers'],
                'disagreements': disagreements,
                'agreement_ratio': agreement_ratio,
            },
            error=None if passed else "Insufficient consensus",
        ), largest_group['answer']
    
    def _group_equivalent_answers(
        self, 
        solutions: dict[str, Any]
    ) -> list[dict]:
        """
        Group solutions by equivalence.
        
        Answers are equivalent if they:
        - Are symbolically equal (after simplification)
        - Are numerically close (within tolerance)
        - Represent the same set of solutions
        """
        from sympy import simplify, N, Abs
        
        groups = []
        
        for solver_name, answer in solutions.items():
            # Check if answer belongs to existing group
            found_group = False
            
            for group in groups:
                if self._answers_equivalent(group['answer'], answer):
                    group['solvers'].append(solver_name)
                    found_group = True
                    break
            
            if not found_group:
                # Create new group
                groups.append({
                    'answer': answer,
                    'solvers': [solver_name],
                })
        
        return groups
    
    def _answers_equivalent(self, a1: Any, a2: Any) -> bool:
        """
        Check if two answers are equivalent.
        
        Handles:
        - Direct equality
        - Symbolic equivalence
        - Numerical equivalence
        - List/set equivalence
        """
        from sympy import simplify, N, Abs, S
        
        # Direct equality
        if a1 == a2:
            return True
        
        # Handle None
        if a1 is None or a2 is None:
            return False
        
        # Handle lists/sets of solutions
        if isinstance(a1, (list, tuple, set)) and isinstance(a2, (list, tuple, set)):
            return self._sets_equivalent(set(a1), set(a2))
        
        # Symbolic equivalence
        try:
            diff = simplify(a1 - a2)
            if diff == 0:
                return True
        except Exception:
            pass
        
        # Numerical equivalence
        try:
            n1 = complex(N(a1))
            n2 = complex(N(a2))
            if abs(n1 - n2) < self.tolerance:
                return True
        except Exception:
            pass
        
        # String comparison as fallback
        try:
            if str(simplify(a1)) == str(simplify(a2)):
                return True
        except Exception:
            pass
        
        return False
    
    def _sets_equivalent(self, s1: set, s2: set) -> bool:
        """
        Check if two sets of solutions are equivalent.
        
        Each element in s1 must have a matching element in s2.
        """
        if len(s1) != len(s2):
            return False
        
        s2_remaining = list(s2)
        
        for elem1 in s1:
            found = False
            for i, elem2 in enumerate(s2_remaining):
                if self._answers_equivalent(elem1, elem2):
                    s2_remaining.pop(i)
                    found = True
                    break
            if not found:
                return False
        
        return True
    
    def get_best_answer(
        self, 
        solutions: dict[str, Any]
    ) -> tuple[Any, str]:
        """
        Get the best answer based on solver trust levels.
        
        Prioritizes:
        1. Formal prover results
        2. Symbolic solver results
        3. Numerical results
        4. LLM results (last resort)
        
        Returns:
            Tuple of (best_answer, source_solver)
        """
        if not solutions:
            return None, ""
        
        # Sort by trust level
        sorted_solvers = sorted(
            solutions.items(),
            key=lambda x: self.SOLVER_TRUST.get(x[0].split('_')[0], 0.5),
            reverse=True,
        )
        
        return sorted_solvers[0][1], sorted_solvers[0][0]
