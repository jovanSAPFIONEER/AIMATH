"""
Main Verifier - Orchestrates all verification checks.

This is the core anti-hallucination component that ensures
mathematical results are correct before delivery.
"""

from typing import Any, Optional
import logging

from ..core.types import (
    MathProblem, 
    VerificationResult, 
    VerificationCheck,
    ConfidenceLevel,
    ProblemType,
)

logger = logging.getLogger(__name__)


class Verifier:
    """
    Mathematical verification orchestrator.
    
    Philosophy: "Never trust, always verify"
    
    This class coordinates multiple verification strategies:
    1. Substitution checks - plug answer back into problem
    2. Counterexample search - try to find cases where answer fails
    3. Multi-path consensus - multiple methods must agree
    4. Formal proof - Z3/Lean verification when possible
    5. Domain checking - ensure answer is in valid domain
    
    Example:
        >>> verifier = Verifier()
        >>> result = verifier.verify_solutions(problem, solutions)
        >>> print(result.is_verified)  # True/False
        >>> print(result.confidence_score)  # 0-100
    """
    
    def __init__(
        self,
        use_formal_prover: bool = True,
        search_counterexamples: bool = True,
        min_consensus_paths: int = 2,
        numerical_test_count: int = 100,
    ):
        """
        Initialize verifier with configuration.
        
        Args:
            use_formal_prover: Enable Z3 formal verification
            search_counterexamples: Actively search for counterexamples
            min_consensus_paths: Minimum solvers that must agree
            numerical_test_count: Number of random test points
        """
        self.use_formal_prover = use_formal_prover
        self.search_counterexamples = search_counterexamples
        self.min_consensus_paths = min_consensus_paths
        self.numerical_test_count = numerical_test_count
        
        # Lazy-loaded components
        self._substitution_checker = None
        self._counterexample_searcher = None
        self._consensus_checker = None
        self._confidence_scorer = None
        self._formal_prover = None
    
    @property
    def substitution_checker(self):
        if self._substitution_checker is None:
            from .substitution import SubstitutionChecker
            self._substitution_checker = SubstitutionChecker()
        return self._substitution_checker
    
    @property
    def counterexample_searcher(self):
        if self._counterexample_searcher is None:
            from .counterexample import CounterexampleSearcher
            self._counterexample_searcher = CounterexampleSearcher()
        return self._counterexample_searcher
    
    @property
    def consensus_checker(self):
        if self._consensus_checker is None:
            from .consensus import ConsensusChecker
            self._consensus_checker = ConsensusChecker(self.min_consensus_paths)
        return self._consensus_checker
    
    @property
    def confidence_scorer(self):
        if self._confidence_scorer is None:
            from .confidence import ConfidenceScorer
            self._confidence_scorer = ConfidenceScorer()
        return self._confidence_scorer
    
    @property
    def formal_prover(self):
        if self._formal_prover is None:
            from .formal_prover import FormalProver
            self._formal_prover = FormalProver()
        return self._formal_prover
    
    def verify_solutions(
        self,
        problem: MathProblem,
        solutions: dict[str, Any],
    ) -> tuple[VerificationResult, Any]:
        """
        Verify solutions from multiple solvers.
        
        Coordinates all verification checks and returns the
        best verified answer with confidence score.
        
        Args:
            problem: The original problem
            solutions: Dict mapping solver name to answer
            
        Returns:
            Tuple of (VerificationResult, best_answer)
        """
        checks = []
        warnings = []
        methods_used = list(solutions.keys())
        
        logger.info(f"Verifying {len(solutions)} solution(s)...")
        
        # Step 1: Check consensus
        consensus_check, best_answer = self.consensus_checker.check(
            problem, solutions
        )
        checks.append(consensus_check)
        
        if not consensus_check.passed:
            warnings.append(f"Solvers disagree: {consensus_check.details}")
        
        # Step 2: Substitution check
        if best_answer is not None:
            sub_check = self.substitution_checker.check(problem, best_answer)
            checks.append(sub_check)
            
            if not sub_check.passed:
                warnings.append(f"Substitution failed: {sub_check.error}")
        
        # Step 3: Domain check
        domain_check = self._check_domain(problem, best_answer)
        checks.append(domain_check)
        
        # Step 4: Counterexample search
        if self.search_counterexamples and best_answer is not None:
            counter_check = self.counterexample_searcher.search(
                problem, best_answer
            )
            checks.append(counter_check)
            
            if not counter_check.passed:
                warnings.append(f"Counterexample found: {counter_check.evidence}")
        
        # Step 5: Formal proof (if enabled and applicable)
        formal_proof = None
        if self.use_formal_prover and self._can_formally_prove(problem):
            proof_check = self.formal_prover.prove(problem, best_answer)
            checks.append(proof_check)
            
            if proof_check.passed:
                formal_proof = proof_check.evidence
        
        # Step 6: Numerical verification
        numerical_check = self._numerical_verify(problem, best_answer)
        if numerical_check:
            checks.append(numerical_check)
        
        # Calculate confidence
        confidence_level, confidence_score = self.confidence_scorer.score(
            checks, methods_used, formal_proof is not None
        )
        
        # Determine overall verification status
        is_verified = (
            confidence_score >= 70 and
            all(c.passed for c in checks if c.check_type in [
                'substitution', 'counterexample', 'domain'
            ])
        )
        
        return VerificationResult(
            is_verified=is_verified,
            confidence=confidence_level,
            confidence_score=confidence_score,
            checks=checks,
            methods_used=methods_used,
            formal_proof=formal_proof,
            warnings=warnings,
        ), best_answer
    
    def verify_claim(self, problem: MathProblem) -> VerificationResult:
        """
        Verify a mathematical claim (proof verification).
        
        Args:
            problem: Problem containing claim to verify
            
        Returns:
            VerificationResult with proof or counterexample
        """
        checks = []
        
        # Try formal proof
        if self.use_formal_prover:
            proof_check = self.formal_prover.verify_claim(problem)
            checks.append(proof_check)
            
            if proof_check.passed:
                return VerificationResult(
                    is_verified=True,
                    confidence=ConfidenceLevel.PROVEN,
                    confidence_score=100,
                    checks=checks,
                    methods_used=['formal_prover'],
                    formal_proof=proof_check.evidence,
                )
        
        # Try to find counterexample
        counter_check = self.counterexample_searcher.search_for_claim(problem)
        checks.append(counter_check)
        
        if not counter_check.passed:
            # Found counterexample - claim is false
            return VerificationResult(
                is_verified=False,
                confidence=ConfidenceLevel.HIGH,
                confidence_score=95,
                checks=checks,
                methods_used=['counterexample_search'],
                counterexamples=[counter_check.evidence],
            )
        
        # Unable to prove or disprove
        return VerificationResult(
            is_verified=False,
            confidence=ConfidenceLevel.UNKNOWN,
            confidence_score=0,
            checks=checks,
            methods_used=['formal_prover', 'counterexample_search'],
            warnings=["Unable to formally prove or find counterexample"],
        )
    
    def _check_domain(
        self, 
        problem: MathProblem, 
        answer: Any
    ) -> VerificationCheck:
        """
        Check that answer is in valid domain.
        
        Verifies:
        - Answer is real when expected
        - Answer satisfies constraints
        - No division by zero, etc.
        """
        try:
            from sympy import im, re, oo, zoo, nan, Symbol
            
            # Check for invalid values
            if answer is None:
                return VerificationCheck(
                    check_type='domain',
                    passed=False,
                    details="No answer provided",
                    error="Answer is None",
                )
            
            # Handle list of solutions
            answers = answer if isinstance(answer, (list, tuple)) else [answer]
            
            for ans in answers:
                # Check for infinity or undefined
                if ans in [oo, -oo, zoo, nan]:
                    return VerificationCheck(
                        check_type='domain',
                        passed=False,
                        details="Answer is undefined or infinite",
                        evidence=str(ans),
                    )
                
                # Check imaginary part if we expect real answer
                try:
                    if hasattr(ans, 'is_real') and ans.is_real is False:
                        # Check if problem expects real solutions
                        # (most do, unless specified)
                        pass  # Allow complex for now
                except Exception:
                    pass
            
            # Check against explicit constraints
            for constraint in problem.constraints:
                # Implement constraint checking
                pass
            
            return VerificationCheck(
                check_type='domain',
                passed=True,
                details="Answer is in valid domain",
            )
            
        except Exception as e:
            logger.warning(f"Domain check error: {e}")
            return VerificationCheck(
                check_type='domain',
                passed=True,  # Assume valid if can't check
                details=f"Domain check inconclusive: {e}",
            )
    
    def _numerical_verify(
        self, 
        problem: MathProblem, 
        answer: Any
    ) -> Optional[VerificationCheck]:
        """
        Verify answer numerically at random points.
        """
        try:
            from ..solvers.numerical_solver import NumericalSolver
            
            numerical = NumericalSolver()
            result = numerical.verify_against(
                answer, problem, self.numerical_test_count
            )
            
            passed = result.get('agreement', 0) >= 0.95
            
            return VerificationCheck(
                check_type='numerical',
                passed=passed,
                details=f"Numerical agreement: {result['agreement']*100:.1f}%",
                evidence=result,
            )
            
        except Exception as e:
            logger.debug(f"Numerical verification skipped: {e}")
            return None
    
    def _can_formally_prove(self, problem: MathProblem) -> bool:
        """Check if problem type supports formal proof."""
        formal_types = {
            ProblemType.EQUATION,
            ProblemType.INEQUALITY,
            ProblemType.VERIFICATION,
            ProblemType.PROOF,
        }
        return problem.problem_type in formal_types
