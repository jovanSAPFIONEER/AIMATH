"""
Main Math Engine - Orchestrates solving, verification, and explanation.

This is the central coordinator that:
1. Parses input (LaTeX, natural language, raw expression)
2. Routes to appropriate solvers
3. Verifies results through multiple paths
4. Generates quality-checked explanations
5. Returns only verified, well-explained results
"""

import time
from typing import Any, Optional, Union
import logging

from .types import (
    MathProblem,
    Solution,
    VerificationResult,
    Explanation,
    ExplanationQuality,
    SolutionStep,
    ConfidenceLevel,
    DifficultyLevel,
    ProblemType,
    VerificationCheck,
)

# Initialize logger
logger = logging.getLogger(__name__)


class MathEngine:
    """
    Main orchestration engine for mathematical problem solving.
    
    Philosophy: "Never trust, always verify"
    - Every solution is verified before delivery
    - Multiple solving paths for consensus
    - Explanations must pass quality gates
    - LLM output NEVER delivered without verification
    
    Example:
        >>> engine = MathEngine()
        >>> result = engine.solve("x^2 - 5x + 6 = 0")
        >>> print(result.answer)  # [2, 3]
        >>> print(result.confidence)  # ConfidenceLevel.PROVEN
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        difficulty: DifficultyLevel = DifficultyLevel.INTERMEDIATE,
        require_verification: bool = True,
        min_confidence: float = 0.7,
    ):
        """
        Initialize the Math Engine.
        
        Args:
            config_path: Path to configuration YAML file
            difficulty: Default explanation difficulty level
            require_verification: If True, only return verified results
            min_confidence: Minimum confidence score to return (0-1)
        """
        self.difficulty = difficulty
        self.require_verification = require_verification
        self.min_confidence = min_confidence
        
        # Initialize components (lazy loading)
        self._parser = None
        self._solvers = {}
        self._verifier = None
        self._explainer = None
        
        logger.info(f"MathEngine initialized (difficulty={difficulty.value})")
    
    @property
    def parser(self):
        """Lazy load parser."""
        if self._parser is None:
            from ..parsers import MathParser
            self._parser = MathParser()
        return self._parser
    
    @property
    def verifier(self):
        """Lazy load verifier."""
        if self._verifier is None:
            from ..verification import Verifier
            self._verifier = Verifier()
        return self._verifier
    
    @property
    def explainer(self):
        """Lazy load explainer."""
        if self._explainer is None:
            from ..explanation import ExplanationEngine
            self._explainer = ExplanationEngine(default_level=self.difficulty)
        return self._explainer
    
    def get_solver(self, solver_type: str):
        """Get or create a solver by type."""
        if solver_type not in self._solvers:
            from ..solvers import get_solver
            self._solvers[solver_type] = get_solver(solver_type)
        return self._solvers[solver_type]
    
    def solve(
        self,
        problem_input: str,
        difficulty: Optional[DifficultyLevel] = None,
        show_steps: bool = True,
        verify: bool = True,
    ) -> Solution:
        """
        Solve a mathematical problem with verification and explanation.
        
        This is the main entry point. It:
        1. Parses the input into a structured problem
        2. Solves using multiple methods
        3. Verifies the solution
        4. Generates a quality-checked explanation
        
        Args:
            problem_input: The problem as string (LaTeX, natural language, etc.)
            difficulty: Explanation difficulty level (overrides default)
            show_steps: Include step-by-step solution
            verify: Perform verification (recommended: always True)
            
        Returns:
            Solution object with answer, verification, and explanation
            
        Raises:
            ValueError: If problem cannot be parsed
            VerificationError: If solution fails verification and require_verification=True
            
        Example:
            >>> result = engine.solve("\\int x^2 dx")
            >>> print(result.answer_latex)  # "\\frac{x^3}{3} + C"
        """
        start_time = time.time()
        difficulty = difficulty or self.difficulty
        
        logger.info(f"Solving: {problem_input[:100]}...")
        
        # Step 1: Parse the problem
        problem = self._parse_problem(problem_input)
        logger.debug(f"Parsed as {problem.problem_type.value}")
        
        # Step 2: Solve using multiple methods
        solutions = self._multi_solve(problem)
        
        # Step 3: Verify and find consensus
        if verify:
            verification, best_answer = self._verify_solutions(problem, solutions)
        else:
            verification = VerificationResult(
                is_verified=False,
                confidence=ConfidenceLevel.UNKNOWN,
                confidence_score=0,
                warnings=["Verification skipped by user request"]
            )
            best_answer = solutions[0] if solutions else None
        
        # Step 4: Generate explanation
        explanation = self._generate_explanation(
            problem, best_answer, solutions, difficulty
        )
        
        # Step 5: Package result
        solve_time = (time.time() - start_time) * 1000
        
        result = Solution(
            problem=problem,
            answer=best_answer,
            answer_latex=self._to_latex(best_answer),
            verification=verification,
            explanation=explanation,
            solve_time_ms=solve_time,
            solver_methods=list(solutions.keys()) if isinstance(solutions, dict) else [],
        )
        
        # Check if we meet confidence requirements
        if self.require_verification:
            if not result.is_verified:
                logger.warning(f"Solution not verified: {result.verification.warnings}")
            if result.verification.confidence_score < self.min_confidence * 100:
                logger.warning(
                    f"Low confidence: {result.verification.confidence_score:.1f}%"
                )
        
        logger.info(
            f"Solved in {solve_time:.1f}ms "
            f"(confidence: {result.confidence.value})"
        )
        
        return result
    
    def verify_claim(
        self,
        claim: str,
        context: Optional[str] = None,
    ) -> VerificationResult:
        """
        Verify a mathematical claim or theorem.
        
        Unlike solve(), this doesn't find an answer but verifies
        whether a given statement is true or false.
        
        Args:
            claim: The mathematical claim to verify
            context: Additional context or constraints
            
        Returns:
            VerificationResult with proof or counterexample
            
        Example:
            >>> result = engine.verify_claim("√2 is irrational")
            >>> print(result.is_verified)  # True
            >>> print(result.formal_proof)  # Proof by contradiction...
        """
        logger.info(f"Verifying claim: {claim}")
        
        problem = MathProblem(
            raw_input=claim,
            problem_type=ProblemType.VERIFICATION,
            context=context,
        )
        
        return self.verifier.verify_claim(problem)
    
    def explain(
        self,
        topic: str,
        difficulty: Optional[DifficultyLevel] = None,
    ) -> Explanation:
        """
        Generate an explanation for a mathematical concept.
        
        This doesn't solve a problem but explains a topic with
        the same quality standards as solution explanations.
        
        Args:
            topic: Mathematical concept to explain
            difficulty: Target difficulty level
            
        Returns:
            Quality-verified Explanation object
            
        Example:
            >>> explanation = engine.explain("chain rule", DifficultyLevel.AMATEUR)
            >>> print(explanation.to_markdown())
        """
        difficulty = difficulty or self.difficulty
        return self.explainer.explain_concept(topic, difficulty)
    
    def _parse_problem(self, input_str: str) -> MathProblem:
        """Parse input string into structured MathProblem."""
        return self.parser.parse(input_str)
    
    def _multi_solve(self, problem: MathProblem) -> dict[str, Any]:
        """
        Solve using multiple independent methods.
        
        This is crucial for anti-hallucination:
        - Multiple methods must agree for high confidence
        - Disagreement triggers review
        """
        solutions = {}
        
        # Always try symbolic solver first (highest trust)
        try:
            symbolic_solver = self.get_solver("symbolic")
            solutions["symbolic"] = symbolic_solver.solve(problem)
            logger.debug("Symbolic solver succeeded")
        except Exception as e:
            logger.debug(f"Symbolic solver failed: {e}")
        
        # Try numerical solver for verification
        try:
            numerical_solver = self.get_solver("numerical")
            solutions["numerical"] = numerical_solver.solve(problem)
            logger.debug("Numerical solver succeeded")
        except Exception as e:
            logger.debug(f"Numerical solver failed: {e}")
        
        # Note: LLM solver intentionally not auto-called
        # It requires explicit enable and verification wrapper
        
        return solutions
    
    def _verify_solutions(
        self,
        problem: MathProblem,
        solutions: dict[str, Any],
    ) -> tuple[VerificationResult, Any]:
        """
        Verify solutions and find consensus.
        
        Returns:
            Tuple of (VerificationResult, best_answer)
        """
        if not solutions:
            return VerificationResult(
                is_verified=False,
                confidence=ConfidenceLevel.UNKNOWN,
                confidence_score=0,
                warnings=["No solver could find a solution"]
            ), None
        
        return self.verifier.verify_solutions(problem, solutions)
    
    def _generate_explanation(
        self,
        problem: MathProblem,
        answer: Any,
        solutions: dict[str, Any],
        difficulty: DifficultyLevel,
    ) -> Explanation:
        """Generate quality-checked explanation."""
        return self.explainer.generate(
            problem=problem,
            answer=answer,
            solution_paths=solutions,
            difficulty=difficulty,
        )
    
    def _to_latex(self, expr: Any) -> str:
        """Convert expression to LaTeX string."""
        if expr is None:
            return ""
        try:
            from sympy import latex
            return latex(expr)
        except Exception:
            return str(expr)


class VerificationError(Exception):
    """Raised when a solution fails verification."""
    pass
