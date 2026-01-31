"""
Confidence Scorer - Calculate confidence levels for results.

Combines multiple verification signals into a single confidence
score and level.
"""

from typing import Optional
import logging

from ..core.types import VerificationCheck, ConfidenceLevel

logger = logging.getLogger(__name__)


class ConfidenceScorer:
    """
    Calculate confidence scores for mathematical results.
    
    Confidence Levels:
    - PROVEN (100%): Formally verified by theorem prover
    - HIGH (95%+): Multiple methods agree + substitution passes
    - MEDIUM (70-95%): Single method, passes basic checks
    - LOW (<70%): Methods disagree, needs human review
    - UNKNOWN: Cannot determine
    
    Scoring factors:
    - Formal proof: +50 points
    - Substitution pass: +20 points
    - Consensus (2+ methods): +15 points
    - No counterexamples: +10 points
    - Domain valid: +5 points
    
    Example:
        >>> scorer = ConfidenceScorer()
        >>> level, score = scorer.score(checks, methods, has_proof=True)
        >>> print(f"{level}: {score}%")  # PROVEN: 100%
    """
    
    # Scoring weights for different check types
    WEIGHTS = {
        'formal_proof': 50,
        'substitution': 20,
        'consensus': 15,
        'counterexample': 10,
        'domain': 5,
        'numerical': 5,
    }
    
    # Penalties for failures
    PENALTIES = {
        'substitution': -40,  # Critical failure
        'counterexample': -50,  # Found counterexample
        'consensus': -20,  # Solvers disagree
        'domain': -10,  # Domain issues
    }
    
    def __init__(self):
        """Initialize scorer."""
        pass
    
    def score(
        self,
        checks: list[VerificationCheck],
        methods_used: list[str],
        has_formal_proof: bool = False,
    ) -> tuple[ConfidenceLevel, float]:
        """
        Calculate confidence score and level.
        
        Args:
            checks: List of verification checks performed
            methods_used: Names of solvers used
            has_formal_proof: Whether a formal proof was obtained
            
        Returns:
            Tuple of (ConfidenceLevel, score 0-100)
        """
        # Start with base score
        score = 0.0
        
        # Formal proof is highest confidence
        if has_formal_proof:
            return ConfidenceLevel.PROVEN, 100.0
        
        # Process each check
        for check in checks:
            check_type = check.check_type
            
            if check.passed:
                score += self.WEIGHTS.get(check_type, 5)
            else:
                score += self.PENALTIES.get(check_type, -5)
        
        # Bonus for multiple methods
        num_methods = len(methods_used)
        if num_methods >= 3:
            score += 10
        elif num_methods >= 2:
            score += 5
        
        # Bonus for high-trust methods
        if 'symbolic' in methods_used:
            score += 5
        if 'formal' in methods_used:
            score += 10
        
        # Penalty for LLM-only
        if methods_used == ['llm']:
            score -= 20
        
        # Clamp score to 0-100 (but not 100 without formal proof)
        score = max(0, min(99, score))
        
        # Determine confidence level
        level = self._score_to_level(score, checks)
        
        return level, score
    
    def _score_to_level(
        self, 
        score: float, 
        checks: list[VerificationCheck]
    ) -> ConfidenceLevel:
        """
        Convert numerical score to confidence level.
        
        Also considers critical failures that override score.
        """
        # Check for critical failures
        for check in checks:
            if check.check_type == 'counterexample' and not check.passed:
                return ConfidenceLevel.LOW
            if check.check_type == 'substitution' and not check.passed:
                return ConfidenceLevel.LOW
        
        # Score-based level
        if score >= 95:
            return ConfidenceLevel.HIGH
        elif score >= 70:
            return ConfidenceLevel.MEDIUM
        elif score >= 30:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.UNKNOWN
    
    def explain_score(
        self,
        checks: list[VerificationCheck],
        methods_used: list[str],
        has_formal_proof: bool = False,
    ) -> str:
        """
        Generate human-readable explanation of confidence score.
        
        Returns:
            Explanation string
        """
        level, score = self.score(checks, methods_used, has_formal_proof)
        
        lines = [
            f"Confidence: {level.value.upper()} ({score:.0f}%)",
            "",
            "Verification breakdown:",
        ]
        
        for check in checks:
            status = "✓" if check.passed else "✗"
            weight = (
                self.WEIGHTS.get(check.check_type, 5) if check.passed
                else self.PENALTIES.get(check.check_type, -5)
            )
            sign = "+" if weight >= 0 else ""
            lines.append(f"  {status} {check.check_type}: {sign}{weight} pts")
        
        lines.append("")
        lines.append(f"Methods used: {', '.join(methods_used)}")
        
        if has_formal_proof:
            lines.append("✓ Formal proof obtained")
        
        return "\n".join(lines)
    
    def get_recommendations(
        self,
        checks: list[VerificationCheck],
        score: float,
    ) -> list[str]:
        """
        Generate recommendations for improving confidence.
        
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Check for specific issues
        for check in checks:
            if not check.passed:
                if check.check_type == 'consensus':
                    recommendations.append(
                        "Try additional solving methods to establish consensus"
                    )
                elif check.check_type == 'substitution':
                    recommendations.append(
                        "Review solution - substitution check failed"
                    )
                elif check.check_type == 'counterexample':
                    recommendations.append(
                        "WARNING: Counterexample found - solution may be incorrect"
                    )
        
        # Score-based recommendations
        if score < 70:
            recommendations.append(
                "Consider manual verification due to low confidence"
            )
        
        if score < 50:
            recommendations.append(
                "Result should not be trusted without additional verification"
            )
        
        return recommendations
