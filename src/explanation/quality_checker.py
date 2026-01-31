"""
Quality Checker - CLEAR rubric implementation.

Scores explanations on:
C - Completeness (0-5): All necessary steps present
L - Logical Flow (0-5): Each step follows from previous
E - Explicit Terms (0-5): All terms defined before use
A - Accessibility (0-5): Matched to learner's level
R - Reasoning (0-5): "Why" provided, not just "how"

Minimum passing: 20/25 total, 3/5 in each category
"""

from dataclasses import dataclass, field
from typing import Optional
import re
import logging

from .anti_patterns import AntiPatternDetector, AntiPatternSeverity

logger = logging.getLogger(__name__)


@dataclass
class CLEARScore:
    """
    CLEAR rubric score for an explanation.
    
    C - Completeness: Are all necessary steps present?
    L - Logical Flow: Does each step follow from the previous?
    E - Explicit Terms: Are all technical terms defined before use?
    A - Accessibility: Is it appropriate for the target audience?
    R - Reasoning: Is the "why" explained, not just the "how"?
    """
    completeness: int = 0      # 0-5
    logical_flow: int = 0      # 0-5
    explicit_terms: int = 0    # 0-5
    accessibility: int = 0     # 0-5
    reasoning: int = 0         # 0-5
    
    # Detailed feedback
    completeness_issues: list[str] = field(default_factory=list)
    logical_flow_issues: list[str] = field(default_factory=list)
    explicit_terms_issues: list[str] = field(default_factory=list)
    accessibility_issues: list[str] = field(default_factory=list)
    reasoning_issues: list[str] = field(default_factory=list)
    
    @property
    def total(self) -> int:
        """Total score out of 25."""
        return (
            self.completeness + 
            self.logical_flow + 
            self.explicit_terms + 
            self.accessibility + 
            self.reasoning
        )
    
    @property
    def passes(self) -> bool:
        """Check if score passes quality gate."""
        MIN_TOTAL = 20
        MIN_CATEGORY = 3
        
        return (
            self.total >= MIN_TOTAL and
            self.completeness >= MIN_CATEGORY and
            self.logical_flow >= MIN_CATEGORY and
            self.explicit_terms >= MIN_CATEGORY and
            self.accessibility >= MIN_CATEGORY and
            self.reasoning >= MIN_CATEGORY
        )
    
    @property
    def weakest_category(self) -> str:
        """Find the weakest scoring category."""
        scores = {
            'completeness': self.completeness,
            'logical_flow': self.logical_flow,
            'explicit_terms': self.explicit_terms,
            'accessibility': self.accessibility,
            'reasoning': self.reasoning,
        }
        return min(scores, key=scores.get)
    
    def to_report(self) -> str:
        """Generate detailed quality report."""
        status = "✓ PASSES" if self.passes else "✗ FAILS"
        
        report = [
            f"CLEAR Quality Score: {self.total}/25 {status}",
            "",
            f"  C - Completeness:   {self.completeness}/5",
        ]
        for issue in self.completeness_issues[:3]:
            report.append(f"      • {issue}")
        
        report.append(f"  L - Logical Flow:   {self.logical_flow}/5")
        for issue in self.logical_flow_issues[:3]:
            report.append(f"      • {issue}")
        
        report.append(f"  E - Explicit Terms: {self.explicit_terms}/5")
        for issue in self.explicit_terms_issues[:3]:
            report.append(f"      • {issue}")
        
        report.append(f"  A - Accessibility:  {self.accessibility}/5")
        for issue in self.accessibility_issues[:3]:
            report.append(f"      • {issue}")
        
        report.append(f"  R - Reasoning:      {self.reasoning}/5")
        for issue in self.reasoning_issues[:3]:
            report.append(f"      • {issue}")
        
        if not self.passes:
            report.append("")
            report.append(f"Weakest area: {self.weakest_category}")
        
        return "\n".join(report)


class QualityChecker:
    """
    Check explanation quality using CLEAR rubric.
    
    Evaluates explanations against five criteria and provides
    detailed feedback for improvement.
    
    Example:
        >>> checker = QualityChecker()
        >>> score = checker.check(explanation_text, problem_type)
        >>> if not score.passes:
        ...     print(score.to_report())
    """
    
    def __init__(self, target_level: str = "intermediate"):
        """
        Initialize quality checker.
        
        Args:
            target_level: Target audience level for accessibility scoring
        """
        self.target_level = target_level
        self.anti_pattern_detector = AntiPatternDetector()
    
    def check(
        self,
        explanation: str,
        problem_type: str = "general",
        defined_terms: Optional[set[str]] = None,
        steps_expected: int = 0,
    ) -> CLEARScore:
        """
        Evaluate explanation quality.
        
        Args:
            explanation: Full explanation text
            problem_type: Type of problem for context
            defined_terms: Set of terms that have been defined
            steps_expected: Minimum expected solution steps
            
        Returns:
            CLEARScore with scores and feedback
        """
        defined_terms = defined_terms or set()
        
        score = CLEARScore()
        
        # Evaluate each dimension
        score.completeness, score.completeness_issues = self._check_completeness(
            explanation, problem_type, steps_expected
        )
        
        score.logical_flow, score.logical_flow_issues = self._check_logical_flow(
            explanation
        )
        
        score.explicit_terms, score.explicit_terms_issues = self._check_explicit_terms(
            explanation, defined_terms
        )
        
        score.accessibility, score.accessibility_issues = self._check_accessibility(
            explanation
        )
        
        score.reasoning, score.reasoning_issues = self._check_reasoning(
            explanation
        )
        
        return score
    
    def _check_completeness(
        self,
        explanation: str,
        problem_type: str,
        steps_expected: int,
    ) -> tuple[int, list[str]]:
        """
        Check if all necessary components are present.
        
        Completeness criteria:
        - Has prerequisite definitions
        - Has concrete example
        - Has step-by-step solution
        - Has edge cases/limitations
        - Has verification/practice problems
        """
        score = 5
        issues = []
        
        # Check for key sections
        has_prereq = bool(re.search(r'prerequisite|before we|first.*define', explanation, re.I))
        has_example = bool(re.search(r'example|for instance|consider|let.*=', explanation, re.I))
        has_steps = bool(re.search(r'step \d|first.*then|1\.|2\.', explanation, re.I))
        has_edge_cases = bool(re.search(r'edge case|limitation|fail|exception|note that|warning|caveat', explanation, re.I))
        has_verification = bool(re.search(r'verify|check|practice|try|exercise', explanation, re.I))
        
        if not has_prereq:
            score -= 1
            issues.append("Missing prerequisite definitions")
        
        if not has_example:
            score -= 1
            issues.append("Missing concrete example")
        
        if not has_steps:
            score -= 1
            issues.append("Missing step-by-step breakdown")
        
        if not has_edge_cases:
            score -= 1
            issues.append("Missing edge cases/limitations")
        
        if not has_verification:
            score -= 0.5
            issues.append("Missing verification/practice problems")
        
        # Check step count
        step_count = len(re.findall(r'step \d|^\d+\.|\n\d+\)', explanation, re.I | re.M))
        if steps_expected > 0 and step_count < steps_expected * 0.7:
            score -= 1
            issues.append(f"Too few steps: {step_count} vs {steps_expected} expected")
        
        return max(0, int(score)), issues
    
    def _check_logical_flow(self, explanation: str) -> tuple[int, list[str]]:
        """
        Check if steps flow logically.
        
        Logical flow criteria:
        - Each step follows from previous
        - No unexplained jumps
        - Clear transitions
        - Consistent notation
        """
        score = 5
        issues = []
        
        # Detect anti-patterns that indicate poor flow
        anti_patterns = self.anti_pattern_detector.scan(explanation)
        
        step_skipping = [
            ap for ap in anti_patterns 
            if ap.pattern_type == 'step_skipping'
        ]
        
        if len(step_skipping) > 0:
            deduction = min(2, len(step_skipping) * 0.5)
            score -= deduction
            issues.append(f"Found {len(step_skipping)} step-skipping pattern(s)")
        
        # Check for transition words
        transition_patterns = [
            r'\b(because|since|therefore|thus|hence|so)\b',
            r'\b(first|then|next|finally)\b',
            r'\b(this means|this implies|from this)\b',
        ]
        
        transition_count = sum(
            len(re.findall(p, explanation, re.I)) 
            for p in transition_patterns
        )
        
        if transition_count < 3:
            score -= 1
            issues.append("Few transition words - connections between steps unclear")
        
        # Check for abrupt topic changes
        paragraphs = explanation.split('\n\n')
        if len(paragraphs) > 1:
            # Simple heuristic: look for paragraphs that don't reference previous
            for i in range(1, len(paragraphs)):
                if not re.search(r'\b(this|that|these|those|it|from above|as shown|previously)\b', 
                                paragraphs[i][:100], re.I):
                    issues.append("Some paragraphs don't connect to previous content")
                    score -= 0.5
                    break
        
        return max(0, int(score)), issues
    
    def _check_explicit_terms(
        self,
        explanation: str,
        defined_terms: set[str],
    ) -> tuple[int, list[str]]:
        """
        Check if technical terms are defined before use.
        
        Explicit terms criteria:
        - All jargon defined
        - Definitions come before use
        - No circular definitions
        - Notation explained
        """
        score = 5
        issues = []
        
        # Check for undefined jargon
        undefined = self.anti_pattern_detector.check_jargon(explanation, defined_terms)
        
        if undefined:
            deduction = min(2, len(undefined) * 0.4)
            score -= deduction
            issues.append(f"Undefined terms: {', '.join(undefined[:5])}")
        
        # Check for definition-hiding patterns
        anti_patterns = self.anti_pattern_detector.scan(explanation)
        
        def_hiding = [
            ap for ap in anti_patterns 
            if ap.pattern_type == 'definition_hiding'
        ]
        
        if len(def_hiding) > 0:
            deduction = min(1.5, len(def_hiding) * 0.5)
            score -= deduction
            issues.append(f"Found {len(def_hiding)} definition-hiding pattern(s)")
        
        # Check for notation explanation
        has_notation = bool(re.search(
            r'where|here|denotes|represents|means|is defined as', 
            explanation, re.I
        ))
        
        # Look for mathematical symbols that should be explained
        symbols = re.findall(r'[∀∃∈∉⊂⊃∪∩∧∨¬→↔≡≠≤≥∞∂∇∫∑∏√]', explanation)
        if symbols and not has_notation:
            score -= 1
            issues.append("Mathematical symbols used without explanation")
        
        return max(0, int(score)), issues
    
    def _check_accessibility(self, explanation: str) -> tuple[int, list[str]]:
        """
        Check if explanation matches target difficulty level.
        
        Accessibility criteria:
        - Appropriate vocabulary
        - Good example-to-abstraction ratio
        - Not too dense
        - Clear structure
        """
        score = 5
        issues = []
        
        # Calculate jargon density
        words = explanation.split()
        total_words = len(words)
        
        if total_words == 0:
            return 0, ["Empty explanation"]
        
        # Check sentence length (long sentences are harder to follow)
        sentences = re.split(r'[.!?]+', explanation)
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
        
        if avg_sentence_length > 30:
            score -= 1
            issues.append(f"Sentences too long (avg {avg_sentence_length:.0f} words)")
        
        # Check for concrete examples
        example_markers = len(re.findall(
            r'\b(example|for instance|e\.g\.|such as|like|consider)\b', 
            explanation, re.I
        ))
        
        if example_markers < 1:
            score -= 1
            issues.append("No concrete examples found")
        
        # Check for analogies (good for accessibility)
        has_analogy = bool(re.search(
            r'\b(like|similar to|analogous|think of|imagine)\b', 
            explanation, re.I
        ))
        
        if not has_analogy and self.target_level in ['amateur', 'intermediate']:
            score -= 0.5
            issues.append("No analogies or intuitive comparisons")
        
        # Check paragraph length (walls of text are hard to read)
        paragraphs = [p for p in explanation.split('\n\n') if p.strip()]
        long_paragraphs = [p for p in paragraphs if len(p.split()) > 150]
        
        if long_paragraphs:
            score -= 0.5
            issues.append(f"{len(long_paragraphs)} paragraph(s) too long")
        
        return max(0, int(score)), issues
    
    def _check_reasoning(self, explanation: str) -> tuple[int, list[str]]:
        """
        Check if "why" is explained, not just "how".
        
        Reasoning criteria:
        - Justification for each step
        - Explains motivation
        - Addresses "why this approach"
        - Shows underlying principle
        """
        score = 5
        issues = []
        
        # Check for "why" indicators
        why_patterns = [
            r'\b(because|since|reason|why|due to|in order to)\b',
            r'\b(this works because|this is true because)\b',
            r'\b(the idea is|the key insight|the motivation)\b',
            r'\b(intuitively|the reason)\b',
        ]
        
        why_count = sum(
            len(re.findall(p, explanation, re.I)) 
            for p in why_patterns
        )
        
        if why_count < 2:
            score -= 2
            issues.append("Missing explanations of WHY (not just how)")
        elif why_count < 4:
            score -= 1
            issues.append("Could use more 'why' explanations")
        
        # Check for trivializing anti-patterns (opposite of reasoning)
        anti_patterns = self.anti_pattern_detector.scan(explanation)
        
        trivializing = [
            ap for ap in anti_patterns 
            if ap.pattern_type == 'trivializing'
        ]
        
        if trivializing:
            deduction = min(2, len(trivializing) * 0.7)
            score -= deduction
            issues.append(f"Found {len(trivializing)} trivializing pattern(s) that skip reasoning")
        
        # Check for vagueness
        vague = [
            ap for ap in anti_patterns 
            if ap.pattern_type == 'vagueness'
        ]
        
        if vague:
            deduction = min(1, len(vague) * 0.3)
            score -= deduction
            issues.append(f"Found {len(vague)} vague statement(s)")
        
        # Check for principle/insight statement
        has_insight = bool(re.search(
            r'\b(principle|insight|key|important|crucial|essential|underlying)\b',
            explanation, re.I
        ))
        
        if not has_insight:
            score -= 0.5
            issues.append("No clear statement of underlying principle")
        
        return max(0, int(score)), issues
    
    def get_improvement_suggestions(self, score: CLEARScore) -> list[str]:
        """
        Generate specific suggestions to improve the explanation.
        
        Args:
            score: CLEAR score from check()
            
        Returns:
            List of improvement suggestions
        """
        suggestions = []
        
        if score.completeness < 4:
            suggestions.append(
                "Add more structure: prerequisites → example → steps → edge cases → practice"
            )
        
        if score.logical_flow < 4:
            suggestions.append(
                "Add transition words (because, therefore, this means) to connect steps"
            )
        
        if score.explicit_terms < 4:
            suggestions.append(
                "Define all technical terms before first use"
            )
        
        if score.accessibility < 4:
            suggestions.append(
                "Add concrete examples and analogies; break up long paragraphs"
            )
        
        if score.reasoning < 4:
            suggestions.append(
                "Explain WHY each step works, not just what the step is"
            )
        
        return suggestions
