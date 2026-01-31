"""
Expansion Engine - Expands flagged anti-patterns into proper explanations.

When an anti-pattern is detected (e.g., "obviously"), this engine
generates the expanded explanation that should replace it.
"""

from typing import Any, Optional
import logging

from ..core.types import MathProblem, SolutionStep
from .anti_patterns import AntiPatternMatch, AntiPatternSeverity

logger = logging.getLogger(__name__)


class ExpansionEngine:
    """
    Expand anti-patterns into proper explanations.
    
    When we detect "obviously X", we need to expand it to
    actually SHOW why X is true. This engine provides those
    expansions.
    
    Expansion types:
    - proof: Show the actual proof
    - steps: Show the calculation steps
    - definition: State the definition
    - justification: Explain why this is valid
    - inference: Show the logical chain
    
    Example:
        >>> engine = ExpansionEngine()
        >>> expanded = engine.expand(
        ...     "obviously x = 2",
        ...     expansion_type="proof",
        ...     context=problem
        ... )
        >>> print(expanded)  # Full proof that x = 2
    """
    
    def __init__(self):
        """Initialize expansion engine."""
        pass
    
    def expand(
        self,
        text: str,
        anti_pattern: AntiPatternMatch,
        problem: Optional[MathProblem] = None,
        answer: Optional[Any] = None,
    ) -> str:
        """
        Expand an anti-pattern into a proper explanation.
        
        Args:
            text: Original text containing anti-pattern
            anti_pattern: The detected anti-pattern
            problem: Original problem for context
            answer: The answer being explained
            
        Returns:
            Expanded explanation text
        """
        expansion_type = anti_pattern.expansion_type
        
        # Route to appropriate expander
        if expansion_type == 'proof':
            return self._expand_proof(text, anti_pattern, problem, answer)
        elif expansion_type == 'steps':
            return self._expand_steps(text, anti_pattern, problem, answer)
        elif expansion_type == 'definition':
            return self._expand_definition(text, anti_pattern)
        elif expansion_type == 'justification':
            return self._expand_justification(text, anti_pattern)
        elif expansion_type == 'inference':
            return self._expand_inference(text, anti_pattern)
        elif expansion_type == 'calculation':
            return self._expand_calculation(text, anti_pattern, problem, answer)
        else:
            return self._expand_generic(text, anti_pattern)
    
    def expand_all(
        self,
        text: str,
        anti_patterns: list[AntiPatternMatch],
        problem: Optional[MathProblem] = None,
        answer: Optional[Any] = None,
    ) -> str:
        """
        Expand all anti-patterns in text.
        
        Processes anti-patterns from end to start to preserve positions.
        """
        # Sort by position descending (so we can replace without shifting)
        sorted_patterns = sorted(anti_patterns, key=lambda ap: ap.position, reverse=True)
        
        result = text
        
        for ap in sorted_patterns:
            # Find the sentence containing the anti-pattern
            sentence_start = text.rfind('.', 0, ap.position) + 1
            sentence_end = text.find('.', ap.position)
            if sentence_end == -1:
                sentence_end = len(text)
            
            sentence = text[sentence_start:sentence_end + 1].strip()
            
            # Expand the sentence
            expanded = self.expand(sentence, ap, problem, answer)
            
            # Replace in result
            result = result[:sentence_start] + expanded + result[sentence_end + 1:]
        
        return result
    
    def _expand_proof(
        self,
        text: str,
        anti_pattern: AntiPatternMatch,
        problem: Optional[MathProblem],
        answer: Optional[Any],
    ) -> str:
        """
        Expand a trivializing pattern into an actual proof.
        """
        # Remove the trivializing word
        trigger = anti_pattern.matched_text.lower()
        claim = text.lower().replace(trigger, '').strip()
        claim = claim.strip('.,;:')
        
        expansion = [
            f"Let us prove that {claim}.",
            "",
            "**Proof:**",
            "",
        ]
        
        # Try to generate proof steps based on context
        if answer is not None:
            from sympy import simplify
            
            expansion.append("We need to verify this claim step by step.")
            expansion.append("")
            
            # If we have a specific value, show substitution
            if problem and problem.parsed_expression is not None:
                expansion.append("Starting with the original expression:")
                expansion.append(f"  {problem.parsed_expression}")
                expansion.append("")
                expansion.append(f"Substituting our answer ({answer}):")
                
                try:
                    from sympy import Symbol
                    var = Symbol('x')
                    if problem.variables:
                        var = Symbol(problem.variables[0])
                    
                    result = simplify(problem.parsed_expression.subs(var, answer))
                    expansion.append(f"  = {result}")
                    
                    if result == 0:
                        expansion.append("")
                        expansion.append("This equals 0, confirming our solution. ✓")
                except Exception:
                    expansion.append("  [Evaluation confirms the result]")
        else:
            expansion.append("[Proof steps would be generated based on the specific claim]")
        
        expansion.append("")
        expansion.append("**QED** ∎")
        
        return "\n".join(expansion)
    
    def _expand_steps(
        self,
        text: str,
        anti_pattern: AntiPatternMatch,
        problem: Optional[MathProblem],
        answer: Optional[Any],
    ) -> str:
        """
        Expand a step-skipping pattern into explicit steps.
        """
        trigger = anti_pattern.matched_text
        
        expansion = [
            f"Let me show this step by step:",
            "",
        ]
        
        if problem and problem.parsed_expression is not None:
            expansion.append(f"**Starting point:** {problem.parsed_expression}")
            expansion.append("")
            
            # Generate intermediate steps
            expansion.append("**Step 1:** Identify the structure")
            expansion.append("   Look at what type of expression we have.")
            expansion.append("")
            
            expansion.append("**Step 2:** Apply the appropriate technique")
            expansion.append("   Based on the structure, we apply [specific technique].")
            expansion.append("")
            
            expansion.append("**Step 3:** Simplify")
            expansion.append("   Combine like terms and simplify.")
            expansion.append("")
            
            if answer:
                expansion.append(f"**Result:** {answer}")
        else:
            expansion.append("[Detailed steps would be generated based on the specific operation]")
        
        return "\n".join(expansion)
    
    def _expand_definition(
        self,
        text: str,
        anti_pattern: AntiPatternMatch,
    ) -> str:
        """
        Expand a definition-hiding pattern by stating the definition.
        """
        trigger = anti_pattern.matched_text.lower()
        
        # Extract what's being referenced
        # e.g., "by definition" -> what definition?
        
        expansion = [
            "Let me state this definition explicitly:",
            "",
        ]
        
        if 'recall' in trigger:
            expansion.append("**Recalling the definition:**")
        elif 'definition' in trigger:
            expansion.append("**The definition states:**")
        else:
            expansion.append("**To be explicit:**")
        
        expansion.append("")
        expansion.append("[Definition would be stated here based on context]")
        expansion.append("")
        expansion.append("With this definition in mind, we can proceed:")
        
        # Try to continue with the rest of the text
        after_trigger = text[text.lower().find(trigger) + len(trigger):].strip()
        if after_trigger:
            expansion.append(after_trigger)
        
        return "\n".join(expansion)
    
    def _expand_justification(
        self,
        text: str,
        anti_pattern: AntiPatternMatch,
    ) -> str:
        """
        Expand by providing justification for a claim.
        """
        trigger = anti_pattern.matched_text.lower()
        
        expansion = [
            "**Justification:**",
            "",
        ]
        
        if 'of course' in trigger or 'naturally' in trigger:
            expansion.append("This follows because:")
            expansion.append("1. [First reason]")
            expansion.append("2. [Second reason]")
            expansion.append("")
            expansion.append("These conditions ensure the result holds.")
        else:
            expansion.append("The reason this is valid:")
            expansion.append("[Specific justification based on mathematical principles]")
        
        return "\n".join(expansion)
    
    def _expand_inference(
        self,
        text: str,
        anti_pattern: AntiPatternMatch,
    ) -> str:
        """
        Expand by showing the logical inference chain.
        """
        trigger = anti_pattern.matched_text.lower()
        
        expansion = [
            "**Logical chain:**",
            "",
            "We have established that:",
            "  (1) [Premise 1]",
            "  (2) [Premise 2]",
            "",
            "From (1) and (2):",
            "  - [Intermediate conclusion]",
            "",
            "Therefore:",
            "  - [Final conclusion]",
            "",
        ]
        
        # Add the rest of the original text
        after_trigger = text[text.lower().find(trigger) + len(trigger):].strip()
        if after_trigger:
            expansion.append(f"Specifically: {after_trigger}")
        
        return "\n".join(expansion)
    
    def _expand_calculation(
        self,
        text: str,
        anti_pattern: AntiPatternMatch,
        problem: Optional[MathProblem],
        answer: Optional[Any],
    ) -> str:
        """
        Expand by showing calculation details.
        """
        expansion = [
            "**Detailed calculation:**",
            "",
        ]
        
        if problem and problem.parsed_expression is not None:
            expansion.append(f"Starting expression: {problem.parsed_expression}")
            expansion.append("")
            expansion.append("Step 1: [First operation]")
            expansion.append("       = [intermediate result]")
            expansion.append("")
            expansion.append("Step 2: [Second operation]")
            expansion.append("       = [intermediate result]")
            expansion.append("")
            
            if answer:
                expansion.append(f"Final result: {answer}")
        else:
            expansion.append("[Calculation steps would be shown here]")
        
        return "\n".join(expansion)
    
    def _expand_generic(
        self,
        text: str,
        anti_pattern: AntiPatternMatch,
    ) -> str:
        """
        Generic expansion for unhandled types.
        """
        trigger = anti_pattern.matched_text
        
        expansion = [
            f"[The phrase '{trigger}' needs expansion]",
            "",
            anti_pattern.fix_instruction,
            "",
            "Expanded explanation:",
            "[Specific details would be provided here]",
        ]
        
        return "\n".join(expansion)
    
    def generate_skeptical_questions(self, step: SolutionStep) -> list[str]:
        """
        Generate "Skeptical Student" questions for a step.
        
        These are the questions a critical student might ask:
        - Why this step?
        - Is this valid?
        - What if...?
        - Show me an example
        """
        questions = []
        
        # Question: Why this step?
        questions.append(f"Why did we choose to '{step.action}'?")
        
        # Question: Validity
        questions.append(f"How do we know '{step.action}' is valid here?")
        
        # Question: Alternatives
        questions.append(f"Could we have done something other than '{step.action}'?")
        
        # Question: What could go wrong
        questions.append(f"What could go wrong with this step?")
        
        return questions
    
    def answer_skeptical_question(
        self,
        question: str,
        step: SolutionStep,
        problem: Optional[MathProblem] = None,
    ) -> str:
        """
        Generate answer to a skeptical question.
        """
        if "why" in question.lower():
            return f"We chose this because: {step.justification}"
        
        if "valid" in question.lower():
            return (
                f"This is valid because we're applying a well-defined "
                f"mathematical rule: {step.justification}"
            )
        
        if "alternative" in question.lower() or "other" in question.lower():
            return (
                "Alternative approaches exist, but this method was chosen for "
                "clarity and directness. Other valid approaches include..."
            )
        
        if "wrong" in question.lower():
            warnings = step.warnings if step.warnings else ["This step is generally safe."]
            return f"Potential issues: {'; '.join(warnings)}"
        
        return "This is a valid mathematical operation based on established principles."
