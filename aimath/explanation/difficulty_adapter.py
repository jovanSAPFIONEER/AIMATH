"""
Difficulty Adapter - Adjust explanations for different skill levels.

Adapts vocabulary, detail level, and examples to match the target
audience (amateur → expert).
"""

from enum import Enum
from typing import Any, Optional
import re
import logging

from ..core.types import DifficultyLevel, SolutionStep

logger = logging.getLogger(__name__)


class DifficultyAdapter:
    """
    Adapt explanations to different mathematical skill levels.
    
    Levels:
    - AMATEUR: High school level, minimal prerequisites
    - INTERMEDIATE: Undergraduate level, knows calculus basics
    - ADVANCED: Graduate level, familiar with advanced topics
    - EXPERT: Research level, can handle abstraction
    
    Adaptations include:
    - Vocabulary simplification/enrichment
    - Example concreteness
    - Step granularity
    - Notation complexity
    
    Example:
        >>> adapter = DifficultyAdapter(DifficultyLevel.AMATEUR)
        >>> simple = adapter.adapt(complex_explanation)
    """
    
    # Vocabulary mappings for simplification
    SIMPLIFY_VOCAB = {
        'differentiate': 'find the derivative of',
        'integrate': 'find the integral of',
        'convergent': 'approaches a specific value',
        'divergent': 'grows without bound',
        'continuous': 'has no gaps or jumps',
        'injective': 'one-to-one',
        'surjective': 'onto',
        'bijective': 'one-to-one and onto',
        'domain': 'valid input values',
        'codomain': 'possible output values',
        'range': 'actual output values',
        'monotonic': 'always increasing or always decreasing',
        'bounded': 'stays within limits',
        'asymptotic': 'approaches but never reaches',
    }
    
    # Prerequisites by level
    LEVEL_PREREQUISITES = {
        DifficultyLevel.AMATEUR: {
            'algebra', 'basic geometry', 'arithmetic', 'fractions',
        },
        DifficultyLevel.INTERMEDIATE: {
            'algebra', 'geometry', 'trigonometry', 'basic calculus',
            'limits', 'derivatives', 'integrals',
        },
        DifficultyLevel.ADVANCED: {
            'calculus', 'linear algebra', 'differential equations',
            'real analysis basics', 'abstract algebra basics',
        },
        DifficultyLevel.EXPERT: {
            # Experts are assumed to know fundamentals
        },
    }
    
    def __init__(self, target_level: DifficultyLevel = DifficultyLevel.INTERMEDIATE):
        """
        Initialize adapter.
        
        Args:
            target_level: Target difficulty level
        """
        self.target_level = target_level
    
    def adapt(
        self,
        explanation: str,
        source_level: DifficultyLevel = DifficultyLevel.ADVANCED,
    ) -> str:
        """
        Adapt explanation to target difficulty level.
        
        Args:
            explanation: Original explanation text
            source_level: Difficulty level of source explanation
            
        Returns:
            Adapted explanation
        """
        if self.target_level == source_level:
            return explanation
        
        # Determine if we're simplifying or enriching
        if self._level_value(self.target_level) < self._level_value(source_level):
            return self._simplify(explanation)
        else:
            return self._enrich(explanation)
    
    def _level_value(self, level: DifficultyLevel) -> int:
        """Convert level to numeric value for comparison."""
        return {
            DifficultyLevel.AMATEUR: 1,
            DifficultyLevel.INTERMEDIATE: 2,
            DifficultyLevel.ADVANCED: 3,
            DifficultyLevel.EXPERT: 4,
        }.get(level, 2)
    
    def _simplify(self, text: str) -> str:
        """
        Simplify explanation for lower level.
        
        - Replace jargon with simpler terms
        - Add more context
        - Break up long sentences
        """
        result = text
        
        # Replace complex vocabulary
        if self.target_level == DifficultyLevel.AMATEUR:
            for complex_term, simple_term in self.SIMPLIFY_VOCAB.items():
                pattern = rf'\b{complex_term}\b'
                result = re.sub(
                    pattern, 
                    f'{simple_term} ({complex_term})', 
                    result, 
                    flags=re.IGNORECASE
                )
        
        # Break up long sentences
        result = self._break_long_sentences(result)
        
        # Add clarifying phrases
        result = self._add_clarifications(result)
        
        return result
    
    def _enrich(self, text: str) -> str:
        """
        Enrich explanation for higher level.
        
        - Use more precise terminology
        - Add formal notation
        - Include deeper insights
        """
        result = text
        
        # Replace informal with formal
        if self.target_level in [DifficultyLevel.ADVANCED, DifficultyLevel.EXPERT]:
            informal_to_formal = {
                'find the derivative': 'differentiate',
                'find the integral': 'integrate',
                'always increasing or always decreasing': 'monotonic',
                'approaches a specific value': 'converges',
            }
            
            for informal, formal in informal_to_formal.items():
                result = result.replace(informal, formal)
        
        return result
    
    def _break_long_sentences(self, text: str, max_words: int = 25) -> str:
        """Break sentences longer than max_words."""
        sentences = re.split(r'([.!?]+\s*)', text)
        result = []
        
        for i in range(0, len(sentences) - 1, 2):
            sentence = sentences[i]
            punctuation = sentences[i + 1] if i + 1 < len(sentences) else '.'
            
            words = sentence.split()
            if len(words) > max_words:
                # Try to find a natural break point
                midpoint = len(words) // 2
                
                # Look for conjunctions near midpoint
                for j in range(midpoint - 3, midpoint + 3):
                    if 0 < j < len(words):
                        if words[j].lower() in ['and', 'but', 'so', 'because', 'which', 'where']:
                            # Split here
                            first_part = ' '.join(words[:j])
                            second_part = ' '.join(words[j:])
                            result.append(first_part + '.')
                            result.append(second_part.capitalize() + punctuation)
                            break
                else:
                    # No good break point, keep as is
                    result.append(sentence + punctuation)
            else:
                result.append(sentence + punctuation)
        
        return ' '.join(result)
    
    def _add_clarifications(self, text: str) -> str:
        """Add clarifying phrases for amateur level."""
        if self.target_level != DifficultyLevel.AMATEUR:
            return text
        
        # Add explanations after first use of certain concepts
        clarifications = {
            r'\bequation\b': 'equation (a mathematical statement with an equals sign)',
            r'\bexpression\b': 'expression (a combination of numbers and symbols)',
            r'\bvariable\b': 'variable (a letter representing an unknown number)',
            r'\bcoefficient\b': 'coefficient (the number multiplying a variable)',
        }
        
        result = text
        for pattern, replacement in clarifications.items():
            # Only replace first occurrence
            if re.search(pattern, result, re.IGNORECASE):
                result = re.sub(pattern, replacement, result, count=1, flags=re.IGNORECASE)
        
        return result
    
    def adapt_step(
        self,
        step: SolutionStep,
    ) -> SolutionStep:
        """
        Adapt a single solution step to target level.
        
        Args:
            step: Original solution step
            
        Returns:
            Adapted solution step
        """
        adapted_action = self.adapt(step.action)
        adapted_justification = self.adapt(step.justification)
        
        # For amateur level, add more context to warnings
        adapted_warnings = []
        for warning in step.warnings:
            adapted_warning = self.adapt(warning)
            if self.target_level == DifficultyLevel.AMATEUR:
                adapted_warning += " (This is important to watch out for!)"
            adapted_warnings.append(adapted_warning)
        
        return SolutionStep(
            action=adapted_action,
            expression=step.expression,  # Keep expression as-is
            justification=adapted_justification,
            from_step=step.from_step,
            warnings=adapted_warnings,
        )
    
    def get_required_prerequisites(self, explanation: str) -> set[str]:
        """
        Identify prerequisites needed to understand explanation.
        
        Returns:
            Set of required prerequisite topics
        """
        prerequisites = set()
        
        # Check for calculus concepts
        if re.search(r'\b(derivative|differentiat|d/dx)\b', explanation, re.I):
            prerequisites.add('derivatives')
        if re.search(r'\b(integral|∫|antiderivative)\b', explanation, re.I):
            prerequisites.add('integrals')
        if re.search(r'\b(limit|lim|→)\b', explanation, re.I):
            prerequisites.add('limits')
        
        # Check for algebra concepts
        if re.search(r'\b(quadratic|x\^2|polynomial)\b', explanation, re.I):
            prerequisites.add('algebra')
        if re.search(r'\b(factor|roots|zeros)\b', explanation, re.I):
            prerequisites.add('factoring')
        
        # Check for trigonometry
        if re.search(r'\b(sin|cos|tan|trig)\b', explanation, re.I):
            prerequisites.add('trigonometry')
        
        # Check for linear algebra
        if re.search(r'\b(matrix|vector|eigenvalue|determinant)\b', explanation, re.I):
            prerequisites.add('linear algebra')
        
        return prerequisites
    
    def check_level_appropriateness(self, explanation: str) -> dict:
        """
        Check if explanation is appropriate for target level.
        
        Returns:
            Dict with assessment and recommendations
        """
        prerequisites = self.get_required_prerequisites(explanation)
        level_prereqs = self.LEVEL_PREREQUISITES.get(self.target_level, set())
        
        # Find missing prerequisites
        missing = prerequisites - level_prereqs
        
        # Calculate jargon density
        words = explanation.split()
        jargon_count = sum(
            1 for word in words 
            if word.lower() in self.SIMPLIFY_VOCAB
        )
        jargon_density = jargon_count / max(len(words), 1)
        
        # Determine appropriateness
        appropriate = len(missing) == 0 and (
            (self.target_level == DifficultyLevel.AMATEUR and jargon_density < 0.05) or
            (self.target_level == DifficultyLevel.INTERMEDIATE and jargon_density < 0.10) or
            (self.target_level in [DifficultyLevel.ADVANCED, DifficultyLevel.EXPERT])
        )
        
        return {
            'appropriate': appropriate,
            'target_level': self.target_level.value,
            'missing_prerequisites': list(missing),
            'jargon_density': jargon_density,
            'recommendation': (
                'Explanation is appropriate for target level' if appropriate
                else f'Consider adding explanations for: {", ".join(missing)}' if missing
                else 'Consider simplifying vocabulary'
            ),
        }
    
    def suggest_examples(self, concept: str) -> list[str]:
        """
        Suggest appropriate examples for target level.
        
        Args:
            concept: Mathematical concept being explained
            
        Returns:
            List of example suggestions
        """
        examples = {
            DifficultyLevel.AMATEUR: {
                'derivative': 'Speed is the derivative of distance. If you drive 60 miles in 1 hour, your speed (derivative) is 60 mph.',
                'integral': 'If you know your speed over time, the integral tells you total distance traveled.',
                'limit': 'Think of getting closer and closer to a wall but never touching it.',
                'equation': 'If you have 3 apples and get some more to have 7, the equation is 3 + x = 7.',
            },
            DifficultyLevel.INTERMEDIATE: {
                'derivative': 'The derivative of f(x) = x² is f\'(x) = 2x, representing the slope at any point.',
                'integral': '∫x² dx = x³/3 + C represents the area under the parabola.',
                'limit': 'lim(x→0) sin(x)/x = 1, fundamental in calculus.',
            },
            DifficultyLevel.ADVANCED: {
                'derivative': 'The Fréchet derivative generalizes differentiation to Banach spaces.',
                'integral': 'Lebesgue integration extends Riemann integration to handle more functions.',
            },
        }
        
        level_examples = examples.get(self.target_level, {})
        return [level_examples.get(concept, f'Example of {concept} at {self.target_level.value} level')]
