"""
Anti-Pattern Detector - Identifies and flags explanation anti-patterns.

This is the core defense against AI explanation failures:
- Detects vague/hand-wavy language
- Flags skipped steps
- Identifies undefined jargon
- Catches circular definitions
- Ensures "why" is present

Every anti-pattern triggers mandatory expansion.
"""

import re
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class AntiPatternSeverity(Enum):
    """Severity levels for anti-patterns."""
    CRITICAL = "critical"    # Must be fixed, blocks explanation
    HIGH = "high"            # Should be fixed
    MEDIUM = "medium"        # Recommended to fix
    LOW = "low"              # Nice to fix


@dataclass
class AntiPatternMatch:
    """
    A detected anti-pattern in explanation text.
    
    Attributes:
        pattern_type: Category of anti-pattern
        severity: How severe the issue is
        matched_text: The text that matched
        position: Character position in text
        fix_instruction: How to fix this
        expansion_type: Type of expansion needed
    """
    pattern_type: str
    severity: AntiPatternSeverity
    matched_text: str
    position: int
    fix_instruction: str
    expansion_type: str  # e.g., "proof", "definition", "steps"


class AntiPatternDetector:
    """
    Detect explanation anti-patterns that indicate poor quality.
    
    Categories:
    1. TRIVIALIZING - "obviously", "clearly", "simply", etc.
    2. STEP_SKIPPING - "it follows", "therefore" without showing work
    3. JARGON_HIDING - undefined technical terms
    4. CIRCULAR - defining terms with themselves
    5. VAGUE - hand-wavy language without substance
    6. MISSING_WHY - steps without justification
    
    Example:
        >>> detector = AntiPatternDetector()
        >>> matches = detector.scan("Obviously, x = 2")
        >>> print(matches[0].pattern_type)  # "trivializing"
        >>> print(matches[0].fix_instruction)  # "Prove it's obvious..."
    """
    
    # ==========================================================================
    # LEVEL 1: TRIVIALIZING PATTERNS (Require immediate proof/expansion)
    # ==========================================================================
    TRIVIALIZING_PATTERNS = {
        r'\b(obviously|obviouly)\b': {
            'fix': "Remove 'obviously' and prove why it's true. If it's obvious, showing it should be easy.",
            'expansion': 'proof',
        },
        r'\b(clearly|cleary)\b': {
            'fix': "Remove 'clearly' and demonstrate clearly. Show the reasoning explicitly.",
            'expansion': 'proof',
        },
        r'\b(trivially|trivial)\b': {
            'fix': "If it's trivial, show the trivial steps. Trivial to experts may not be trivial to learners.",
            'expansion': 'proof',
        },
        r'\bsimply\b(?!\s+put)': {  # "simply put" is OK
            'fix': "Remove 'simply' and show the simple steps explicitly.",
            'expansion': 'steps',
        },
        r'\b(easy to see|easily seen)\b': {
            'fix': "Show what's easy to see. Make it visible, not just claimed.",
            'expansion': 'proof',
        },
        r'\b(straightforward)\b': {
            'fix': "Show the straightforward process step by step.",
            'expansion': 'steps',
        },
        r'\b(of course)\b': {
            'fix': "Don't assume shared knowledge. Explain why 'of course'.",
            'expansion': 'justification',
        },
        r'\b(naturally)\b': {
            'fix': "Explain what makes this 'natural'. Why does it follow?",
            'expansion': 'justification',
        },
    }
    
    # ==========================================================================
    # LEVEL 2: STEP-SKIPPING PATTERNS (Require showing the inference)
    # ==========================================================================
    STEP_SKIPPING_PATTERNS = {
        r'\b(it follows that)\b': {
            'fix': "Show HOW it follows. Make the logical chain explicit.",
            'expansion': 'inference',
        },
        r'\b(therefore|thus|hence)\b(?!.*because)': {
            'fix': "Before 'therefore', ensure all premises are stated and the inference is clear.",
            'expansion': 'inference',
        },
        r'\b(we get|we have|we obtain)\b(?!.*by|.*from|.*since)': {
            'fix': "Show HOW we get this. What operation or rule was applied?",
            'expansion': 'steps',
        },
        r'\b(this gives us)\b': {
            'fix': "Explain what operation gives us this result.",
            'expansion': 'steps',
        },
        r'\bwhich\s+(gives|yields|produces)\b': {
            'fix': "Show the calculation or transformation that produces this.",
            'expansion': 'calculation',
        },
        r'\b(after simplification|simplifying)\b(?!.*we get)': {
            'fix': "Show the simplification steps, not just the result.",
            'expansion': 'steps',
        },
        r'\b(by algebra|algebraically)\b': {
            'fix': "Show the algebraic manipulation explicitly.",
            'expansion': 'calculation',
        },
    }
    
    # ==========================================================================
    # LEVEL 3: DEFINITION-HIDING PATTERNS (Require stating definitions)
    # ==========================================================================
    DEFINITION_HIDING_PATTERNS = {
        r'\b(by definition)\b': {
            'fix': "State the definition being invoked. Don't just reference it.",
            'expansion': 'definition',
        },
        r'\b(recall that)\b': {
            'fix': "State what we're recalling. The reader may not remember.",
            'expansion': 'definition',
        },
        r'\b(as we know|as is known)\b': {
            'fix': "State what is known. Don't assume shared knowledge.",
            'expansion': 'statement',
        },
        r'\b(using the fact that)\b': {
            'fix': "State the fact and why it's true or where it comes from.",
            'expansion': 'justification',
        },
        r'\b(by|from) the (\w+) (theorem|lemma|property|rule)\b': {
            'fix': "State what this theorem/property says, not just its name.",
            'expansion': 'statement',
        },
        r'\b(similarly|analogously)\b': {
            'fix': "Show the parallel explicitly. What's similar and why?",
            'expansion': 'parallel',
        },
        r'\b(by a similar argument)\b': {
            'fix': "Write out the similar argument. Don't just reference it.",
            'expansion': 'proof',
        },
    }
    
    # ==========================================================================
    # LEVEL 4: VAGUENESS PATTERNS (Require substantiation)
    # ==========================================================================
    VAGUENESS_PATTERNS = {
        r'\b(it is well[- ]known that)\b': {
            'fix': "Either prove it or cite a source. 'Well-known' is not a proof.",
            'expansion': 'proof_or_citation',
        },
        r'\b(it can be shown that)\b': {
            'fix': "Show it. Don't just claim it can be shown.",
            'expansion': 'proof',
        },
        r'\b(one can (prove|verify|show) that)\b': {
            'fix': "Do the proof/verification here, not hypothetically.",
            'expansion': 'proof',
        },
        r'\b(it turns out that)\b': {
            'fix': "Show how it turns out. What's the derivation?",
            'expansion': 'derivation',
        },
        r'\b(it is (easy|possible) to show)\b': {
            'fix': "If it's easy/possible, do it. Show the work.",
            'expansion': 'proof',
        },
        r'\b(the reader (can|may) verify)\b': {
            'fix': "Verify it yourself and show the verification.",
            'expansion': 'verification',
        },
        r'\b(left as (an )?exercise)\b': {
            'fix': "Complete the exercise. This is an explanation, not a textbook.",
            'expansion': 'solution',
        },
        r'\b(details (are )?omitted)\b': {
            'fix': "Include the details. Omitted details mean incomplete understanding.",
            'expansion': 'details',
        },
        r'\b(without loss of generality|wlog)\b': {
            'fix': "Explain why there's no loss of generality. What cases are being ignored and why?",
            'expansion': 'justification',
        },
    }
    
    # ==========================================================================
    # LEVEL 5: PASSIVE VOICE HIDING (Require active voice with agent)
    # ==========================================================================
    PASSIVE_HIDING_PATTERNS = {
        r'\b(it is (assumed|given|stated))\b': {
            'fix': "State who assumes this or where it's given.",
            'expansion': 'attribution',
        },
        r'\b(it (has been|was) (shown|proven|established))\b': {
            'fix': "Show it here or cite where it was shown.",
            'expansion': 'proof_or_citation',
        },
        r'\b(is defined as)\b(?!.*where)': {
            'fix': "After defining, explain what the definition means intuitively.",
            'expansion': 'intuition',
        },
    }
    
    def __init__(self, strict_mode: bool = True):
        """
        Initialize detector.
        
        Args:
            strict_mode: If True, treat all patterns as requiring expansion
        """
        self.strict_mode = strict_mode
        
        # Compile all patterns
        self._compiled_patterns = {}
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile all regex patterns for efficiency."""
        pattern_groups = [
            ('trivializing', self.TRIVIALIZING_PATTERNS, AntiPatternSeverity.CRITICAL),
            ('step_skipping', self.STEP_SKIPPING_PATTERNS, AntiPatternSeverity.HIGH),
            ('definition_hiding', self.DEFINITION_HIDING_PATTERNS, AntiPatternSeverity.HIGH),
            ('vagueness', self.VAGUENESS_PATTERNS, AntiPatternSeverity.MEDIUM),
            ('passive_hiding', self.PASSIVE_HIDING_PATTERNS, AntiPatternSeverity.LOW),
        ]
        
        for group_name, patterns, severity in pattern_groups:
            for pattern, info in patterns.items():
                compiled = re.compile(pattern, re.IGNORECASE)
                self._compiled_patterns[compiled] = {
                    'group': group_name,
                    'severity': severity,
                    'fix': info['fix'],
                    'expansion': info['expansion'],
                }
    
    def scan(self, text: str) -> list[AntiPatternMatch]:
        """
        Scan text for anti-patterns.
        
        Args:
            text: Explanation text to scan
            
        Returns:
            List of detected anti-patterns
        """
        matches = []
        
        for pattern, info in self._compiled_patterns.items():
            for match in pattern.finditer(text):
                matches.append(AntiPatternMatch(
                    pattern_type=info['group'],
                    severity=info['severity'],
                    matched_text=match.group(0),
                    position=match.start(),
                    fix_instruction=info['fix'],
                    expansion_type=info['expansion'],
                ))
        
        # Sort by position
        matches.sort(key=lambda m: m.position)
        
        return matches
    
    def has_critical_issues(self, text: str) -> bool:
        """Check if text has critical anti-patterns."""
        matches = self.scan(text)
        return any(m.severity == AntiPatternSeverity.CRITICAL for m in matches)
    
    def get_expansion_requirements(self, text: str) -> list[dict]:
        """
        Get list of expansions needed to fix anti-patterns.
        
        Returns:
            List of dicts with expansion instructions
        """
        matches = self.scan(text)
        
        requirements = []
        for match in matches:
            requirements.append({
                'type': match.expansion_type,
                'trigger': match.matched_text,
                'instruction': match.fix_instruction,
                'severity': match.severity.value,
            })
        
        return requirements
    
    def suggest_fixes(self, text: str) -> str:
        """
        Generate suggested fixes for all anti-patterns.
        
        Returns:
            String with suggestions for each issue
        """
        matches = self.scan(text)
        
        if not matches:
            return "No anti-patterns detected. ✓"
        
        lines = [f"Found {len(matches)} anti-pattern(s):\n"]
        
        for i, match in enumerate(matches, 1):
            lines.append(f"{i}. [{match.severity.value.upper()}] \"{match.matched_text}\"")
            lines.append(f"   Fix: {match.fix_instruction}")
            lines.append(f"   Needs: {match.expansion_type}")
            lines.append("")
        
        return "\n".join(lines)
    
    def check_jargon(self, text: str, defined_terms: set[str]) -> list[str]:
        """
        Find technical terms that aren't defined.
        
        Args:
            text: Text to check
            defined_terms: Set of terms that have been defined
            
        Returns:
            List of undefined technical terms
        """
        # Common mathematical jargon
        MATH_JARGON = {
            'bijective', 'injective', 'surjective', 'homeomorphism',
            'isomorphism', 'homomorphism', 'topology', 'manifold',
            'continuous', 'differentiable', 'integrable', 'convergent',
            'divergent', 'bounded', 'compact', 'dense', 'open', 'closed',
            'limit', 'derivative', 'integral', 'series', 'sequence',
            'vector space', 'linear', 'orthogonal', 'eigenvalue',
            'determinant', 'matrix', 'tensor', 'field', 'ring', 'group',
            'abelian', 'polynomial', 'rational', 'irrational', 'algebraic',
            'transcendental', 'prime', 'composite', 'factorial',
            'permutation', 'combination', 'probability', 'distribution',
            'variance', 'covariance', 'correlation', 'hypothesis',
            'theorem', 'lemma', 'corollary', 'axiom', 'proof', 'conjecture',
        }
        
        undefined = []
        text_lower = text.lower()
        defined_lower = {t.lower() for t in defined_terms}
        
        for term in MATH_JARGON:
            if term in text_lower and term not in defined_lower:
                # Check if it appears as a standalone word
                pattern = rf'\b{re.escape(term)}\b'
                if re.search(pattern, text_lower):
                    undefined.append(term)
        
        return undefined
    
    def check_circular_definition(
        self, 
        term: str, 
        definition: str
    ) -> bool:
        """
        Check if a definition is circular (uses the term being defined).
        
        Args:
            term: The term being defined
            definition: The definition text
            
        Returns:
            True if circular (BAD), False if ok
        """
        term_lower = term.lower()
        def_lower = definition.lower()
        
        # Direct self-reference
        if term_lower in def_lower:
            return True
        
        # Check for variations (e.g., "limit" in "limiting")
        # This is a simple check - could be more sophisticated
        pattern = rf'\b{re.escape(term_lower)}\w*\b'
        if re.search(pattern, def_lower):
            return True
        
        return False
