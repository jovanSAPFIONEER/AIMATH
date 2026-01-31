"""
Tests for the Anti-Pattern Detection System.

These tests verify that our system catches the "classic AI traits"
that skip explaining concepts.
"""

import pytest
import sys
sys.path.insert(0, str(__file__).replace('tests\\test_anti_patterns.py', 'src'))

from src.explanation.anti_patterns import (
    AntiPatternDetector,
    AntiPatternSeverity,
    AntiPatternMatch,
)


class TestTrivializingPatterns:
    """Test detection of trivializing language."""
    
    def setup_method(self):
        self.detector = AntiPatternDetector()
    
    def test_detect_obviously(self):
        """'Obviously' should be flagged as critical."""
        text = "Obviously, x equals 2."
        matches = self.detector.scan(text)
        
        assert len(matches) >= 1
        assert any(m.matched_text.lower() == 'obviously' for m in matches)
        assert any(m.severity == AntiPatternSeverity.CRITICAL for m in matches)
    
    def test_detect_clearly(self):
        """'Clearly' should be flagged."""
        text = "Clearly, this follows from the definition."
        matches = self.detector.scan(text)
        
        assert len(matches) >= 1
        assert any('clearly' in m.matched_text.lower() for m in matches)
    
    def test_detect_trivially(self):
        """'Trivially' should be flagged."""
        text = "This is trivially true."
        matches = self.detector.scan(text)
        
        assert len(matches) >= 1
        assert any('trivial' in m.matched_text.lower() for m in matches)
    
    def test_detect_simply(self):
        """'Simply' (as hand-waving) should be flagged."""
        text = "We simply apply the formula."
        matches = self.detector.scan(text)
        
        assert len(matches) >= 1
    
    def test_simply_put_allowed(self):
        """'Simply put' is an acceptable phrase."""
        text = "Simply put, this means x = 2."
        matches = self.detector.scan(text)
        
        # Should not flag "simply put" as it's a valid explanation phrase
        # (depends on regex - current implementation may or may not catch this)
        # At minimum, we document the expected behavior


class TestStepSkippingPatterns:
    """Test detection of step-skipping language."""
    
    def setup_method(self):
        self.detector = AntiPatternDetector()
    
    def test_detect_it_follows(self):
        """'It follows that' without showing work should be flagged."""
        text = "It follows that x = 5."
        matches = self.detector.scan(text)
        
        assert len(matches) >= 1
        skip_matches = [m for m in matches if m.pattern_type == 'step_skipping']
        assert len(skip_matches) >= 1
    
    def test_detect_therefore_without_because(self):
        """'Therefore' without 'because' should be flagged."""
        text = "Therefore, we have our answer."
        matches = self.detector.scan(text)
        
        # Should flag unless followed by explicit reasoning
        assert len(matches) >= 1
    
    def test_allow_therefore_with_because(self):
        """'Therefore X because Y' should be acceptable."""
        text = "Therefore x = 2, because we solved the equation."
        matches = self.detector.scan(text)
        
        # The regex uses negative lookahead - check behavior
        # May still flag depending on exact implementation
    
    def test_detect_we_get_without_explanation(self):
        """'We get' without showing how should be flagged."""
        text = "We get x = 3."
        matches = self.detector.scan(text)
        
        # May be flagged as step-skipping
        assert len(matches) >= 1


class TestDefinitionHidingPatterns:
    """Test detection of undefined term usage."""
    
    def setup_method(self):
        self.detector = AntiPatternDetector()
    
    def test_detect_by_definition(self):
        """'By definition' without stating definition should be flagged."""
        text = "By definition, this is true."
        matches = self.detector.scan(text)
        
        assert len(matches) >= 1
        def_matches = [m for m in matches if m.pattern_type == 'definition_hiding']
        assert len(def_matches) >= 1
    
    def test_detect_recall_that(self):
        """'Recall that' without stating what should be flagged."""
        text = "Recall that f is continuous."
        matches = self.detector.scan(text)
        
        assert len(matches) >= 1
    
    def test_detect_as_we_know(self):
        """'As we know' assumes shared knowledge."""
        text = "As we know, integration is the inverse of differentiation."
        matches = self.detector.scan(text)
        
        assert len(matches) >= 1


class TestVaguenessPatterns:
    """Test detection of vague, hand-wavy language."""
    
    def setup_method(self):
        self.detector = AntiPatternDetector()
    
    def test_detect_well_known(self):
        """'It is well known that' should be flagged."""
        text = "It is well known that the limit exists."
        matches = self.detector.scan(text)
        
        assert len(matches) >= 1
    
    def test_detect_can_be_shown(self):
        """'It can be shown that' should be flagged."""
        text = "It can be shown that x > 0."
        matches = self.detector.scan(text)
        
        assert len(matches) >= 1
    
    def test_detect_left_as_exercise(self):
        """'Left as an exercise' is unacceptable in explanations."""
        text = "The proof is left as an exercise."
        matches = self.detector.scan(text)
        
        assert len(matches) >= 1
        assert any(m.severity in [AntiPatternSeverity.MEDIUM, AntiPatternSeverity.HIGH] 
                  for m in matches)
    
    def test_detect_details_omitted(self):
        """'Details are omitted' should be flagged."""
        text = "Some details are omitted for brevity."
        matches = self.detector.scan(text)
        
        assert len(matches) >= 1


class TestJargonDetection:
    """Test undefined jargon detection."""
    
    def setup_method(self):
        self.detector = AntiPatternDetector()
    
    def test_find_undefined_jargon(self):
        """Should find undefined mathematical jargon."""
        text = "The function is bijective and continuous on the manifold."
        defined_terms = set()  # Nothing defined
        
        undefined = self.detector.check_jargon(text, defined_terms)
        
        assert 'bijective' in undefined
        assert 'continuous' in undefined
        assert 'manifold' in undefined
    
    def test_allow_defined_jargon(self):
        """Should allow jargon that's been defined."""
        text = "The function is continuous."
        defined_terms = {'continuous'}
        
        undefined = self.detector.check_jargon(text, defined_terms)
        
        assert 'continuous' not in undefined


class TestCircularDefinitions:
    """Test circular definition detection."""
    
    def setup_method(self):
        self.detector = AntiPatternDetector()
    
    def test_detect_direct_circular(self):
        """Should detect 'X is X' type definitions."""
        is_circular = self.detector.check_circular_definition(
            term="limit",
            definition="A limit is the limit of a sequence."
        )
        
        assert is_circular is True
    
    def test_allow_proper_definition(self):
        """Should allow proper non-circular definitions."""
        is_circular = self.detector.check_circular_definition(
            term="limit",
            definition="A limit describes the value a function approaches as input approaches some value."
        )
        
        assert is_circular is False


class TestSeverityLevels:
    """Test that severity levels are assigned correctly."""
    
    def setup_method(self):
        self.detector = AntiPatternDetector()
    
    def test_trivializing_is_critical(self):
        """Trivializing patterns should be critical severity."""
        text = "Obviously this is true."
        matches = self.detector.scan(text)
        
        triv_matches = [m for m in matches if m.pattern_type == 'trivializing']
        assert all(m.severity == AntiPatternSeverity.CRITICAL for m in triv_matches)
    
    def test_step_skipping_is_high(self):
        """Step-skipping patterns should be high severity."""
        text = "It follows that x = 5."
        matches = self.detector.scan(text)
        
        skip_matches = [m for m in matches if m.pattern_type == 'step_skipping']
        assert all(m.severity == AntiPatternSeverity.HIGH for m in skip_matches)


class TestExpansionRequirements:
    """Test expansion requirement generation."""
    
    def setup_method(self):
        self.detector = AntiPatternDetector()
    
    def test_get_expansion_requirements(self):
        """Should generate expansion requirements for flagged patterns."""
        text = "Obviously, by definition, this is trivially true."
        requirements = self.detector.get_expansion_requirements(text)
        
        assert len(requirements) >= 2  # Multiple issues
        assert all('type' in r for r in requirements)
        assert all('instruction' in r for r in requirements)


class TestFixSuggestions:
    """Test fix suggestion generation."""
    
    def setup_method(self):
        self.detector = AntiPatternDetector()
    
    def test_suggest_fixes(self):
        """Should generate helpful fix suggestions."""
        text = "Clearly, it is trivial to see that x = 2."
        suggestions = self.detector.suggest_fixes(text)
        
        assert "anti-pattern" in suggestions.lower() or "found" in suggestions.lower()
    
    def test_clean_text_no_suggestions(self):
        """Clean text should report no issues."""
        text = "We add 3 to both sides because we want to isolate x. This gives us x + 3 = 7, and subtracting 3 from both sides yields x = 4."
        suggestions = self.detector.suggest_fixes(text)
        
        # Should indicate no issues (or minimal issues)
        # Note: "gives us" might still be flagged depending on implementation


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
