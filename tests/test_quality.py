"""
Tests for the CLEAR Quality Rubric.

The CLEAR rubric ensures explanations meet quality standards:
- Completeness
- Logical flow  
- Explicit terms
- Accessibility
- Reasoning
"""

import pytest
import sys
sys.path.insert(0, str(__file__).replace('tests\\test_quality.py', 'src'))

from src.explanation.quality_checker import QualityChecker, CLEARScore


class TestCLEARScore:
    """Test CLEARScore data structure."""
    
    def test_total_calculation(self):
        """Test total score calculation."""
        score = CLEARScore(
            completeness=4,
            logical_flow=5,
            explicit_terms=4,
            accessibility=4,
            reasoning=5,
        )
        
        assert score.total == 22
    
    def test_passes_with_good_scores(self):
        """Test passing with good scores (20+, 3+ in each)."""
        score = CLEARScore(
            completeness=4,
            logical_flow=4,
            explicit_terms=4,
            accessibility=4,
            reasoning=4,
        )
        
        assert score.passes is True
    
    def test_fails_with_low_total(self):
        """Test failing with low total score."""
        score = CLEARScore(
            completeness=3,
            logical_flow=3,
            explicit_terms=3,
            accessibility=3,
            reasoning=3,
        )
        
        # Total = 15, below 20
        assert score.passes is False
    
    def test_fails_with_single_low_category(self):
        """Test failing when one category is below 3."""
        score = CLEARScore(
            completeness=5,
            logical_flow=5,
            explicit_terms=2,  # Below threshold
            accessibility=5,
            reasoning=5,
        )
        
        # Total = 22, but explicit_terms < 3
        assert score.passes is False
    
    def test_weakest_category(self):
        """Test finding the weakest category."""
        score = CLEARScore(
            completeness=5,
            logical_flow=4,
            explicit_terms=2,  # Lowest
            accessibility=4,
            reasoning=5,
        )
        
        assert score.weakest_category == 'explicit_terms'
    
    def test_report_generation(self):
        """Test quality report generation."""
        score = CLEARScore(
            completeness=4,
            logical_flow=4,
            explicit_terms=4,
            accessibility=4,
            reasoning=4,
        )
        
        report = score.to_report()
        
        assert "CLEAR Quality Score" in report
        assert "20/25" in report
        assert "PASSES" in report


class TestQualityChecker:
    """Test QualityChecker functionality."""
    
    def setup_method(self):
        self.checker = QualityChecker()
    
    def test_check_good_explanation(self):
        """Test checking a high-quality explanation."""
        good_explanation = """
        **Prerequisites:**
        Before we begin, let's define what we mean by 'derivative'.
        A derivative measures how fast something changes.
        
        **Example:**
        For instance, consider f(x) = x². Let's see what happens at x = 2.
        
        **Step 1:** Identify the function type
        This is a polynomial, so we use the power rule.
        
        **Step 2:** Apply the power rule
        The power rule states: d/dx[x^n] = n*x^(n-1)
        
        Because our function is x², we get: 2*x^(2-1) = 2x
        
        **Edge case:**
        Note that this rule only applies for n ≠ 0.
        
        **Verify:**
        Try computing f'(3) = 2*3 = 6. This represents the slope at x=3.
        
        **Why it works:**
        The power rule works because the derivative measures instantaneous
        rate of change, and for polynomials, this reduces to this simple formula.
        """
        
        score = self.checker.check(good_explanation, 'derivative')
        
        # Should score reasonably well
        assert score.total >= 15  # May not be perfect but should be decent
    
    def test_check_poor_explanation(self):
        """Test checking a poor-quality explanation."""
        poor_explanation = """
        Obviously, we just apply the formula.
        It's trivially true that the answer is 2x.
        By definition, this is correct.
        """
        
        score = self.checker.check(poor_explanation, 'derivative')
        
        # Should score poorly
        assert score.total < 15
        assert not score.passes
    
    def test_detect_missing_prerequisites(self):
        """Test detection of missing prerequisites."""
        no_prereq = """
        Step 1: We differentiate.
        Step 2: We get the answer.
        """
        
        score = self.checker.check(no_prereq, 'derivative')
        
        # Should have completeness issues
        assert score.completeness < 5
        assert any('prerequisite' in issue.lower() 
                  for issue in score.completeness_issues)
    
    def test_detect_missing_examples(self):
        """Test detection of missing examples."""
        no_example = """
        In general, to solve quadratic equations, we use the quadratic formula.
        The formula gives us the roots of any quadratic equation.
        This is because of the algebraic properties of equations.
        """
        
        score = self.checker.check(no_example, 'equation')
        
        # Should detect missing examples
        assert any('example' in issue.lower() 
                  for issue in score.completeness_issues + score.accessibility_issues)
    
    def test_detect_missing_reasoning(self):
        """Test detection of missing 'why' explanations."""
        no_why = """
        Step 1: Do this.
        Step 2: Then do that.
        Step 3: Get the answer.
        """
        
        score = self.checker.check(no_why, 'general')
        
        # Should have reasoning issues
        assert score.reasoning < 5
        assert any('why' in issue.lower() 
                  for issue in score.reasoning_issues)


class TestCompletenessChecking:
    """Test completeness dimension specifically."""
    
    def setup_method(self):
        self.checker = QualityChecker()
    
    def test_complete_structure(self):
        """Test that complete structure is recognized."""
        complete = """
        Before we start, let's define our terms.
        
        For example, consider x = 5.
        
        Step 1: First, we identify the problem.
        Step 2: Then, we apply our method.
        
        Note that this doesn't work when x < 0 (edge case).
        
        Verify by substituting back.
        """
        
        score = self.checker.check(complete, 'general')
        assert score.completeness >= 3


class TestLogicalFlowChecking:
    """Test logical flow dimension specifically."""
    
    def setup_method(self):
        self.checker = QualityChecker()
    
    def test_good_flow_with_transitions(self):
        """Test that transition words improve flow score."""
        good_flow = """
        First, we identify the equation type.
        Then, because it's quadratic, we use the formula.
        This means we compute the discriminant.
        Therefore, we can find the roots.
        """
        
        score = self.checker.check(good_flow, 'equation')
        assert score.logical_flow >= 3
    
    def test_poor_flow_detected(self):
        """Test that poor flow is detected."""
        poor_flow = """
        It follows that x = 2.
        We get y = 3.
        Hence z = 5.
        """
        
        score = self.checker.check(poor_flow, 'general')
        # Should detect step-skipping patterns
        assert score.logical_flow < 5


class TestAccessibilityChecking:
    """Test accessibility dimension specifically."""
    
    def setup_method(self):
        self.checker = QualityChecker(target_level='amateur')
    
    def test_long_sentences_penalized(self):
        """Test that overly long sentences are penalized."""
        long_sentences = """
        This is an extremely long sentence that goes on and on and on without
        any breaks or pauses and contains many many words that make it very
        difficult to follow and understand especially for beginners who may
        not be familiar with the material being presented in this explanation.
        """
        
        score = self.checker.check(long_sentences, 'general')
        assert score.accessibility < 5
        assert any('sentence' in issue.lower() 
                  for issue in score.accessibility_issues)


class TestReasoningChecking:
    """Test reasoning dimension specifically."""
    
    def setup_method(self):
        self.checker = QualityChecker()
    
    def test_why_present(self):
        """Test that 'why' explanations are recognized."""
        with_why = """
        We add 5 because we want to isolate x on one side.
        The reason we use this method is that it's most efficient.
        Intuitively, this makes sense since addition undoes subtraction.
        """
        
        score = self.checker.check(with_why, 'general')
        assert score.reasoning >= 3
    
    def test_why_missing(self):
        """Test that missing 'why' is detected."""
        without_why = """
        Add 5.
        Multiply by 2.
        The answer is 10.
        """
        
        score = self.checker.check(without_why, 'general')
        assert score.reasoning < 4


class TestImprovementSuggestions:
    """Test improvement suggestion generation."""
    
    def setup_method(self):
        self.checker = QualityChecker()
    
    def test_suggestions_for_poor_score(self):
        """Test that suggestions are generated for poor scores."""
        poor = "Obviously x = 2."
        score = self.checker.check(poor, 'equation')
        
        suggestions = self.checker.get_improvement_suggestions(score)
        
        assert len(suggestions) > 0
        assert all(isinstance(s, str) for s in suggestions)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
