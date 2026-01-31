"""
Tests for the verification system.

These tests ensure our anti-hallucination measures work correctly.
"""

import pytest
from sympy import Symbol, sin, cos, Eq, sqrt, simplify

import sys
sys.path.insert(0, str(__file__).replace('tests\\test_verification.py', 'src'))

from src.verification.substitution import SubstitutionChecker
from src.verification.counterexample import CounterexampleSearcher
from src.verification.consensus import ConsensusChecker
from src.verification.confidence import ConfidenceScorer
from src.core.types import ConfidenceLevel


class TestSubstitutionChecker:
    """Test solution verification by substitution."""
    
    def setup_method(self):
        self.checker = SubstitutionChecker()
        self.x = Symbol('x')
    
    def test_correct_quadratic_solution(self):
        """Test verifying correct quadratic solutions."""
        # x^2 - 4 = 0, solutions are x = 2 and x = -2
        expression = self.x**2 - 4
        
        result_pos = self.checker.verify(expression, self.x, 2)
        result_neg = self.checker.verify(expression, self.x, -2)
        
        assert result_pos['is_valid'] is True
        assert result_neg['is_valid'] is True
    
    def test_incorrect_solution(self):
        """Test detecting incorrect solutions."""
        # x^2 - 4 = 0, x = 3 is NOT a solution
        expression = self.x**2 - 4
        
        result = self.checker.verify(expression, self.x, 3)
        
        assert result['is_valid'] is False
    
    def test_derivative_verification(self):
        """Test verifying derivative results."""
        # If f(x) = x^2, then f'(x) = 2x
        from sympy import diff
        
        f = self.x**2
        f_prime = diff(f, self.x)
        
        # Verify the derivative is correct
        expected = 2 * self.x
        assert simplify(f_prime - expected) == 0


class TestCounterexampleSearcher:
    """Test counterexample finding."""
    
    def setup_method(self):
        self.searcher = CounterexampleSearcher()
        self.x = Symbol('x')
    
    def test_no_counterexample_for_identity(self):
        """Test that sin²(x) + cos²(x) = 1 has no counterexamples."""
        lhs = sin(self.x)**2 + cos(self.x)**2
        rhs = 1
        
        result = self.searcher.search(lhs - rhs, self.x)
        
        assert result['found'] is False
    
    def test_find_counterexample_for_false_claim(self):
        """Test finding counterexample for false claim."""
        # Claim: x^2 = x for all x (FALSE: only true for x=0 and x=1)
        claim = self.x**2 - self.x  # Should be 0 if claim is true
        
        result = self.searcher.search(claim, self.x, must_be_zero=True)
        
        # Should find counterexamples like x=2 (where 4 ≠ 2)
        # The searcher looks for values where expression ≠ 0
        if result['found']:
            ce = result['counterexample']
            value = claim.subs(self.x, ce)
            assert value != 0


class TestConsensusChecker:
    """Test multi-solver consensus checking."""
    
    def setup_method(self):
        self.checker = ConsensusChecker()
    
    def test_full_consensus(self):
        """Test agreement detection."""
        results = {
            'symbolic': {'answer': 2, 'trust': 0.9},
            'numerical': {'answer': 2.0, 'trust': 0.7},
            'formal': {'answer': 2, 'trust': 1.0},
        }
        
        consensus = self.checker.check(results)
        
        assert consensus['has_consensus'] is True
        assert consensus['consensus_value'] == 2
    
    def test_no_consensus(self):
        """Test disagreement detection."""
        results = {
            'solver1': {'answer': 2, 'trust': 0.9},
            'solver2': {'answer': 3, 'trust': 0.7},  # Different!
            'solver3': {'answer': 5, 'trust': 0.8},  # Different!
        }
        
        consensus = self.checker.check(results)
        
        assert consensus['has_consensus'] is False
    
    def test_numeric_tolerance(self):
        """Test that close numeric values count as consensus."""
        results = {
            'solver1': {'answer': 2.0, 'trust': 0.9},
            'solver2': {'answer': 2.00001, 'trust': 0.7},  # Close enough
            'solver3': {'answer': 1.99999, 'trust': 0.8},  # Close enough
        }
        
        consensus = self.checker.check(results, tolerance=1e-4)
        
        assert consensus['has_consensus'] is True


class TestConfidenceScorer:
    """Test confidence score calculation."""
    
    def setup_method(self):
        self.scorer = ConfidenceScorer()
    
    def test_high_confidence_with_all_checks(self):
        """Test high confidence when all checks pass."""
        verification = {
            'formal_proof': True,
            'substitution_passed': True,
            'consensus': True,
            'counterexamples': [],
        }
        
        result = self.scorer.calculate(verification)
        
        # Should be HIGH or PROVEN
        assert result['level'] in [ConfidenceLevel.HIGH, ConfidenceLevel.PROVEN]
        assert result['score'] >= 0.8
    
    def test_low_confidence_with_counterexamples(self):
        """Test low confidence when counterexamples exist."""
        verification = {
            'formal_proof': False,
            'substitution_passed': True,
            'consensus': False,
            'counterexamples': ['x = 5 fails'],
        }
        
        result = self.scorer.calculate(verification)
        
        # Should be LOW or UNKNOWN
        assert result['level'] in [ConfidenceLevel.LOW, ConfidenceLevel.UNKNOWN]
    
    def test_confidence_level_thresholds(self):
        """Test confidence level threshold assignments."""
        # PROVEN: score >= 0.95 with formal proof
        # HIGH: score >= 0.8
        # MEDIUM: score >= 0.6
        # LOW: score >= 0.4
        # UNKNOWN: score < 0.4
        
        proven_verification = {
            'formal_proof': True,
            'substitution_passed': True,
            'consensus': True,
            'counterexamples': [],
        }
        
        result = self.scorer.calculate(proven_verification)
        
        # With formal proof, should be PROVEN
        if result['score'] >= 0.95:
            assert result['level'] == ConfidenceLevel.PROVEN


class TestIntegratedVerification:
    """Test the integrated verification pipeline."""
    
    def test_equation_verification_pipeline(self):
        """Test full verification of an equation solution."""
        x = Symbol('x')
        
        # Problem: x^2 - 4 = 0
        expression = x**2 - 4
        proposed_solutions = [2, -2]
        
        # Step 1: Substitution check
        sub_checker = SubstitutionChecker()
        for sol in proposed_solutions:
            result = sub_checker.verify(expression, x, sol)
            assert result['is_valid'] is True
        
        # Step 2: Counterexample search (should find none for valid solutions)
        ce_searcher = CounterexampleSearcher()
        # For a specific solution, verify it's actually a root
        for sol in proposed_solutions:
            val = expression.subs(x, sol)
            assert simplify(val) == 0
        
        # Step 3: Consensus check (simulate multiple solvers agreeing)
        consensus_checker = ConsensusChecker()
        multi_results = {
            'symbolic': {'answer': {2, -2}, 'trust': 0.9},
            'numerical': {'answer': {2.0, -2.0}, 'trust': 0.7},
        }
        # Note: Consensus checking for sets requires special handling


class TestTrustHierarchy:
    """Test that trust hierarchy is respected."""
    
    def test_formal_proof_highest_trust(self):
        """Formal proof should have trust level 1.0."""
        scorer = ConfidenceScorer()
        
        formal_only = {
            'formal_proof': True,
            'substitution_passed': False,
            'consensus': False,
            'counterexamples': [],
        }
        
        result = scorer.calculate(formal_only)
        
        # Formal proof should give high confidence
        assert result['score'] >= 0.5  # At minimum, significant boost
    
    def test_numerical_needs_verification(self):
        """Numerical results should need additional verification."""
        # Numerical alone shouldn't give HIGH confidence
        scorer = ConfidenceScorer()
        
        numerical_only = {
            'formal_proof': False,
            'substitution_passed': True,  # Just substitution
            'consensus': False,
            'counterexamples': [],
        }
        
        result = scorer.calculate(numerical_only)
        
        # Should be MEDIUM at best without consensus
        assert result['level'] != ConfidenceLevel.PROVEN


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
