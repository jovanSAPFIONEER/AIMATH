"""
Tests for the AI Math Verification System.

Run with: pytest tests/ -v
"""

import pytest
from sympy import Symbol, sin, cos, Eq, solve, diff, integrate

# Import system components
import sys
sys.path.insert(0, str(__file__).replace('tests\\test_core.py', 'src'))

from src.core.types import (
    MathProblem, 
    ProblemType, 
    DifficultyLevel,
    ConfidenceLevel,
    SolutionStep,
)


class TestMathProblem:
    """Test MathProblem data structure."""
    
    def test_create_equation_problem(self):
        """Test creating an equation problem."""
        problem = MathProblem(
            raw_input="x^2 - 4 = 0",
            problem_type=ProblemType.EQUATION,
        )
        
        assert problem.raw_input == "x^2 - 4 = 0"
        assert problem.problem_type == ProblemType.EQUATION
    
    def test_default_problem_type(self):
        """Test that default problem type is UNKNOWN."""
        problem = MathProblem(raw_input="something")
        assert problem.problem_type == ProblemType.UNKNOWN
    
    def test_difficulty_hint(self):
        """Test difficulty hint."""
        problem = MathProblem(
            raw_input="∫x^2 dx",
            difficulty_hint=DifficultyLevel.INTERMEDIATE,
        )
        assert problem.difficulty_hint == DifficultyLevel.INTERMEDIATE


class TestConfidenceLevel:
    """Test confidence level enumeration."""
    
    def test_confidence_levels_exist(self):
        """Test all expected confidence levels exist."""
        assert ConfidenceLevel.PROVEN
        assert ConfidenceLevel.HIGH
        assert ConfidenceLevel.MEDIUM
        assert ConfidenceLevel.LOW
        assert ConfidenceLevel.UNKNOWN
    
    def test_confidence_ordering(self):
        """Test that confidence levels have meaningful values."""
        # These are used for comparison in the system
        levels = [
            ConfidenceLevel.PROVEN,
            ConfidenceLevel.HIGH,
            ConfidenceLevel.MEDIUM,
            ConfidenceLevel.LOW,
            ConfidenceLevel.UNKNOWN,
        ]
        
        # Each level should have a string value
        for level in levels:
            assert isinstance(level.value, str)


class TestSolutionStep:
    """Test SolutionStep data structure."""
    
    def test_create_step(self):
        """Test creating a solution step."""
        step = SolutionStep(
            action="Subtract 4 from both sides",
            expression="x^2 = 4",
            justification="Isolating the variable term",
        )
        
        assert step.action == "Subtract 4 from both sides"
        assert step.expression == "x^2 = 4"
        assert step.justification == "Isolating the variable term"
    
    def test_step_with_warnings(self):
        """Test step with warnings."""
        step = SolutionStep(
            action="Take square root",
            expression="x = ±2",
            justification="Square root of both sides",
            warnings=["Don't forget the ± when taking square roots!"],
        )
        
        assert len(step.warnings) == 1
        assert "±" in step.warnings[0]


class TestSymPyIntegration:
    """Test SymPy integration for verification."""
    
    def test_equation_solving(self):
        """Test basic equation solving with SymPy."""
        x = Symbol('x')
        equation = Eq(x**2 - 4, 0)
        solutions = solve(equation, x)
        
        assert len(solutions) == 2
        assert 2 in solutions
        assert -2 in solutions
    
    def test_derivative(self):
        """Test derivative calculation."""
        x = Symbol('x')
        f = x**3
        f_prime = diff(f, x)
        
        assert f_prime == 3*x**2
    
    def test_integral(self):
        """Test integral calculation."""
        x = Symbol('x')
        f = x**2
        F = integrate(f, x)
        
        assert F == x**3 / 3
    
    def test_trig_identity_verification(self):
        """Test trig identity verification."""
        x = Symbol('x')
        lhs = sin(x)**2 + cos(x)**2
        
        from sympy import simplify, trigsimp
        result = trigsimp(lhs)
        
        assert result == 1


class TestProblemTypes:
    """Test problem type classification."""
    
    def test_all_types_exist(self):
        """Test all problem types exist."""
        expected_types = [
            'EQUATION', 'DERIVATIVE', 'INTEGRAL', 'LIMIT',
            'PROOF', 'SIMPLIFY', 'UNKNOWN',
        ]
        
        for type_name in expected_types:
            assert hasattr(ProblemType, type_name)
    
    def test_type_values(self):
        """Test problem type string values."""
        assert ProblemType.EQUATION.value == 'equation'
        assert ProblemType.DERIVATIVE.value == 'derivative'
        assert ProblemType.INTEGRAL.value == 'integral'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
