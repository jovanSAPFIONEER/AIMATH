"""
Natural Language Parser - Parse math problems in plain English.

Converts natural language mathematical descriptions to structured
problems and SymPy expressions.
"""

import re
from typing import Any, Optional, Tuple
import logging

from ..core.types import ProblemType

logger = logging.getLogger(__name__)


class NaturalLanguageParser:
    """
    Parser for natural language mathematical problems.
    
    Converts plain English descriptions to structured mathematical
    problems that can be solved.
    
    Example:
        >>> parser = NaturalLanguageParser()
        >>> problem = parser.parse("find the derivative of x squared plus 3x")
        >>> print(problem)  # Derivative(x**2 + 3*x, x)
    """
    
    # Patterns for number words
    NUMBER_WORDS = {
        'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
        'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
        'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13,
        'fourteen': 14, 'fifteen': 15, 'sixteen': 16, 'seventeen': 17,
        'eighteen': 18, 'nineteen': 19, 'twenty': 20, 'thirty': 30,
        'forty': 40, 'fifty': 50, 'sixty': 60, 'seventy': 70,
        'eighty': 80, 'ninety': 90, 'hundred': 100, 'thousand': 1000,
        'half': 0.5, 'quarter': 0.25, 'third': 1/3,
    }
    
    # Ordinal to power mappings
    ORDINALS = {
        'squared': 2, 'cubed': 3, 'square': 2, 'cube': 3,
        'first': 1, 'second': 2, 'third': 3, 'fourth': 4, 'fifth': 5,
    }
    
    # Operation keywords
    OPERATIONS = {
        'plus': '+', 'add': '+', 'added to': '+', 'sum': '+',
        'minus': '-', 'subtract': '-', 'less': '-', 'difference': '-',
        'times': '*', 'multiply': '*', 'multiplied by': '*', 'product': '*',
        'divided by': '/', 'over': '/', 'quotient': '/',
        'to the power of': '**', 'raised to': '**',
    }
    
    def __init__(self):
        """Initialize natural language parser."""
        pass
    
    def parse(self, text: str) -> Any:
        """
        Parse natural language math problem to SymPy expression.
        
        Args:
            text: Natural language mathematical description
            
        Returns:
            SymPy expression or structured problem
        """
        text = text.lower().strip()
        
        # Detect problem type and extract components
        problem_type, expression_text = self._extract_problem_structure(text)
        
        # Convert to mathematical expression
        expr = self._text_to_expression(expression_text)
        
        # Wrap in appropriate operation
        expr = self._apply_operation(problem_type, expr, text)
        
        return expr
    
    def _extract_problem_structure(self, text: str) -> Tuple[ProblemType, str]:
        """
        Extract problem type and the expression from text.
        
        Returns:
            Tuple of (ProblemType, expression_text)
        """
        # Derivative patterns
        derivative_patterns = [
            (r"(?:find |compute |calculate )?(?:the )?derivative of (.+?)(?:\s+with respect to \w+)?$", ProblemType.DIFFERENTIATION),
            (r"differentiate (.+)", ProblemType.DIFFERENTIATION),
            (r"d/d\w+ of (.+)", ProblemType.DIFFERENTIATION),
        ]
        
        # Integral patterns  
        integral_patterns = [
            (r"(?:find |compute |calculate )?(?:the )?integral of (.+?)(?:\s*d\w+)?$", ProblemType.INTEGRATION),
            (r"integrate (.+)", ProblemType.INTEGRATION),
            (r"antiderivative of (.+)", ProblemType.INTEGRATION),
        ]
        
        # Solve patterns
        solve_patterns = [
            (r"solve (.+?)(?:\s+for \w+)?$", ProblemType.EQUATION),
            (r"find (?:the )?(?:value of |solution to )?(.+)", ProblemType.EQUATION),
        ]
        
        # Limit patterns
        limit_patterns = [
            (r"(?:find |evaluate )?(?:the )?limit (?:of )?(.+?)(?:\s+as .+)?$", ProblemType.LIMIT),
            (r"(?:find |evaluate )?lim(?:it)? (.+)", ProblemType.LIMIT),
        ]
        
        # Simplify patterns
        simplify_patterns = [
            (r"simplify (.+)", ProblemType.SIMPLIFICATION),
            (r"reduce (.+)", ProblemType.SIMPLIFICATION),
        ]
        
        # Proof patterns
        proof_patterns = [
            (r"prove (?:that )?(.+)", ProblemType.PROOF),
            (r"show (?:that )?(.+)", ProblemType.PROOF),
        ]
        
        all_patterns = (
            derivative_patterns + integral_patterns + solve_patterns +
            limit_patterns + simplify_patterns + proof_patterns
        )
        
        for pattern, problem_type in all_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return problem_type, match.group(1).strip()
        
        # Default: treat as general expression
        return ProblemType.GENERAL, text
    
    def _text_to_expression(self, text: str) -> Any:
        """
        Convert natural language expression to SymPy.
        
        Example: "x squared plus 3 x" -> x**2 + 3*x
        """
        from sympy import Symbol, sympify, sqrt, sin, cos, tan, log, exp, pi, E
        from sympy.parsing.sympy_parser import parse_expr
        
        expr_str = text
        
        # Replace number words
        for word, num in self.NUMBER_WORDS.items():
            expr_str = re.sub(rf'\b{word}\b', str(num), expr_str)
        
        # Replace ordinals/powers
        # "x squared" -> "x**2"
        for word, power in self.ORDINALS.items():
            expr_str = re.sub(rf'(\w+)\s+{word}', rf'\1**{power}', expr_str)
        
        # Replace operations
        for word, op in self.OPERATIONS.items():
            expr_str = re.sub(rf'\b{word}\b', f' {op} ', expr_str)
        
        # Handle "x to the n" -> "x**n"
        expr_str = re.sub(r'(\w+)\s+to\s+the\s+(\d+)', r'\1**\2', expr_str)
        
        # Handle "nth power of x" -> "x**n"
        expr_str = re.sub(r'(\d+)(?:st|nd|rd|th)\s+power\s+of\s+(\w+)', r'\2**\1', expr_str)
        
        # Handle "square root of x" -> "sqrt(x)"
        expr_str = re.sub(r'square root of\s+([^\s]+)', r'sqrt(\1)', expr_str)
        expr_str = re.sub(r'sqrt of\s+([^\s]+)', r'sqrt(\1)', expr_str)
        
        # Handle "cube root of x" -> "x**(1/3)"
        expr_str = re.sub(r'cube root of\s+([^\s]+)', r'(\1)**(1/3)', expr_str)
        
        # Handle trig functions: "sin of x" -> "sin(x)"
        for func in ['sin', 'cos', 'tan', 'log', 'ln', 'exp']:
            expr_str = re.sub(rf'{func}\s+of\s+([^\s]+)', rf'{func}(\1)', expr_str)
            expr_str = re.sub(rf'{func}\s+([^\s\+\-\*\/]+)', rf'{func}(\1)', expr_str)
        
        # Handle "e to the x" -> "exp(x)"
        expr_str = re.sub(r'e\s+to\s+the\s+([^\s]+)', r'exp(\1)', expr_str)
        
        # Handle "natural log" -> "log"
        expr_str = re.sub(r'natural\s+log(?:arithm)?', 'log', expr_str)
        
        # Clean up spacing
        expr_str = re.sub(r'\s+', ' ', expr_str).strip()
        
        # Add implicit multiplication
        expr_str = re.sub(r'(\d)\s+([a-zA-Z])', r'\1*\2', expr_str)
        expr_str = re.sub(r'([a-zA-Z])\s+(\d)', r'\1*\2', expr_str)
        expr_str = re.sub(r'([a-zA-Z])\s+([a-zA-Z])', r'\1*\2', expr_str)
        
        # Replace remaining spaces with nothing or *
        expr_str = re.sub(r'\s+', '', expr_str)
        
        try:
            from sympy.parsing.sympy_parser import (
                standard_transformations,
                implicit_multiplication_application,
                convert_xor,
            )
            transformations = (
                standard_transformations + 
                (implicit_multiplication_application, convert_xor)
            )
            return parse_expr(expr_str, transformations=transformations)
        except Exception as e:
            logger.warning(f"Expression parsing failed: {e}")
            # Return as string for further processing
            return expr_str
    
    def _apply_operation(self, problem_type: ProblemType, expr: Any, original_text: str) -> Any:
        """
        Wrap expression in appropriate mathematical operation.
        """
        from sympy import (
            Symbol, Derivative, Integral, Limit, Eq,
            symbols, oo, simplify
        )
        
        # Extract variable (default to x)
        var_match = re.search(r'with respect to (\w+)', original_text)
        var = Symbol(var_match.group(1) if var_match else 'x')
        
        if problem_type == ProblemType.DIFFERENTIATION:
            if hasattr(expr, 'free_symbols'):
                # Use the variable from the expression if only one
                free = list(expr.free_symbols)
                if len(free) == 1:
                    var = free[0]
            return Derivative(expr, var)
        
        elif problem_type == ProblemType.INTEGRATION:
            if hasattr(expr, 'free_symbols'):
                free = list(expr.free_symbols)
                if len(free) == 1:
                    var = free[0]
            return Integral(expr, var)
        
        elif problem_type == ProblemType.LIMIT:
            # Extract limit point
            limit_match = re.search(r'as (\w+) (?:goes to|approaches|→) ([\w\d\+\-]+)', original_text)
            if limit_match:
                var = Symbol(limit_match.group(1))
                point_str = limit_match.group(2)
                if point_str in ['infinity', 'inf', '∞']:
                    point = oo
                elif point_str in ['-infinity', '-inf', '-∞']:
                    point = -oo
                else:
                    try:
                        point = int(point_str) if point_str.lstrip('-').isdigit() else Symbol(point_str)
                    except ValueError:
                        point = Symbol(point_str)
                return Limit(expr, var, point)
            return Limit(expr, var, 0)  # Default limit at 0
        
        elif problem_type == ProblemType.EQUATION:
            # Check if it's already an equation
            if '=' in str(expr):
                # Parse as equation
                parts = str(expr).split('=')
                if len(parts) == 2:
                    from sympy import Eq
                    from sympy.parsing.sympy_parser import parse_expr
                    try:
                        lhs = parse_expr(parts[0].strip())
                        rhs = parse_expr(parts[1].strip())
                        return Eq(lhs, rhs)
                    except Exception:
                        pass
            return expr
        
        elif problem_type == ProblemType.SIMPLIFICATION:
            return simplify(expr)
        
        return expr
    
    def extract_variable(self, text: str) -> str:
        """Extract the main variable from problem text."""
        # Look for explicit mention
        match = re.search(r'(?:for|solve for|find)\s+(\w)', text, re.IGNORECASE)
        if match:
            return match.group(1)
        
        # Look for common variable letters
        for var in ['x', 'y', 'z', 't', 'n', 'k']:
            if var in text.lower():
                return var
        
        return 'x'  # Default
