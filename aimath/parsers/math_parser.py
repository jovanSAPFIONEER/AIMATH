"""
Main Math Parser - Unified parsing interface.

Automatically detects input type (LaTeX, natural language, raw expression)
and routes to the appropriate specialized parser.
"""

import re
from typing import Optional, Any
import logging

from ..core.types import MathProblem, ProblemType, DifficultyLevel

logger = logging.getLogger(__name__)


class MathParser:
    """
    Unified mathematical input parser.
    
    Handles multiple input formats:
    - LaTeX: "\\frac{d}{dx} x^2"
    - Natural language: "find the derivative of x squared"
    - Raw expression: "diff(x**2, x)"
    - Mixed: "solve x^2 - 5x + 6 = 0 for x"
    
    Example:
        >>> parser = MathParser()
        >>> problem = parser.parse("x^2 - 5x + 6 = 0")
        >>> print(problem.problem_type)  # ProblemType.EQUATION
    """
    
    def __init__(self):
        """Initialize parser with specialized sub-parsers."""
        self._latex_parser = None
        self._nl_parser = None
    
    @property
    def latex_parser(self):
        """Lazy load LaTeX parser."""
        if self._latex_parser is None:
            from .latex_parser import LaTeXParser
            self._latex_parser = LaTeXParser()
        return self._latex_parser
    
    @property
    def nl_parser(self):
        """Lazy load natural language parser."""
        if self._nl_parser is None:
            from .natural_language import NaturalLanguageParser
            self._nl_parser = NaturalLanguageParser()
        return self._nl_parser
    
    def parse(self, input_str: str) -> MathProblem:
        """
        Parse input string into a MathProblem.
        
        Automatically detects the input format and routes to
        the appropriate parser.
        
        Args:
            input_str: Mathematical input in any supported format
            
        Returns:
            MathProblem with parsed expression and metadata
            
        Raises:
            ValueError: If input cannot be parsed
        """
        input_str = input_str.strip()
        
        if not input_str:
            raise ValueError("Input cannot be empty")
        
        logger.debug(f"Parsing: {input_str[:50]}...")
        
        # Detect input type
        input_type = self._detect_input_type(input_str)
        logger.debug(f"Detected input type: {input_type}")
        
        # Route to appropriate parser
        if input_type == "latex":
            parsed = self.latex_parser.parse(input_str)
        elif input_type == "natural_language":
            parsed = self.nl_parser.parse(input_str)
        else:
            parsed = self._parse_raw_expression(input_str)
        
        # Detect problem type
        problem_type = self._detect_problem_type(input_str, parsed)
        
        # Extract variables
        variables = self._extract_variables(parsed)
        
        # Extract constraints
        constraints = self._extract_constraints(input_str)
        
        return MathProblem(
            raw_input=input_str,
            parsed_expression=parsed,
            problem_type=problem_type,
            variables=variables,
            constraints=constraints,
        )
    
    def _detect_input_type(self, input_str: str) -> str:
        """
        Detect the type of input.
        
        Returns:
            "latex", "natural_language", or "raw"
        """
        # Check for SymPy function calls first - treat as raw
        if re.match(r'^(Sum|Product|Integral|Derivative|Limit)\s*\(', input_str):
            return "raw"
        
        # Check for LaTeX commands
        latex_patterns = [
            r'\\frac', r'\\sqrt', r'\\int', r'\\sum', r'\\prod',
            r'\\lim', r'\\sin', r'\\cos', r'\\tan', r'\\log',
            r'\\ln', r'\\exp', r'\\partial', r'\\infty',
            r'\^{', r'_{', r'\\left', r'\\right', r'\\begin',
        ]
        
        for pattern in latex_patterns:
            if re.search(pattern, input_str):
                return "latex"
        
        # Check for natural language indicators
        nl_patterns = [
            r'\b(solve|find|calculate|compute|evaluate)\b',
            r'\b(derivative|integral|limit|sum|product)\b',
            r'\b(what is|prove that|show that|verify)\b',
            r'\b(simplify|factor|expand)\b',
            r'\bof\b.*\bwith respect to\b',
        ]
        
        for pattern in nl_patterns:
            if re.search(pattern, input_str, re.IGNORECASE):
                return "natural_language"
        
        return "raw"
    
    def _detect_problem_type(self, input_str: str, parsed: Any) -> ProblemType:
        """
        Detect the type of mathematical problem.
        """
        input_lower = input_str.lower()
        
        # Check for SymPy expression types first
        from sympy import Sum, Product, Derivative, Integral
        if isinstance(parsed, Sum) or isinstance(parsed, Product):
            return ProblemType.SERIES
        if isinstance(parsed, Derivative):
            return ProblemType.DIFFERENTIATION
        if isinstance(parsed, Integral):
            return ProblemType.INTEGRATION
        
        # Check for explicit problem type keywords
        type_patterns = {
            ProblemType.DIFFERENTIATION: [
                r'\bderivative\b', r'\bdifferentiate\b', r"\\frac{d}{d",
                r'\bdiff\b', r"d/dx", r"∂"
            ],
            ProblemType.INTEGRATION: [
                r'\bintegral\b', r'\bintegrate\b', r'\\int',
                r'\bantiderivative\b'
            ],
            ProblemType.LIMIT: [
                r'\blimit\b', r'\\lim', r'→', r'->'
            ],
            ProblemType.PROOF: [
                r'\bprove\b', r'\bshow that\b', r'\bdemonstrate\b'
            ],
            ProblemType.VERIFICATION: [
                r'\bverify\b', r'\bcheck\b', r'\bis .* true\b'
            ],
            ProblemType.SIMPLIFICATION: [
                r'\bsimplify\b', r'\breduce\b'
            ],
            ProblemType.OPTIMIZATION: [
                r'\bmaximize\b', r'\bminimize\b', r'\boptimize\b',
                r'\bfind .* (max|min)\b', r'\bextreme\b'
            ],
            ProblemType.EQUATION: [
                r'=', r'\bsolve\b', r'\bfind .* (x|y|z|value)\b'
            ],
            ProblemType.INEQUALITY: [
                r'<', r'>', r'≤', r'≥', r'\\le', r'\\ge'
            ],
            ProblemType.SYSTEM: [
                r'\bsystem\b', r'\bsimultaneous\b'
            ],
            ProblemType.SERIES: [
                r'\bseries\b', r'\bsequence\b', r'\\sum', r'∑', r'\bSum\('
            ],
            ProblemType.MATRIX: [
                r'\bmatrix\b', r'\bdeterminant\b', r'\beigenvalue\b',
                r'\\begin{matrix}', r'\\begin{pmatrix}'
            ],
        }
        
        for problem_type, patterns in type_patterns.items():
            for pattern in patterns:
                if re.search(pattern, input_str, re.IGNORECASE):
                    return problem_type
        
        return ProblemType.GENERAL
    
    def _parse_raw_expression(self, input_str: str) -> Any:
        """
        Parse raw mathematical expression using SymPy.
        """
        try:
            from sympy import sympify, Symbol, Eq, Sum, Product, oo, pi, E, I
            from sympy.parsing.sympy_parser import (
                parse_expr,
                standard_transformations,
                implicit_multiplication_application,
                convert_xor,
            )
            from sympy.abc import n, k, i, j, x, y, z
            
            # Check if it's a SymPy expression (Sum, Product, etc.)
            if input_str.strip().startswith(('Sum(', 'Product(', 'Integral(', 'Derivative(')):
                # Use sympify with local dict for SymPy functions
                local_dict = {
                    'Sum': Sum, 'Product': Product, 
                    'oo': oo, 'inf': oo, 'pi': pi, 'E': E, 'I': I,
                    'n': n, 'k': k, 'i': i, 'j': j, 'x': x, 'y': y, 'z': z,
                }
                return sympify(input_str, locals=local_dict)
            
            # Apply transformations for common notation
            transformations = (
                standard_transformations + 
                (implicit_multiplication_application, convert_xor)
            )
            
            # Handle common patterns
            expr_str = input_str
            
            # Replace common notation
            expr_str = expr_str.replace('^', '**')  # ^ to **
            
            # Handle equations with = sign
            if '=' in expr_str and expr_str.count('=') == 1:
                lhs, rhs = expr_str.split('=')
                lhs = lhs.strip()
                rhs = rhs.strip()
                
                # Parse both sides
                lhs_expr = parse_expr(lhs, transformations=transformations)
                rhs_expr = parse_expr(rhs, transformations=transformations)
                
                return Eq(lhs_expr, rhs_expr)
            
            # For single expressions
            expr_str = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', expr_str)  # 2x -> 2*x
            
            return parse_expr(expr_str, transformations=transformations)
            
        except Exception as e:
            logger.warning(f"Raw parsing failed: {e}")
            # Return as string if parsing fails
            return input_str
    
    def _extract_variables(self, parsed: Any) -> list[str]:
        """Extract variable names from parsed expression."""
        try:
            from sympy import Symbol
            if hasattr(parsed, 'free_symbols'):
                return [str(s) for s in parsed.free_symbols]
        except Exception:
            pass
        
        # Fallback: find single letters that look like variables
        if isinstance(parsed, str):
            vars_found = set(re.findall(r'\b([a-zA-Z])\b', parsed))
            # Exclude common function names
            vars_found -= {'e', 'i', 'd', 'f', 'g'}
            return list(vars_found)
        
        return []
    
    def _extract_constraints(self, input_str: str) -> list[Any]:
        """Extract domain constraints from input."""
        constraints = []
        
        # Look for explicit constraints
        constraint_patterns = [
            (r'where\s+(\w+)\s*>\s*(\d+)', 'greater'),
            (r'where\s+(\w+)\s*<\s*(\d+)', 'less'),
            (r'where\s+(\w+)\s*>=\s*(\d+)', 'geq'),
            (r'where\s+(\w+)\s*<=\s*(\d+)', 'leq'),
            (r'for\s+(\w+)\s*>\s*(\d+)', 'greater'),
            (r'for\s+(\w+)\s*in\s*\[([^\]]+)\]', 'interval'),
            (r'(\w+)\s*≠\s*(\d+)', 'neq'),
        ]
        
        for pattern, constraint_type in constraint_patterns:
            matches = re.findall(pattern, input_str, re.IGNORECASE)
            for match in matches:
                constraints.append({
                    'type': constraint_type,
                    'variable': match[0],
                    'value': match[1],
                })
        
        return constraints
