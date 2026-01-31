"""
LaTeX Parser - Parse LaTeX mathematical notation.

Converts LaTeX input to SymPy expressions for computation.
"""

import re
from typing import Any, Optional
import logging

logger = logging.getLogger(__name__)


class LaTeXParser:
    """
    Parser for LaTeX mathematical notation.
    
    Converts LaTeX strings to SymPy expressions that can be
    manipulated and computed.
    
    Example:
        >>> parser = LaTeXParser()
        >>> expr = parser.parse(r"\\frac{x^2 + 1}{x - 1}")
        >>> print(expr)  # (x**2 + 1)/(x - 1)
    """
    
    def __init__(self):
        """Initialize LaTeX parser."""
        self._latex2sympy_available = None
    
    @property
    def latex2sympy_available(self) -> bool:
        """Check if latex2sympy2 is available."""
        if self._latex2sympy_available is None:
            try:
                import latex2sympy2
                self._latex2sympy_available = True
            except ImportError:
                self._latex2sympy_available = False
                logger.warning(
                    "latex2sympy2 not installed. "
                    "Install with: pip install latex2sympy2"
                )
        return self._latex2sympy_available
    
    def parse(self, latex_str: str) -> Any:
        """
        Parse LaTeX string to SymPy expression.
        
        Args:
            latex_str: LaTeX mathematical expression
            
        Returns:
            SymPy expression
            
        Example:
            >>> parse(r"\\int x^2 dx")  # Integral(x**2, x)
        """
        latex_str = self._preprocess(latex_str)
        
        if self.latex2sympy_available:
            return self._parse_with_latex2sympy(latex_str)
        else:
            return self._parse_manual(latex_str)
    
    def _preprocess(self, latex_str: str) -> str:
        """
        Preprocess LaTeX string for parsing.
        
        Normalizes common variations and fixes common issues.
        """
        # Remove display math delimiters
        latex_str = re.sub(r'^\$\$?|\$\$?$', '', latex_str.strip())
        latex_str = re.sub(r'^\\[|\\\]', '', latex_str)
        latex_str = re.sub(r'^\\begin{equation\*?}|\\end{equation\*?}$', '', latex_str)
        
        # Normalize spacing
        latex_str = re.sub(r'\s+', ' ', latex_str)
        
        # Common substitutions
        replacements = [
            (r'\\cdot', '*'),
            (r'\\times', '*'),
            (r'\\div', '/'),
            (r'\\pm', '±'),
            (r'\\mp', '∓'),
            (r'\\leq', '<='),
            (r'\\geq', '>='),
            (r'\\neq', '!='),
            (r'\\infty', 'oo'),
            (r'\\pi', 'pi'),
            (r'\\alpha', 'alpha'),
            (r'\\beta', 'beta'),
            (r'\\gamma', 'gamma'),
            (r'\\theta', 'theta'),
            (r'\\phi', 'phi'),
            (r'\\lambda', 'lamda'),  # Note: sympy uses 'lamda'
        ]
        
        for pattern, replacement in replacements:
            latex_str = re.sub(pattern, replacement, latex_str)
        
        return latex_str.strip()
    
    def _parse_with_latex2sympy(self, latex_str: str) -> Any:
        """Parse using latex2sympy2 library."""
        try:
            from latex2sympy2 import latex2sympy
            return latex2sympy(latex_str)
        except Exception as e:
            logger.warning(f"latex2sympy failed: {e}, falling back to manual")
            return self._parse_manual(latex_str)
    
    def _parse_manual(self, latex_str: str) -> Any:
        """
        Manual LaTeX parsing fallback.
        
        Handles common LaTeX patterns when latex2sympy is unavailable.
        """
        from sympy import (
            Symbol, sympify, sqrt, Rational, oo,
            sin, cos, tan, log, ln, exp, pi, E,
            Integral, Derivative, Limit, Sum, Product,
            symbols
        )
        from sympy.parsing.sympy_parser import parse_expr
        
        expr_str = latex_str
        
        # Handle fractions: \frac{a}{b} -> (a)/(b)
        while r'\frac' in expr_str:
            expr_str = re.sub(
                r'\\frac\{([^{}]+)\}\{([^{}]+)\}',
                r'((\1)/(\2))',
                expr_str
            )
        
        # Handle sqrt: \sqrt{a} -> sqrt(a), \sqrt[n]{a} -> a**(1/n)
        expr_str = re.sub(r'\\sqrt\[(\d+)\]\{([^{}]+)\}', r'((\2)**(1/\1))', expr_str)
        expr_str = re.sub(r'\\sqrt\{([^{}]+)\}', r'sqrt(\1)', expr_str)
        
        # Handle powers: x^{n} -> x**(n)
        expr_str = re.sub(r'\^{([^{}]+)}', r'**(\1)', expr_str)
        expr_str = re.sub(r'\^(\d)', r'**\1', expr_str)
        
        # Handle subscripts (often indices, convert to symbol)
        expr_str = re.sub(r'_\{([^{}]+)\}', r'_\1', expr_str)
        expr_str = re.sub(r'_(\d)', r'_\1', expr_str)
        
        # Handle trig functions
        trig_funcs = ['sin', 'cos', 'tan', 'cot', 'sec', 'csc',
                      'arcsin', 'arccos', 'arctan', 'sinh', 'cosh', 'tanh']
        for func in trig_funcs:
            expr_str = re.sub(rf'\\{func}\s*\{{([^{{}}]+)\}}', rf'{func}(\1)', expr_str)
            expr_str = re.sub(rf'\\{func}\s+', rf'{func}', expr_str)
        
        # Handle log and ln
        expr_str = re.sub(r'\\ln\s*\{([^{}]+)\}', r'log(\1)', expr_str)
        expr_str = re.sub(r'\\log\s*\{([^{}]+)\}', r'log(\1, 10)', expr_str)
        expr_str = re.sub(r'\\exp\s*\{([^{}]+)\}', r'exp(\1)', expr_str)
        
        # Handle e^x
        expr_str = re.sub(r'e\^{([^{}]+)}', r'exp(\1)', expr_str)
        
        # Remove remaining braces
        expr_str = expr_str.replace('{', '(').replace('}', ')')
        
        # Handle implicit multiplication
        expr_str = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', expr_str)
        expr_str = re.sub(r'([a-zA-Z])(\d)', r'\1*\2', expr_str)
        expr_str = re.sub(r'\)(\w)', r')*\1', expr_str)
        expr_str = re.sub(r'(\w)\(', r'\1*(', expr_str)
        
        try:
            return parse_expr(expr_str)
        except Exception as e:
            logger.error(f"Manual parsing failed: {e}")
            raise ValueError(f"Could not parse LaTeX: {latex_str}")
    
    def to_latex(self, expr: Any) -> str:
        """
        Convert SymPy expression to LaTeX.
        
        Args:
            expr: SymPy expression
            
        Returns:
            LaTeX string representation
        """
        try:
            from sympy import latex
            return latex(expr)
        except Exception:
            return str(expr)
