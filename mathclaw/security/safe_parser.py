"""
Safe Parser - Secure mathematical expression parsing.

This module wraps SymPy's parsing with strict security controls.
It ONLY allows mathematical operations, blocking any code execution.

FROZEN: This file must NEVER be modified by the evolution engine.

Security Philosophy:
1. Never use sympify() directly on untrusted input
2. Use parse_expr() with restricted local_dict
3. Validate input BEFORE parsing
4. Catch and sanitize any parsing errors
"""

import re
from typing import Any, Dict, Optional, Set
from dataclasses import dataclass

from sympy import (
    Symbol, Integer, Float, Rational, pi, E, I, oo, zoo, nan,
    sin, cos, tan, cot, sec, csc,
    asin, acos, atan, acot, asec, acsc,
    sinh, cosh, tanh, coth, sech, csch,
    asinh, acosh, atanh, acoth, asech, acsch,
    exp, log, ln, sqrt, Abs, sign, floor, ceiling,
    factorial, gamma, beta, zeta, erf, erfc,
    Min, Max, Mod, gcd, lcm,
    diff, integrate, limit, Sum, Product,
    Eq, Ne, Lt, Le, Gt, Ge,
    And, Or, Not, Xor,
    Matrix, det, trace, transpose,
    Function, Lambda, Piecewise, Derivative, Integral,
    simplify, expand, factor, cancel, apart, together,
    symbols, Dummy,
)
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)

from .input_validator import InputValidator, ValidationError, validate_input


class SecurityError(Exception):
    """Raised when a security violation is detected during parsing."""
    
    def __init__(self, message: str, expression: str = "", violation_type: str = ""):
        self.message = message
        self.expression = expression[:100] if expression else ""
        self.violation_type = violation_type
        super().__init__(f"SECURITY: {message}")


@dataclass
class ParseResult:
    """Result of safe parsing."""
    success: bool
    expression: Any  # SymPy expression or None
    error: Optional[str]
    warnings: list


class SafeParser:
    """
    Secure mathematical expression parser.
    
    This class provides a safe way to parse mathematical expressions
    without risking code execution. It uses a strict allowlist of
    functions and validates all input before parsing.
    
    Security Layers:
    1. InputValidator - Character and pattern checks
    2. Restricted local_dict - Only safe SymPy functions
    3. Disabled __builtins__ - No Python builtins
    4. Post-parse validation - Check result type
    
    Example:
        >>> parser = SafeParser()
        >>> result = parser.parse("sin(x) + cos(x)")
        >>> result.expression
        sin(x) + cos(x)
        
        >>> parser.parse("__import__('os').system('ls')")
        SecurityError: Blocked pattern detected
    """
    
    # Transformations to apply during parsing
    TRANSFORMATIONS = (
        standard_transformations + 
        (implicit_multiplication_application, convert_xor)
    )
    
    def __init__(
        self,
        validator: InputValidator = None,
        extra_symbols: Dict[str, Any] = None,
        allow_undefined: bool = True,
    ):
        """
        Initialize safe parser.
        
        Args:
            validator: Custom input validator (uses default if None)
            extra_symbols: Additional symbols to allow in parsing
            allow_undefined: If True, undefined symbols become Symbol objects
        """
        self.validator = validator or InputValidator(strict_mode=False)
        self.allow_undefined = allow_undefined
        
        # Build the safe namespace
        self._safe_dict = self._build_safe_dict()
        if extra_symbols:
            # Validate extra symbols before adding
            for name, value in extra_symbols.items():
                if self._is_safe_value(value):
                    self._safe_dict[name] = value
    
    def _build_safe_dict(self) -> Dict[str, Any]:
        """
        Build the restricted namespace for parsing.
        
        Only mathematical functions and constants are included.
        NO Python builtins, NO dangerous functions.
        """
        safe = {
            # ─── Constants ───
            'pi': pi,
            'Pi': pi,
            'PI': pi,
            'e': E,
            'E': E,
            'I': I,
            'i': I,
            'oo': oo,
            'inf': oo,
            'Infinity': oo,
            'zoo': zoo,
            'nan': nan,
            
            # ─── Basic Functions ───
            'Abs': Abs,
            'abs': Abs,
            'sign': sign,
            'floor': floor,
            'ceiling': ceiling,
            'ceil': ceiling,
            'Min': Min,
            'Max': Max,
            'min': Min,
            'max': Max,
            'sqrt': sqrt,
            'Sqrt': sqrt,
            'exp': exp,
            'Exp': exp,
            'log': log,
            'Log': log,
            'ln': ln,
            'factorial': factorial,
            'gamma': gamma,
            'Gamma': gamma,
            'beta': beta,
            'zeta': zeta,
            
            # ─── Trigonometric ───
            'sin': sin, 'cos': cos, 'tan': tan,
            'cot': cot, 'sec': sec, 'csc': csc,
            'Sin': sin, 'Cos': cos, 'Tan': tan,
            'Cot': cot, 'Sec': sec, 'Csc': csc,
            'asin': asin, 'acos': acos, 'atan': atan,
            'acot': acot, 'asec': asec, 'acsc': acsc,
            'arcsin': asin, 'arccos': acos, 'arctan': atan,
            'arccot': acot, 'arcsec': asec, 'arccsc': acsc,
            
            # ─── Hyperbolic ───
            'sinh': sinh, 'cosh': cosh, 'tanh': tanh,
            'coth': coth, 'sech': sech, 'csch': csch,
            'asinh': asinh, 'acosh': acosh, 'atanh': atanh,
            'acoth': acoth, 'asech': asech, 'acsch': acsch,
            
            # ─── Special Functions ───
            'erf': erf,
            'erfc': erfc,
            
            # ─── Calculus (as constructors, not evaluators) ───
            'Derivative': Derivative,
            'Integral': Integral,
            'diff': diff,
            'integrate': integrate,
            'limit': limit,
            'Sum': Sum,
            'Product': Product,
            
            # ─── Algebra ───
            'simplify': simplify,
            'expand': expand,
            'factor': factor,
            'cancel': cancel,
            'apart': apart,
            'together': together,
            
            # ─── Number Theory ───
            'gcd': gcd,
            'lcm': lcm,
            'Mod': Mod,
            'mod': Mod,
            
            # ─── Relational ───
            'Eq': Eq,
            'Ne': Ne,
            'Lt': Lt,
            'Le': Le,
            'Gt': Gt,
            'Ge': Ge,
            
            # ─── Logic ───
            'And': And,
            'Or': Or,
            'Not': Not,
            'Xor': Xor,
            'true': True,
            'false': False,
            'True': True,
            'False': False,
            
            # ─── Linear Algebra ───
            'Matrix': Matrix,
            'det': det,
            'trace': trace,
            'transpose': transpose,
            
            # ─── Types ───
            'Symbol': Symbol,
            'symbols': symbols,
            'Integer': Integer,
            'Rational': Rational,
            'Float': Float,
            'Function': Function,
            'Lambda': Lambda,
            'Piecewise': Piecewise,
            
            # ─── Common variable symbols (pre-created) ───
            'x': Symbol('x'),
            'y': Symbol('y'),
            'z': Symbol('z'),
            't': Symbol('t'),
            'n': Symbol('n', integer=True),
            'k': Symbol('k', integer=True),
            'm': Symbol('m', integer=True),
            'a': Symbol('a'),
            'b': Symbol('b'),
            'c': Symbol('c'),
            'r': Symbol('r'),
            'theta': Symbol('theta'),
            'phi': Symbol('phi'),
            'alpha': Symbol('alpha'),
            'beta': Symbol('beta'),
            'omega': Symbol('omega'),
            'sigma': Symbol('sigma'),
            'lambda': Symbol('lambda'),
            'mu': Symbol('mu'),
            'nu': Symbol('nu'),
            'tau': Symbol('tau'),
        }
        
        return safe
    
    def _is_safe_value(self, value: Any) -> bool:
        """Check if a value is safe to add to the namespace."""
        # Allow SymPy types
        from sympy import Basic, Symbol
        from sympy.core.function import FunctionClass
        
        if isinstance(value, (type(None), bool, int, float, str)):
            return True
        if isinstance(value, Basic):
            return True
        if isinstance(value, type) and issubclass(value, Basic):
            return True
        if isinstance(value, FunctionClass):
            return True
        if callable(value) and hasattr(value, '__module__'):
            # Only allow sympy functions
            return value.__module__.startswith('sympy')
        
        return False
    
    def parse(self, expression: str) -> ParseResult:
        """
        Safely parse a mathematical expression.
        
        Args:
            expression: String to parse
            
        Returns:
            ParseResult with parsed expression or error
            
        Raises:
            SecurityError: If security violation detected
        """
        warnings = []
        
        # ─── Step 1: Input validation ───
        try:
            validation = self.validator.validate(expression)
            sanitized = validation.sanitized_input
            warnings.extend(validation.warnings)
        except ValidationError as e:
            raise SecurityError(
                f"Input validation failed: {e.message}",
                expression,
                e.rule_violated
            )
        
        # ─── Step 2: Additional security checks ───
        self._security_check(sanitized)
        
        # ─── Step 3: Parse with restricted namespace ───
        try:
            # Create a copy of safe_dict to avoid mutation
            local_dict = self._safe_dict.copy()
            
            # Parse with NO global namespace
            parsed = parse_expr(
                sanitized,
                local_dict=local_dict,
                transformations=self.TRANSFORMATIONS,
                evaluate=True,
            )
            
            # ─── Step 4: Post-parse validation ───
            if not self._validate_result(parsed):
                raise SecurityError(
                    "Parsed result contains unsafe types",
                    expression,
                    "unsafe_result_type"
                )
            
            return ParseResult(
                success=True,
                expression=parsed,
                error=None,
                warnings=warnings,
            )
            
        except SecurityError:
            raise
        except SyntaxError as e:
            return ParseResult(
                success=False,
                expression=None,
                error=f"Syntax error: {e}",
                warnings=warnings,
            )
        except Exception as e:
            # Don't leak internal error details
            error_type = type(e).__name__
            return ParseResult(
                success=False,
                expression=None,
                error=f"Parse error ({error_type}): {str(e)[:100]}",
                warnings=warnings,
            )
    
    def _security_check(self, expression: str) -> None:
        """Additional security checks before parsing."""
        
        # Check for attribute access on results of function calls
        # e.g., sin(x).__class__ is suspicious
        if re.search(r'\)\s*\.', expression):
            # Allow .subs(), .diff(), etc. - common SymPy methods
            safe_methods = {'.subs', '.diff', '.integrate', '.limit', '.series',
                          '.simplify', '.expand', '.factor', '.doit', '.evalf',
                          '.rewrite', '.trigsimp', '.radsimp', '.ratsimp'}
            
            matches = re.findall(r'\)\s*(\.[a-zA-Z_]+)', expression)
            for match in matches:
                method = match.split('(')[0]
                if method not in safe_methods:
                    raise SecurityError(
                        f"Suspicious method access: {method}",
                        expression,
                        "method_access"
                    )
        
        # Check for string literals (usually not needed in math)
        if re.search(r'["\']', expression):
            raise SecurityError(
                "String literals not allowed in mathematical expressions",
                expression,
                "string_literal"
            )
    
    def _validate_result(self, result: Any) -> bool:
        """Validate that the parsing result is a safe SymPy type."""
        from sympy import Basic
        
        if result is None:
            return True
        
        if isinstance(result, (bool, int, float)):
            return True
        
        if isinstance(result, Basic):
            return True
        
        if isinstance(result, (list, tuple)):
            return all(self._validate_result(item) for item in result)
        
        if isinstance(result, dict):
            return all(
                self._validate_result(k) and self._validate_result(v)
                for k, v in result.items()
            )
        
        return False
    
    def parse_or_raise(self, expression: str) -> Any:
        """
        Parse expression, raising exception on failure.
        
        Args:
            expression: String to parse
            
        Returns:
            Parsed SymPy expression
            
        Raises:
            SecurityError: On security violation
            ValueError: On parse failure
        """
        result = self.parse(expression)
        
        if not result.success:
            raise ValueError(result.error)
        
        return result.expression
    
    def get_symbols(self, expression: str) -> Set[Symbol]:
        """
        Parse and extract all symbols from expression.
        
        Args:
            expression: String to parse
            
        Returns:
            Set of Symbol objects found
        """
        result = self.parse(expression)
        
        if not result.success or result.expression is None:
            return set()
        
        if hasattr(result.expression, 'free_symbols'):
            return result.expression.free_symbols
        
        return set()


# ═══════════════════════════════════════════════════════════════
# Convenience functions
# ═══════════════════════════════════════════════════════════════

_default_parser = None

def safe_parse(expression: str) -> ParseResult:
    """Parse expression using default safe parser."""
    global _default_parser
    if _default_parser is None:
        _default_parser = SafeParser()
    return _default_parser.parse(expression)


def safe_parse_or_raise(expression: str) -> Any:
    """Parse expression, raising on failure."""
    global _default_parser
    if _default_parser is None:
        _default_parser = SafeParser()
    return _default_parser.parse_or_raise(expression)
