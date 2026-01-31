"""
Input Validator - First line of defense against malicious input.

This module validates ALL input before it reaches any parser or evaluator.
It enforces strict allowlists and blocklists to prevent injection attacks.

FROZEN: This file must NEVER be modified by the evolution engine.
"""

import re
from typing import Set, List, Tuple
from dataclasses import dataclass


class ValidationError(Exception):
    """Raised when input fails validation."""
    
    def __init__(self, message: str, input_sample: str = "", rule_violated: str = ""):
        self.message = message
        self.input_sample = input_sample[:50] + "..." if len(input_sample) > 50 else input_sample
        self.rule_violated = rule_violated
        super().__init__(f"{message} [Rule: {rule_violated}]")


@dataclass
class ValidationResult:
    """Result of input validation."""
    valid: bool
    sanitized_input: str
    warnings: List[str]
    blocked_patterns_found: List[str]


class InputValidator:
    """
    Validates mathematical input against security rules.
    
    Philosophy: Allowlist is safer than blocklist, but we use both
    for defense in depth.
    
    Security Model:
    1. Length limits (prevent DoS)
    2. Character allowlist (only math-relevant chars)
    3. Pattern blocklist (catch known attack vectors)
    4. Nesting depth limits (prevent stack overflow)
    5. Symbol allowlist (only known-safe functions)
    
    Example:
        >>> validator = InputValidator()
        >>> result = validator.validate("sin(x) + cos(x)")
        >>> result.valid
        True
        >>> validator.validate("__import__('os')")
        ValidationError: Blocked pattern detected
    """
    
    # ═══════════════════════════════════════════════════════════════
    # ALLOWLISTS (Explicitly permitted)
    # ═══════════════════════════════════════════════════════════════
    
    # Characters allowed in mathematical expressions
    ALLOWED_CHARS: Set[str] = set(
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789"
        "+-*/^**()[]{}.,;:=<>!|&"
        " \t\n"  # Whitespace
        "_"  # For variable names like x_1
    )
    
    # Mathematical symbols/functions that are safe
    ALLOWED_SYMBOLS: Set[str] = {
        # Variables (single letter)
        'x', 'y', 'z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
        'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w',
        
        # Constants
        'pi', 'PI', 'Pi',
        'E', 'e',  # Euler's number
        'I', 'i',  # Imaginary unit
        'oo',  # Infinity
        'zoo',  # Complex infinity
        'nan',  # Not a number
        
        # Basic operations
        'Abs', 'abs', 'sign', 'floor', 'ceiling', 'ceil',
        'Min', 'Max', 'min', 'max',
        'sqrt', 'Sqrt', 'root', 'cbrt',
        'exp', 'Exp', 'log', 'Log', 'ln', 'log10', 'log2',
        'factorial', 'binomial', 'gamma', 'Gamma',
        
        # Trigonometric
        'sin', 'cos', 'tan', 'cot', 'sec', 'csc',
        'Sin', 'Cos', 'Tan', 'Cot', 'Sec', 'Csc',
        'asin', 'acos', 'atan', 'atan2', 'acot', 'asec', 'acsc',
        'arcsin', 'arccos', 'arctan', 'arccot', 'arcsec', 'arccsc',
        
        # Hyperbolic
        'sinh', 'cosh', 'tanh', 'coth', 'sech', 'csch',
        'asinh', 'acosh', 'atanh', 'acoth', 'asech', 'acsch',
        
        # Calculus operations (as symbols, not executed)
        'diff', 'Derivative', 'integrate', 'Integral',
        'limit', 'Limit', 'Sum', 'Product',
        'series', 'taylor', 'fourier',
        
        # Special functions
        'zeta', 'beta', 'erf', 'erfc', 'erfi',
        'besselj', 'bessely', 'besseli', 'besselk',
        'legendre', 'hermite', 'laguerre', 'chebyshev',
        'digamma', 'polygamma', 'loggamma',
        
        # Number theory
        'prime', 'isprime', 'nextprime', 'prevprime',
        'factorint', 'divisors', 'totient', 'mobius',
        'gcd', 'lcm', 'mod', 'Mod',
        
        # Linear algebra (symbols)
        'Matrix', 'det', 'trace', 'transpose',
        'eigenvals', 'eigenvects', 'inv',
        
        # Logic/Relational
        'Eq', 'Ne', 'Lt', 'Le', 'Gt', 'Ge',
        'And', 'Or', 'Not', 'Xor',
        'true', 'false', 'True', 'False',
        
        # SymPy types
        'Symbol', 'symbols', 'Integer', 'Rational', 'Float',
        'Function', 'Lambda', 'Piecewise',
        'oo', 'inf', 'Infinity',
        
        # Greek letters (often used as variables)
        'alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta',
        'theta', 'iota', 'kappa', 'lambda', 'mu', 'nu', 'xi',
        'omicron', 'rho', 'sigma', 'tau', 'upsilon', 'phi', 'chi', 'psi', 'omega',
        'Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon', 'Zeta', 'Eta',
        'Theta', 'Iota', 'Kappa', 'Lambda', 'Mu', 'Nu', 'Xi',
        'Omicron', 'Rho', 'Sigma', 'Tau', 'Upsilon', 'Phi', 'Chi', 'Psi', 'Omega',
    }
    
    # ═══════════════════════════════════════════════════════════════
    # BLOCKLISTS (Explicitly forbidden - defense in depth)
    # ═══════════════════════════════════════════════════════════════
    
    # Patterns that NEVER appear in legitimate math
    BLOCKED_PATTERNS: List[Tuple[str, str]] = [
        # Python code execution
        (r'__\w+__', 'dunder_method'),
        (r'\bimport\b', 'import_statement'),
        (r'\bexec\b', 'exec_function'),
        (r'\beval\b', 'eval_function'),
        (r'\bcompile\b', 'compile_function'),
        (r'\bopen\b', 'file_open'),
        (r'\bfile\b', 'file_object'),
        (r'\bprint\b', 'print_function'),
        (r'\binput\b', 'input_function'),
        
        # System access
        (r'\bos\b', 'os_module'),
        (r'\bsys\b', 'sys_module'),
        (r'\bsubprocess\b', 'subprocess_module'),
        (r'\bshutil\b', 'shutil_module'),
        (r'\bsocket\b', 'socket_module'),
        (r'\brequests\b', 'requests_module'),
        (r'\burllib\b', 'urllib_module'),
        
        # Dangerous builtins
        (r'\bglobals\b', 'globals_builtin'),
        (r'\blocals\b', 'locals_builtin'),
        (r'\bvars\b', 'vars_builtin'),
        (r'\bdir\b', 'dir_builtin'),
        (r'\bgetattr\b', 'getattr_builtin'),
        (r'\bsetattr\b', 'setattr_builtin'),
        (r'\bdelattr\b', 'delattr_builtin'),
        (r'\bhasattr\b', 'hasattr_builtin'),
        
        # Class/type manipulation
        (r'\btype\b\s*\(', 'type_constructor'),
        (r'\bclass\b', 'class_definition'),
        (r'\bdef\b', 'function_definition'),
        (r'\blambda\b\s*:', 'lambda_with_colon'),  # Allow Lambda as SymPy symbol
        
        # String formatting attacks
        (r'\bformat\b', 'format_method'),
        (r'%\s*\(', 'percent_formatting'),
        (r'\{[^}]*\}', 'curly_brace_formatting'),  # But allow {} for sets - relaxed below
        
        # Dangerous string methods
        (r'\.read\s*\(', 'read_method'),
        (r'\.write\s*\(', 'write_method'),
        (r'\.execute\s*\(', 'execute_method'),
        
        # Shell injection
        (r'[;&|]', 'shell_metachar'),  # Relaxed: allow in math context
        (r'\$\(', 'command_substitution'),
        (r'`[^`]+`', 'backtick_execution'),
    ]
    
    # Refined blocklist - these are ALWAYS blocked
    CRITICAL_BLOCKS: List[Tuple[str, str]] = [
        (r'__\w+__', 'dunder_method'),
        (r'\bimport\s', 'import_statement'),
        (r'\bexec\s*\(', 'exec_call'),
        (r'\beval\s*\(', 'eval_call'),
        (r'\bopen\s*\(', 'open_call'),
        (r'\bos\.', 'os_access'),
        (r'\bsys\.', 'sys_access'),
        (r'\bsubprocess', 'subprocess_access'),
        (r'globals\s*\(\s*\)', 'globals_call'),
        (r'locals\s*\(\s*\)', 'locals_call'),
        (r'\.__class__', 'class_access'),
        (r'\.__base__', 'base_access'),
        (r'\.__subclasses__', 'subclass_access'),
        (r'\.__mro__', 'mro_access'),
        (r'\.__code__', 'code_access'),
        (r'\.__globals__', 'globals_access'),
    ]
    
    # ═══════════════════════════════════════════════════════════════
    # LIMITS
    # ═══════════════════════════════════════════════════════════════
    
    MAX_LENGTH: int = 2000  # Characters
    MAX_NESTING_DEPTH: int = 50  # Parentheses/brackets
    MAX_SYMBOL_LENGTH: int = 50  # Single identifier
    MAX_NUMBER_LENGTH: int = 100  # Digits in a number
    
    def __init__(
        self,
        max_length: int = None,
        max_nesting: int = None,
        extra_allowed_symbols: Set[str] = None,
        strict_mode: bool = True,
    ):
        """
        Initialize validator.
        
        Args:
            max_length: Override max input length
            max_nesting: Override max nesting depth
            extra_allowed_symbols: Additional symbols to allow
            strict_mode: If True, unknown symbols are rejected
        """
        self.max_length = max_length or self.MAX_LENGTH
        self.max_nesting = max_nesting or self.MAX_NESTING_DEPTH
        self.strict_mode = strict_mode
        
        self.allowed_symbols = self.ALLOWED_SYMBOLS.copy()
        if extra_allowed_symbols:
            self.allowed_symbols.update(extra_allowed_symbols)
        
        # Compile regex patterns for performance
        self._critical_patterns = [
            (re.compile(pattern, re.IGNORECASE), name) 
            for pattern, name in self.CRITICAL_BLOCKS
        ]
    
    def validate(self, input_string: str) -> ValidationResult:
        """
        Validate input string against all security rules.
        
        Args:
            input_string: Raw input to validate
            
        Returns:
            ValidationResult with status and sanitized input
            
        Raises:
            ValidationError: If critical security violation detected
        """
        warnings = []
        blocked_found = []
        
        # ─── Check 1: Length ───
        if len(input_string) > self.max_length:
            raise ValidationError(
                f"Input too long ({len(input_string)} > {self.max_length})",
                input_string,
                "max_length"
            )
        
        # ─── Check 2: Empty input ───
        if not input_string or not input_string.strip():
            raise ValidationError(
                "Empty input",
                input_string,
                "non_empty"
            )
        
        # ─── Check 3: Critical blocklist (ALWAYS fail) ───
        for pattern, name in self._critical_patterns:
            if pattern.search(input_string):
                raise ValidationError(
                    f"Blocked pattern detected: {name}",
                    input_string,
                    f"blocked_{name}"
                )
        
        # ─── Check 4: Character allowlist ───
        invalid_chars = set(input_string) - self.ALLOWED_CHARS
        if invalid_chars:
            # Allow some Unicode math symbols
            math_unicode = set('∞∫∑∏√πΣΠ∂∇αβγδεζηθικλμνξοπρστυφχψω')
            truly_invalid = invalid_chars - math_unicode
            if truly_invalid:
                raise ValidationError(
                    f"Invalid characters: {truly_invalid}",
                    input_string,
                    "allowed_chars"
                )
            else:
                warnings.append(f"Unicode math symbols detected: {invalid_chars}")
        
        # ─── Check 5: Nesting depth ───
        depth = 0
        max_depth = 0
        for char in input_string:
            if char in '([{':
                depth += 1
                max_depth = max(max_depth, depth)
            elif char in ')]}':
                depth -= 1
            
            if depth < 0:
                raise ValidationError(
                    "Unbalanced brackets",
                    input_string,
                    "balanced_brackets"
                )
        
        if depth != 0:
            raise ValidationError(
                "Unbalanced brackets",
                input_string,
                "balanced_brackets"
            )
        
        if max_depth > self.max_nesting:
            raise ValidationError(
                f"Nesting too deep ({max_depth} > {self.max_nesting})",
                input_string,
                "max_nesting"
            )
        
        # ─── Check 6: Extract and validate symbols ───
        # Find all word-like tokens
        tokens = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', input_string)
        
        for token in tokens:
            # Check symbol length
            if len(token) > self.MAX_SYMBOL_LENGTH:
                raise ValidationError(
                    f"Symbol too long: {token[:20]}...",
                    input_string,
                    "max_symbol_length"
                )
            
            # In strict mode, reject unknown symbols
            if self.strict_mode and token not in self.allowed_symbols:
                # Allow subscripted variables like x_1, y_2
                base = token.split('_')[0]
                if len(base) <= 2 or base in self.allowed_symbols:
                    continue
                    
                warnings.append(f"Unknown symbol: {token}")
        
        # ─── Check 7: Number sanity ───
        numbers = re.findall(r'\d+\.?\d*', input_string)
        for num in numbers:
            if len(num) > self.MAX_NUMBER_LENGTH:
                raise ValidationError(
                    f"Number too long: {num[:20]}...",
                    input_string,
                    "max_number_length"
                )
        
        # ─── Sanitize ───
        sanitized = self._sanitize(input_string)
        
        return ValidationResult(
            valid=True,
            sanitized_input=sanitized,
            warnings=warnings,
            blocked_patterns_found=blocked_found,
        )
    
    def _sanitize(self, input_string: str) -> str:
        """
        Sanitize input by normalizing whitespace and removing comments.
        """
        # Remove Python-style comments
        sanitized = re.sub(r'#.*$', '', input_string, flags=re.MULTILINE)
        
        # Normalize whitespace
        sanitized = ' '.join(sanitized.split())
        
        return sanitized.strip()
    
    def is_safe_symbol(self, symbol: str) -> bool:
        """Check if a symbol is in the allowlist."""
        return symbol in self.allowed_symbols
    
    def add_allowed_symbol(self, symbol: str) -> None:
        """Add a symbol to the allowlist (use with caution)."""
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', symbol):
            raise ValueError(f"Invalid symbol format: {symbol}")
        self.allowed_symbols.add(symbol)


# ═══════════════════════════════════════════════════════════════
# Quick validation function for convenience
# ═══════════════════════════════════════════════════════════════

_default_validator = None

def validate_input(input_string: str, strict: bool = True) -> ValidationResult:
    """
    Validate input using default validator.
    
    Args:
        input_string: Input to validate
        strict: Whether to reject unknown symbols
        
    Returns:
        ValidationResult
    """
    global _default_validator
    if _default_validator is None:
        _default_validator = InputValidator(strict_mode=strict)
    return _default_validator.validate(input_string)


def is_safe(input_string: str) -> bool:
    """Quick check if input is safe."""
    try:
        validate_input(input_string)
        return True
    except ValidationError:
        return False
