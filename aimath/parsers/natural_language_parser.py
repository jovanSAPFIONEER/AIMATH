"""
Universal Natural Language Math Parser

Uses LLM translation pattern to convert natural language to SymPy code.
Falls back to pattern matching for common phrases.
"""

import re as regex  # Rename to avoid SymPy conflict
import sympy
from sympy import *
from typing import Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


# Common natural language patterns mapped to SymPy templates
NL_PATTERNS = {
    # Derivatives
    r"(?:find\s+)?(?:the\s+)?derivative\s+of\s+(.+?)(?:\s+with\s+respect\s+to\s+(\w))?$": 
        lambda m: f"diff({m.group(1)}, {m.group(2) or 'x'})",
    r"(?:find\s+)?d/d(\w)\s*\[?(.+?)\]?$":
        lambda m: f"diff({m.group(2)}, {m.group(1)})",
    r"differentiate\s+(.+?)(?:\s+with\s+respect\s+to\s+(\w))?$":
        lambda m: f"diff({m.group(1)}, {m.group(2) or 'x'})",
    
    # Integrals
    r"(?:find\s+)?(?:the\s+)?integral\s+of\s+(.+?)(?:\s+from\s+(.+?)\s+to\s+(.+))?$":
        lambda m: f"integrate({m.group(1)}, (x, {m.group(2)}, {m.group(3)}))" if m.group(2) else f"integrate({m.group(1)}, x)",
    r"(?:find\s+)?(?:the\s+)?area\s+under\s+(.+?)\s+from\s+(.+?)\s+to\s+(.+)$":
        lambda m: f"integrate({m.group(1)}, (x, {m.group(2)}, {m.group(3)}))",
    r"integrate\s+(.+?)(?:\s+from\s+(.+?)\s+to\s+(.+))?$":
        lambda m: f"integrate({m.group(1)}, (x, {m.group(2)}, {m.group(3)}))" if m.group(2) else f"integrate({m.group(1)}, x)",
    
    # Equations
    r"solve\s+(.+?)\s*=\s*(.+?)(?:\s+for\s+(\w))?$":
        lambda m: f"solve(Eq({m.group(1)}, {m.group(2)}), {m.group(3) or 'x'})",
    r"(?:find\s+)?(?:the\s+)?(?:roots?|zeros?|solutions?)\s+of\s+(.+)$":
        lambda m: f"solve({m.group(1)}, x)",
    r"(?:find\s+)?where\s+(.+?)\s+(?:equals?|=|is)\s+(?:zero|0)$":
        lambda m: f"solve({m.group(1)}, x)",
    r"(?:find\s+)?where\s+(?:the\s+)?(?:curve|function|graph)\s+crosses\s+(?:the\s+)?x[- ]?axis":
        lambda m: "solve(f, x)",  # Needs context
    
    # Limits
    r"(?:find\s+)?(?:the\s+)?limit\s+(?:of\s+)?(.+?)\s+as\s+(\w)\s*(?:->|→|approaches?)\s*(.+)$":
        lambda m: f"limit({m.group(1)}, {m.group(2)}, {m.group(3)})",
    r"(?:evaluate\s+)?lim\s*(?:\(|_{)?(\w)\s*(?:->|→)\s*(.+?)(?:\)|})?(?:\s+of)?\s*(.+)$":
        lambda m: f"limit({m.group(3)}, {m.group(1)}, {m.group(2)})",
    
    # Series
    r"(?:find\s+)?(?:the\s+)?sum\s+of\s+(.+?)\s+from\s+(\w)\s*=\s*(.+?)\s+to\s+(.+)$":
        lambda m: f"Sum({m.group(1)}, ({m.group(2)}, {m.group(3)}, {m.group(4)})).doit()",
    r"(?:evaluate\s+)?(?:the\s+)?series\s+(.+?)\s+from\s+(\w)\s*=\s*(.+?)\s+to\s+(.+)$":
        lambda m: f"Sum({m.group(1)}, ({m.group(2)}, {m.group(3)}, {m.group(4)})).doit()",
    
    # Linear Algebra
    r"(?:find\s+)?(?:the\s+)?eigenvalues?\s+of\s+(?:matrix\s+)?\[\[(.+)\]\]$":
        lambda m: f"Matrix([{m.group(1)}]).eigenvals()",
    r"(?:find\s+)?(?:the\s+)?determinant\s+of\s+(?:matrix\s+)?\[\[(.+)\]\]$":
        lambda m: f"Matrix([{m.group(1)}]).det()",
    r"(?:find\s+)?(?:the\s+)?inverse\s+of\s+(?:matrix\s+)?\[\[(.+)\]\]$":
        lambda m: f"Matrix([{m.group(1)}]).inv()",
    
    # Simplification
    r"simplify\s+(.+)$":
        lambda m: f"simplify({m.group(1)})",
    r"expand\s+(.+)$":
        lambda m: f"expand({m.group(1)})",
    r"factor\s+(.+)$":
        lambda m: f"factor({m.group(1)})",
    
    # Taylor/Maclaurin
    r"(?:find\s+)?(?:the\s+)?taylor\s+(?:series\s+)?(?:expansion\s+)?of\s+(.+?)(?:\s+around\s+(\w)\s*=\s*(.+?))?(?:\s+to\s+(?:order\s+)?(\d+))?$":
        lambda m: f"series({m.group(1)}, {m.group(2) or 'x'}, {m.group(3) or '0'}, {m.group(4) or '6'})",
}


def normalize_math_input(text: str) -> str:
    """Normalize common math notations to SymPy format."""
    text = text.strip().lower()
    
    # Common substitutions
    replacements = {
        r'\^': '**',
        r'×': '*',
        r'·': '*',
        r'÷': '/',
        r'√': 'sqrt',
        r'π': 'pi',
        r'∞': 'oo',
        r'infinity': 'oo',
        r'e\^': 'exp(',
        r'ln\(': 'log(',
        r'log\(': 'log(',  # Natural log in math
        r'arcsin': 'asin',
        r'arccos': 'acos',
        r'arctan': 'atan',
        r'x squared': 'x**2',
        r'x cubed': 'x**3',
        r'square root of (.+)': r'sqrt(\1)',
        r'cube root of (.+)': r'(\1)**(1/3)',
    }
    
    for pattern, replacement in replacements.items():
        text = regex.sub(pattern, replacement, text)
    
    return text


def parse_natural_language(query: str, llm_client=None) -> Tuple[str, str]:
    """
    Parse natural language math query to SymPy code.
    
    Args:
        query: Natural language math question
        llm_client: Optional LLM client for advanced parsing
        
    Returns:
        Tuple of (sympy_code, method_used)
    """
    normalized = normalize_math_input(query)
    
    # Try pattern matching first (fast, no API calls)
    for pattern, template in NL_PATTERNS.items():
        match = regex.match(pattern, normalized, regex.IGNORECASE)
        if match:
            try:
                code = template(match)
                logger.info(f"Pattern matched: '{query}' -> `{code}`")
                return code, "pattern_match"
            except Exception as e:
                logger.warning(f"Pattern template error: {e}")
                continue
    
    # If LLM client provided, use it for complex queries
    if llm_client:
        code = _llm_translate(query, llm_client)
        if code:
            return code, "llm_translation"
    
    # Fallback: try to interpret as raw SymPy
    return query, "raw_sympy"


def _llm_translate(query: str, llm_client) -> Optional[str]:
    """Use LLM to translate natural language to SymPy code."""
    system_prompt = """You are a strictly formal Python code generator for the SymPy library.
Convert the user's math request into a single executable Python expression.

RULES:
- Assume 'from sympy import *' is already executed
- Assume symbols x, y, z, t, n, k are already defined
- OUTPUT ONLY THE CODE. No markdown, no explanations, no backticks.
- Use proper SymPy syntax

EXAMPLES:
- "Find the derivative of x squared" -> diff(x**2, x)
- "Solve x^2 = 4" -> solve(Eq(x**2, 4), x)
- "Area under sin(x) from 0 to pi" -> integrate(sin(x), (x, 0, pi))
- "Sum of 1/n^2 from n=1 to infinity" -> Sum(1/n**2, (n, 1, oo)).doit()
- "Eigenvalues of [[1, 2], [2, 1]]" -> Matrix([[1, 2], [2, 1]]).eigenvals()
- "Limit of sin(x)/x as x approaches 0" -> limit(sin(x)/x, x, 0)
"""
    
    try:
        # This is a generic interface - adapt to your LLM client
        if hasattr(llm_client, 'generate'):
            response = llm_client.generate(system_prompt, query)
            code = response.text.strip()
        elif hasattr(llm_client, 'chat'):
            response = llm_client.chat([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ])
            code = response.strip()
        else:
            return None
        
        # Clean up any markdown artifacts
        code = code.strip('`').strip()
        if code.startswith('python'):
            code = code[6:].strip()
        
        logger.info(f"LLM translated: '{query}' -> `{code}`")
        return code
    except Exception as e:
        logger.error(f"LLM translation error: {e}")
        return None


def execute_safe(code_str: str) -> Any:
    """
    Execute SymPy code in a controlled sandbox.
    
    Args:
        code_str: Valid SymPy code string
        
    Returns:
        Result of execution or error message
    """
    # Create safe execution context with all SymPy functions
    x, y, z, t, n, k, a, b, c = symbols('x y z t n k a b c')
    f = Function('f')
    
    context = {
        # Symbols
        'x': x, 'y': y, 'z': z, 't': t, 'n': n, 'k': k,
        'a': a, 'b': b, 'c': c, 'f': f,
        # Core SymPy
        'Symbol': Symbol, 'symbols': symbols, 'Function': Function,
        'Eq': Eq, 'Ne': Ne, 'Lt': Lt, 'Le': Le, 'Gt': Gt, 'Ge': Ge,
        # Numbers
        'pi': pi, 'E': E, 'I': I, 'oo': oo, 'zoo': zoo,
        'Rational': Rational, 'Integer': Integer, 'Float': Float,
        # Functions
        'sin': sin, 'cos': cos, 'tan': tan, 'cot': cot, 'sec': sec, 'csc': csc,
        'asin': asin, 'acos': acos, 'atan': atan, 'atan2': atan2,
        'sinh': sinh, 'cosh': cosh, 'tanh': tanh,
        'exp': exp, 'log': log, 'ln': log, 'sqrt': sqrt, 'Abs': Abs,
        'factorial': factorial, 'binomial': binomial, 'gamma': gamma,
        # Calculus
        'diff': diff, 'Derivative': Derivative,
        'integrate': integrate, 'Integral': Integral,
        'limit': limit, 'Limit': Limit,
        'Sum': Sum, 'Product': Product, 'summation': summation,
        'series': series,
        # Algebra
        'solve': solve, 'solveset': solveset, 'linsolve': linsolve,
        'simplify': simplify, 'expand': expand, 'factor': factor,
        'collect': collect, 'cancel': cancel, 'apart': apart, 'together': together,
        'trigsimp': trigsimp, 'powsimp': powsimp, 'radsimp': radsimp,
        # Linear Algebra
        'Matrix': Matrix, 'eye': eye, 'zeros': zeros, 'ones': ones,
        'diag': diag, 'det': lambda M: M.det(),
        # ODEs
        'dsolve': dsolve, 'classify_ode': classify_ode,
        # Misc
        'N': N, 'evalf': lambda e, n=15: e.evalf(n),
        'latex': latex, 'pprint': pprint,
        'S': S, 'sympify': sympify,
    }
    
    try:
        # Use compile + eval for better error messages
        compiled = compile(code_str, '<aimath>', 'eval')
        result = eval(compiled, {"__builtins__": {}}, context)
        return result
    except SyntaxError as e:
        return f"Syntax Error: {e}"
    except NameError as e:
        return f"Unknown symbol: {e}"
    except Exception as e:
        return f"Execution Error: {e}"


class NaturalLanguageParser:
    """
    Natural Language Math Parser with optional LLM support.
    
    Usage:
        parser = NaturalLanguageParser()
        result = parser.parse_and_execute("Find the derivative of x cubed")
        # Returns: 3*x**2
    """
    
    def __init__(self, llm_client=None):
        """
        Initialize parser.
        
        Args:
            llm_client: Optional LLM client for advanced parsing
        """
        self.llm_client = llm_client
        self._cache = {}
    
    def parse(self, query: str) -> Tuple[str, str]:
        """
        Parse natural language to SymPy code.
        
        Returns:
            Tuple of (sympy_code, method_used)
        """
        # Check cache
        if query in self._cache:
            return self._cache[query], "cached"
        
        code, method = parse_natural_language(query, self.llm_client)
        self._cache[query] = code
        return code, method
    
    def parse_and_execute(self, query: str) -> Any:
        """
        Parse and execute a natural language math query.
        
        Args:
            query: Natural language math question
            
        Returns:
            SymPy result or error message
        """
        code, method = self.parse(query)
        logger.info(f"Executing ({method}): {code}")
        result = execute_safe(code)
        return result
    
    def explain_parse(self, query: str) -> dict:
        """
        Parse and explain the translation process.
        
        Returns:
            Dict with parsing details
        """
        code, method = self.parse(query)
        result = execute_safe(code)
        
        return {
            "query": query,
            "normalized": normalize_math_input(query),
            "sympy_code": code,
            "method": method,
            "result": result,
            "result_latex": latex(result) if hasattr(result, 'atoms') else str(result)
        }


# Convenience function
def ask_math(query: str, llm_client=None) -> Any:
    """
    Simple interface: ask a math question in English, get the answer.
    
    Examples:
        >>> ask_math("derivative of x^3")
        3*x**2
        >>> ask_math("solve x^2 - 4 = 0")
        [-2, 2]
        >>> ask_math("integral of sin(x) from 0 to pi")
        2
    """
    parser = NaturalLanguageParser(llm_client)
    return parser.parse_and_execute(query)
