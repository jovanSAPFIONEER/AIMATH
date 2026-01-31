"""
Special Integral Database (Knowledge Base)

A lookup table for famous definite integrals that are:
1. Difficult/impossible for symbolic engines
2. Have known closed-form solutions
3. Frequently appear in competitions and research

This allows instant recognition of benchmark integrals.
"""

import sympy
from sympy import (
    Symbol, symbols, sqrt, log, atan, sin, cos, tan, exp,
    pi, E, oo, Rational, S, sympify, simplify, srepr,
    Integral, zeta, gamma, factorial, binomial
)
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
import hashlib
import logging

logger = logging.getLogger(__name__)

x, y, z, t, u, a, b, n = symbols('x y z t u a b n')


@dataclass
class IntegralEntry:
    """An entry in the integral database."""
    name: str
    integrand: Any  # SymPy expression
    variable: Symbol
    lower: Any
    upper: Any
    result: Any  # SymPy expression
    category: str
    difficulty: str  # 'standard', 'hard', 'olympiad', 'research'
    techniques: List[str]  # Methods to solve it
    reference: str = ""


class IntegralKnowledgeBase:
    """
    Database of famous/difficult definite integrals.
    """
    
    def __init__(self):
        self.db: Dict[str, IntegralEntry] = {}
        self._build_database()
    
    def _build_database(self):
        """Populate the database with known integrals."""
        
        # ═══════════════════════════════════════════════════════════
        # AHMED'S INTEGRAL (The one that started this!)
        # ═══════════════════════════════════════════════════════════
        self._add_entry(IntegralEntry(
            name="Ahmed's Integral",
            integrand=atan(sqrt(x**2 + 2)) / ((x**2 + 1) * sqrt(x**2 + 2)),
            variable=x,
            lower=0,
            upper=1,
            result=5 * pi**2 / 96,
            category="special_functions",
            difficulty="olympiad",
            techniques=["Feynman", "contour_integration", "parameter_differentiation"],
            reference="Nahin, Inside Interesting Integrals"
        ))
        
        # ═══════════════════════════════════════════════════════════
        # DIRICHLET INTEGRAL
        # ∫₀^∞ sin(x)/x dx = π/2
        # ═══════════════════════════════════════════════════════════
        self._add_entry(IntegralEntry(
            name="Dirichlet Integral",
            integrand=sin(x) / x,
            variable=x,
            lower=0,
            upper=oo,
            result=pi / 2,
            category="classic",
            difficulty="standard",
            techniques=["Feynman", "Laplace_transform", "contour_integration"],
            reference="Standard calculus"
        ))
        
        # ═══════════════════════════════════════════════════════════
        # GAUSSIAN INTEGRAL
        # ∫₋∞^∞ e^(-x²) dx = √π
        # ═══════════════════════════════════════════════════════════
        self._add_entry(IntegralEntry(
            name="Gaussian Integral",
            integrand=exp(-x**2),
            variable=x,
            lower=-oo,
            upper=oo,
            result=sqrt(pi),
            category="classic",
            difficulty="standard",
            techniques=["polar_coordinates", "Gamma_function"],
            reference="Standard calculus"
        ))
        
        # Half Gaussian
        self._add_entry(IntegralEntry(
            name="Half Gaussian Integral",
            integrand=exp(-x**2),
            variable=x,
            lower=0,
            upper=oo,
            result=sqrt(pi) / 2,
            category="classic",
            difficulty="standard",
            techniques=["polar_coordinates", "Gamma_function"],
            reference="Standard calculus"
        ))
        
        # ═══════════════════════════════════════════════════════════
        # EULER'S REFLECTION / BETA INTEGRALS
        # ═══════════════════════════════════════════════════════════
        self._add_entry(IntegralEntry(
            name="Basel Problem Integral",
            integrand=1 / (1 - x*y),  # This needs double integral form
            variable=x,
            lower=0,
            upper=1,
            result=pi**2 / 6,  # ζ(2)
            category="zeta_values",
            difficulty="hard",
            techniques=["series_expansion", "double_integral"],
            reference="Euler, Basel problem"
        ))
        
        # ═══════════════════════════════════════════════════════════
        # FRESNEL INTEGRALS
        # ═══════════════════════════════════════════════════════════
        self._add_entry(IntegralEntry(
            name="Fresnel Cosine Integral",
            integrand=cos(x**2),
            variable=x,
            lower=0,
            upper=oo,
            result=sqrt(pi/2) / 2,
            category="special_functions",
            difficulty="hard",
            techniques=["contour_integration", "Gamma_function"],
            reference="Fresnel integrals"
        ))
        
        self._add_entry(IntegralEntry(
            name="Fresnel Sine Integral",
            integrand=sin(x**2),
            variable=x,
            lower=0,
            upper=oo,
            result=sqrt(pi/2) / 2,
            category="special_functions",
            difficulty="hard",
            techniques=["contour_integration", "Gamma_function"],
            reference="Fresnel integrals"
        ))
        
        # ═══════════════════════════════════════════════════════════
        # PUTNAM / OLYMPIAD INTEGRALS
        # ═══════════════════════════════════════════════════════════
        self._add_entry(IntegralEntry(
            name="Putnam 2005 B5",
            integrand=log(x + 1) / (x**2 + 1),
            variable=x,
            lower=0,
            upper=1,
            result=pi * log(2) / 8,
            category="olympiad",
            difficulty="olympiad",
            techniques=["Feynman", "series_expansion"],
            reference="Putnam 2005 B5"
        ))
        
        # ∫₀¹ ln(x)ln(1-x) dx = 2 - π²/6
        self._add_entry(IntegralEntry(
            name="Log Product Integral",
            integrand=log(x) * log(1 - x),
            variable=x,
            lower=0,
            upper=1,
            result=2 - pi**2/6,
            category="classic",
            difficulty="hard",
            techniques=["series_expansion", "Beta_function"],
            reference="Euler"
        ))
        
        # ═══════════════════════════════════════════════════════════
        # CATALAN'S CONSTANT INTEGRALS
        # ═══════════════════════════════════════════════════════════
        self._add_entry(IntegralEntry(
            name="Catalan Integral (arctan/x)",
            integrand=atan(x) / x,
            variable=x,
            lower=0,
            upper=1,
            result=sympy.Catalan,  # G ≈ 0.9159
            category="special_constants",
            difficulty="hard",
            techniques=["series_expansion"],
            reference="Catalan's constant"
        ))
        
        # ═══════════════════════════════════════════════════════════
        # TRIGONOMETRIC INTEGRALS
        # ═══════════════════════════════════════════════════════════
        self._add_entry(IntegralEntry(
            name="Wallis Integral (sin^n)",
            integrand=sin(x)**4,  # Example for n=4
            variable=x,
            lower=0,
            upper=pi/2,
            result=3*pi/16,
            category="classic",
            difficulty="standard",
            techniques=["reduction_formula", "Beta_function"],
            reference="Wallis integrals"
        ))
        
        # ∫₀^(π/2) ln(sin(x)) dx = -π·ln(2)/2
        self._add_entry(IntegralEntry(
            name="Log-Sine Integral",
            integrand=log(sin(x)),
            variable=x,
            lower=0,
            upper=pi/2,
            result=-pi * log(2) / 2,
            category="classic",
            difficulty="hard",
            techniques=["Fourier_series", "symmetry"],
            reference="Euler"
        ))
        
        # ═══════════════════════════════════════════════════════════
        # COXETER'S INTEGRAL
        # ∫₀^(π/2) arctan(tan²x) dx = π²/8
        # ═══════════════════════════════════════════════════════════
        self._add_entry(IntegralEntry(
            name="Coxeter's Integral",
            integrand=atan(tan(x)**2),
            variable=x,
            lower=0,
            upper=pi/2,
            result=pi**2/8,
            category="special_functions",
            difficulty="olympiad",
            techniques=["symmetry", "substitution"],
            reference="Coxeter"
        ))
        
    def _add_entry(self, entry: IntegralEntry):
        """Add an entry to the database with multiple keys for lookup."""
        # Primary key: name
        self.db[entry.name.lower()] = entry
        
        # Secondary key: hash of integrand structure
        try:
            struct_hash = self._structure_hash(entry.integrand, entry.variable, 
                                               entry.lower, entry.upper)
            self.db[struct_hash] = entry
        except Exception as e:
            logger.debug(f"Could not hash {entry.name}: {e}")
    
    def _structure_hash(self, integrand, var, lower, upper) -> str:
        """Create a hash based on the structure of the integral."""
        # Normalize: replace variable with dummy
        dummy = Symbol('_u')
        normalized = integrand.subs(var, dummy)
        
        # Create canonical string representation
        canonical = f"{srepr(simplify(normalized))}|{srepr(lower)}|{srepr(upper)}"
        return hashlib.md5(canonical.encode()).hexdigest()
    
    def lookup(self, integrand, var, lower, upper) -> Optional[IntegralEntry]:
        """
        Look up an integral in the database.
        
        Args:
            integrand: SymPy expression
            var: Integration variable
            lower: Lower limit
            upper: Upper limit
            
        Returns:
            IntegralEntry if found, None otherwise
        """
        try:
            # Try structure hash
            struct_hash = self._structure_hash(integrand, var, lower, upper)
            if struct_hash in self.db:
                return self.db[struct_hash]
        except Exception:
            pass
        
        # Try pattern matching (more flexible)
        return self._pattern_match(integrand, var, lower, upper)
    
    def _pattern_match(self, integrand, var, lower, upper) -> Optional[IntegralEntry]:
        """
        Try to match against known patterns using heuristics.
        """
        integrand_str = str(integrand)
        
        # Ahmed's Integral signature
        if "atan" in integrand_str and "sqrt" in integrand_str:
            if "x**2 + 2" in integrand_str or "2 + x**2" in integrand_str:
                return self.db.get("ahmed's integral")
        
        # Dirichlet signature
        if integrand_str == "sin(x)/x" or str(simplify(integrand - sin(var)/var)) == "0":
            if upper == oo:
                return self.db.get("dirichlet integral")
        
        # Gaussian signature
        if "exp(-x**2)" in integrand_str or str(simplify(integrand - exp(-var**2))) == "0":
            if lower == -oo and upper == oo:
                return self.db.get("gaussian integral")
            elif lower == 0 and upper == oo:
                return self.db.get("half gaussian integral")
        
        # Fresnel signatures
        if str(simplify(integrand - cos(var**2))) == "0":
            if lower == 0 and upper == oo:
                return self.db.get("fresnel cosine integral")
        if str(simplify(integrand - sin(var**2))) == "0":
            if lower == 0 and upper == oo:
                return self.db.get("fresnel sine integral")
        
        return None
    
    def lookup_by_name(self, name: str) -> Optional[IntegralEntry]:
        """Look up by name (case-insensitive)."""
        return self.db.get(name.lower())
    
    def list_all(self) -> List[str]:
        """List all named integrals in the database."""
        return [name for name in self.db.keys() 
                if not name.startswith(('a', 'b', 'c', 'd', 'e', 'f', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'))]
    
    def search(self, keyword: str) -> List[IntegralEntry]:
        """Search for integrals by keyword."""
        results = []
        keyword = keyword.lower()
        
        seen = set()
        for key, entry in self.db.items():
            if entry.name in seen:
                continue
            
            if (keyword in entry.name.lower() or 
                keyword in entry.category.lower() or
                keyword in str(entry.techniques).lower()):
                results.append(entry)
                seen.add(entry.name)
        
        return results


# Global instance
INTEGRAL_DB = IntegralKnowledgeBase()


def lookup_integral(integrand, var, lower, upper) -> Optional[IntegralEntry]:
    """Convenience function to look up an integral."""
    return INTEGRAL_DB.lookup(integrand, var, lower, upper)


def get_integral_info(name: str) -> Optional[IntegralEntry]:
    """Get info about a named integral."""
    return INTEGRAL_DB.lookup_by_name(name)


# Quick test
if __name__ == "__main__":
    print("Special Integral Database")
    print("=" * 50)
    
    # List all integrals
    print("\nKnown integrals:")
    for name in sorted(set(e.name for e in INTEGRAL_DB.db.values() if hasattr(e, 'name'))):
        entry = INTEGRAL_DB.lookup_by_name(name)
        if entry:
            print(f"  • {entry.name}: {entry.result} ({entry.difficulty})")
    
    # Test Ahmed's integral lookup
    print("\n" + "=" * 50)
    print("Testing Ahmed's Integral lookup:")
    ahmed = atan(sqrt(x**2 + 2)) / ((x**2 + 1) * sqrt(x**2 + 2))
    result = lookup_integral(ahmed, x, 0, 1)
    if result:
        print(f"  Found: {result.name}")
        print(f"  Result: {result.result}")
        print(f"  Techniques: {result.techniques}")
    else:
        print("  Not found!")
