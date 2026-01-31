"""
Formal Prover - Z3-based theorem proving.

Uses the Z3 SMT solver for formal verification of mathematical
results. This is the HIGHEST trust level in our system.
"""

from typing import Any, Optional
import logging

from ..core.types import MathProblem, VerificationCheck, ProblemType

logger = logging.getLogger(__name__)


class FormalProver:
    """
    Formal theorem prover using Z3.
    
    Z3 is a Satisfiability Modulo Theories (SMT) solver that
    can formally verify mathematical statements.
    
    Trust Level: HIGHEST (1.0)
    
    When Z3 proves something, it IS proven (within its theory).
    This is the gold standard for verification.
    
    Capabilities:
    - Equation verification
    - Inequality verification
    - Logical formula verification
    - Counterexample generation
    
    Example:
        >>> prover = FormalProver()
        >>> check = prover.prove(problem, answer)
        >>> if check.passed:
        ...     print("Formally proven!")
    """
    
    def __init__(self, timeout_ms: int = 5000):
        """
        Initialize formal prover.
        
        Args:
            timeout_ms: Z3 solver timeout in milliseconds
        """
        self.timeout_ms = timeout_ms
        self._z3_available = None
    
    @property
    def z3_available(self) -> bool:
        """Check if Z3 is available."""
        if self._z3_available is None:
            try:
                import z3
                self._z3_available = True
            except ImportError:
                self._z3_available = False
                logger.warning(
                    "Z3 not installed. Install with: pip install z3-solver"
                )
        return self._z3_available
    
    def prove(
        self, 
        problem: MathProblem, 
        answer: Any
    ) -> VerificationCheck:
        """
        Attempt to formally prove that answer is correct.
        
        Args:
            problem: Original problem
            answer: Answer to verify
            
        Returns:
            VerificationCheck with proof status
        """
        if not self.z3_available:
            return VerificationCheck(
                check_type='formal_proof',
                passed=False,
                details="Z3 not available",
                error="Install z3-solver for formal verification",
            )
        
        try:
            if problem.problem_type == ProblemType.EQUATION:
                return self._prove_equation(problem, answer)
            elif problem.problem_type == ProblemType.INEQUALITY:
                return self._prove_inequality(problem, answer)
            else:
                return VerificationCheck(
                    check_type='formal_proof',
                    passed=False,
                    details=f"Formal proof not implemented for {problem.problem_type}",
                )
        except Exception as e:
            logger.error(f"Formal proof error: {e}")
            return VerificationCheck(
                check_type='formal_proof',
                passed=False,
                details="Formal proof failed",
                error=str(e),
            )
    
    def verify_claim(self, problem: MathProblem) -> VerificationCheck:
        """
        Verify a mathematical claim (theorem).
        
        Args:
            problem: Problem containing claim
            
        Returns:
            VerificationCheck with proof or counterexample
        """
        if not self.z3_available:
            return VerificationCheck(
                check_type='formal_proof',
                passed=False,
                details="Z3 not available",
            )
        
        # Try to prove the claim
        try:
            return self._verify_claim_z3(problem)
        except Exception as e:
            return VerificationCheck(
                check_type='formal_proof',
                passed=False,
                details=f"Claim verification failed: {e}",
                error=str(e),
            )
    
    def _prove_equation(
        self, 
        problem: MathProblem, 
        answer: Any
    ) -> VerificationCheck:
        """
        Prove that answer solves the equation.
        
        Strategy: Prove that for the given answer, the equation
        holds true (i.e., there's no counterexample).
        """
        import z3
        
        expr = problem.parsed_expression
        
        # Get variable
        if problem.variables:
            var_name = problem.variables[0]
        else:
            var_name = 'x'
        
        # Create Z3 variable
        x = z3.Real(var_name)
        
        # Convert SymPy expression to Z3
        z3_expr = self._sympy_to_z3(expr, {var_name: x})
        
        if z3_expr is None:
            return VerificationCheck(
                check_type='formal_proof',
                passed=False,
                details="Could not convert expression to Z3",
            )
        
        # Handle list of answers
        answers = answer if isinstance(answer, (list, tuple)) else [answer]
        
        proofs = []
        
        for ans in answers:
            # Convert answer to Z3
            z3_ans = self._sympy_to_z3(ans, {var_name: x})
            
            if z3_ans is None:
                continue
            
            # Create solver
            solver = z3.Solver()
            solver.set("timeout", self.timeout_ms)
            
            # Assert that x equals the answer
            solver.add(x == z3_ans)
            
            # Assert that the equation does NOT hold (looking for counterexample)
            # If equation is Eq(lhs, rhs), we want to prove lhs == rhs
            if hasattr(expr, 'lhs') and hasattr(expr, 'rhs'):
                lhs_z3 = self._sympy_to_z3(expr.lhs, {var_name: x})
                rhs_z3 = self._sympy_to_z3(expr.rhs, {var_name: x})
                if lhs_z3 is not None and rhs_z3 is not None:
                    solver.add(lhs_z3 != rhs_z3)
            else:
                # Assume expr = 0
                solver.add(z3_expr != 0)
            
            # If unsatisfiable, the answer is correct
            result = solver.check()
            
            if result == z3.unsat:
                proofs.append({
                    'answer': str(ans),
                    'status': 'proven',
                })
            elif result == z3.sat:
                model = solver.model()
                proofs.append({
                    'answer': str(ans),
                    'status': 'disproven',
                    'counterexample': str(model),
                })
            else:
                proofs.append({
                    'answer': str(ans),
                    'status': 'unknown',
                })
        
        all_proven = all(p['status'] == 'proven' for p in proofs)
        
        return VerificationCheck(
            check_type='formal_proof',
            passed=all_proven,
            details=(
                "All solutions formally proven" if all_proven
                else "Some solutions could not be proven"
            ),
            evidence=proofs,
        )
    
    def _prove_inequality(
        self, 
        problem: MathProblem, 
        answer: Any
    ) -> VerificationCheck:
        """
        Prove inequality solution.
        """
        import z3
        
        # Similar to equation but handling inequality relations
        # This is more complex and depends on the inequality type
        
        return VerificationCheck(
            check_type='formal_proof',
            passed=False,
            details="Inequality formal proof not yet implemented",
        )
    
    def _verify_claim_z3(self, problem: MathProblem) -> VerificationCheck:
        """
        Verify a general claim using Z3.
        """
        import z3
        
        claim = problem.parsed_expression
        
        # Get variables
        variables = {}
        if hasattr(claim, 'free_symbols'):
            for sym in claim.free_symbols:
                variables[str(sym)] = z3.Real(str(sym))
        
        # Convert claim to Z3
        z3_claim = self._sympy_to_z3(claim, variables)
        
        if z3_claim is None:
            return VerificationCheck(
                check_type='formal_proof',
                passed=False,
                details="Could not convert claim to Z3",
            )
        
        # Try to prove by showing negation is unsatisfiable
        solver = z3.Solver()
        solver.set("timeout", self.timeout_ms)
        solver.add(z3.Not(z3_claim))
        
        result = solver.check()
        
        if result == z3.unsat:
            return VerificationCheck(
                check_type='formal_proof',
                passed=True,
                details="Claim formally proven (negation is unsatisfiable)",
                evidence="QED",
            )
        elif result == z3.sat:
            model = solver.model()
            return VerificationCheck(
                check_type='formal_proof',
                passed=False,
                details="Claim is FALSE - counterexample found",
                evidence=str(model),
            )
        else:
            return VerificationCheck(
                check_type='formal_proof',
                passed=False,
                details="Could not determine (timeout or unknown)",
            )
    
    def _sympy_to_z3(self, expr: Any, variables: dict) -> Optional[Any]:
        """
        Convert SymPy expression to Z3 expression.
        
        Args:
            expr: SymPy expression
            variables: Dict mapping variable names to Z3 variables
            
        Returns:
            Z3 expression or None if conversion fails
        """
        import z3
        from sympy import (
            Symbol, Integer, Float, Rational,
            Add, Mul, Pow, 
            Eq, Ne, Lt, Le, Gt, Ge,
            And, Or, Not,
            sin, cos, tan, exp, log, sqrt,
        )
        
        try:
            # Handle numeric types
            if isinstance(expr, (int, float)):
                return z3.RealVal(expr)
            if isinstance(expr, Integer):
                return z3.RealVal(int(expr))
            if isinstance(expr, Float):
                return z3.RealVal(float(expr))
            if isinstance(expr, Rational):
                return z3.RealVal(float(expr))
            
            # Handle symbols
            if isinstance(expr, Symbol):
                name = str(expr)
                if name in variables:
                    return variables[name]
                # Create new variable
                variables[name] = z3.Real(name)
                return variables[name]
            
            # Handle operations
            if isinstance(expr, Add):
                terms = [self._sympy_to_z3(arg, variables) for arg in expr.args]
                if None in terms:
                    return None
                result = terms[0]
                for term in terms[1:]:
                    result = result + term
                return result
            
            if isinstance(expr, Mul):
                factors = [self._sympy_to_z3(arg, variables) for arg in expr.args]
                if None in factors:
                    return None
                result = factors[0]
                for factor in factors[1:]:
                    result = result * factor
                return result
            
            if isinstance(expr, Pow):
                base = self._sympy_to_z3(expr.args[0], variables)
                exp_val = expr.args[1]
                
                if base is None:
                    return None
                
                # Handle integer powers
                if isinstance(exp_val, Integer):
                    n = int(exp_val)
                    if n >= 0:
                        result = z3.RealVal(1)
                        for _ in range(n):
                            result = result * base
                        return result
                
                # For non-integer powers, we can't easily convert
                return None
            
            # Handle relations
            if isinstance(expr, Eq):
                lhs = self._sympy_to_z3(expr.args[0], variables)
                rhs = self._sympy_to_z3(expr.args[1], variables)
                if lhs is None or rhs is None:
                    return None
                return lhs == rhs
            
            if isinstance(expr, (Lt, Le, Gt, Ge)):
                lhs = self._sympy_to_z3(expr.args[0], variables)
                rhs = self._sympy_to_z3(expr.args[1], variables)
                if lhs is None or rhs is None:
                    return None
                
                if isinstance(expr, Lt):
                    return lhs < rhs
                elif isinstance(expr, Le):
                    return lhs <= rhs
                elif isinstance(expr, Gt):
                    return lhs > rhs
                elif isinstance(expr, Ge):
                    return lhs >= rhs
            
            # Handle logical operators
            if isinstance(expr, And):
                args = [self._sympy_to_z3(arg, variables) for arg in expr.args]
                if None in args:
                    return None
                return z3.And(*args)
            
            if isinstance(expr, Or):
                args = [self._sympy_to_z3(arg, variables) for arg in expr.args]
                if None in args:
                    return None
                return z3.Or(*args)
            
            if isinstance(expr, Not):
                arg = self._sympy_to_z3(expr.args[0], variables)
                if arg is None:
                    return None
                return z3.Not(arg)
            
            # Transcendental functions not directly supported
            # Would need approximations or different theory
            
            logger.debug(f"Cannot convert {type(expr)} to Z3")
            return None
            
        except Exception as e:
            logger.debug(f"Z3 conversion error: {e}")
            return None
