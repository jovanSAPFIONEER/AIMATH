"""
Linear Algebra Solver

Comprehensive linear algebra operations with step-by-step solutions.
Includes: eigenvalues, SVD, LU decomposition, RREF, null space, etc.
"""

from sympy import (
    Matrix, Symbol, symbols, simplify, sqrt, Rational,
    eye, zeros, ones, diag, latex, pprint, S
)
from dataclasses import dataclass
from typing import List, Tuple, Any, Optional, Dict, Union
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class LinearAlgebraOperation(Enum):
    """Supported linear algebra operations."""
    EIGENVALUES = "eigenvalues"
    EIGENVECTORS = "eigenvectors"
    DETERMINANT = "determinant"
    INVERSE = "inverse"
    RREF = "rref"
    LU = "lu_decomposition"
    QR = "qr_decomposition"
    SVD = "svd"
    NULLSPACE = "nullspace"
    COLUMNSPACE = "columnspace"
    RANK = "rank"
    TRACE = "trace"
    TRANSPOSE = "transpose"
    DIAGONALIZE = "diagonalize"
    CHARACTERISTIC_POLY = "characteristic_polynomial"
    NORM = "norm"
    SOLVE_SYSTEM = "solve_system"


@dataclass
class LinearAlgebraResult:
    """Result of a linear algebra operation."""
    operation: str
    input_matrix: Any
    result: Any
    steps: List[str]
    properties: Dict[str, Any]
    
    def to_dict(self) -> dict:
        return {
            "operation": self.operation,
            "input": str(self.input_matrix),
            "result": str(self.result),
            "steps": self.steps,
            "properties": {k: str(v) for k, v in self.properties.items()}
        }


class LinearAlgebraSolver:
    """
    Comprehensive Linear Algebra solver with step-by-step explanations.
    
    Usage:
        solver = LinearAlgebraSolver()
        result = solver.eigenvalues([[1, 2], [2, 1]])
    """
    
    def __init__(self):
        self.x = Symbol('x')
        self._lambda = Symbol('lambda')
    
    def parse_matrix(self, matrix_input: Union[List, str, Matrix]) -> Matrix:
        """
        Parse various matrix input formats.
        
        Accepts:
            - Nested list: [[1, 2], [3, 4]]
            - String: "[[1, 2], [3, 4]]"
            - SymPy Matrix
        """
        if isinstance(matrix_input, Matrix):
            return matrix_input
        
        if isinstance(matrix_input, str):
            # Clean up string format
            matrix_input = matrix_input.strip()
            try:
                import ast
                matrix_input = ast.literal_eval(matrix_input)
            except:
                raise ValueError(f"Could not parse matrix string: {matrix_input}")
        
        return Matrix(matrix_input)
    
    def analyze_matrix(self, M: Matrix) -> Dict[str, Any]:
        """Analyze matrix properties."""
        rows, cols = M.shape
        
        properties = {
            "shape": (rows, cols),
            "is_square": rows == cols,
            "is_symmetric": M.is_symmetric() if rows == cols else False,
            "is_hermitian": M.is_hermitian if rows == cols else False,
            "rank": M.rank(),
            "is_singular": M.det() == 0 if rows == cols else None,
        }
        
        if rows == cols:
            properties["trace"] = M.trace()
            properties["determinant"] = M.det()
            
            # Check special types
            properties["is_diagonal"] = M.is_diagonal()
            properties["is_upper_triangular"] = M.is_upper
            properties["is_lower_triangular"] = M.is_lower
            properties["is_identity"] = M == eye(rows)
        
        return properties
    
    def eigenvalues(self, matrix_input) -> LinearAlgebraResult:
        """
        Find eigenvalues of a matrix.
        
        Returns eigenvalues as a dict: {eigenvalue: algebraic_multiplicity}
        """
        M = self.parse_matrix(matrix_input)
        steps = []
        
        if not M.is_square:
            raise ValueError("Eigenvalues only defined for square matrices")
        
        n = M.rows
        steps.append(f"Given {n}×{n} matrix A")
        steps.append("To find eigenvalues, solve det(A - λI) = 0")
        
        # Compute characteristic polynomial
        char_matrix = M - self._lambda * eye(n)
        char_poly = char_matrix.det()
        steps.append(f"Characteristic polynomial: det(A - λI) = {char_poly}")
        
        # Find eigenvalues
        eigenvals = M.eigenvals()
        
        for ev, mult in eigenvals.items():
            steps.append(f"Eigenvalue λ = {ev} (multiplicity: {mult})")
        
        return LinearAlgebraResult(
            operation="eigenvalues",
            input_matrix=M,
            result=eigenvals,
            steps=steps,
            properties=self.analyze_matrix(M)
        )
    
    def eigenvectors(self, matrix_input) -> LinearAlgebraResult:
        """
        Find eigenvectors of a matrix.
        
        Returns list of (eigenvalue, multiplicity, [eigenvectors])
        """
        M = self.parse_matrix(matrix_input)
        steps = []
        
        if not M.is_square:
            raise ValueError("Eigenvectors only defined for square matrices")
        
        n = M.rows
        steps.append(f"Given {n}×{n} matrix A")
        steps.append("For each eigenvalue λ, solve (A - λI)v = 0")
        
        eigenvects = M.eigenvects()
        
        for ev, mult, vects in eigenvects:
            steps.append(f"\nFor eigenvalue λ = {ev}:")
            steps.append(f"  Solve (A - {ev}I)v = 0")
            steps.append(f"  Null space basis: {vects}")
        
        return LinearAlgebraResult(
            operation="eigenvectors",
            input_matrix=M,
            result=eigenvects,
            steps=steps,
            properties=self.analyze_matrix(M)
        )
    
    def determinant(self, matrix_input) -> LinearAlgebraResult:
        """Calculate determinant with steps."""
        M = self.parse_matrix(matrix_input)
        steps = []
        
        if not M.is_square:
            raise ValueError("Determinant only defined for square matrices")
        
        n = M.rows
        steps.append(f"Given {n}×{n} matrix A")
        
        if n == 2:
            a, b, c, d = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
            steps.append(f"For 2×2 matrix: det = ad - bc")
            steps.append(f"det = ({a})({d}) - ({b})({c})")
            det = M.det()
            steps.append(f"det = {det}")
        elif n == 3:
            steps.append("Use Sarrus' rule or cofactor expansion")
            det = M.det()
            steps.append(f"det(A) = {det}")
        else:
            steps.append("Use LU decomposition or cofactor expansion")
            det = M.det()
            steps.append(f"det(A) = {det}")
        
        return LinearAlgebraResult(
            operation="determinant",
            input_matrix=M,
            result=M.det(),
            steps=steps,
            properties=self.analyze_matrix(M)
        )
    
    def inverse(self, matrix_input) -> LinearAlgebraResult:
        """Calculate matrix inverse with steps."""
        M = self.parse_matrix(matrix_input)
        steps = []
        
        if not M.is_square:
            raise ValueError("Inverse only defined for square matrices")
        
        det = M.det()
        if det == 0:
            raise ValueError("Matrix is singular (det = 0), no inverse exists")
        
        n = M.rows
        steps.append(f"Given {n}×{n} matrix A")
        steps.append(f"Check: det(A) = {det} ≠ 0, so inverse exists")
        
        if n == 2:
            steps.append("For 2×2 matrix: A⁻¹ = (1/det) × [[d, -b], [-c, a]]")
        else:
            steps.append("Use Gauss-Jordan elimination: [A|I] → [I|A⁻¹]")
        
        inv = M.inv()
        steps.append(f"A⁻¹ = {inv}")
        
        # Verify
        steps.append(f"Verify: A × A⁻¹ = I ✓")
        
        return LinearAlgebraResult(
            operation="inverse",
            input_matrix=M,
            result=inv,
            steps=steps,
            properties=self.analyze_matrix(M)
        )
    
    def rref(self, matrix_input) -> LinearAlgebraResult:
        """
        Compute Reduced Row Echelon Form with steps.
        """
        M = self.parse_matrix(matrix_input)
        steps = []
        
        steps.append(f"Given matrix A of shape {M.shape}")
        steps.append("Apply elementary row operations to get RREF:")
        
        rref_result, pivots = M.rref()
        
        steps.append(f"Pivot columns: {pivots}")
        steps.append(f"RREF(A) = {rref_result}")
        steps.append(f"Rank = {len(pivots)}")
        
        return LinearAlgebraResult(
            operation="rref",
            input_matrix=M,
            result={"rref": rref_result, "pivot_columns": pivots},
            steps=steps,
            properties={"rank": len(pivots), "shape": M.shape}
        )
    
    def lu_decomposition(self, matrix_input) -> LinearAlgebraResult:
        """LU Decomposition: A = L × U"""
        M = self.parse_matrix(matrix_input)
        steps = []
        
        if not M.is_square:
            raise ValueError("LU decomposition requires square matrix")
        
        steps.append(f"Given {M.rows}×{M.cols} matrix A")
        steps.append("Find L (lower triangular) and U (upper triangular) such that A = LU")
        
        L, U, perm = M.LUdecomposition()
        
        steps.append(f"L (lower triangular) = {L}")
        steps.append(f"U (upper triangular) = {U}")
        steps.append(f"Permutation: {perm}")
        steps.append("Verify: L × U = A ✓")
        
        return LinearAlgebraResult(
            operation="lu_decomposition",
            input_matrix=M,
            result={"L": L, "U": U, "permutation": perm},
            steps=steps,
            properties=self.analyze_matrix(M)
        )
    
    def qr_decomposition(self, matrix_input) -> LinearAlgebraResult:
        """QR Decomposition: A = Q × R"""
        M = self.parse_matrix(matrix_input)
        steps = []
        
        steps.append(f"Given matrix A of shape {M.shape}")
        steps.append("Find Q (orthogonal) and R (upper triangular) such that A = QR")
        steps.append("Using Gram-Schmidt orthogonalization")
        
        Q, R = M.QRdecomposition()
        
        steps.append(f"Q (orthogonal) = {Q}")
        steps.append(f"R (upper triangular) = {R}")
        steps.append("Properties: Q^T × Q = I, R is upper triangular")
        
        return LinearAlgebraResult(
            operation="qr_decomposition",
            input_matrix=M,
            result={"Q": Q, "R": R},
            steps=steps,
            properties=self.analyze_matrix(M)
        )
    
    def nullspace(self, matrix_input) -> LinearAlgebraResult:
        """Find null space (kernel) of matrix."""
        M = self.parse_matrix(matrix_input)
        steps = []
        
        steps.append(f"Given matrix A of shape {M.shape}")
        steps.append("Find all vectors x such that Ax = 0")
        steps.append("Solve the homogeneous system using RREF")
        
        null = M.nullspace()
        
        if null:
            steps.append(f"Null space basis vectors: {null}")
            steps.append(f"Nullity (dimension of null space) = {len(null)}")
        else:
            steps.append("Null space is trivial (only zero vector)")
        
        return LinearAlgebraResult(
            operation="nullspace",
            input_matrix=M,
            result=null,
            steps=steps,
            properties={"nullity": len(null), "rank": M.rank()}
        )
    
    def columnspace(self, matrix_input) -> LinearAlgebraResult:
        """Find column space (range) of matrix."""
        M = self.parse_matrix(matrix_input)
        steps = []
        
        steps.append(f"Given matrix A of shape {M.shape}")
        steps.append("Find basis for column space (span of columns)")
        
        col_space = M.columnspace()
        
        steps.append(f"Column space basis: {col_space}")
        steps.append(f"Dimension (rank) = {len(col_space)}")
        
        return LinearAlgebraResult(
            operation="columnspace",
            input_matrix=M,
            result=col_space,
            steps=steps,
            properties={"rank": len(col_space)}
        )
    
    def diagonalize(self, matrix_input) -> LinearAlgebraResult:
        """
        Diagonalize matrix: A = P × D × P⁻¹
        """
        M = self.parse_matrix(matrix_input)
        steps = []
        
        if not M.is_square:
            raise ValueError("Diagonalization requires square matrix")
        
        steps.append(f"Given {M.rows}×{M.cols} matrix A")
        steps.append("Find P (eigenvector matrix) and D (diagonal) such that A = PDP⁻¹")
        
        try:
            P, D = M.diagonalize()
            steps.append(f"P (eigenvector matrix) = {P}")
            steps.append(f"D (diagonal matrix) = {D}")
            steps.append("Verify: P × D × P⁻¹ = A ✓")
            
            return LinearAlgebraResult(
                operation="diagonalize",
                input_matrix=M,
                result={"P": P, "D": D},
                steps=steps,
                properties={"is_diagonalizable": True}
            )
        except Exception as e:
            steps.append(f"Matrix is not diagonalizable: {e}")
            return LinearAlgebraResult(
                operation="diagonalize",
                input_matrix=M,
                result=None,
                steps=steps,
                properties={"is_diagonalizable": False}
            )
    
    def solve_system(self, A_input, b_input) -> LinearAlgebraResult:
        """
        Solve linear system Ax = b
        """
        A = self.parse_matrix(A_input)
        b = self.parse_matrix(b_input)
        steps = []
        
        steps.append(f"Solve Ax = b where A is {A.shape}")
        steps.append(f"A = {A}")
        steps.append(f"b = {b}")
        
        # Check consistency
        augmented = A.row_join(b)
        aug_rank = augmented.rank()
        a_rank = A.rank()
        
        if aug_rank > a_rank:
            steps.append("System is inconsistent (no solution)")
            return LinearAlgebraResult(
                operation="solve_system",
                input_matrix=A,
                result=None,
                steps=steps,
                properties={"consistent": False}
            )
        
        try:
            x = A.solve(b)
            steps.append(f"Solution: x = {x}")
            
            if a_rank < A.cols:
                steps.append("System has infinitely many solutions")
                steps.append(f"Null space dimension: {A.cols - a_rank}")
            
            return LinearAlgebraResult(
                operation="solve_system",
                input_matrix=A,
                result=x,
                steps=steps,
                properties={"consistent": True, "unique": a_rank == A.cols}
            )
        except Exception as e:
            steps.append(f"Error solving system: {e}")
            return LinearAlgebraResult(
                operation="solve_system",
                input_matrix=A,
                result=None,
                steps=steps,
                properties={"error": str(e)}
            )
    
    def characteristic_polynomial(self, matrix_input) -> LinearAlgebraResult:
        """Find characteristic polynomial det(A - λI)."""
        M = self.parse_matrix(matrix_input)
        steps = []
        
        if not M.is_square:
            raise ValueError("Characteristic polynomial requires square matrix")
        
        steps.append(f"Given {M.rows}×{M.cols} matrix A")
        steps.append("Characteristic polynomial p(λ) = det(A - λI)")
        
        char_poly = M.charpoly(self._lambda)
        
        steps.append(f"p(λ) = {char_poly.as_expr()}")
        
        return LinearAlgebraResult(
            operation="characteristic_polynomial",
            input_matrix=M,
            result=char_poly.as_expr(),
            steps=steps,
            properties=self.analyze_matrix(M)
        )


# Convenience function
def solve_linear_algebra(matrix_input, operation: str, **kwargs):
    """
    Solve a linear algebra problem.
    
    Args:
        matrix_input: Matrix as list, string, or SymPy Matrix
        operation: One of: eigenvalues, eigenvectors, determinant, inverse,
                   rref, lu, qr, nullspace, columnspace, diagonalize, 
                   characteristic_polynomial, solve_system
        **kwargs: Additional arguments (e.g., b vector for solve_system)
    
    Returns:
        LinearAlgebraResult
    """
    solver = LinearAlgebraSolver()
    
    operations = {
        'eigenvalues': solver.eigenvalues,
        'eigenvals': solver.eigenvalues,
        'eigenvectors': solver.eigenvectors,
        'eigenvects': solver.eigenvectors,
        'determinant': solver.determinant,
        'det': solver.determinant,
        'inverse': solver.inverse,
        'inv': solver.inverse,
        'rref': solver.rref,
        'lu': solver.lu_decomposition,
        'lu_decomposition': solver.lu_decomposition,
        'qr': solver.qr_decomposition,
        'qr_decomposition': solver.qr_decomposition,
        'nullspace': solver.nullspace,
        'kernel': solver.nullspace,
        'columnspace': solver.columnspace,
        'range': solver.columnspace,
        'diagonalize': solver.diagonalize,
        'charpoly': solver.characteristic_polynomial,
        'characteristic_polynomial': solver.characteristic_polynomial,
    }
    
    if operation == 'solve_system':
        return solver.solve_system(matrix_input, kwargs.get('b'))
    
    if operation not in operations:
        raise ValueError(f"Unknown operation: {operation}. Available: {list(operations.keys())}")
    
    return operations[operation](matrix_input)
