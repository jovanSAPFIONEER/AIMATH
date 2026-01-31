"""
Expression Tree - Abstract syntax tree for mathematical expressions.

Provides structured representation of expressions for analysis
and manipulation.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Union
import logging

logger = logging.getLogger(__name__)


class NodeType(Enum):
    """Types of nodes in expression tree."""
    NUMBER = "number"
    SYMBOL = "symbol"
    OPERATOR = "operator"
    FUNCTION = "function"
    EQUATION = "equation"
    RELATION = "relation"


class OperatorType(Enum):
    """Mathematical operators."""
    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "/"
    POW = "**"
    NEG = "neg"  # Unary negation


class RelationType(Enum):
    """Relational operators."""
    EQ = "="      # Equal
    NE = "!="     # Not equal
    LT = "<"      # Less than
    LE = "<="     # Less than or equal
    GT = ">"      # Greater than
    GE = ">="     # Greater than or equal


@dataclass
class ExpressionNode:
    """
    Node in an expression tree.
    
    Attributes:
        node_type: Type of this node
        value: Value (number, symbol name, operator, or function name)
        children: Child nodes (for operators and functions)
        metadata: Additional information (e.g., domain constraints)
    """
    node_type: NodeType
    value: Any
    children: list['ExpressionNode'] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    
    def is_leaf(self) -> bool:
        """Check if node is a leaf (no children)."""
        return len(self.children) == 0
    
    def depth(self) -> int:
        """Calculate depth of subtree."""
        if self.is_leaf():
            return 1
        return 1 + max(child.depth() for child in self.children)
    
    def count_nodes(self) -> int:
        """Count total nodes in subtree."""
        return 1 + sum(child.count_nodes() for child in self.children)
    
    def collect_symbols(self) -> set[str]:
        """Collect all symbol names in subtree."""
        symbols = set()
        if self.node_type == NodeType.SYMBOL:
            symbols.add(str(self.value))
        for child in self.children:
            symbols.update(child.collect_symbols())
        return symbols
    
    def to_string(self) -> str:
        """Convert subtree to string representation."""
        if self.node_type == NodeType.NUMBER:
            return str(self.value)
        elif self.node_type == NodeType.SYMBOL:
            return str(self.value)
        elif self.node_type == NodeType.OPERATOR:
            if len(self.children) == 1:  # Unary
                return f"({self.value}{self.children[0].to_string()})"
            elif len(self.children) == 2:  # Binary
                left = self.children[0].to_string()
                right = self.children[1].to_string()
                return f"({left} {self.value} {right})"
        elif self.node_type == NodeType.FUNCTION:
            args = ", ".join(child.to_string() for child in self.children)
            return f"{self.value}({args})"
        elif self.node_type == NodeType.RELATION:
            left = self.children[0].to_string()
            right = self.children[1].to_string()
            return f"{left} {self.value} {right}"
        return str(self.value)


class ExpressionTree:
    """
    Expression tree builder and manipulator.
    
    Converts SymPy expressions to tree representation and provides
    analysis methods.
    
    Example:
        >>> tree = ExpressionTree.from_sympy(sympify("x**2 + 2*x + 1"))
        >>> print(tree.root.depth())  # 4
        >>> print(tree.root.collect_symbols())  # {'x'}
    """
    
    def __init__(self, root: Optional[ExpressionNode] = None):
        """Initialize expression tree."""
        self.root = root
    
    @classmethod
    def from_sympy(cls, expr: Any) -> 'ExpressionTree':
        """
        Build expression tree from SymPy expression.
        
        Args:
            expr: SymPy expression
            
        Returns:
            ExpressionTree instance
        """
        root = cls._sympy_to_node(expr)
        return cls(root)
    
    @classmethod
    def _sympy_to_node(cls, expr: Any) -> ExpressionNode:
        """Recursively convert SymPy expression to tree node."""
        from sympy import (
            Symbol, Integer, Float, Rational, 
            Add, Mul, Pow, 
            sin, cos, tan, log, exp, sqrt,
            Function, Eq, Ne, Lt, Le, Gt, Ge,
        )
        
        # Numbers
        if isinstance(expr, (int, float)):
            return ExpressionNode(NodeType.NUMBER, expr)
        elif isinstance(expr, Integer):
            return ExpressionNode(NodeType.NUMBER, int(expr))
        elif isinstance(expr, Float):
            return ExpressionNode(NodeType.NUMBER, float(expr))
        elif isinstance(expr, Rational):
            return ExpressionNode(NodeType.NUMBER, float(expr))
        
        # Symbols
        elif isinstance(expr, Symbol):
            return ExpressionNode(NodeType.SYMBOL, str(expr))
        
        # Addition
        elif isinstance(expr, Add):
            children = [cls._sympy_to_node(arg) for arg in expr.args]
            return ExpressionNode(NodeType.OPERATOR, '+', children)
        
        # Multiplication
        elif isinstance(expr, Mul):
            children = [cls._sympy_to_node(arg) for arg in expr.args]
            return ExpressionNode(NodeType.OPERATOR, '*', children)
        
        # Power
        elif isinstance(expr, Pow):
            base = cls._sympy_to_node(expr.args[0])
            exponent = cls._sympy_to_node(expr.args[1])
            return ExpressionNode(NodeType.OPERATOR, '**', [base, exponent])
        
        # Functions
        elif isinstance(expr, Function) or hasattr(expr, 'func'):
            func_name = expr.func.__name__ if hasattr(expr, 'func') else type(expr).__name__
            children = [cls._sympy_to_node(arg) for arg in expr.args]
            return ExpressionNode(NodeType.FUNCTION, func_name, children)
        
        # Equations and relations
        elif isinstance(expr, (Eq, Ne, Lt, Le, Gt, Ge)):
            left = cls._sympy_to_node(expr.args[0])
            right = cls._sympy_to_node(expr.args[1])
            rel_map = {Eq: '=', Ne: '!=', Lt: '<', Le: '<=', Gt: '>', Ge: '>='}
            rel_symbol = rel_map.get(type(expr), '=')
            return ExpressionNode(NodeType.RELATION, rel_symbol, [left, right])
        
        # Fallback
        else:
            # Try to handle as generic expression
            if hasattr(expr, 'args') and expr.args:
                children = [cls._sympy_to_node(arg) for arg in expr.args]
                return ExpressionNode(
                    NodeType.FUNCTION, 
                    type(expr).__name__, 
                    children
                )
            return ExpressionNode(NodeType.SYMBOL, str(expr))
    
    def to_sympy(self) -> Any:
        """Convert tree back to SymPy expression."""
        if self.root is None:
            return None
        return self._node_to_sympy(self.root)
    
    def _node_to_sympy(self, node: ExpressionNode) -> Any:
        """Recursively convert tree node to SymPy."""
        from sympy import (
            Symbol, Integer, Float, Add, Mul, Pow,
            sin, cos, tan, log, exp, sqrt, Eq,
        )
        
        if node.node_type == NodeType.NUMBER:
            if isinstance(node.value, int):
                return Integer(node.value)
            return Float(node.value)
        
        elif node.node_type == NodeType.SYMBOL:
            return Symbol(node.value)
        
        elif node.node_type == NodeType.OPERATOR:
            children = [self._node_to_sympy(c) for c in node.children]
            if node.value == '+':
                return Add(*children)
            elif node.value == '*':
                return Mul(*children)
            elif node.value == '**':
                return Pow(children[0], children[1])
            elif node.value == '-' and len(children) == 2:
                return children[0] - children[1]
            elif node.value == '/' and len(children) == 2:
                return children[0] / children[1]
        
        elif node.node_type == NodeType.FUNCTION:
            children = [self._node_to_sympy(c) for c in node.children]
            func_map = {
                'sin': sin, 'cos': cos, 'tan': tan,
                'log': log, 'exp': exp, 'sqrt': sqrt,
            }
            func = func_map.get(node.value.lower())
            if func:
                return func(*children)
        
        elif node.node_type == NodeType.RELATION:
            children = [self._node_to_sympy(c) for c in node.children]
            if node.value == '=' and len(children) == 2:
                return Eq(children[0], children[1])
        
        # Fallback
        return Symbol(str(node.value))
    
    def complexity_score(self) -> int:
        """
        Calculate complexity score of expression.
        
        Higher score = more complex expression.
        """
        if self.root is None:
            return 0
        
        return self._node_complexity(self.root)
    
    def _node_complexity(self, node: ExpressionNode) -> int:
        """Calculate complexity of a node and its subtree."""
        # Base complexity by node type
        type_weights = {
            NodeType.NUMBER: 1,
            NodeType.SYMBOL: 2,
            NodeType.OPERATOR: 3,
            NodeType.FUNCTION: 5,
            NodeType.RELATION: 2,
            NodeType.EQUATION: 2,
        }
        
        # Operator complexity weights
        op_weights = {
            '+': 1, '-': 1, '*': 2, '/': 3, '**': 4,
        }
        
        # Function complexity weights
        func_weights = {
            'sin': 3, 'cos': 3, 'tan': 4, 'log': 4, 'exp': 3,
            'sqrt': 2, 'Integral': 10, 'Derivative': 8, 'Limit': 8,
        }
        
        base = type_weights.get(node.node_type, 1)
        
        if node.node_type == NodeType.OPERATOR:
            base += op_weights.get(str(node.value), 2)
        elif node.node_type == NodeType.FUNCTION:
            base += func_weights.get(str(node.value), 3)
        
        # Add children complexity
        child_complexity = sum(
            self._node_complexity(c) for c in node.children
        )
        
        return base + child_complexity
