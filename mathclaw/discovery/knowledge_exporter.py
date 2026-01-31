"""
Knowledge Exporter - Export discovered theorems to various formats.

Supports:
- Markdown documentation
- LaTeX papers
- JSON for APIs
- Jupyter notebooks
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

from .theorem_store import TheoremStore, Theorem


class KnowledgeExporter:
    """
    Export MathClaw discoveries to various formats.
    
    Generates:
    - Markdown files for documentation
    - LaTeX for academic papers
    - JSON for API consumption
    - Jupyter notebooks for exploration
    
    Example:
        >>> store = TheoremStore()
        >>> exporter = KnowledgeExporter(store)
        >>> 
        >>> # Export to markdown
        >>> exporter.to_markdown("discoveries.md")
        >>> 
        >>> # Export to LaTeX
        >>> exporter.to_latex("paper.tex")
    """
    
    def __init__(self, theorem_store: TheoremStore = None):
        """
        Initialize exporter.
        
        Args:
            theorem_store: TheoremStore to export from
        """
        self.store = theorem_store or TheoremStore()
    
    def to_markdown(
        self,
        output_path: Path = None,
        domain: str = None,
        include_proofs: bool = True,
    ) -> str:
        """
        Export to Markdown format.
        
        Args:
            output_path: Path to write file (optional)
            domain: Filter by domain (optional)
            include_proofs: Include proof steps
            
        Returns:
            Markdown content as string
        """
        if domain:
            theorems = self.store.get_by_domain(domain)
            title = f"MathClaw: {domain.replace('_', ' ').title()} Theorems"
        else:
            theorems = self.store.get_all()
            title = "MathClaw Discovered Theorems"
        
        stats = self.store.get_stats()
        
        lines = [
            f"# {title}",
            "",
            f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
            "",
            "## Summary",
            "",
            f"- **Total Theorems**: {stats['total_theorems']}",
            f"- **Domains Covered**: {len(stats['by_domain'])}",
            f"- **Avg Verification Time**: {stats['avg_verification_time_ms']:.2f}ms",
            "",
        ]
        
        # Group by domain
        by_domain: Dict[str, List[Theorem]] = {}
        for t in theorems:
            d = t.domain or 'general'
            if d not in by_domain:
                by_domain[d] = []
            by_domain[d].append(t)
        
        for domain_name, domain_theorems in sorted(by_domain.items()):
            lines.append(f"## {domain_name.replace('_', ' ').title()}")
            lines.append("")
            
            for t in domain_theorems:
                lines.append(f"### {t.statement}")
                lines.append("")
                
                if t.natural_language and t.natural_language != t.statement:
                    lines.append(f"*{t.natural_language}*")
                    lines.append("")
                
                lines.append(f"- **Proof Method**: {t.proof_method or 'N/A'}")
                lines.append(f"- **Discovered**: {t.discovered_at[:10]}")
                lines.append(f"- **Strategy**: {t.strategy_id or 'N/A'}")
                
                if t.variables:
                    lines.append(f"- **Variables**: {', '.join(t.variables)}")
                
                if t.assumptions:
                    lines.append(f"- **Assumptions**: {', '.join(t.assumptions)}")
                
                if include_proofs and t.proof_steps:
                    lines.append("")
                    lines.append("**Proof:**")
                    lines.append("```")
                    for step in t.proof_steps:
                        lines.append(f"  {step}")
                    lines.append("```")
                
                lines.append("")
        
        content = '\n'.join(lines)
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(content, encoding='utf-8')
        
        return content
    
    def to_latex(
        self,
        output_path: Path = None,
        domain: str = None,
        document_class: str = "article",
    ) -> str:
        """
        Export to LaTeX format.
        
        Args:
            output_path: Path to write file
            domain: Filter by domain
            document_class: LaTeX document class
            
        Returns:
            LaTeX content as string
        """
        if domain:
            theorems = self.store.get_by_domain(domain)
            title = f"MathClaw: {domain.replace('_', ' ').title()} Discoveries"
        else:
            theorems = self.store.get_all()
            title = "MathClaw Mathematical Discoveries"
        
        lines = [
            f"\\documentclass{{{document_class}}}",
            "\\usepackage{amsmath}",
            "\\usepackage{amssymb}",
            "\\usepackage{amsthm}",
            "",
            "\\newtheorem{theorem}{Theorem}[section]",
            "",
            f"\\title{{{title}}}",
            "\\author{MathClaw Autonomous Discovery System}",
            f"\\date{{{datetime.now().strftime('%B %d, %Y')}}}",
            "",
            "\\begin{document}",
            "\\maketitle",
            "",
            "\\begin{abstract}",
            f"This document contains {len(theorems)} mathematical theorems",
            "discovered and verified by the MathClaw system.",
            "\\end{abstract}",
            "",
        ]
        
        # Group by domain
        by_domain: Dict[str, List[Theorem]] = {}
        for t in theorems:
            d = t.domain or 'general'
            if d not in by_domain:
                by_domain[d] = []
            by_domain[d].append(t)
        
        for domain_name, domain_theorems in sorted(by_domain.items()):
            section_name = domain_name.replace('_', ' ').title()
            lines.append(f"\\section{{{section_name}}}")
            lines.append("")
            
            for t in domain_theorems:
                # Convert to LaTeX-safe format
                statement = self._sympy_to_latex(t.statement)
                
                lines.append("\\begin{theorem}")
                if t.natural_language and t.natural_language != t.statement:
                    lines.append(f"({t.natural_language})")
                lines.append(f"$${statement}$$")
                lines.append("\\end{theorem}")
                lines.append("")
                
                if t.proof_steps:
                    lines.append("\\begin{proof}")
                    for step in t.proof_steps:
                        step_latex = self._sympy_to_latex(step)
                        lines.append(f"{step_latex}\\\\")
                    lines.append("\\end{proof}")
                    lines.append("")
        
        lines.append("\\end{document}")
        
        content = '\n'.join(lines)
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(content, encoding='utf-8')
        
        return content
    
    def to_json(
        self,
        output_path: Path = None,
        domain: str = None,
        include_stats: bool = True,
    ) -> str:
        """
        Export to JSON format.
        
        Args:
            output_path: Path to write file
            domain: Filter by domain
            include_stats: Include statistics
            
        Returns:
            JSON content as string
        """
        if domain:
            theorems = self.store.get_by_domain(domain)
        else:
            theorems = self.store.get_all()
        
        data = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'generator': 'MathClaw',
                'version': '1.0.0',
                'total_theorems': len(theorems),
            },
            'theorems': [t.to_dict() for t in theorems],
        }
        
        if include_stats:
            data['statistics'] = self.store.get_stats()
        
        content = json.dumps(data, indent=2, default=str)
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(content, encoding='utf-8')
        
        return content
    
    def to_jupyter(
        self,
        output_path: Path = None,
        domain: str = None,
    ) -> str:
        """
        Export to Jupyter notebook format.
        
        Args:
            output_path: Path to write file
            domain: Filter by domain
            
        Returns:
            Notebook JSON content
        """
        if domain:
            theorems = self.store.get_by_domain(domain)
            title = f"MathClaw: {domain.replace('_', ' ').title()}"
        else:
            theorems = self.store.get_all()
            title = "MathClaw Discoveries"
        
        cells = [
            # Title cell
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    f"# {title}\n",
                    f"\n",
                    f"*Generated: {datetime.now().isoformat()}*\n",
                    f"\n",
                    f"This notebook contains {len(theorems)} verified theorems.\n",
                ]
            },
            # Setup cell
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "import sympy as sp\n",
                    "from sympy import *\n",
                    "\n",
                    "# Initialize pretty printing\n",
                    "sp.init_printing()\n",
                ]
            },
        ]
        
        # Add theorems
        for i, t in enumerate(theorems[:50], 1):  # Limit to 50
            # Markdown cell with theorem
            cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    f"## Theorem {i}: {t.natural_language or t.statement}\n",
                    f"\n",
                    f"**Domain**: {t.domain}  \n",
                    f"**Proof Method**: {t.proof_method}  \n",
                ]
            })
            
            # Code cell to verify
            cells.append({
                "cell_type": "code",
                "metadata": {},
                "source": [
                    f"# Theorem: {t.statement}\n",
                    self._generate_verification_code(t),
                ]
            })
        
        notebook = {
            "nbformat": 4,
            "nbformat_minor": 4,
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                }
            },
            "cells": cells
        }
        
        content = json.dumps(notebook, indent=2)
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(content, encoding='utf-8')
        
        return content
    
    def _sympy_to_latex(self, expr: str) -> str:
        """Convert SymPy expression to LaTeX."""
        # Basic conversions
        conversions = [
            ('**', '^'),
            ('*', ' \\cdot '),
            ('sqrt(', '\\sqrt{'),
            ('sin(', '\\sin('),
            ('cos(', '\\cos('),
            ('tan(', '\\tan('),
            ('log(', '\\log('),
            ('exp(', '\\exp('),
            ('pi', '\\pi'),
            ('oo', '\\infty'),
        ]
        
        result = expr
        for old, new in conversions:
            result = result.replace(old, new)
        
        # Escape underscores
        result = result.replace('_', '\\_')
        
        return result
    
    def _generate_verification_code(self, theorem: Theorem) -> str:
        """Generate Python code to verify a theorem."""
        # Extract variable names
        vars_str = ', '.join(theorem.variables) if theorem.variables else 'x'
        
        if '=' in theorem.statement:
            lhs, rhs = theorem.statement.split('=', 1)
            code = f"""
{vars_str} = symbols('{vars_str}')
lhs = {lhs.strip()}
rhs = {rhs.strip()}
print(f"LHS: {{lhs}}")
print(f"RHS: {{rhs}}")
print(f"Difference simplified: {{simplify(lhs - rhs)}}")
"""
        else:
            code = f"""
{vars_str} = symbols('{vars_str}')
expr = {theorem.statement}
print(f"Expression: {{expr}}")
print(f"Simplified: {{simplify(expr)}}")
"""
        
        return code.strip()
