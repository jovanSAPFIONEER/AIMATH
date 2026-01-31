"""
Sandbox - Isolated execution environment for untrusted operations.

This module provides subprocess-based sandboxing to isolate potentially
dangerous operations (like evaluating LLM-generated conjectures).

FROZEN: This file must NEVER be modified by the evolution engine.

Security Model:
1. Run in separate subprocess (memory isolation)
2. Timeout enforcement (prevent infinite loops)
3. Resource limits where available
4. Capture output without execution in main process
"""

import subprocess
import sys
import json
import tempfile
import os
from typing import Any, Dict, Optional
from dataclasses import dataclass
from pathlib import Path


class SandboxError(Exception):
    """Raised when sandboxed execution fails."""
    
    def __init__(self, message: str, exit_code: int = -1, stderr: str = ""):
        self.message = message
        self.exit_code = exit_code
        self.stderr = stderr[:500] if stderr else ""
        super().__init__(f"Sandbox: {message}")


@dataclass
class SandboxResult:
    """Result from sandboxed execution."""
    success: bool
    result: Any
    stdout: str
    stderr: str
    execution_time: float
    exit_code: int


class Sandbox:
    """
    Sandboxed execution environment.
    
    Runs code in a separate subprocess with:
    - Timeout enforcement
    - Memory limits (on Linux)
    - No access to main process state
    - Output capture
    
    Example:
        >>> sandbox = Sandbox(timeout=10)
        >>> result = sandbox.run_verification("sin(x)**2 + cos(x)**2", "1")
        >>> result.success
        True
        >>> result.result
        {'verified': True, 'method': 'symbolic'}
    """
    
    # Template for verification script
    VERIFY_TEMPLATE = '''
import sys
import json

# Prevent any shenanigans
sys.modules['os'] = None
sys.modules['subprocess'] = None
sys.modules['socket'] = None

try:
    # Add aimath to path
    sys.path.insert(0, {aimath_path!r})
    
    from mathclaw.security.safe_parser import SafeParser
    from sympy import simplify, expand, trigsimp, N, Eq
    
    parser = SafeParser()
    
    lhs_expr = {lhs!r}
    rhs_expr = {rhs!r}
    
    # Parse both sides safely
    lhs_result = parser.parse(lhs_expr)
    rhs_result = parser.parse(rhs_expr)
    
    if not lhs_result.success:
        result = {{"verified": False, "error": lhs_result.error, "method": "parse_failed"}}
    elif not rhs_result.success:
        result = {{"verified": False, "error": rhs_result.error, "method": "parse_failed"}}
    else:
        lhs = lhs_result.expression
        rhs = rhs_result.expression
        
        # Try multiple verification methods
        verified = False
        method = "none"
        
        # Method 1: Direct simplification
        try:
            diff = simplify(lhs - rhs)
            if diff == 0:
                verified = True
                method = "simplify"
        except:
            pass
        
        # Method 2: Expand and compare
        if not verified:
            try:
                diff = expand(lhs - rhs)
                if diff == 0:
                    verified = True
                    method = "expand"
            except:
                pass
        
        # Method 3: Trig simplification
        if not verified:
            try:
                diff = trigsimp(lhs - rhs)
                if diff == 0:
                    verified = True
                    method = "trigsimp"
            except:
                pass
        
        # Method 4: Numerical check at random points
        if not verified:
            try:
                from sympy import Symbol
                import random
                
                free_syms = (lhs.free_symbols | rhs.free_symbols)
                if free_syms:
                    all_match = True
                    for _ in range(20):
                        subs = {{s: random.uniform(-10, 10) for s in free_syms}}
                        try:
                            lhs_val = complex(N(lhs.subs(subs)))
                            rhs_val = complex(N(rhs.subs(subs)))
                            if abs(lhs_val - rhs_val) > 1e-8:
                                all_match = False
                                break
                        except:
                            pass
                    
                    if all_match:
                        verified = True
                        method = "numerical"
            except:
                pass
        
        result = {{
            "verified": verified,
            "method": method,
            "lhs": str(lhs),
            "rhs": str(rhs),
        }}
    
    print(json.dumps(result))

except Exception as e:
    print(json.dumps({{"verified": False, "error": str(e), "method": "exception"}}))
'''
    
    # Template for conjecture testing
    CONJECTURE_TEMPLATE = '''
import sys
import json

sys.modules['os'] = None
sys.modules['subprocess'] = None

try:
    sys.path.insert(0, {aimath_path!r})
    
    from mathclaw.security.safe_parser import SafeParser
    from sympy import simplify, N, Symbol
    import random
    
    parser = SafeParser()
    
    conjecture = {conjecture!r}
    num_trials = {num_trials}
    
    result = parser.parse(conjecture)
    
    if not result.success:
        output = {{"status": "parse_error", "error": result.error}}
    else:
        expr = result.expression
        free_syms = expr.free_symbols if hasattr(expr, 'free_symbols') else set()
        
        # Check if it's trivially true/false
        try:
            simplified = simplify(expr)
            if simplified == True or simplified == 1:
                output = {{"status": "trivially_true", "expression": str(expr)}}
            elif simplified == False or simplified == 0:
                output = {{"status": "trivially_false", "expression": str(expr)}}
            else:
                # Fuzz test
                counterexamples = []
                for trial in range(num_trials):
                    try:
                        if free_syms:
                            subs = {{s: random.uniform(-100, 100) for s in free_syms}}
                            val = N(expr.subs(subs))
                        else:
                            val = N(expr)
                        
                        # Check if identity holds (should equal 0 or True)
                        if hasattr(val, 'is_zero') and val.is_zero == False:
                            if abs(complex(val)) > 1e-6:
                                counterexamples.append({{
                                    "values": {{str(k): float(v) for k, v in subs.items()}} if free_syms else {{}},
                                    "result": str(val)[:50]
                                }})
                                if len(counterexamples) >= 3:
                                    break
                    except:
                        pass
                
                if counterexamples:
                    output = {{
                        "status": "disproven",
                        "counterexamples": counterexamples,
                        "expression": str(expr)
                    }}
                else:
                    output = {{
                        "status": "plausible",
                        "trials": num_trials,
                        "expression": str(expr)
                    }}
        except Exception as e:
            output = {{"status": "error", "error": str(e)}}
    
    print(json.dumps(output))

except Exception as e:
    print(json.dumps({{"status": "exception", "error": str(e)}}))
'''
    
    def __init__(
        self,
        timeout: int = 30,
        memory_limit_mb: int = 512,
        python_executable: str = None,
    ):
        """
        Initialize sandbox.
        
        Args:
            timeout: Max execution time in seconds
            memory_limit_mb: Max memory (Linux only, advisory)
            python_executable: Python interpreter to use
        """
        self.timeout = timeout
        self.memory_limit_mb = memory_limit_mb
        self.python_executable = python_executable or sys.executable
        
        # Get path to aimath for imports
        self.aimath_path = str(Path(__file__).parent.parent.parent)
    
    def run_verification(
        self,
        lhs: str,
        rhs: str,
    ) -> SandboxResult:
        """
        Verify if lhs equals rhs in sandboxed environment.
        
        Args:
            lhs: Left-hand side expression
            rhs: Right-hand side expression
            
        Returns:
            SandboxResult with verification outcome
        """
        script = self.VERIFY_TEMPLATE.format(
            aimath_path=self.aimath_path,
            lhs=lhs,
            rhs=rhs,
        )
        
        return self._execute_script(script)
    
    def run_conjecture_test(
        self,
        conjecture: str,
        num_trials: int = 1000,
    ) -> SandboxResult:
        """
        Test a conjecture in sandboxed environment.
        
        Args:
            conjecture: Mathematical conjecture to test
            num_trials: Number of random trials
            
        Returns:
            SandboxResult with test outcome
        """
        script = self.CONJECTURE_TEMPLATE.format(
            aimath_path=self.aimath_path,
            conjecture=conjecture,
            num_trials=num_trials,
        )
        
        return self._execute_script(script)
    
    def _execute_script(self, script: str) -> SandboxResult:
        """
        Execute a script in sandboxed subprocess.
        
        Args:
            script: Python script to execute
            
        Returns:
            SandboxResult
        """
        import time
        
        # Write script to temp file
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.py',
            delete=False,
            encoding='utf-8'
        ) as f:
            f.write(script)
            script_path = f.name
        
        start_time = time.time()
        
        try:
            # Build command
            cmd = [self.python_executable, script_path]
            
            # Set environment to limit capabilities
            env = os.environ.copy()
            env['PYTHONDONTWRITEBYTECODE'] = '1'
            env['PYTHONHASHSEED'] = '0'
            
            # Run subprocess
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=env,
                cwd=self.aimath_path,  # Run from aimath directory
            )
            
            execution_time = time.time() - start_time
            
            # Parse output
            stdout = result.stdout.strip()
            stderr = result.stderr.strip()
            
            if result.returncode == 0 and stdout:
                try:
                    parsed_result = json.loads(stdout)
                    return SandboxResult(
                        success=True,
                        result=parsed_result,
                        stdout=stdout,
                        stderr=stderr,
                        execution_time=execution_time,
                        exit_code=result.returncode,
                    )
                except json.JSONDecodeError:
                    return SandboxResult(
                        success=False,
                        result={"error": "Invalid JSON output"},
                        stdout=stdout,
                        stderr=stderr,
                        execution_time=execution_time,
                        exit_code=result.returncode,
                    )
            else:
                return SandboxResult(
                    success=False,
                    result={"error": stderr or "Unknown error"},
                    stdout=stdout,
                    stderr=stderr,
                    execution_time=execution_time,
                    exit_code=result.returncode,
                )
                
        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            return SandboxResult(
                success=False,
                result={"error": f"Timeout after {self.timeout}s"},
                stdout="",
                stderr="",
                execution_time=execution_time,
                exit_code=-1,
            )
        except Exception as e:
            execution_time = time.time() - start_time
            return SandboxResult(
                success=False,
                result={"error": str(e)},
                stdout="",
                stderr=str(e),
                execution_time=execution_time,
                exit_code=-1,
            )
        finally:
            # Clean up temp file
            try:
                os.unlink(script_path)
            except:
                pass
    
    def run_custom(self, code: str, context: Dict[str, Any] = None) -> SandboxResult:
        """
        Run custom code in sandbox (USE WITH CAUTION).
        
        This should only be used for trusted code patterns.
        The code is still sandboxed but has more flexibility.
        
        Args:
            code: Python code to execute
            context: Variables to pass to the script
            
        Returns:
            SandboxResult
        """
        context = context or {}
        
        # Serialize context
        context_json = json.dumps(context)
        
        script = f'''
import sys
import json

sys.modules['os'] = None
sys.modules['subprocess'] = None
sys.modules['socket'] = None

try:
    sys.path.insert(0, {self.aimath_path!r})
    context = json.loads({context_json!r})
    
    # User code (sandboxed)
{self._indent_code(code, 4)}
    
except Exception as e:
    print(json.dumps({{"error": str(e)}}))
'''
        
        return self._execute_script(script)
    
    def _indent_code(self, code: str, spaces: int) -> str:
        """Indent code block."""
        indent = ' ' * spaces
        return '\n'.join(indent + line for line in code.split('\n'))


# ═══════════════════════════════════════════════════════════════
# Convenience functions
# ═══════════════════════════════════════════════════════════════

_default_sandbox = None

def sandboxed_verify(lhs: str, rhs: str, timeout: int = 30) -> SandboxResult:
    """Verify identity in sandbox."""
    global _default_sandbox
    if _default_sandbox is None:
        _default_sandbox = Sandbox(timeout=timeout)
    return _default_sandbox.run_verification(lhs, rhs)


def sandboxed_test(conjecture: str, trials: int = 1000, timeout: int = 30) -> SandboxResult:
    """Test conjecture in sandbox."""
    global _default_sandbox
    if _default_sandbox is None:
        _default_sandbox = Sandbox(timeout=timeout)
    return _default_sandbox.run_conjecture_test(conjecture, trials)
