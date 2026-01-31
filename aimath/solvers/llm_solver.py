"""
LLM Solver - AI-assisted mathematical reasoning.

Trust Level: LOW (0.3) - ALWAYS requires verification

This solver uses LLM capabilities for reasoning about problems
that are difficult to solve symbolically. However, LLM output
is NEVER trusted without verification.
"""

import time
from typing import Any, Optional
import logging

from .solver_base import BaseSolver, SolverResult, SolverStatus
from ..core.types import MathProblem, ProblemType, SolutionStep

logger = logging.getLogger(__name__)


class LLMSolver(BaseSolver):
    """
    LLM-based mathematical reasoning solver.
    
    Trust Level: LOW (0.3)
    
    ⚠️ CRITICAL: Results from this solver MUST be verified
    before being presented to users. This solver is useful for:
    - Suggesting solution approaches
    - Handling natural language problems
    - Providing intuition for complex problems
    
    But its outputs require verification by SymPy or formal methods.
    
    Example:
        >>> solver = LLMSolver(api_key="...")
        >>> result = solver.solve(problem)
        >>> # MUST verify before using
        >>> verified = verifier.verify(result.answer)
    """
    
    name = "llm"
    trust_level = 0.3  # Low trust - requires verification
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4",
        temperature: float = 0.1,
        timeout: Optional[float] = 30.0,
    ):
        """
        Initialize LLM solver.
        
        Args:
            api_key: API key (or uses environment variable)
            model: Model to use
            temperature: Sampling temperature (lower = more deterministic)
            timeout: Request timeout
        """
        super().__init__(timeout)
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self._client = None
    
    def can_solve(self, problem: MathProblem) -> bool:
        """LLM can attempt any problem type (but needs verification)."""
        return True
    
    def solve(self, problem: MathProblem) -> SolverResult:
        """
        Solve using LLM reasoning.
        
        Returns result with LOW confidence - verification required.
        """
        start_time = time.time()
        
        try:
            # Generate prompt
            prompt = self._create_prompt(problem)
            
            # Call LLM
            response = self._call_llm(prompt)
            
            if response is None:
                return self._create_error_result("LLM call failed")
            
            # Parse response
            answer, steps = self._parse_response(response, problem)
            
            elapsed = (time.time() - start_time) * 1000
            
            return SolverResult(
                status=SolverStatus.SUCCESS,
                answer=answer,
                steps=steps,
                method_name=self.name,
                confidence=self.trust_level,
                computation_time_ms=elapsed,
                metadata={
                    'model': self.model,
                    'requires_verification': True,  # ALWAYS
                    'raw_response': response,
                }
            )
            
        except Exception as e:
            logger.error(f"LLM solver error: {e}")
            return self._create_error_result(str(e))
    
    def _create_prompt(self, problem: MathProblem) -> str:
        """
        Create a structured prompt for mathematical problem solving.
        
        The prompt enforces step-by-step reasoning to reduce errors.
        """
        prompt = f"""You are a precise mathematical problem solver. 
Solve the following problem step by step, showing all work.

IMPORTANT RULES:
1. Show every step explicitly - no skipping
2. Justify each step with the rule or theorem used
3. Check your answer by substitution when possible
4. Express final answer clearly
5. If uncertain, say so

PROBLEM:
{problem.raw_input}

PROBLEM TYPE: {problem.problem_type.value}

Solve this step by step:
"""
        return prompt
    
    def _call_llm(self, prompt: str) -> Optional[str]:
        """
        Call the LLM API.
        
        Handles API initialization and error handling.
        """
        # Try to use OpenAI
        try:
            import openai
            
            if self.api_key:
                client = openai.OpenAI(api_key=self.api_key)
            else:
                # Will use OPENAI_API_KEY environment variable
                client = openai.OpenAI()
            
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a precise mathematical assistant. "
                            "Show all work step by step. Be explicit about "
                            "every operation and why it's valid."
                        )
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=2000,
            )
            
            return response.choices[0].message.content
            
        except ImportError:
            logger.warning("OpenAI package not installed")
            return self._mock_response(prompt)
        except Exception as e:
            logger.error(f"LLM API error: {e}")
            return None
    
    def _mock_response(self, prompt: str) -> str:
        """
        Mock response for testing without API.
        
        Returns a placeholder that indicates verification is needed.
        """
        return """
SOLUTION (Mock - Requires Verification):

Step 1: Parse the problem
The problem asks us to solve the given expression.

Step 2: Apply relevant rules
[This is a mock response - actual LLM would provide reasoning]

Step 3: Compute result
[Mock computation]

ANSWER: [REQUIRES SYMBOLIC VERIFICATION]

Note: This is a mock response. Enable LLM API for actual solving.
"""
    
    def _parse_response(
        self, 
        response: str, 
        problem: MathProblem
    ) -> tuple[Any, list[SolutionStep]]:
        """
        Parse LLM response to extract answer and steps.
        
        Returns:
            Tuple of (answer, steps)
        """
        steps = []
        answer = None
        
        # Try to extract answer
        import re
        
        # Look for common answer patterns
        answer_patterns = [
            r'(?:answer|solution|result)\s*(?:is|=|:)\s*(.+?)(?:\n|$)',
            r'(?:therefore|thus|hence)\s*,?\s*(.+?)(?:\n|$)',
            r'=\s*(\d+(?:\.\d+)?|\w+)\s*$',
        ]
        
        for pattern in answer_patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
            if match:
                answer = match.group(1).strip()
                break
        
        # Try to parse answer with SymPy
        if answer:
            try:
                from sympy.parsing.sympy_parser import parse_expr
                answer = parse_expr(answer)
            except Exception:
                pass  # Keep as string
        
        # Extract steps
        step_pattern = r'(?:step\s*\d+|first|then|next|finally)[:\s]+(.+?)(?=(?:step\s*\d+|first|then|next|finally|$))'
        step_matches = re.findall(step_pattern, response, re.IGNORECASE | re.DOTALL)
        
        for i, step_text in enumerate(step_matches):
            steps.append(SolutionStep(
                action=f"Step {i+1}",
                expression=step_text.strip()[:200],  # Truncate
                justification="[From LLM - requires verification]",
                warnings=["This step was generated by AI and needs verification"],
            ))
        
        # Always add verification warning
        steps.append(SolutionStep(
            action="⚠️ Verification Required",
            expression="LLM output must be verified",
            justification=(
                "AI-generated solutions can contain errors. "
                "This answer must be verified by symbolic computation "
                "or formal proof before being trusted."
            ),
            warnings=["UNVERIFIED - Do not trust without verification"],
        ))
        
        return answer, steps
    
    def suggest_approach(self, problem: MathProblem) -> dict:
        """
        Suggest an approach for solving without computing answer.
        
        Useful for guiding symbolic solvers or explaining strategy.
        """
        prompt = f"""For the following math problem, suggest the best approach to solve it.
Don't solve it - just explain the strategy and relevant concepts.

PROBLEM: {problem.raw_input}

What approach should be used? What theorems or techniques are relevant?
"""
        response = self._call_llm(prompt)
        
        return {
            'approach': response,
            'problem_type': problem.problem_type.value,
            'confidence': 'low - suggestion only',
        }
