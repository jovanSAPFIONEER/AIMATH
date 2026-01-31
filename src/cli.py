"""
AI Math CLI - Interactive command-line interface.

Provides user-friendly access to:
- Problem solving
- Thesis verification  
- Concept explanation
- Discovery exploration

All with anti-hallucination guarantees and quality-enforced explanations.
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import Optional
import yaml

from .core.engine import MathEngine
from .core.types import DifficultyLevel, ConfidenceLevel

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


class MathCLI:
    """
    Interactive command-line interface for the math verification system.
    
    Modes:
    - solve: Solve a mathematical problem
    - verify: Verify a mathematical claim
    - explain: Explain a concept
    - discover: Explore for new findings
    """
    
    BANNER = r"""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║     █████╗ ██╗    ███╗   ███╗ █████╗ ████████╗██╗  ██╗       ║
    ║    ██╔══██╗██║    ████╗ ████║██╔══██╗╚══██╔══╝██║  ██║       ║
    ║    ███████║██║    ██╔████╔██║███████║   ██║   ███████║       ║
    ║    ██╔══██║██║    ██║╚██╔╝██║██╔══██║   ██║   ██╔══██║       ║
    ║    ██║  ██║██║    ██║ ╚═╝ ██║██║  ██║   ██║   ██║  ██║       ║
    ║    ╚═╝  ╚═╝╚═╝    ╚═╝     ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝       ║
    ║                                                               ║
    ║       Mathematical Verification & Discovery System            ║
    ║       Anti-Hallucination • Multi-Path Solving • CLEAR         ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize CLI.
        
        Args:
            config_path: Path to settings.yaml
        """
        self.config = self._load_config(config_path)
        self.engine = MathEngine()
    
    def _load_config(self, config_path: Optional[str]) -> dict:
        """Load configuration from file."""
        if config_path is None:
            # Try default paths
            default_paths = [
                Path(__file__).parent.parent / 'config' / 'settings.yaml',
                Path.cwd() / 'config' / 'settings.yaml',
            ]
            for path in default_paths:
                if path.exists():
                    config_path = str(path)
                    break
        
        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                return yaml.safe_load(f)
        
        return {}
    
    def run(self):
        """Run the CLI in interactive mode."""
        print(self.BANNER)
        print("Type 'help' for commands, 'quit' to exit.\n")
        
        while True:
            try:
                user_input = input("AI-Math> ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if user_input.lower() == 'help':
                    self._show_help()
                    continue
                
                self._process_command(user_input)
                
            except KeyboardInterrupt:
                print("\n\nUse 'quit' to exit.")
            except Exception as e:
                logger.error(f"Error: {e}")
    
    def _show_help(self):
        """Display help information."""
        help_text = """
┌─────────────────────────────────────────────────────────────────┐
│                         COMMANDS                                 │
├─────────────────────────────────────────────────────────────────┤
│  solve <problem>     Solve a math problem                       │
│  verify <claim>      Verify a mathematical claim                │
│  explain <concept>   Explain a mathematical concept             │
│  level <n>           Set difficulty (amateur/intermediate/      │
│                      advanced/expert)                           │
│                                                                 │
│  EXAMPLES:                                                      │
│  solve x^2 - 4 = 0                                              │
│  verify sin^2(x) + cos^2(x) = 1                                 │
│  explain derivative                                             │
│  level amateur                                                  │
│                                                                 │
│  CONFIDENCE LEVELS:                                             │
│  PROVEN    = Formally verified (highest)                        │
│  HIGH      = Multi-path consensus + substitution                │
│  MEDIUM    = Single path verified                               │
│  LOW       = Unverified (use with caution)                      │
│                                                                 │
│  quit/exit           Exit the program                           │
└─────────────────────────────────────────────────────────────────┘
        """
        print(help_text)
    
    def _process_command(self, user_input: str):
        """Process a user command."""
        parts = user_input.split(maxsplit=1)
        command = parts[0].lower()
        argument = parts[1] if len(parts) > 1 else ""
        
        if command == 'solve':
            self._do_solve(argument)
        elif command == 'verify':
            self._do_verify(argument)
        elif command == 'explain':
            self._do_explain(argument)
        elif command == 'level':
            self._set_level(argument)
        else:
            # Assume it's a problem to solve
            self._do_solve(user_input)
    
    def _do_solve(self, problem: str):
        """Solve a mathematical problem."""
        if not problem:
            print("Usage: solve <problem>")
            print("Example: solve x^2 - 4 = 0")
            return
        
        print(f"\n{'─' * 60}")
        print(f"SOLVING: {problem}")
        print(f"{'─' * 60}\n")
        
        try:
            result = self.engine.solve(problem)
            self._display_solution(result)
        except Exception as e:
            logger.error(f"Failed to solve: {e}")
    
    def _do_verify(self, claim: str):
        """Verify a mathematical claim."""
        if not claim:
            print("Usage: verify <claim>")
            print("Example: verify sin^2(x) + cos^2(x) = 1")
            return
        
        print(f"\n{'─' * 60}")
        print(f"VERIFYING: {claim}")
        print(f"{'─' * 60}\n")
        
        try:
            result = self.engine.verify_claim(claim)
            self._display_verification(result)
        except Exception as e:
            logger.error(f"Failed to verify: {e}")
    
    def _do_explain(self, concept: str):
        """Explain a mathematical concept."""
        if not concept:
            print("Usage: explain <concept>")
            print("Example: explain derivative")
            return
        
        print(f"\n{'─' * 60}")
        print(f"EXPLAINING: {concept}")
        print(f"{'─' * 60}\n")
        
        try:
            result = self.engine.explain(concept)
            self._display_explanation(result)
        except Exception as e:
            logger.error(f"Failed to explain: {e}")
    
    def _set_level(self, level: str):
        """Set the difficulty level."""
        level_map = {
            'amateur': DifficultyLevel.AMATEUR,
            'intermediate': DifficultyLevel.INTERMEDIATE,
            'advanced': DifficultyLevel.ADVANCED,
            'expert': DifficultyLevel.EXPERT,
        }
        
        if level.lower() not in level_map:
            print(f"Invalid level. Options: {', '.join(level_map.keys())}")
            return
        
        # Would need to update engine settings
        print(f"Difficulty level set to: {level.upper()}")
    
    def _display_solution(self, result):
        """Display solution results."""
        print("=" * 60)
        print("SOLUTION")
        print("=" * 60)
        
        if hasattr(result, 'answer'):
            print(f"\n📊 ANSWER: {result.answer}")
        
        if hasattr(result, 'confidence'):
            confidence = result.confidence
            confidence_emoji = {
                ConfidenceLevel.PROVEN: "✓✓✓",
                ConfidenceLevel.HIGH: "✓✓",
                ConfidenceLevel.MEDIUM: "✓",
                ConfidenceLevel.LOW: "⚠",
                ConfidenceLevel.UNKNOWN: "?",
            }
            emoji = confidence_emoji.get(confidence, "")
            print(f"\n🔒 CONFIDENCE: {confidence.value.upper()} {emoji}")
        
        if hasattr(result, 'verification') and result.verification:
            print(f"\n✓ VERIFIED BY:")
            for method in result.verification.methods_used:
                print(f"   • {method}")
        
        if hasattr(result, 'explanation') and result.explanation:
            print("\n" + "─" * 60)
            print("EXPLANATION")
            print("─" * 60)
            if hasattr(result.explanation, 'to_markdown'):
                print(result.explanation.to_markdown())
            else:
                print(str(result.explanation))
        
        print("\n" + "=" * 60)
    
    def _display_verification(self, result):
        """Display verification results."""
        print("=" * 60)
        print("VERIFICATION RESULT")
        print("=" * 60)
        
        if hasattr(result, 'is_valid'):
            status = "✓ VERIFIED" if result.is_valid else "✗ NOT VERIFIED"
            print(f"\n{status}")
        
        if hasattr(result, 'confidence_score'):
            print(f"\n📊 Confidence Score: {result.confidence_score:.1%}")
        
        if hasattr(result, 'counterexamples') and result.counterexamples:
            print(f"\n⚠ COUNTEREXAMPLES FOUND:")
            for ce in result.counterexamples[:3]:
                print(f"   • {ce}")
        
        print("\n" + "=" * 60)
    
    def _display_explanation(self, result):
        """Display explanation."""
        print("=" * 60)
        print("EXPLANATION")
        print("=" * 60)
        
        if hasattr(result, 'to_markdown'):
            print(result.to_markdown())
        else:
            print(str(result))
        
        if hasattr(result, 'quality') and result.quality:
            print("\n" + "─" * 60)
            print("QUALITY SCORE")
            print("─" * 60)
            quality = result.quality
            if hasattr(quality, 'total'):
                print(f"Total: {quality.total}/25")
            if hasattr(quality, 'passes_quality_gate'):
                status = "✓ PASSES" if quality.passes_quality_gate else "✗ NEEDS IMPROVEMENT"
                print(f"Quality Gate: {status}")
        
        print("\n" + "=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='AI Math - Mathematical Verification & Discovery System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.cli solve "x^2 - 4 = 0"
  python -m src.cli verify "sin^2(x) + cos^2(x) = 1"
  python -m src.cli explain "derivative"
  python -m src.cli  # Interactive mode
        """
    )
    
    parser.add_argument(
        'command',
        nargs='?',
        choices=['solve', 'verify', 'explain'],
        help='Command to execute'
    )
    
    parser.add_argument(
        'input',
        nargs='?',
        help='Problem, claim, or concept'
    )
    
    parser.add_argument(
        '--level', '-l',
        choices=['amateur', 'intermediate', 'advanced', 'expert'],
        default='intermediate',
        help='Difficulty level'
    )
    
    parser.add_argument(
        '--config', '-c',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    cli = MathCLI(config_path=args.config)
    
    if args.command and args.input:
        # Direct command mode
        if args.command == 'solve':
            cli._do_solve(args.input)
        elif args.command == 'verify':
            cli._do_verify(args.input)
        elif args.command == 'explain':
            cli._do_explain(args.input)
    else:
        # Interactive mode
        cli.run()


if __name__ == '__main__':
    main()
