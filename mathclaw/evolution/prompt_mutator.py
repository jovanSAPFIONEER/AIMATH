"""
Prompt Mutator - Evolves text prompts for better discovery.

This module can ONLY mutate TEXT - it never generates or modifies code.
Mutations are limited to:
- Changing words in prompts
- Adjusting numerical values
- Combining successful prompts
- Reordering instructions

This module is EVOLVABLE - the evolution engine can modify mutation
strategies (but still text-only).
"""

import re
import random
from typing import List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from .strategy_store import Strategy, StrategyStore


@dataclass
class MutationResult:
    """Result of a prompt mutation."""
    original_id: str
    mutated_prompt: str
    mutation_type: str
    changes_made: List[str]
    new_strategy: Optional[Strategy] = None


class PromptMutator:
    """
    Mutates prompts to discover better strategies.
    
    CRITICAL: This class can ONLY modify TEXT content.
    It cannot generate, modify, or inject code.
    
    Mutation strategies:
    1. Synonym replacement - change words while preserving meaning
    2. Number tweaking - adjust numerical values slightly
    3. Instruction reordering - change order of instructions
    4. Crossover - combine parts of successful prompts
    5. Emphasis adjustment - add/remove emphasis words
    
    Example:
        >>> mutator = PromptMutator()
        >>> result = mutator.mutate(strategy, mutation_type='synonym')
        >>> print(result.mutated_prompt)
    """
    
    # Word synonyms for mutation (safe, mathematical context)
    SYNONYMS = {
        'generate': ['create', 'produce', 'propose', 'formulate', 'devise'],
        'identity': ['equation', 'equality', 'relation', 'formula'],
        'plausible': ['reasonable', 'possible', 'likely', 'believable'],
        'mathematical': ['math', 'numeric', 'algebraic', 'analytic'],
        'relationship': ['connection', 'relation', 'link', 'correspondence'],
        'discover': ['find', 'uncover', 'identify', 'reveal'],
        'explore': ['investigate', 'examine', 'study', 'analyze'],
        'consider': ['think about', 'examine', 'contemplate', 'evaluate'],
        'output': ['provide', 'give', 'return', 'produce'],
        'involving': ['with', 'containing', 'including', 'featuring'],
    }
    
    # Emphasis words that can be added/removed
    EMPHASIS_WORDS = [
        'very', 'extremely', 'highly', 'particularly', 
        'especially', 'notably', 'clearly', 'obviously',
    ]
    
    # Instruction starters that can be reordered
    INSTRUCTION_PATTERNS = [
        r'(?:^|\n)\d+\.\s*[^\n]+',  # Numbered lists
        r'(?:^|\n)-\s*[^\n]+',      # Bullet points
        r'(?:^|\n)•\s*[^\n]+',      # Bullet points with •
    ]
    
    def __init__(self, store: StrategyStore = None):
        """
        Initialize mutator.
        
        Args:
            store: Strategy store for saving mutations
        """
        self.store = store or StrategyStore()
    
    def mutate(
        self,
        strategy: Strategy,
        mutation_type: str = 'random',
    ) -> MutationResult:
        """
        Mutate a strategy's prompt.
        
        Args:
            strategy: Strategy to mutate
            mutation_type: Type of mutation ('synonym', 'number', 'reorder', 
                          'crossover', 'emphasis', 'random')
                          
        Returns:
            MutationResult with mutated prompt
        """
        if mutation_type == 'random':
            mutation_type = random.choice([
                'synonym', 'number', 'emphasis', 'reorder'
            ])
        
        original_prompt = strategy.prompt_template
        changes = []
        
        if mutation_type == 'synonym':
            mutated, changes = self._mutate_synonyms(original_prompt)
        elif mutation_type == 'number':
            mutated, changes = self._mutate_numbers(original_prompt)
        elif mutation_type == 'emphasis':
            mutated, changes = self._mutate_emphasis(original_prompt)
        elif mutation_type == 'reorder':
            mutated, changes = self._mutate_reorder(original_prompt)
        elif mutation_type == 'crossover':
            # Get another successful strategy for crossover
            other = self.store.get_random_strategy(
                domain=strategy.domain,
                weighted=True
            )
            if other and other.id != strategy.id:
                mutated, changes = self._crossover(original_prompt, other.prompt_template)
            else:
                mutated, changes = original_prompt, ["No suitable crossover partner"]
        else:
            mutated, changes = original_prompt, [f"Unknown mutation type: {mutation_type}"]
        
        return MutationResult(
            original_id=strategy.id,
            mutated_prompt=mutated,
            mutation_type=mutation_type,
            changes_made=changes,
        )
    
    def mutate_and_save(
        self,
        strategy: Strategy,
        mutation_type: str = 'random',
    ) -> MutationResult:
        """
        Mutate a strategy and save as new strategy.
        
        Args:
            strategy: Strategy to mutate
            mutation_type: Type of mutation
            
        Returns:
            MutationResult with new strategy
        """
        result = self.mutate(strategy, mutation_type)
        
        if result.mutated_prompt != strategy.prompt_template:
            # Create new strategy
            new_id = f"{strategy.id}_mut_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            new_strategy = Strategy(
                id=new_id,
                name=f"{strategy.name} (Mutated)",
                description=f"Mutated from {strategy.id} via {mutation_type}",
                prompt_template=result.mutated_prompt,
                domain=strategy.domain,
                success_rate=0.0,
                total_uses=0,
                total_successes=0,
                is_baseline=False,
            )
            
            self.store.add_strategy(new_strategy)
            result.new_strategy = new_strategy
        
        return result
    
    def _mutate_synonyms(self, prompt: str) -> Tuple[str, List[str]]:
        """Replace words with synonyms."""
        changes = []
        mutated = prompt
        
        for word, synonyms in self.SYNONYMS.items():
            if word.lower() in prompt.lower():
                # Only mutate with 30% probability per word
                if random.random() < 0.3:
                    new_word = random.choice(synonyms)
                    # Case-preserving replacement
                    pattern = re.compile(re.escape(word), re.IGNORECASE)
                    mutated = pattern.sub(new_word, mutated, count=1)
                    changes.append(f"'{word}' -> '{new_word}'")
        
        return mutated, changes
    
    def _mutate_numbers(self, prompt: str) -> Tuple[str, List[str]]:
        """Slightly adjust numbers in the prompt."""
        changes = []
        
        def adjust_number(match):
            num_str = match.group(0)
            try:
                if '.' in num_str:
                    num = float(num_str)
                    # Adjust by ±10%
                    adjusted = num * random.uniform(0.9, 1.1)
                    changes.append(f"{num} -> {adjusted:.2f}")
                    return f"{adjusted:.2f}"
                else:
                    num = int(num_str)
                    if num > 1:
                        # Adjust by ±1 or ±10%
                        delta = max(1, int(num * 0.1))
                        adjusted = num + random.randint(-delta, delta)
                        changes.append(f"{num} -> {adjusted}")
                        return str(adjusted)
            except ValueError:
                pass
            return num_str
        
        # Only mutate numbers that look like parameters (not in examples)
        # Look for numbers not in example outputs
        mutated = re.sub(r'\b\d+\.?\d*\b', adjust_number, prompt)
        
        return mutated, changes
    
    def _mutate_emphasis(self, prompt: str) -> Tuple[str, List[str]]:
        """Add or remove emphasis words."""
        changes = []
        mutated = prompt
        
        # 50% chance to add, 50% to remove
        if random.random() < 0.5:
            # Add emphasis
            emphasis = random.choice(self.EMPHASIS_WORDS)
            # Find a good place to add (before adjectives)
            adjectives = ['important', 'key', 'critical', 'useful', 'good', 'new', 'different']
            for adj in adjectives:
                if adj in mutated.lower() and emphasis not in mutated.lower():
                    pattern = re.compile(f'\\b({adj})\\b', re.IGNORECASE)
                    mutated = pattern.sub(f'{emphasis} \\1', mutated, count=1)
                    changes.append(f"Added '{emphasis}' before '{adj}'")
                    break
        else:
            # Remove emphasis
            for emphasis in self.EMPHASIS_WORDS:
                if emphasis in mutated.lower():
                    pattern = re.compile(f'\\b{emphasis}\\s+', re.IGNORECASE)
                    mutated = pattern.sub('', mutated, count=1)
                    changes.append(f"Removed '{emphasis}'")
                    break
        
        return mutated, changes
    
    def _mutate_reorder(self, prompt: str) -> Tuple[str, List[str]]:
        """Reorder instructions in the prompt."""
        changes = []
        
        # Find numbered instructions
        numbered = re.findall(r'(\d+)\.\s*([^\n]+)', prompt)
        
        if len(numbered) >= 2:
            # Shuffle middle items (keep first and last)
            items = list(numbered)
            if len(items) > 2:
                middle = items[1:-1]
                random.shuffle(middle)
                items = [items[0]] + middle + [items[-1]]
            else:
                random.shuffle(items)
            
            # Rebuild with new numbers
            mutated = prompt
            for i, (old_num, text) in enumerate(numbered):
                new_num = i + 1
                mutated = mutated.replace(f"{old_num}. {text}", f"{new_num}. {items[i][1]}", 1)
            
            changes.append(f"Reordered {len(numbered)} instructions")
            return mutated, changes
        
        return prompt, ["No numbered instructions to reorder"]
    
    def _crossover(self, prompt1: str, prompt2: str) -> Tuple[str, List[str]]:
        """Combine two prompts (simple crossover)."""
        changes = []
        
        # Split prompts into sections
        sections1 = prompt1.split('\n\n')
        sections2 = prompt2.split('\n\n')
        
        if len(sections1) >= 2 and len(sections2) >= 2:
            # Take some sections from each
            crossover_point = random.randint(1, min(len(sections1), len(sections2)) - 1)
            
            new_sections = sections1[:crossover_point] + sections2[crossover_point:]
            mutated = '\n\n'.join(new_sections)
            
            changes.append(f"Crossover at section {crossover_point}")
            return mutated, changes
        
        return prompt1, ["Prompts not suitable for crossover"]
