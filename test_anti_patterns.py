"""Test anti-pattern detection."""
from src.explanation.anti_patterns import AntiPatternDetector

detector = AntiPatternDetector()

print('='*70)
print('  ANTI-PATTERN DETECTION TEST')
print('='*70)

# Bad explanation (typical AI hand-waving)
bad = '''
Obviously, the derivative of x squared is 2x.
This is trivially true by definition.
It follows that the integral is x squared plus C.
Clearly, the reader can verify this as an exercise.
By similar argument, this works for all polynomials.
'''

print('\nBAD EXPLANATION (typical AI response):')
print('-'*50)
print(bad)
print('-'*50)

matches = detector.scan(bad)
print(f'\nFound {len(matches)} anti-patterns:\n')

for m in matches:
    print(f'  [{m.severity.value.upper():8}] "{m.matched_text}"')
    print(f'             Fix: {m.fix_instruction[:65]}...')
    print()

print('='*70)
print('  GOOD EXPLANATION (proper mathematical writing)')
print('='*70)

good = '''
To find the derivative of x squared, we apply the power rule.

The power rule states: d/dx[x^n] = n * x^(n-1)

For f(x) = x^2, we have n = 2:
  d/dx[x^2] = 2 * x^(2-1) = 2x

This works because the derivative measures instantaneous rate of change,
and for polynomials, the limit definition simplifies to this formula.

We can verify: at x = 3, the slope is 2(3) = 6.
'''

print('\nGOOD EXPLANATION:')
print('-'*50)
print(good)
print('-'*50)

matches2 = detector.scan(good)
print(f'\nFound {len(matches2)} anti-patterns ✓')
if matches2:
    for m in matches2:
        print(f'  - "{m.matched_text}" ({m.pattern_type})')
