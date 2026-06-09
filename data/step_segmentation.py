"""
Utilities for segmenting multi-step math solutions into individual steps.
"""
import re


def segment_steps(solution: str) -> list[str]:
    """
    Split a math solution string into individual reasoning steps.
    Handles GSM8K-style, MATH-style, and free-form solutions.
    Returns list of non-empty step strings.
    """
    # Try numbered steps first: "Step 1:", "1.", "Step 1 -"
    numbered = re.split(r'(?:Step\s+\d+[:.]\s*|\n\d+\.\s+)', solution)
    if len(numbered) > 2:
        return [s.strip() for s in numbered if s.strip()]

    # Fall back to double-newline paragraph splits
    paragraphs = [s.strip() for s in solution.split("\n\n") if s.strip()]
    if len(paragraphs) > 1:
        return paragraphs

    # Fall back to single newlines
    lines = [s.strip() for s in solution.split("\n") if s.strip()]
    return lines if lines else [solution.strip()]


def rejoin_steps(steps: list[str]) -> str:
    return "\n".join(steps)
