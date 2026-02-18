"""Output formatting utilities."""
from typing import List
from ..core.data_structures import GradualPattern


def format_patterns(patterns: List[GradualPattern], show_support: bool = True) -> List[str]:
    """Format patterns as list of strings.

    Args:
        patterns: List of GradualPattern objects
        show_support: Include support values

    Returns:
        List of formatted strings
    """
    if show_support:
        return [f"{p.to_string()} : {p.support}" for p in patterns]
    else:
        return [str(p.to_string()) for p in patterns]


def format_pattern_string(pattern: GradualPattern, show_support: bool = True) -> str:
    """Format a single pattern as string.

    Args:
        pattern: GradualPattern object
        show_support: Include support value

    Returns:
        Formatted string
    """
    if show_support:
        return f"{pattern.to_string()} : {pattern.support}"
    else:
        return str(pattern.to_string())
