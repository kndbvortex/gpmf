"""
ParaMiner Gradual - Python implementation of gradual pattern mining.

This module implements the mining of closed frequent gradual itemsets (GRI).
Gradual patterns express co-variations between attributes, such as:
"When X increases and Y decreases, then Z increases"

The implementation is based on the ParaMiner C++ implementation with the following
key components:
- Boolean matrix representation of transaction pairs
- Longest path computation for support counting
- Closure operator for mining closed patterns

Performance-critical components can use optional Rust acceleration for 10-50x speedup.
"""

from .datatypes import Transaction, TransactionTable, GradualItem, GradualPattern, Variation, TransactionPair

# Try to import Rust implementations
RUST_AVAILABLE = False
RustGradualMiner = None
RustGradualItem = None
RustGradualPatternResult = None
RustVariation = None

try:
    from .paraminer_gradual_rust import (
        BoolMatrix as RustBoolMatrix,
        compute_gradual_support as rust_compute_gradual_support,
        compute_path_lengths as rust_compute_path_lengths,
        compute_gradual_support_parallel as rust_compute_gradual_support_parallel,
        compute_path_lengths_parallel as rust_compute_path_lengths_parallel,
        RustGradualMiner,
        GradualItem as RustGradualItem,
        GradualPatternResult as RustGradualPatternResult,
        Variation as RustVariation,
    )
    RUST_AVAILABLE = True

    # Default to Rust implementations
    BoolMatrix = RustBoolMatrix
    compute_gradual_support = rust_compute_gradual_support
    compute_path_lengths = rust_compute_path_lengths

except ImportError:
    # Fall back to Python implementations
    from .bool_matrix import BoolMatrix
    from .support import compute_gradual_support, compute_path_lengths

# Always import the miner (it uses the above based on availability)
from .gradual_mining import GradualMiner

__version__ = "0.1.3"

__all__ = [
    "Transaction",
    "TransactionTable",
    "GradualItem",
    "GradualPattern",
    "Variation",
    "TransactionPair",
    "BoolMatrix",
    "GradualMiner",
    "compute_gradual_support",
    "compute_path_lengths",
    "RUST_AVAILABLE",
]

def use_rust(enable: bool = True):
    """
    Enable or disable Rust acceleration.

    Args:
        enable: True to use Rust (if available), False to use pure Python

    Note:
        This must be called before creating a GradualMiner instance.
    """
    global BoolMatrix, compute_gradual_support, compute_path_lengths

    if enable:
        if RUST_AVAILABLE:
            BoolMatrix = RustBoolMatrix
            compute_gradual_support = rust_compute_gradual_support
            compute_path_lengths = rust_compute_path_lengths
        else:
            raise ImportError("Rust extensions not available. Install with: maturin develop --release")
    else:
        from .bool_matrix import BoolMatrix as PyBoolMatrix
        from .support import compute_gradual_support as py_compute_gradual_support
        from .support import compute_path_lengths as py_compute_path_lengths

        BoolMatrix = PyBoolMatrix
        compute_gradual_support = py_compute_gradual_support
        compute_path_lengths = py_compute_path_lengths
