"""
Command-line interface for gradual pattern mining.
"""

import sys
import argparse
from pathlib import Path
from .gradual_mining import GradualMiner
from . import __version__, RUST_AVAILABLE


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Mine closed frequent gradual patterns from numerical datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Mine with relative support (30%)
  python -m paraminer_gradual data.txt 0.3

  # Mine with absolute support (10 transactions)
  python -m paraminer_gradual data.txt 10

  # Show version
  python -m paraminer_gradual --version

Input file format:
  - First line is skipped (header)
  - Each line contains space-separated numerical values
  - All lines must have the same number of attributes

Gradual patterns express co-variations like:
  - {Age+, Salary+}: "As age increases, salary increases"
  - {Experience+, Errors-}: "As experience increases, errors decrease"
        """
    )

    parser.add_argument(
        "dataset",
        nargs="?",
        help="Path to the dataset file"
    )

    parser.add_argument(
        "minsup",
        nargs="?",
        type=float,
        help="Minimum support (0-1 for relative, >1 for absolute)"
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"paraminer-gradual {__version__}"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    parser.add_argument(
        "--no-rust",
        action="store_true",
        help="Disable Rust acceleration (use pure Python)"
    )

    parser.add_argument(
        "-j", "--threads",
        type=int,
        metavar="N",
        help="Number of threads for parallel processing (default: auto)"
    )

    args = parser.parse_args()

    # Check if required arguments are provided
    if not args.dataset or args.minsup is None:
        parser.print_help()
        return 1

    # Validate inputs
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Error: Dataset file not found: {args.dataset}", file=sys.stderr)
        return 1

    if args.minsup <= 0:
        print(f"Error: Minimum support must be positive", file=sys.stderr)
        return 1

    # Run mining
    try:
        # Show configuration
        from . import RUST_AVAILABLE
        use_rust = not args.no_rust and RUST_AVAILABLE

        if args.verbose:
            print(f"Configuration:")
            print(f"  Rust acceleration: {'enabled' if use_rust else 'disabled'}")
            if RUST_AVAILABLE and args.no_rust:
                print(f"    (Rust is available but disabled by --no-rust)")
            elif not RUST_AVAILABLE:
                print(f"    (Rust not available - using Python)")
            if args.threads:
                print(f"  Threads: {args.threads}")
            print()

        miner = GradualMiner(
            min_support=args.minsup,
            num_threads=args.threads,
            use_rust=use_rust
        )
        miner.load_data(str(dataset_path))
        patterns = miner.mine()

        # Print summary
        print("\n" + "="*60)
        print(f"SUMMARY: Found {len(patterns)} closed frequent gradual patterns")
        print("="*60)

        if patterns:
            print("\nPatterns:")
            for i, pattern in enumerate(patterns, 1):
                print(f"{i:3d}. {pattern}")

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
