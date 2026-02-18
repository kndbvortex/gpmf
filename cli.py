"""Command-line interface for gradual pattern mining."""
import argparse
import sys
import json
from pathlib import Path

from .miner import GradualPatternMiner
from .factory import AlgorithmRegistry
from .config import config


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Gradual Pattern Mining - Unified CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available algorithms
  gradual-mine --list

  # Mine patterns with GRAANK
  gradual-mine graank data.csv --min-support 0.5

  # Save results to file
  gradual-mine graank data.csv --min-support 0.5 --output results.json

  # Use parallel processing
  gradual-mine graank data.csv --min-support 0.5 --n-jobs -1

  # Verbose output
  gradual-mine graank data.csv --min-support 0.5 --verbose
        """
    )

    parser.add_argument('--list', action='store_true',
                        help='List available algorithms and exit')
    parser.add_argument('--version', action='store_true',
                        help='Show version and exit')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    parser.add_argument('algorithm', nargs='?',
                        help='Algorithm name (use --list to see available)')
    parser.add_argument('data', nargs='?',
                        help='Path to input CSV file')
    parser.add_argument('--min-support', '-s', type=float, default=0.5,
                        help='Minimum support threshold (0.0 to 1.0, default: 0.5)')
    parser.add_argument('--n-jobs', '-j', type=int, default=1,
                        help='Number of parallel jobs (-1 for all cores, default: 1)')
    parser.add_argument('--output', '-o', type=str,
                        help='Output file path (JSON format)')
    parser.add_argument('--csv', type=str,
                        help='Output file path (CSV format)')
    parser.add_argument('--no-print', action='store_true',
                        help='Don\'t print patterns to stdout')

    args = parser.parse_args()

    if args.version:
        from . import __version__
        print(f"gradual-mining version {__version__}")
        sys.exit(0)

    if args.list:
        algorithms = AlgorithmRegistry.list_algorithms()
        print("Available algorithms:")
        for algo in algorithms:
            print(f"  - {algo}")
        sys.exit(0)

    if not args.algorithm:
        parser.error("algorithm is required (use --list to see available)")
    if not args.data:
        parser.error("data file path is required")

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: File not found: {args.data}", file=sys.stderr)
        sys.exit(1)

    if args.verbose:
        config.verbose = True
        config.suppress_prints = False
        config.log_level = "INFO"
        config.setup_logging()

    try:
        print(f"Mining patterns with {args.algorithm} (min_support={args.min_support})...")

        miner = GradualPatternMiner(
            algorithm=args.algorithm,
            data=args.data,
            min_support=args.min_support,
            n_jobs=args.n_jobs
        )

        result = miner.mine_and_get_result()

        print(result.summary())

        if not args.no_print:
            print("\nPatterns:")
            for i, pattern in enumerate(result.patterns, 1):
                print(f"  {i}. {pattern.to_string()} : {pattern.support}")

        if args.output:
            result.save_json(args.output)
            print(f"\nResults saved to: {args.output}")

        if args.csv:
            result.save_csv(args.csv)
            print(f"Results saved to: {args.csv}")

        sys.exit(0)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
