#!/usr/bin/env python3
"""Unified entry point for NA²Q and HiT-MAC algorithms."""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="DSN Lab: Multi-Agent RL for Directional Sensor Networks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Algorithms:
  na2q    - Neural Attention Additive Q-Learning
  hitmac  - Hierarchical Twin-Actor Multi-Agent Coordination

Examples:
  python main.py --algorithm na2q --mode train --scenario 1
  python main.py --algorithm hitmac --mode train --scenario 1
""",
    )

    parser.add_argument(
        "--algorithm",
        "-a",
        type=str,
        default="na2q",
        choices=["na2q", "hitmac"],
        help="Algorithm to run (default: na2q)",
    )

    # Parse only known args so algorithm-specific args pass through
    args, remaining = parser.parse_known_args()

    # Reconstruct sys.argv for the algorithm's parser
    sys.argv = [sys.argv[0]] + remaining

    if args.algorithm == "na2q":
        print("Running NA²Q...")
        from na2q.main import main as na2q_main

        na2q_main()
    elif args.algorithm == "hitmac":
        print("Running HiT-MAC...")
        from hitmac.main import main as hitmac_main

        hitmac_main()


if __name__ == "__main__":
    main()
