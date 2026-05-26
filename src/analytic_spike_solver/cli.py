from __future__ import annotations

import argparse

from .tools.benchmark import run_benchmark


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="analytic-spike-solver")
    sub = parser.add_subparsers(dest="cmd", required=True)
    bench = sub.add_parser("benchmark")
    bench.add_argument("path")
    args = parser.parse_args(argv)
    if args.cmd == "benchmark":
        print(run_benchmark(args.path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
