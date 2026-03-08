"""CLI for the modular CCPP USDR+ project."""

from __future__ import annotations

import argparse

from usdr_plus.ccpp.runner import run_full_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="usdr-plus ccpp",
        description=(
            "USDR+ CCPP pipeline\n"
            "------------------\n"
            "Runs the modular CCPP pipeline from start to finish."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    subparsers.required = True

    subparsers.add_parser("run", help="Run the full CCPP pipeline")
    subparsers.add_parser("where", help="Print CCPP package root path")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "where":
        import usdr_plus.ccpp as pkg
        from pathlib import Path

        print(Path(pkg.__file__).resolve().parent)
        return 0

    if args.command == "run":
        print("\n" + "=" * 60)
        print("  USDR+ CCPP PROJECT  |  command: run")
        print("=" * 60 + "\n")
        try:
            run_full_pipeline()
            return 0
        except RuntimeError as exc:
            print(str(exc))
            return 1

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
