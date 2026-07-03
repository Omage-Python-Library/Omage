"""
Omage CLI — Command-line interface  (FIX #2: import order / RuntimeWarning)
"""

from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="[Omage] %(message)s",
        level=level,
        stream=sys.stdout,
    )


def cmd_run(args: argparse.Namespace) -> None:
    path = Path(args.file)
    if not path.exists():
        print(f"[Omage] Error: File not found: '{args.file}'")
        sys.exit(1)
    print(f"[Omage] Running: {args.file}")
    exec(path.read_text(encoding="utf-8"), {"__name__": "__main__"})


def cmd_compile(args: argparse.Namespace) -> None:
    # Import here — AFTER logging is set up — to avoid RuntimeWarning (FIX #2)
    from omage.compiler.transpiler import compile_file, TranspileError
    try:
        out = compile_file(args.file, args.output)
        print(f"[Omage] Compiled → {out}")
    except TranspileError as exc:
        print(f"[Omage] Compile error: {exc}")
        sys.exit(1)


def cmd_zoo_list(args: argparse.Namespace) -> None:
    import omage as og
    og.list_models(filter_type=getattr(args, "type", None))


def cmd_version(args: argparse.Namespace) -> None:
    import omage as og
    print(f"Omage v{og.__version__}")
    print(f"Device: {og.get_device()}")
    print(f"GPU:    {'Yes' if og.is_gpu() else 'No'}")


def cmd_train(args: argparse.Namespace) -> None:
    print(f"[Omage] Training: {args.file}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print("[Omage] Run your training script directly with: omage run <script.py>")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Omage — AI-First Python Library",
        prog="omage",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    subparsers = parser.add_subparsers(dest="command")

    # run
    p_run = subparsers.add_parser("run", help="Run a Python file")
    p_run.add_argument("file", help="Python file to run")

    # compile
    p_compile = subparsers.add_parser("compile", help="Compile .omg to .py")
    p_compile.add_argument("file", help=".omg source file")
    p_compile.add_argument("--output", "-o", help="Output .py file path")

    # zoo
    p_zoo = subparsers.add_parser("zoo", help="Model Zoo commands")
    zoo_sub = p_zoo.add_subparsers(dest="zoo_command")
    p_zoo_list = zoo_sub.add_parser("list", help="List available models")
    p_zoo_list.add_argument("--type", help="Filter by model type")

    # train
    p_train = subparsers.add_parser("train", help="Train a model")
    p_train.add_argument("file", help="Training script")
    p_train.add_argument("--epochs", type=int, default=10)
    p_train.add_argument("--batch-size", type=int, default=32)

    # version
    subparsers.add_parser("version", help="Show version info")

    args = parser.parse_args()
    _setup_logging(getattr(args, "verbose", False))

    dispatch = {
        "run":     cmd_run,
        "compile": cmd_compile,
        "zoo":     lambda a: cmd_zoo_list(a) if getattr(a, "zoo_command") == "list" else p_zoo.print_help(),
        "train":   cmd_train,
        "version": cmd_version,
    }

    if args.command in dispatch:
        dispatch[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
