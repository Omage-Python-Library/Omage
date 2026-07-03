"""
Omage Transpiler — Main entry point
"""

from __future__ import annotations
import logging
from pathlib import Path

from .parser import parse_omage
from .generator import generate_python

logger = logging.getLogger("omage")


class TranspileError(Exception):
    pass


def transpile(code: str) -> str:
    """Transpile Omage source code to Python"""
    logger.info("Transpiling Omage code...")
    ast = parse_omage(code)
    python_code = generate_python(ast)
    logger.info("Transpile complete!")
    return python_code


def compile_file(input_path: str, output_path: str | None = None) -> str:
    """Compile a .omg file to a .py file  (FIX #1)"""
    in_path = Path(input_path)

    if not in_path.exists():
        raise TranspileError(f"Input file not found: '{input_path}'")
    if in_path.suffix != ".omg":
        raise TranspileError(f"Expected a .omg file, got: '{input_path}'")

    if output_path is None:
        output_path = str(in_path.with_suffix(".py"))

    logger.info(f"Compiling: {input_path}")

    try:
        omage_code = in_path.read_text(encoding="utf-8")
    except Exception as exc:
        raise TranspileError(f"Failed to read '{input_path}': {exc}") from exc

    python_code = transpile(omage_code)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        out_path.write_text(python_code, encoding="utf-8")   # FIX #1: was writing empty
    except Exception as exc:
        raise TranspileError(f"Failed to write '{output_path}': {exc}") from exc

    logger.info(f"Output written: {output_path}")
    return output_path
