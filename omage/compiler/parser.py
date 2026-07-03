"""
Omage Parser — Parse .omg syntax into AST
"""

from __future__ import annotations
import re
from typing import Any, Dict, List, Optional


class ParseError(Exception):
    def __init__(self, message: str, line_number: int = 0):
        self.line_number = line_number
        super().__init__(f"Line {line_number}: {message}" if line_number else message)


def _strip_comments(code: str) -> str:
    """Remove // line comments and @{...}@ block comments"""
    # Block comments first
    code = re.sub(r"@\{.*?\}@", "", code, flags=re.DOTALL)
    # Line comments
    code = re.sub(r"//[^\n]*", "", code)
    return code


def parse_omage(code: str) -> List[Dict[str, Any]]:
    """Parse Omage code and return AST node list"""
    code = _strip_comments(code)
    lines = code.split("\n")
    ast: List[Dict[str, Any]] = []
    i = 0

    while i < len(lines):
        raw = lines[i]
        line = raw.strip()
        lineno = i + 1
        i += 1

        if not line:
            continue

        # ---- make / lock variable ------------------------------------ #
        m = re.match(r"^(make|lock)\s+(\w+)\s*:\s*(\w+)\s*=\s*(.+)$", line)
        if m:
            ast.append({
                "type":      "variable",
                "kind":      m.group(1),       # make | lock
                "var_type":  m.group(2),       # number, text, model …
                "name":      m.group(3),
                "value":     m.group(4).strip(),
                "line":      lineno,
            })
            continue

        # ---- show(...) ----------------------------------------------- #
        m = re.match(r"^show\s*\((.+)\)$", line)
        if m:
            ast.append({"type": "show", "content": m.group(1).strip(), "line": lineno})
            continue

        # ---- name.train { ... } (single-line stub) ------------------- #
        m = re.match(r"^(\w+)\.train\b", line)
        if m:
            ast.append({"type": "train", "model": m.group(1), "config": line, "line": lineno})
            continue

        # ---- name.evaluate(...) -------------------------------------- #
        m = re.match(r"^(\w+)\.evaluate\s*\((\w+)\)", line)
        if m:
            ast.append({"type": "evaluate", "model": m.group(1), "data": m.group(2), "line": lineno})
            continue

        # ---- evaluate model on data ---------------------------------- #
        m = re.match(r"^evaluate\s+(\w+)\s+on\s+(\w+)$", line)
        if m:
            ast.append({"type": "evaluate", "model": m.group(1), "data": m.group(2), "line": lineno})
            continue

        # ---- save model --> "path" ----------------------------------- #
        m = re.match(r'^save\s+(\w+)\s*-->\s*"([^"]+)"', line)
        if m:
            ast.append({"type": "save", "model": m.group(1), "path": m.group(2), "line": lineno})
            continue

        # ---- model.save("path") ------------------------------------- #
        m = re.match(r'^(\w+)\.save\s*\("([^"]+)"\)', line)
        if m:
            ast.append({"type": "save", "model": m.group(1), "path": m.group(2), "line": lineno})
            continue

        # ---- result = model <-- "input" ----------------------------- #
        m = re.match(r'^(?:make\s+result\s*:\s*(\w+)\s*=\s*)?(\w+)\s*<--\s*(.+)$', line)
        if m:
            ast.append({
                "type":       "predict",
                "result_var": m.group(1),
                "model":      m.group(2),
                "input":      m.group(3).strip(),
                "line":       lineno,
            })
            continue

        # ---- anything else: emit as raw Python ----------------------- #
        ast.append({"type": "raw", "code": line, "line": lineno})

    return ast
