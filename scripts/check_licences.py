#!/usr/bin/env python3
"""Licence gate: nothing copyleft, non-commercial or custom arrives unannounced.

Three checks, each against NOTICE:

1. Locks. Every entry in vendor/sources.lock.json and vendor/weights.lock.json
   has a licence, and NOTICE names both the component (its repository, or the
   weight set's id when it has none) and that licence.
2. Python. Every distribution installed in the running environment whose
   licence metadata reads as GPL, LGPL, AGPL, non-commercial or custom is named
   in NOTICE. Run it with the engine's environment's interpreter; CI installs
   ".[test,meshfix]".
3. npm. The same for every package in frontend/package-lock.json and
   vendor/mesh2motion-app/package-lock.json.

A licence expression that can be satisfied without a flagged licence passes:
"MIT OR GPL-3.0" does, "GPL-3.0 AND (MIT OR Apache-2.0)" does not. A package
with no licence metadata is listed as a warning and does not fail the gate.

    python scripts/check_licences.py [--root DIR] [--skip-python] [--skip-npm]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FLAGGED = re.compile(
    r"\b(A|L)?GPL|General Public Licen[cs]e|Affero|\bSSPL\b|\bEUPL\b"
    r"|non-?commercial|\bCC-BY-NC|Commons Clause|\bBUSL\b|Business Source"
    r"|LicenseRef-|proprietary",
    re.IGNORECASE,
)
TOKEN = re.compile(r"\(|\)|[^\s()]+")
# Platform packages that NOTICE names by their parent package.
FAMILIES = (("@img/sharp-", "sharp"),)


def _acceptable(tokens: list[str]) -> bool:
    """Evaluate an SPDX-style expression: a licence is acceptable when it is
    not flagged, AND needs every side acceptable, OR needs one. `X WITH
    exception` is judged by X. Raises ValueError on anything else."""
    position = 0

    def peek() -> str | None:
        return tokens[position] if position < len(tokens) else None

    def take() -> str:
        nonlocal position
        position += 1
        return tokens[position - 1]

    def factor() -> bool:
        token = peek()
        if token is None or token.upper() in {"AND", "OR", "WITH", ")"}:
            raise ValueError("expected a licence")
        if token == "(":
            take()
            value = disjunction()
            if peek() != ")":
                raise ValueError("unbalanced parenthesis")
            take()
            return value
        value = not FLAGGED.search(take())
        if (peek() or "").upper() == "WITH":
            take()
            if peek() is None or peek() in {"(", ")"}:
                raise ValueError("expected an exception")
            take()
        return value

    def conjunction() -> bool:
        value = factor()
        while (peek() or "").upper() == "AND":
            take()
            value = factor() and value
        return value

    def disjunction() -> bool:
        value = conjunction()
        while (peek() or "").upper() == "OR":
            take()
            value = conjunction() or value
        return value

    value = disjunction()
    if position != len(tokens):
        raise ValueError("trailing tokens")
    return value


def flagged(expression: str) -> bool:
    """True when the licence expression cannot be satisfied without a flagged
    licence. SPDX-style expressions are parsed (AND binds tighter than OR, and
    `A/B` reads as `A OR B`), so `GPL-3.0 AND (MIT OR Apache-2.0)` is flagged
    while `(MIT OR GPL-3.0)` is not. Free text that does not parse, such as
    "GNU General Public License v3 (GPLv3)", is flagged when it mentions a
    flagged licence anywhere."""
    text = expression.replace("/", " OR ").strip()
    if not text:
        return False
    try:
        return not _acceptable(TOKEN.findall(text))
    except ValueError:
        return FLAGGED.search(expression) is not None


def named_in(notice: str, name: str) -> bool:
    for prefix, parent in FAMILIES:
        if name.startswith(prefix):
            name = parent
    wanted = name.lower().replace("_", "-")
    text = notice.lower().replace("_", "-")
    return re.search(rf"(?<![\w-]){re.escape(wanted)}(?![\w-])", text) is not None


def check_locks(root: Path, notice: str) -> list[str]:
    problems = []
    sources = json.loads((root / "vendor" / "sources.lock.json").read_text(encoding="utf-8"))["sources"]
    for source in sources:
        repository = re.sub(r"^https://github\.com/|\.git$", "", source["repository"])
        licence = source.get("licence") or ""
        if not licence:
            problems.append(f"sources lock: {source['name']} has no licence")
        elif licence not in notice:
            problems.append(f"sources lock: {source['name']}'s licence {licence!r} is not in NOTICE")
        if repository not in notice:
            problems.append(f"sources lock: {source['name']} ({repository}) is not named in NOTICE")
    weights = json.loads((root / "vendor" / "weights.lock.json").read_text(encoding="utf-8"))["weights"]
    for weight in weights:
        name = weight.get("repository") or weight["id"]
        licence = (weight.get("licence") or "").split(" (", 1)[0]
        if not licence:
            problems.append(f"weights lock: {weight['id']} has no licence")
        elif licence not in notice:
            problems.append(f"weights lock: {weight['id']}'s licence {licence!r} is not in NOTICE")
        if name not in notice:
            problems.append(f"weights lock: {weight['id']} ({name}) is not named in NOTICE")
    return problems


def python_licence(distribution: metadata.Distribution) -> tuple[str, bool]:
    """The distribution's licence as text, and whether it is flagged. Several
    licence classifiers are alternatives (dual licensing): flagged only when
    every one of them is."""
    meta = distribution.metadata
    expression = meta.get("License-Expression")
    if expression:
        return expression, flagged(expression)
    classifiers = [c.split(" :: ")[-1] for c in meta.get_all("Classifier") or [] if c.startswith("License :: ")]
    if classifiers:
        return "; ".join(classifiers), all(FLAGGED.search(c) for c in classifiers)
    text = (meta.get("License") or "").strip()
    first = text.splitlines()[0][:200] if text else ""
    return first, bool(first) and flagged(first)


def check_python(notice: str) -> tuple[list[str], list[str]]:
    problems, unknown = [], []
    seen = set()
    for distribution in metadata.distributions():
        name = distribution.metadata.get("Name") or ""
        if not name or name.lower() in seen:
            continue
        seen.add(name.lower())
        licence, is_flagged = python_licence(distribution)
        if not licence or licence.upper() == "UNKNOWN":
            unknown.append(f"python: {name}")
        elif is_flagged and not named_in(notice, name):
            problems.append(f"python: {name} is {licence!r} and is not named in NOTICE")
    return problems, unknown


def check_npm(root: Path, notice: str) -> tuple[list[str], list[str]]:
    problems, unknown = [], []
    for lock in ("frontend/package-lock.json", "vendor/mesh2motion-app/package-lock.json"):
        path = root / lock
        if not path.is_file():
            problems.append(f"npm: {lock} is missing")
            continue
        packages = json.loads(path.read_text(encoding="utf-8")).get("packages", {})
        for key, package in packages.items():
            if not key:
                continue  # the project itself
            name = package.get("name") or key.rsplit("node_modules/", 1)[-1]
            licence = package.get("license")
            if isinstance(licence, dict):
                licence = licence.get("type")
            if not licence:
                if not package.get("link"):
                    unknown.append(f"npm ({lock}): {name}")
            elif flagged(str(licence)) and not named_in(notice, name):
                problems.append(f"npm ({lock}): {name} is {licence!r} and is not named in NOTICE")
    return problems, unknown


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--skip-python", action="store_true")
    parser.add_argument("--skip-npm", action="store_true")
    args = parser.parse_args(argv)

    notice = (args.root / "NOTICE").read_text(encoding="utf-8")
    problems = check_locks(args.root, notice)
    unknown: list[str] = []
    if not args.skip_python:
        found, missing = check_python(notice)
        problems += found
        unknown += missing
    if not args.skip_npm:
        found, missing = check_npm(args.root, notice)
        problems += found
        unknown += missing
    for item in sorted(set(unknown)):
        print(f"warning: no licence metadata: {item}")
    for problem in problems:
        print(f"problem: {problem}")
    print(f"licence gate: {len(problems)} problems, {len(set(unknown))} packages without licence metadata")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
