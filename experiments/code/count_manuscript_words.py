#!/usr/bin/env python3
"""Generate the declared main-text word count with a pinned convention."""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
from pathlib import Path


STUDY = Path(__file__).resolve().parents[2]
PAPER = STUDY / "paper"
FILES = (
    "sections/abstract.tex",
    "sections/introduction.tex",
    "sections/related-work.tex",
    "sections/method.tex",
    "sections/experimental-setup.tex",
    "sections/results.tex",
    "sections/discussion.tex",
    "sections/limitations.tex",
    "sections/conclusion.tex",
)
OUTPUT = STUDY / "experiments/derived/results/manuscript_word_count.json"
TOKEN_PATTERN = r"\b(?:[A-Za-z0-9]+(?:[-'’][A-Za-z0-9]+)*)\b"


def main() -> None:
    version = subprocess.check_output(
        ["pandoc", "--version"], text=True
    ).splitlines()[0]
    with tempfile.NamedTemporaryFile(suffix=".txt") as destination:
        subprocess.run(
            [
                "pandoc",
                *FILES,
                "-f",
                "latex",
                "-t",
                "plain",
                "-o",
                destination.name,
            ],
            cwd=PAPER,
            check=True,
        )
        plain = Path(destination.name).read_text()
    count = len(re.findall(TOKEN_PATTERN, plain))
    main_tex = (PAPER / "main.tex").read_text()
    printed = re.search(r"\\textbf\{Word count:\}\s*([0-9,]+)", main_tex)
    if printed is None:
        raise RuntimeError("main.tex has no printed word count")
    printed_count = int(printed.group(1).replace(",", ""))
    if printed_count != count:
        raise RuntimeError(
            f"printed word count {printed_count} != generated count {count}"
        )
    payload = {
        "schema_version": 1,
        "count": count,
        "scope": (
            "abstract through conclusion; excludes captions, tables, "
            "references, and declarations"
        ),
        "source_files": list(FILES),
        "converter": version,
        "converter_command": (
            "pandoc <source_files> -f latex -t plain -o <temporary.txt>"
        ),
        "token_pattern": TOKEN_PATTERN,
        "printed_count_matches": True,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"PASS: generated and printed word count = {count}")


if __name__ == "__main__":
    main()
