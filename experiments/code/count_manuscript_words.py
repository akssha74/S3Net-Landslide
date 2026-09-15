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
EXCLUDED_ENVIRONMENTS = ("table", "table*", "figure", "figure*", "equation", "align")


def main() -> None:
    version = subprocess.check_output(
        ["pandoc", "--version"], text=True
    ).splitlines()[0]
    with tempfile.TemporaryDirectory() as temporary:
        temporary_root = Path(temporary)
        sanitized_sources = []
        for relative in FILES:
            text = (PAPER / relative).read_text()
            for environment in EXCLUDED_ENVIRONMENTS:
                text = re.sub(
                    rf"\\begin\{{{re.escape(environment)}\}}.*?"
                    rf"\\end\{{{re.escape(environment)}\}}",
                    "",
                    text,
                    flags=re.DOTALL,
                )
            destination = temporary_root / Path(relative).name
            destination.write_text(text)
            sanitized_sources.append(str(destination))
        plain_path = temporary_root / "manuscript.txt"
        subprocess.run(
            [
                "pandoc",
                *sanitized_sources,
                "-f",
                "latex",
                "-t",
                "plain",
                "-o",
                str(plain_path),
            ],
            check=True,
        )
        plain = plain_path.read_text()
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
            "abstract through conclusion; excludes table/figure environments, "
            "displayed equations, references, and declarations"
        ),
        "source_files": list(FILES),
        "converter": version,
        "converter_command": (
            "pandoc <source_files> -f latex -t plain -o <temporary.txt>"
        ),
        "token_pattern": TOKEN_PATTERN,
        "excluded_environments": list(EXCLUDED_ENVIRONMENTS),
        "printed_count_matches": True,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"PASS: generated and printed word count = {count}")


if __name__ == "__main__":
    main()
