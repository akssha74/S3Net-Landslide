#!/usr/bin/env python3
"""Freeze the installed transitive environment with RECORD-file hashes."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import re
import sys
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


STUDY = Path(__file__).resolve().parents[2]
DIRECT = STUDY / "environment/requirements-lock.txt"
OUTPUT = STUDY / "environment/environment-freeze.json"


def record_hash(distribution: importlib.metadata.Distribution) -> str | None:
    for file in distribution.files or []:
        if str(file).endswith(".dist-info/RECORD"):
            path = distribution.locate_file(file)
            if path.is_file():
                return hashlib.sha256(path.read_bytes()).hexdigest()
    return None


def main() -> None:
    direct = []
    for line in DIRECT.read_text().splitlines():
        match = re.match(r"^([A-Za-z0-9_.-]+)==(.+)$", line.strip())
        if match:
            direct.append(canonicalize_name(match.group(1)))
    distributions = {
        canonicalize_name(dist.metadata["Name"]): dist
        for dist in importlib.metadata.distributions()
        if dist.metadata.get("Name")
    }
    queue = list(direct)
    selected: set[str] = set()
    while queue:
        name = queue.pop()
        if name in selected:
            continue
        distribution = distributions.get(name)
        if distribution is None:
            raise RuntimeError(f"Locked distribution not installed: {name}")
        selected.add(name)
        for requirement_text in distribution.requires or []:
            requirement = Requirement(requirement_text)
            if requirement.marker and not requirement.marker.evaluate(
                {"extra": ""}
            ):
                continue
            dependency = canonicalize_name(requirement.name)
            if dependency in distributions:
                queue.append(dependency)
    records = []
    for name in sorted(selected):
        distribution = distributions[name]
        records.append(
            {
                "name": name,
                "version": distribution.version,
                "record_sha256": record_hash(distribution),
            }
        )
    payload = {
        "schema_version": 1,
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "direct_requirements_sha256": hashlib.sha256(
            DIRECT.read_bytes()
        ).hexdigest(),
        "direct_distributions": sorted(direct),
        "transitive_distribution_count": len(records),
        "distributions": records,
        "scope": (
            "Installed macOS arm64 environment; RECORD hashes bind installed "
            "files and complement exact direct version pins."
        ),
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {OUTPUT} with {len(records)} distributions")


if __name__ == "__main__":
    main()
