#!/usr/bin/env python3
"""Build a deterministic, publicly uploadable scientific-source snapshot."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path


STUDY_PREFIX = "studies/disaster-hrgldd-landslide"
ROOTS = (
    "paper",
    "experiments/code",
    "experiments/configs",
    "experiments/derived",
    "experiments/logs",
    "evidence",
    "environment",
    "research",
)
FILES = (
    "experiments/run-ledger.jsonl",
    "experiments/tree.jsonl",
    "study.json",
)


def git(repo: Path, *args: str, text: bool = True) -> str | bytes:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args],
        text=text,
        stderr=subprocess.STDOUT,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.expanduser().resolve()
    commit = str(git(repo, "rev-parse", f"{args.commit}^{{commit}}")).strip()
    selectors = [
        f"{STUDY_PREFIX}/{path}" for path in (*ROOTS, *FILES)
    ]
    listed = str(
        git(repo, "ls-tree", "-r", "--name-only", commit, "--", *selectors)
    ).splitlines()
    records = []
    payloads: list[tuple[str, bytes]] = []
    for full_path in sorted(set(listed)):
        data = bytes(git(repo, "show", f"{commit}:{full_path}", text=False))
        relative = full_path.removeprefix(f"{STUDY_PREFIX}/")
        payloads.append((relative, data))
        records.append(
            {
                "path": relative,
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    manifest = {
        "schema_version": 1,
        "source_commit": commit,
        "study_prefix": STUDY_PREFIX,
        "file_count": len(records),
        "files": records,
    }
    manifest_bytes = (json.dumps(manifest, indent=2) + "\n").encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                for name, data in [
                    ("SOURCE_SNAPSHOT_MANIFEST.json", manifest_bytes),
                    *payloads,
                ]:
                    info = tarfile.TarInfo(name)
                    info.size = len(data)
                    info.mtime = 0
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    info.mode = 0o644
                    archive.addfile(info, io.BytesIO(data))
    print(
        f"Wrote {args.output} with {len(records)} files; "
        f"sha256={hashlib.sha256(args.output.read_bytes()).hexdigest()}"
    )


if __name__ == "__main__":
    main()
