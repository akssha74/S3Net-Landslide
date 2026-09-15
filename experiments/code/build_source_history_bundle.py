#!/usr/bin/env python3
"""Build a standalone two-commit Git bundle for protocol/source provenance."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

from build_scientific_source_snapshot import FILES, ROOTS, STUDY_PREFIX, git


PROTOCOL_PATHS = (
    "research/lrd-boundary-confirmation-preregistration.md",
    "research/dataset-metadata/lrd-prospective-confirmation/"
    "protected_event_folds.json",
    "experiments/code/fetch_lrd_optical_folds.py",
    "experiments/code/run_lrd_prospective_confirmation.py",
)


def commit(
    repository: Path, message: str, date: str
) -> str:
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "Scientific Snapshot",
        "GIT_AUTHOR_EMAIL": "snapshot@example.invalid",
        "GIT_COMMITTER_NAME": "Scientific Snapshot",
        "GIT_COMMITTER_EMAIL": "snapshot@example.invalid",
        "GIT_AUTHOR_DATE": date,
        "GIT_COMMITTER_DATE": date,
    }
    subprocess.run(["git", "-C", str(repository), "add", "-A"], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "commit", "-m", message],
        check=True,
        env=env,
        stdout=subprocess.DEVNULL,
    )
    return subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()


def write_commit_file(
    source_repo: Path,
    source_commit: str,
    relative: str,
    destination_root: Path,
) -> None:
    full = f"{STUDY_PREFIX}/{relative}"
    data = bytes(git(source_repo, "show", f"{source_commit}:{full}", text=False))
    destination = destination_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--protocol-commit", required=True)
    parser.add_argument("--scientific-commit", required=True)
    parser.add_argument("--scientific-tree-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_repo = args.repo.expanduser().resolve()
    protocol_commit = str(
        git(source_repo, "rev-parse", f"{args.protocol_commit}^{{commit}}")
    ).strip()
    scientific_commit = str(
        git(source_repo, "rev-parse", f"{args.scientific_commit}^{{commit}}")
    ).strip()
    protocol_date = str(
        git(source_repo, "show", "-s", "--format=%aI", protocol_commit)
    ).strip()
    scientific_date = str(
        git(source_repo, "show", "-s", "--format=%aI", scientific_commit)
    ).strip()

    with tempfile.TemporaryDirectory() as temporary:
        history = Path(temporary) / "history"
        history.mkdir()
        subprocess.run(["git", "-C", str(history), "init", "-q"], check=True)
        for relative in PROTOCOL_PATHS:
            write_commit_file(
                source_repo, protocol_commit, relative, history
            )
        (history / "PROVENANCE.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "stage": "local-protocol-snapshot",
                    "original_commit": protocol_commit,
                    "original_author_date": protocol_date,
                    "public_timestamp_caveat": (
                        "This history bundle was published after evaluation and "
                        "does not create an independent pre-access timestamp."
                    ),
                },
                indent=2,
            )
            + "\n"
        )
        public_protocol_commit = commit(
            history, "Protocol snapshot (retrospective publication)", protocol_date
        )

        selectors = [
            f"{STUDY_PREFIX}/{path}" for path in (*ROOTS, *FILES)
        ]
        listed = str(
            git(
                source_repo,
                "ls-tree",
                "-r",
                "--name-only",
                scientific_commit,
                "--",
                *selectors,
            )
        ).splitlines()
        for full_path in listed:
            relative = full_path.removeprefix(f"{STUDY_PREFIX}/")
            write_commit_file(
                source_repo, scientific_commit, relative, history
            )
        (history / "PROVENANCE.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "stage": "scientific-source-snapshot",
                    "original_scientific_commit": scientific_commit,
                    "original_scientific_tree_sha256": args.scientific_tree_sha256,
                    "protocol_snapshot_commit": public_protocol_commit,
                    "public_timestamp_caveat": (
                        "The bundle proves content and internal ancestry, not "
                        "third-party pre-access publication time."
                    ),
                },
                indent=2,
            )
            + "\n"
        )
        public_scientific_commit = commit(
            history, "Final scientific source snapshot", scientific_date
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(history),
                "bundle",
                "create",
                str(args.output.resolve()),
                "--all",
            ],
            check=True,
        )
    print(
        json.dumps(
            {
                "bundle": str(args.output),
                "original_protocol_commit": protocol_commit,
                "original_scientific_commit": scientific_commit,
                "bundle_protocol_commit": public_protocol_commit,
                "bundle_scientific_commit": public_scientific_commit,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
