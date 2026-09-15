#!/usr/bin/env python3
"""Download, index, and guarded-extract frozen Sen12 Sentinel-2 files."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tarfile
import urllib.request
from collections import defaultdict
from pathlib import Path


STUDY = Path(__file__).resolve().parents[2]
METADATA = STUDY / "research/dataset-metadata/sen12-s2-confirmation"
ARCHIVE_MANIFEST = METADATA / "archive_manifest.json"
MEMBER_INDEX = METADATA / "member_index.json"
ACCESS_AUDIT = STUDY / "research/sen12-s2-access-audit.json"
AUTHORIZATION = (
    STUDY
    / "experiments/derived/results/sen12_s2_confirmation/"
    "protected_access_authorization.json"
)
AUTHORIZATION_RECEIPT = (
    STUDY
    / "research/dataset-metadata/sen12-s2-confirmation/"
    "public_authorization_receipt.json"
)
REVISION = "40af2dd6b4e568edb6640d6e14dc67ebd01038a4"
BASE_URL = "https://huggingface.co/datasets/paulhoehn/Sen12Landslides/resolve"
MAX_PER_INVENTORY = 256


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def inventories() -> tuple[list[str], list[str]]:
    audit = json.loads(ACCESS_AUDIT.read_text())
    development = (
        audit["development_train_inventories"]
        + audit["development_validation_inventories"]
    )
    return development, audit["untouched_protected_inventories"]


def download(archive_dir: Path) -> None:
    archive_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(ARCHIVE_MANIFEST.read_text())
    for row in manifest["archives"]:
        destination = archive_dir / Path(row["path"]).name
        if destination.exists() and destination.stat().st_size == row["size"]:
            if sha256(destination) == row["sha256"]:
                print(f"verified {destination.name}")
                continue
        url = f"{BASE_URL}/{REVISION}/{row['path']}?download=true"
        subprocess.run(
            [
                "curl",
                "-fL",
                "--retry",
                "8",
                "--retry-delay",
                "5",
                "-C",
                "-",
                "-o",
                str(destination),
                url,
            ],
            check=True,
        )
        actual = sha256(destination)
        if actual != row["sha256"]:
            raise RuntimeError(
                f"{destination}: sha256 {actual} != {row['sha256']}"
            )
        print(f"downloaded {destination.name}")


def index(archive_dir: Path) -> None:
    manifest = json.loads(ARCHIVE_MANIFEST.read_text())
    development, protected = inventories()
    allowed = set(development + protected)
    records = []
    for row in manifest["archives"]:
        archive = archive_dir / Path(row["path"]).name
        if not archive.exists():
            raise FileNotFoundError(archive)
        with tarfile.open(archive, "r:gz") as payload:
            for member in payload:
                if not member.isfile() or not member.name.endswith(".nc"):
                    continue
                filename = Path(member.name).name
                if "_s2_" not in filename:
                    continue
                inventory = filename.split("_s2_", 1)[0]
                if inventory not in allowed:
                    continue
                records.append(
                    {
                        "archive": archive.name,
                        "member": member.name,
                        "filename": filename,
                        "inventory": inventory,
                        "size": member.size,
                    }
                )
    selected = []
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        grouped[row["inventory"]].append(row)
    for inventory, rows in sorted(grouped.items()):
        rows.sort(
            key=lambda row: hashlib.sha256(
                f"sen12-s2-confirmation-v1:{row['filename']}".encode()
            ).hexdigest()
        )
        selected.extend(rows[:MAX_PER_INVENTORY])
    output = {
        "schema_version": 1,
        "dataset_revision": REVISION,
        "selection_salt": "sen12-s2-confirmation-v1",
        "maximum_per_inventory": MAX_PER_INVENTORY,
        "records_seen": len(records),
        "selected": selected,
        "selected_counts": {
            key: sum(row["inventory"] == key for row in selected)
            for key in sorted(grouped)
        },
    }
    MEMBER_INDEX.parent.mkdir(parents=True, exist_ok=True)
    MEMBER_INDEX.write_text(json.dumps(output, indent=2) + "\n")
    print(f"indexed {len(records)} files; selected {len(selected)}")


def verify_authorization() -> None:
    if not AUTHORIZATION.exists():
        raise RuntimeError("protected authorization does not exist")
    authorization = json.loads(AUTHORIZATION.read_text())
    if authorization.get("status") != "authorized":
        raise RuntimeError("protected authorization is not authorized")
    if authorization.get("dataset_revision") != REVISION:
        raise RuntimeError("protected authorization dataset revision mismatch")
    if authorization.get("member_index_sha256") != sha256(MEMBER_INDEX):
        raise RuntimeError("protected authorization member-index mismatch")
    if not AUTHORIZATION_RECEIPT.exists():
        raise RuntimeError("public authorization receipt does not exist")
    receipt = json.loads(AUTHORIZATION_RECEIPT.read_text())
    if receipt.get("authorization_sha256") != sha256(AUTHORIZATION):
        raise RuntimeError("public authorization receipt hash mismatch")
    public_commit = receipt.get("public_authorization_commit", "")
    with urllib.request.urlopen(
        "https://api.github.com/repos/akssha74/S3Net-Landslide/commits/"
        + public_commit
    ) as response:
        observed = json.load(response)
    if observed.get("sha") != public_commit:
        raise RuntimeError("public authorization commit is not resolvable")
    for checkpoint in authorization["checkpoints"]:
        path = STUDY / checkpoint["path"]
        if sha256(path) != checkpoint["sha256"]:
            raise RuntimeError(f"checkpoint changed: {path}")


def extract(archive_dir: Path, output_dir: Path, fold: str) -> None:
    development, protected = inventories()
    targets = set(development if fold == "development" else protected)
    if fold == "protected":
        verify_authorization()
    payload = json.loads(MEMBER_INDEX.read_text())
    selected = [row for row in payload["selected"] if row["inventory"] in targets]
    by_archive: dict[str, list[dict]] = defaultdict(list)
    for row in selected:
        by_archive[row["archive"]].append(row)
    for archive_name, rows in sorted(by_archive.items()):
        wanted = {row["member"]: row for row in rows}
        with tarfile.open(archive_dir / archive_name, "r:gz") as archive:
            for member in archive:
                row = wanted.get(member.name)
                if row is None:
                    continue
                source = archive.extractfile(member)
                if source is None:
                    raise RuntimeError(f"cannot extract {member.name}")
                destination = (
                    output_dir / fold / row["inventory"] / row["filename"]
                )
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(source.read())
    print(f"extracted {len(selected)} {fold} files")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    download_parser = subparsers.add_parser("download")
    download_parser.add_argument("--archive-dir", type=Path, required=True)
    index_parser = subparsers.add_parser("index")
    index_parser.add_argument("--archive-dir", type=Path, required=True)
    extract_parser = subparsers.add_parser("extract")
    extract_parser.add_argument(
        "--fold", choices=("development", "protected"), required=True
    )
    extract_parser.add_argument("--archive-dir", type=Path, required=True)
    extract_parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "download":
        download(args.archive_dir)
    elif args.command == "index":
        index(args.archive_dir)
    else:
        extract(args.archive_dir, args.output_dir, args.fold)


if __name__ == "__main__":
    main()
