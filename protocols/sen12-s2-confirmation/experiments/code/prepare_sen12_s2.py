#!/usr/bin/env python3
"""Download, index, and guarded-extract frozen Sen12 Sentinel-2 files."""

from __future__ import annotations

import argparse
import datetime as dt
import email.utils
import hashlib
import json
import subprocess
import tarfile
import time
import urllib.request
from collections import defaultdict
from pathlib import Path


STUDY = Path(__file__).resolve().parents[2]
METADATA = STUDY / "research/dataset-metadata/sen12-s2-confirmation"
ARCHIVE_MANIFEST = METADATA / "archive_manifest.json"
MEMBER_INDEX = METADATA / "member_index.json"
TASK_MEMBERSHIP = METADATA / "s12ls_ld_s2_membership.json"
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
UPSTREAM_CODE_REVISION = "d26a25edc8e0b69550696cfb97bb5a983eaa2fde"
PUBLIC_REPOSITORY = "akssha74/S3Net-Landslide"
PUBLIC_PROTOCOL_PREFIX = "protocols/sen12-s2-confirmation"
PUBLIC_AUTHORIZATION_PATH = (
    f"{PUBLIC_PROTOCOL_PREFIX}/experiments/derived/results/"
    "sen12_s2_confirmation/protected_access_authorization.json"
)
TASK_MEMBERSHIP_SHA256 = (
    "9d538889e86cd2e1c4c61bb7bf201ecd86765c6428b9c87833f7341fdddbec0e"
)
BASE_URL = "https://huggingface.co/datasets/paulhoehn/Sen12Landslides/resolve"
MAX_PER_INVENTORY = 256


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def public_push_event(commit: str) -> dict | None:
    if len(commit) != 40 or any(value not in "0123456789abcdef" for value in commit):
        raise RuntimeError("public commit must be a full lowercase SHA")
    for page in range(1, 3):
        url = (
            f"https://api.github.com/repos/{PUBLIC_REPOSITORY}/events"
            f"?per_page=100&page={page}"
        )
        with urllib.request.urlopen(url) as response:
            events = json.load(response)
        for event in events:
            payload = event.get("payload", {})
            commit_ids = [row.get("sha") for row in payload.get("commits", [])]
            if event.get("type") == "PushEvent" and (
                payload.get("head") == commit or commit in commit_ids
            ):
                return event
        if not events:
            break
    return None


def verify_public_sources(commit: str, expected: dict[str, str]) -> None:
    for relative, expected_hash in expected.items():
        public_path = f"{PUBLIC_PROTOCOL_PREFIX}/{relative}"
        url = (
            f"https://raw.githubusercontent.com/{PUBLIC_REPOSITORY}/"
            f"{commit}/{public_path}"
        )
        with urllib.request.urlopen(url) as response:
            actual_hash = hashlib.sha256(response.read()).hexdigest()
        if actual_hash != expected_hash:
            raise RuntimeError(f"public source mismatch: {public_path}")


def inventories() -> tuple[list[str], list[str]]:
    audit = json.loads(ACCESS_AUDIT.read_text())
    development = (
        audit["development_train_inventories"]
        + audit["development_validation_inventories"]
    )
    protected = audit["untouched_protected_inventories"]
    if set(development) & set(protected):
        raise RuntimeError("development and protected inventories overlap")
    if len(set(development + protected)) != 13:
        raise RuntimeError("expected exactly 13 disjoint inventory identifiers")
    return development, protected


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
                "--http1.1",
                "--retry",
                "8",
                "--retry-all-errors",
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
    if sha256(TASK_MEMBERSHIP) != TASK_MEMBERSHIP_SHA256:
        raise RuntimeError("official S12LS-LD task membership hash mismatch")
    task_rows = json.loads(TASK_MEMBERSHIP.read_text())["members"]
    if len(task_rows) != 4988:
        raise RuntimeError("official S12LS-LD task membership count mismatch")
    task_membership = {
        row["filename"]: row["inventory"]
        for row in task_rows
        if row["inventory"] in allowed
    }
    if len(task_membership) != len(task_rows):
        raise RuntimeError("official task membership contains duplicates or inventories")
    records = []
    for row in manifest["archives"]:
        archive = archive_dir / Path(row["path"]).name
        if not archive.exists():
            raise FileNotFoundError(archive)
        if archive.stat().st_size != row["size"] or sha256(archive) != row["sha256"]:
            raise RuntimeError(f"archive identity mismatch: {archive}")
        with tarfile.open(archive, "r:gz") as payload:
            for member in payload:
                if not member.isfile() or not member.name.endswith(".nc"):
                    continue
                filename = Path(member.name).name
                if "_s2_" not in filename:
                    continue
                inventory = task_membership.get(filename)
                if inventory is None or inventory not in allowed:
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
    observed = {row["filename"] for row in records}
    missing = sorted(set(task_membership) - observed)
    if missing:
        raise RuntimeError(
            f"{len(missing)} official S12LS-LD files absent from archives"
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
        "upstream_code_revision": UPSTREAM_CODE_REVISION,
        "task_membership_sha256": TASK_MEMBERSHIP_SHA256,
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
        f"https://api.github.com/repos/{PUBLIC_REPOSITORY}/commits/"
        + public_commit
    ) as response:
        observed = json.load(response)
        current_observed_at = email.utils.parsedate_to_datetime(
            response.headers["Date"]
        ).astimezone(dt.timezone.utc)
    if observed.get("sha") != public_commit:
        raise RuntimeError("public authorization commit is not resolvable")
    push_event = public_push_event(public_commit)
    if push_event is not None and receipt.get(
        "public_authorization_push_event_id"
    ) not in (None, push_event["id"]):
        raise RuntimeError("public authorization push evidence mismatch")
    protocol_observed_at = dt.datetime.fromisoformat(
        authorization["public_protocol_api_observed_at"].replace("Z", "+00:00")
    )
    authorized_at = dt.datetime.fromisoformat(
        authorization["authorized_at"].replace("Z", "+00:00")
    )
    receipt_observed_at = dt.datetime.fromisoformat(
        receipt["public_authorization_api_observed_at"].replace("Z", "+00:00")
    )
    if not (
        protocol_observed_at
        < authorized_at
        < receipt_observed_at
        <= current_observed_at
    ):
        raise RuntimeError("public protocol/authorization chronology is invalid")
    public_authorization_url = (
        f"https://raw.githubusercontent.com/{PUBLIC_REPOSITORY}/"
        f"{public_commit}/{PUBLIC_AUTHORIZATION_PATH}"
    )
    with urllib.request.urlopen(public_authorization_url) as response:
        public_authorization_hash = hashlib.sha256(response.read()).hexdigest()
    if public_authorization_hash != receipt["authorization_sha256"]:
        raise RuntimeError("public authorization artifact hash mismatch")
    fit_decisions = STUDY / (
        "experiments/derived/results/sen12_s2_confirmation/fit_decisions.json"
    )
    if authorization.get("fit_decisions_sha256") != sha256(fit_decisions):
        raise RuntimeError("fit decisions changed after authorization")
    development_manifest_path = METADATA / "development_extraction_manifest.json"
    if authorization.get("development_extraction_manifest_sha256") != sha256(
        development_manifest_path
    ):
        raise RuntimeError("development extraction manifest changed")
    development_manifest = json.loads(development_manifest_path.read_text())
    for row in development_manifest["files"]:
        path = STUDY / row["path"]
        if path.stat().st_size != row["size"] or sha256(path) != row["sha256"]:
            raise RuntimeError(f"development file changed: {path}")
    for relative, expected_hash in authorization["source_artifacts"].items():
        path = STUDY / relative
        if sha256(path) != expected_hash:
            raise RuntimeError(f"source artifact changed: {path}")
    verify_public_sources(
        authorization["public_protocol_commit"],
        authorization["source_artifacts"],
    )
    for checkpoint in authorization["checkpoints"]:
        path = STUDY / checkpoint["path"]
        if sha256(path) != checkpoint["sha256"]:
            raise RuntimeError(f"checkpoint changed: {path}")
        prediction_path = STUDY / checkpoint["validation_predictions"]
        if sha256(prediction_path) != checkpoint["validation_predictions_sha256"]:
            raise RuntimeError(f"validation predictions changed: {prediction_path}")


def extract(archive_dir: Path, output_dir: Path, fold: str) -> None:
    archive_dir = archive_dir.resolve()
    output_dir = output_dir.resolve()
    development, protected = inventories()
    targets = set(development if fold == "development" else protected)
    if fold == "protected":
        verify_authorization()
    payload = json.loads(MEMBER_INDEX.read_text())
    selected = [row for row in payload["selected"] if row["inventory"] in targets]
    archive_identities = {
        Path(row["path"]).name: row
        for row in json.loads(ARCHIVE_MANIFEST.read_text())["archives"]
    }
    by_archive: dict[str, list[dict]] = defaultdict(list)
    for row in selected:
        by_archive[row["archive"]].append(row)
    extraction_metadata = METADATA / f"{fold}_extraction_manifest.json"
    if extraction_metadata.exists():
        raise RuntimeError(f"{fold} extraction manifest already exists")
    if fold == "protected":
        access_started = METADATA / "protected_access_started.json"
        if access_started.exists():
            started = json.loads(access_started.read_text())
            if (
                started["authorization_sha256"] != sha256(AUTHORIZATION)
                or started["selected_count"] != len(selected)
                or started["inventories"] != sorted(targets)
            ):
                raise RuntimeError("protected access-start identity mismatch")
        else:
            access_started.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "started_at": time.strftime(
                            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                        ),
                        "authorization_sha256": sha256(AUTHORIZATION),
                        "public_authorization_receipt_sha256": sha256(
                            AUTHORIZATION_RECEIPT
                        ),
                        "selected_count": len(selected),
                        "inventories": sorted(targets),
                    },
                    indent=2,
                )
                + "\n"
            )
    extracted = []
    for archive_name, rows in sorted(by_archive.items()):
        archive_path = archive_dir / archive_name
        identity = archive_identities[archive_name]
        if (
            archive_path.stat().st_size != identity["size"]
            or sha256(archive_path) != identity["sha256"]
        ):
            raise RuntimeError(f"archive identity mismatch: {archive_path}")
        wanted = {row["member"]: row for row in rows}
        with tarfile.open(archive_path, "r:gz") as archive:
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
                content = source.read()
                if len(content) != row["size"]:
                    raise RuntimeError(f"member size mismatch: {member.name}")
                destination.write_bytes(content)
                extracted.append(
                    {
                        **row,
                        "path": str(destination.relative_to(STUDY)),
                        "sha256": hashlib.sha256(content).hexdigest(),
                    }
                )
    if len(extracted) != len(selected):
        raise RuntimeError(
            f"extracted {len(extracted)} files but expected {len(selected)}"
        )
    extraction_metadata.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "fold": fold,
                "completed_at": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                ),
                "dataset_revision": REVISION,
                "member_index_sha256": sha256(MEMBER_INDEX),
                "authorization_sha256": (
                    sha256(AUTHORIZATION) if fold == "protected" else None
                ),
                "files": extracted,
            },
            indent=2,
        )
        + "\n"
    )
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
