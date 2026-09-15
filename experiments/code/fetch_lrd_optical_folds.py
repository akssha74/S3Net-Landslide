#!/usr/bin/env python3
"""Range-fetch named Sentinel-2 RGB and masks for frozen LRD event folds."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import urllib.request
from pathlib import Path
from zipfile import ZipFile


STUDY = Path(__file__).resolve().parents[2]
METADATA = STUDY / "research/dataset-metadata/lrd-prospective-confirmation"
FOLDS = METADATA / "protected_event_folds.json"
OUTPUT = STUDY / "experiments/raw/external/lrd-optical"
URL = (
    "https://zenodo.org/api/records/17007637/files/"
    "s1s2_landslide_reference_data.zip/content"
)
BANDS = ("B02", "B03", "B04")


class HTTPRangeFile(io.RawIOBase):
    """Minimal seekable HTTP range reader for zipfile."""

    def __init__(self, url: str) -> None:
        self.url = url
        self.position = 0
        request = urllib.request.Request(
            url, method="HEAD", headers={"User-Agent": "paper-activities/1"}
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            self.length = int(response.headers["Content-Length"])

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self.position + offset
        elif whence == io.SEEK_END:
            position = self.length + offset
        else:
            raise ValueError(f"Unsupported whence: {whence}")
        if position < 0:
            raise ValueError("Negative seek")
        self.position = position
        return position

    def read(self, size: int = -1) -> bytes:
        if self.position >= self.length:
            return b""
        end = (
            self.length - 1
            if size is None or size < 0
            else min(self.length - 1, self.position + size - 1)
        )
        request = urllib.request.Request(
            self.url,
            headers={
                "Range": f"bytes={self.position}-{end}",
                "User-Agent": "paper-activities/1",
            },
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            if response.status != 206:
                raise RuntimeError(
                    f"Server ignored byte range: HTTP {response.status}"
                )
            data = response.read()
        expected = end - self.position + 1
        if len(data) != expected:
            raise RuntimeError(f"Range length {len(data)} != {expected}")
        self.position = end + 1
        return data


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def selected_names(eid: str) -> list[str]:
    metadata_path = METADATA / "archive-metadata" / f"{eid}__files_meta.csv"
    rows = list(csv.DictReader(metadata_path.open(), delimiter=";"))
    selected = []
    for band in BANDS:
        candidates = [
            row
            for row in rows
            if row["sensor"] == "S2L2A"
            and row["band"] == band
            and row["patch_type"] == "POST1"
        ]
        if len(candidates) != 1:
            raise RuntimeError(
                f"{eid} {band}: expected one POST1 row, got {len(candidates)}"
            )
        selected.append(candidates[0]["file_name"] + ".tif")
    masks = [row for row in rows if row["band"] == "MASK"]
    if len(masks) != 1:
        raise RuntimeError(f"{eid}: expected one mask, got {len(masks)}")
    selected.append(masks[0]["file_name"] + ".tif")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fold", choices=("development", "validation", "protected"), required=True
    )
    parser.add_argument(
        "--protected-authorization",
        type=Path,
        help="Required marker generated after all model/threshold decisions.",
    )
    args = parser.parse_args()
    folds = json.loads(FOLDS.read_text())
    key = {
        "development": "development_eids",
        "validation": "validation_eids",
        "protected": "protected_test_eids",
    }[args.fold]
    eids = list(folds[key])
    if args.fold == "protected":
        marker = args.protected_authorization
        if marker is None or not marker.is_file():
            raise RuntimeError("Protected fetch requires an authorization marker")
        authorization = json.loads(marker.read_text())
        if authorization.get("status") != "all-decisions-frozen":
            raise RuntimeError("Invalid protected-access authorization")

    records = []
    remote = HTTPRangeFile(URL)
    with ZipFile(remote) as archive:
        available = set(archive.namelist())
        for eid in eids:
            for filename in selected_names(eid):
                member = (
                    "s1s2_landslide_reference_data/original_scenes/"
                    f"{eid}/{filename}"
                )
                if member not in available:
                    raise RuntimeError(f"Missing archive member: {member}")
                destination = OUTPUT / args.fold / eid / filename
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(archive.read(member))
                records.append(
                    {
                        "eid": eid,
                        "member": member,
                        "path": str(destination.relative_to(STUDY)),
                        "bytes": destination.stat().st_size,
                        "sha256": sha256(destination),
                    }
                )
                print(f"fetched {eid}/{filename}", flush=True)
    manifest = {
        "schema_version": 1,
        "source_url": URL,
        "fold": args.fold,
        "eids": eids,
        "records": records,
    }
    manifest_path = METADATA / f"{args.fold}_optical_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"Wrote {manifest_path}; files={len(records)}; "
        f"bytes={sum(row['bytes'] for row in records)}"
    )


if __name__ == "__main__":
    main()
