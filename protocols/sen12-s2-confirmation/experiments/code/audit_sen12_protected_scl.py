#!/usr/bin/env python3
"""Audit protected SCL codes after the authorized v5 loader failure."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_sen12_s2_confirmation as run


STUDY = Path(__file__).resolve().parents[2]
OUTPUT = (
    STUDY
    / "experiments/derived/results/sen12_v5_protected_scl_diagnostic.json"
)


def main() -> None:
    run.verify_authorization_state()
    manifest = json.loads(
        (run.METADATA / "protected_extraction_manifest.json").read_text()
    )
    totals: Counter[int] = Counter()
    by_inventory: dict[str, Counter[int]] = {}
    unavailable_files: dict[str, list[str]] = {}
    for row in manifest["files"]:
        path = STUDY / row["path"]
        inventory = path.parent.name
        with xr.open_dataset(path, engine="h5netcdf") as dataset:
            post = run.parse_post(dataset.attrs["pre_post_dates"])
            scl = np.asarray(dataset["SCL"].isel(time=post).values)
        values, counts = np.unique(scl, return_counts=True)
        observed = Counter(
            {int(value): int(count) for value, count in zip(values, counts)}
        )
        totals.update(observed)
        by_inventory.setdefault(inventory, Counter()).update(observed)
        if observed.get(255, 0):
            unavailable_files.setdefault(inventory, []).append(path.name)
    payload = {
        "schema_version": 1,
        "status": "authorized-post-access-descriptive-scl-diagnostic",
        "trigger": (
            "v5 evaluation stopped before model inference when strict SCL "
            "validation encountered code 255"
        ),
        "official_scl_codes": list(range(12)),
        "dataset_specific_unavailable_sentinel": 255,
        "treatment": (
            "retain code 255 in histograms; exclude it from descriptive SCL "
            "cloud denominators; do not exclude files or alter model inputs"
        ),
        "file_count": len(manifest["files"]),
        "pixel_counts": {
            str(code): count for code, count in sorted(totals.items())
        },
        "by_inventory_pixel_counts": {
            inventory: {
                str(code): count for code, count in sorted(counter.items())
            }
            for inventory, counter in sorted(by_inventory.items())
        },
        "files_with_255": {
            inventory: sorted(filenames)
            for inventory, filenames in sorted(unavailable_files.items())
        },
        "files_with_255_count": sum(map(len, unavailable_files.values())),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {
                "file_count": payload["file_count"],
                "observed_codes": sorted(totals),
                "files_with_255_count": payload["files_with_255_count"],
                "pixels_255": totals[255],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
