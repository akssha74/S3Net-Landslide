#!/usr/bin/env python3
"""Freeze compact bootstrap outputs needed for stdlib-only claim verification."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.ndimage import binary_dilation, binary_erosion


STUDY = Path(__file__).resolve().parents[2]
RESULTS = STUDY / "experiments/derived/results"
SEN12 = (
    RESULTS
    / "sen12_s2_confirmation_v5/sen12_s2_confirmation_summary.json"
)
OUTPUT = RESULTS / "public_claim_sufficient_statistics.json"
PROTECTED_MANIFEST = (
    STUDY
    / "research/dataset-metadata/sen12-s2-confirmation/"
    "protected_extraction_manifest.json"
)
SOURCE_PATHS = [
    RESULTS / "reviewer_remediation/reviewer_remediation_summary.json",
    RESULTS
    / "reviewer_remediation_bgrn/reviewer_remediation_summary.json",
    RESULTS
    / "cas_boundary_confirmation/cas_boundary_confirmation_summary.json",
    RESULTS / "lrd_boundary_confirmation/endpoint_sensitivity.json",
    SEN12,
    PROTECTED_MANIFEST,
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bootstrap_indices(size: int, salt: str) -> np.ndarray:
    rng = np.random.default_rng(
        int(hashlib.sha256(salt.encode()).hexdigest()[:16], 16)
    )
    return rng.integers(0, size, size=(10000, size)).astype(np.uint8)


def boundary(mask: np.ndarray) -> np.ndarray:
    structure = np.ones((3, 3), dtype=bool)
    return binary_dilation(mask, structure=structure) ^ binary_erosion(
        mask, structure=structure, border_value=0
    )


def boundary_scores(probabilities: np.ndarray, targets: np.ndarray) -> list[float]:
    structure = np.ones((3, 3), dtype=bool)
    output = []
    for prediction, target in zip(
        probabilities >= 0.5, targets >= 0.5
    ):
        predicted_boundary = boundary(prediction)
        target_boundary = boundary(target)
        if not predicted_boundary.any() and not target_boundary.any():
            output.append(1.0)
            continue
        precision = np.sum(
            predicted_boundary
            & binary_dilation(target_boundary, structure=structure)
        ) / (np.sum(predicted_boundary) + 1e-12)
        recall = np.sum(
            target_boundary
            & binary_dilation(predicted_boundary, structure=structure)
        ) / (np.sum(target_boundary) + 1e-12)
        output.append(
            float(2 * precision * recall / (precision + recall + 1e-12))
        )
    return output


def main() -> None:
    sen12 = json.loads(SEN12.read_text())
    inventories = sen12["inventories"]
    bootstrap = {}
    for name, row in sen12["contrasts"].items():
        indices = bootstrap_indices(len(inventories), name)
        bootstrap[name] = {
            "salt": name,
            "population_order": inventories,
            "draw_count": int(indices.shape[0]),
            "draw_width": int(indices.shape[1]),
            "indices_uint8_base64": base64.b64encode(
                indices.tobytes()
            ).decode("ascii"),
        }

    generic_indices = bootstrap_indices(len(inventories), "generic-average")
    bootstrap["generic_boundary_average"] = {
        "salt": "generic-average",
        "population_order": inventories,
        "draw_count": int(generic_indices.shape[0]),
        "draw_width": int(generic_indices.shape[1]),
        "indices_uint8_base64": base64.b64encode(
            generic_indices.tobytes()
        ).decode("ascii"),
    }

    manifest = json.loads(PROTECTED_MANIFEST.read_text())
    paths_by_inventory: dict[str, list[Path]] = {}
    for row in manifest["files"]:
        path = STUDY / row["path"]
        assert sha256(path) == row["sha256"]
        paths_by_inventory.setdefault(path.parent.name, []).append(path)
    targets = {}
    for inventory, paths in paths_by_inventory.items():
        arrays = []
        for path in sorted(paths):
            with xr.open_dataset(path, engine="h5netcdf") as dataset:
                arrays.append(
                    np.asarray(dataset["MASK"].isel(time=0).values).astype(
                        np.float32
                    )
                )
        targets[inventory] = np.stack(arrays)

    boundary_f1_per_patch: dict[str, dict[str, dict[str, list[float]]]] = {}
    for run in sen12["runs"]:
        arm = run["arm"]
        seed = str(run["seed"])
        boundary_f1_per_patch.setdefault(arm, {})[seed] = {}
        for inventory, reported in run["protected"].items():
            probability_path = STUDY / reported["probabilities"]
            assert sha256(probability_path) == reported["probabilities_sha256"]
            probabilities = np.load(probability_path, allow_pickle=False)
            scores = boundary_scores(probabilities, targets[inventory])
            assert np.isclose(
                np.mean(scores), reported["boundary_f1"], atol=1e-12, rtol=0
            )
            boundary_f1_per_patch[arm][seed][inventory] = scores

    payload = {
        "schema_version": 1,
        "purpose": (
            "Compact frozen bootstrap outputs for dependency-free recomputation "
            "of all manuscript point estimates, envelopes, and conditions."
        ),
        "source_artifacts": {
            str(path.relative_to(STUDY)): sha256(path)
            for path in SOURCE_PATHS
        },
        "sen12_bootstrap": bootstrap,
        "sen12_boundary_f1_per_patch": boundary_f1_per_patch,
    }
    OUTPUT.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    )
    print(
        json.dumps(
            {
                "status": "pass",
                "output": str(OUTPUT.relative_to(STUDY)),
                "sha256": sha256(OUTPUT),
                "bootstrap_series": len(bootstrap),
                "draws_per_series": 10000,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
