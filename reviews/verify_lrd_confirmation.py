#!/usr/bin/env python3
"""Independently recompute the prospective LRD confirmation metrics."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import binary_dilation, binary_erosion


STUDY = Path(__file__).resolve().parent.parent
REPO = STUDY.parents[1]
DATA = STUDY / "experiments/raw/external/lrd-optical/protected"
SUMMARY = (
    STUDY
    / "experiments/derived/results/lrd_boundary_confirmation/"
    "lrd_confirmation_summary.json"
)
OUTPUT = STUDY / "reviews/verified-lrd-confirmation.json"
AUTHORIZATION = (
    STUDY
    / "experiments/derived/results/lrd_boundary_confirmation/"
    "protected_access_authorization.json"
)
ACCESS_LOG = (
    STUDY
    / "research/dataset-metadata/lrd-prospective-confirmation/"
    "protected_access_log.json"
)
PROTOCOL_FILES = (
    "studies/disaster-hrgldd-landslide/"
    "research/lrd-boundary-confirmation-preregistration.md",
    "studies/disaster-hrgldd-landslide/"
    "research/dataset-metadata/lrd-prospective-confirmation/"
    "protected_event_folds.json",
    "studies/disaster-hrgldd-landslide/"
    "experiments/code/fetch_lrd_optical_folds.py",
    "studies/disaster-hrgldd-landslide/"
    "experiments/code/run_lrd_prospective_confirmation.py",
)
SEEDS = (42, 43, 44)


def one(root: Path, pattern: str) -> Path:
    matches = list(root.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"{root} {pattern}: expected 1, got {len(matches)}")
    return matches[0]


def selected_masks(
    eid: str, crop_size: int, limit: int, salt: str
) -> tuple[np.ndarray, list[str]]:
    mask = tifffile.imread(one(DATA / eid, "*__MASK.tif"))
    while mask.ndim > 2:
        mask = mask[..., 0]
    truth = mask > 0
    height, width = truth.shape
    padded_height = ((height + crop_size - 1) // crop_size) * crop_size
    padded_width = ((width + crop_size - 1) // crop_size) * crop_size
    truth = np.pad(
        truth,
        ((0, padded_height - height), (0, padded_width - width)),
        mode="constant",
        constant_values=False,
    )
    coordinates = [
        (y, x)
        for y in range(0, padded_height, crop_size)
        for x in range(0, padded_width, crop_size)
    ]
    coordinates.sort(
        key=lambda row: hashlib.sha256(
            f"{salt}:{eid}:{row[0]}:{row[1]}".encode()
        ).hexdigest()
    )
    coordinates = coordinates[:limit]
    masks = np.stack(
        [truth[y : y + crop_size, x : x + crop_size] for y, x in coordinates]
    )
    ids = [f"{eid}:{y}:{x}:{height}:{width}" for y, x in coordinates]
    return masks, ids


def boundary(mask: np.ndarray) -> np.ndarray:
    structure = np.ones((3, 3), dtype=bool)
    return binary_dilation(mask, structure=structure) ^ binary_erosion(
        mask, structure=structure, border_value=0
    )


def boundary_f1(
    probabilities: np.ndarray, targets: np.ndarray, threshold: float
) -> float:
    prediction = probabilities >= threshold
    structure = np.ones((3, 3), dtype=bool)
    values = []
    for pred, truth in zip(prediction, targets):
        pred_boundary = boundary(pred)
        truth_boundary = boundary(truth)
        if not pred_boundary.any() and not truth_boundary.any():
            values.append(1.0)
            continue
        precision = np.sum(
            pred_boundary & binary_dilation(truth_boundary, structure=structure)
        ) / (np.sum(pred_boundary) + 1e-12)
        recall = np.sum(
            truth_boundary & binary_dilation(pred_boundary, structure=structure)
        ) / (np.sum(truth_boundary) + 1e-12)
        values.append(2 * precision * recall / (precision + recall + 1e-12))
    return float(np.mean(values))


def metrics(
    probabilities: np.ndarray, targets: np.ndarray, threshold: float
) -> dict[str, float | int]:
    prediction = probabilities >= threshold
    truth = targets >= 0.5
    tp = int(np.sum(prediction & truth))
    fp = int(np.sum(prediction & ~truth))
    fn = int(np.sum(~prediction & truth))
    tn = int(np.sum(~prediction & ~truth))
    return {
        "threshold": threshold,
        "f1": 2 * tp / (2 * tp + fp + fn + 1e-12),
        "iou": tp / (tp + fp + fn + 1e-12),
        "precision": tp / (tp + fp + 1e-12),
        "recall": tp / (tp + fn + 1e-12),
        "background_fpr": fp / (fp + tn + 1e-12),
        "false_positive_pixels": fp,
        "boundary_f1": boundary_f1(probabilities, targets, threshold),
    }


def close(first: float | int, second: float | int) -> bool:
    return bool(np.isclose(first, second, rtol=0, atol=1e-10))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    payload = json.loads(SUMMARY.read_text())
    config = payload["config"]
    authorization = json.loads(AUTHORIZATION.read_text())
    access_log = json.loads(ACCESS_LOG.read_text())
    protocol_commit = config["protocol_commit"]
    subprocess.run(
        ["git", "-C", str(REPO), "merge-base", "--is-ancestor", protocol_commit, "HEAD"],
        check=True,
    )
    protocol_diff = subprocess.run(
        [
            "git",
            "-C",
            str(REPO),
            "diff",
            "--quiet",
            protocol_commit,
            "--",
            *PROTOCOL_FILES,
        ]
    ).returncode
    process_checks = {
        "authorization_precedes_access": (
            authorization["status"] == "all-decisions-frozen"
            and access_log["status"] == "valid-prospective-protected-access"
        ),
        "authorization_hash_matches": (
            access_log["protected_access_authorization_sha256"]
            == sha256(AUTHORIZATION)
        ),
        "protocol_files_unchanged": protocol_diff == 0,
        "process_condition_passed": (
            payload["process_condition"]["status"] == "passed"
            and access_log["test_driven_retuning"] is False
        ),
    }
    if not all(process_checks.values()):
        raise RuntimeError(f"LRD process checks failed: {process_checks}")
    targets = {}
    crop_ids = {}
    for eid in config["protected_test_eids"]:
        targets[eid], crop_ids[eid] = selected_masks(
            eid,
            config["crop_size"],
            config["crops_per_event"],
            config["crop_salt"],
        )
    checks = []
    for run in payload["training_records"]:
        for eid, reported in run["test_events"].items():
            probabilities = np.load(
                STUDY / reported["probabilities"], mmap_mode="r"
            )
            recomputed = metrics(
                probabilities,
                targets[eid],
                float(run["validation_threshold"]),
            )
            for metric, value in recomputed.items():
                checks.append(
                    {
                        "loss_mode": run["loss_mode"],
                        "seed": run["seed"],
                        "eid": eid,
                        "metric": metric,
                        "reported": reported[metric],
                        "recomputed": value,
                        "matched": close(reported[metric], value),
                    }
                )
            checks.append(
                {
                    "loss_mode": run["loss_mode"],
                    "seed": run["seed"],
                    "eid": eid,
                    "metric": "crop_id_sha256",
                    "reported": reported["crop_id_sha256"],
                    "recomputed": hashlib.sha256(
                        "\n".join(crop_ids[eid]).encode()
                    ).hexdigest(),
                    "matched": (
                        reported["crop_id_sha256"]
                        == hashlib.sha256(
                            "\n".join(crop_ids[eid]).encode()
                        ).hexdigest()
                    ),
                }
            )
    effects = {}
    for seed in SEEDS:
        base = next(
            row
            for row in payload["training_records"]
            if row["loss_mode"] == "base" and row["seed"] == seed
        )
        boundary_run = next(
            row
            for row in payload["training_records"]
            if row["loss_mode"] == "boundary" and row["seed"] == seed
        )
        per_event = {
            eid: {
                "delta_f1": (
                    boundary_run["test_events"][eid]["f1"]
                    - base["test_events"][eid]["f1"]
                ),
                "delta_boundary_f1": (
                    boundary_run["test_events"][eid]["boundary_f1"]
                    - base["test_events"][eid]["boundary_f1"]
                ),
            }
            for eid in config["protected_test_eids"]
        }
        effects[str(seed)] = {
            "event_macro_delta_f1": float(
                np.mean([row["delta_f1"] for row in per_event.values()])
            ),
            "event_macro_delta_boundary_f1": float(
                np.mean(
                    [row["delta_boundary_f1"] for row in per_event.values()]
                )
            ),
        }
    for seed in SEEDS:
        for metric, value in effects[str(seed)].items():
            reported = payload["effects"][str(seed)][metric]
            checks.append(
                {
                    "loss_mode": "contrast",
                    "seed": seed,
                    "eid": "event-macro",
                    "metric": metric,
                    "reported": reported,
                    "recomputed": value,
                    "matched": close(reported, value),
                }
            )
    failed = [check for check in checks if not check["matched"]]
    output = {
        "run": "LRD-prospective-confirmation",
        "protocol_commit": config["protocol_commit"],
        "independence_unit": "LRD EID",
        "protected_eids": config["protected_test_eids"],
        "process_checks": process_checks,
        "checks": checks,
        "failed": failed,
        "status": "passed" if not failed else "failed",
    }
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n")
    if failed:
        raise SystemExit(f"{len(failed)} LRD checks failed")
    print(f"PASS: {len(checks)} LRD metrics independently recomputed")


if __name__ == "__main__":
    main()
