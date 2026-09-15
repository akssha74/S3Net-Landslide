#!/usr/bin/env python3
"""Independent arithmetic verification of the Sen12 v6 protected result."""

from __future__ import annotations

import ast
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyproj
import xarray as xr
from scipy.ndimage import binary_dilation, binary_erosion


STUDY = Path(__file__).resolve().parents[1]
RESULT = (
    STUDY
    / "experiments/derived/results/sen12_s2_confirmation_v5/"
    "sen12_s2_confirmation_summary.json"
)
MANIFEST = (
    STUDY
    / "research/dataset-metadata/sen12-s2-confirmation/"
    "protected_extraction_manifest.json"
)
OUTPUT = STUDY / "reviews/verified-sen12-v6.json"
THRESHOLD = 0.5
SEEDS = (42, 43, 44)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def pooled_f1(probabilities: np.ndarray, targets: np.ndarray) -> float:
    prediction = probabilities >= THRESHOLD
    target = targets >= 0.5
    true_positive = np.sum(prediction & target)
    false_positive = np.sum(prediction & ~target)
    false_negative = np.sum(~prediction & target)
    return float(
        2
        * true_positive
        / (2 * true_positive + false_positive + false_negative + 1e-12)
    )


def boundary(mask: np.ndarray) -> np.ndarray:
    structure = np.ones((3, 3), dtype=bool)
    return binary_dilation(mask, structure=structure) ^ binary_erosion(
        mask, structure=structure, border_value=0
    )


def boundary_f1(probabilities: np.ndarray, targets: np.ndarray) -> float:
    structure = np.ones((3, 3), dtype=bool)
    values = []
    for prediction, target in zip(
        probabilities >= THRESHOLD, targets >= 0.5
    ):
        predicted_boundary = boundary(prediction)
        target_boundary = boundary(target)
        if not predicted_boundary.any() and not target_boundary.any():
            values.append(1.0)
            continue
        precision = np.sum(
            predicted_boundary
            & binary_dilation(target_boundary, structure=structure)
        ) / (np.sum(predicted_boundary) + 1e-12)
        recall = np.sum(
            target_boundary
            & binary_dilation(predicted_boundary, structure=structure)
        ) / (np.sum(target_boundary) + 1e-12)
        values.append(2 * precision * recall / (precision + recall + 1e-12))
    return float(np.mean(values))


def interval(values: list[float], salt: str) -> list[float]:
    seed = int(hashlib.sha256(salt.encode()).hexdigest()[:16], 16)
    generator = np.random.default_rng(seed)
    array = np.asarray(values)
    draws = np.mean(
        array[
            generator.integers(
                0, len(array), size=(10_000, len(array))
            )
        ],
        axis=1,
    )
    return [
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    ]


def assert_close(actual: float, expected: float, label: str) -> None:
    if not np.isclose(actual, expected, atol=1e-10, rtol=0):
        raise RuntimeError(f"{label}: {actual} != {expected}")


def summarize(values: dict[str, float], salt: str) -> dict[str, Any]:
    ordered = [values[key] for key in sorted(values)]
    return {
        "mean": float(np.mean(ordered)),
        "interval": interval(ordered, salt),
    }


def parse_post(value: Any) -> int:
    if isinstance(value, str):
        value = ast.literal_eval(value)
    if isinstance(value, (list, tuple)):
        value = value[0]
    if not isinstance(value, dict):
        raise RuntimeError("invalid pre_post_dates")
    return int(value["post"])


def high_confidence(value: Any) -> bool:
    if isinstance(value, str):
        values = [float(item) for item in value.split(",")]
    elif isinstance(value, (list, tuple, np.ndarray)):
        values = [float(item) for item in value]
    else:
        values = [float(value)]
    return all(item == 1.0 for item in values)


def footprint(dataset: xr.Dataset) -> dict[str, Any]:
    x = np.asarray(dataset["x"].values, dtype=float)
    y = np.asarray(dataset["y"].values, dtype=float)
    x_resolution = float(np.median(np.abs(np.diff(x))))
    y_resolution = float(np.median(np.abs(np.diff(y))))
    native = [
        float(x.min() - x_resolution / 2),
        float(y.min() - y_resolution / 2),
        float(x.max() + x_resolution / 2),
        float(y.max() + y_resolution / 2),
    ]
    crs = str(dataset.attrs["crs"])
    transformer = pyproj.Transformer.from_crs(
        crs, "EPSG:4326", always_xy=True
    )
    left, bottom, right, top = transformer.transform_bounds(
        *native, densify_pts=64
    )
    longitude_intervals = (
        [[float(left), 180.0], [-180.0, float(right)]]
        if right < left
        else [[float(left), float(right)]]
    )
    return {
        "crs": crs,
        "native": native,
        "longitude_intervals": longitude_intervals,
        "latitude_interval": [float(bottom), float(top)],
    }


def transformed_overlaps(records: list[dict[str, Any]]) -> list[dict[str, str]]:
    overlaps = []
    for left_index, left in enumerate(records):
        for right in records[left_index + 1 :]:
            if left["inventory"] == right["inventory"]:
                continue
            latitude_overlap = max(
                left["footprint"]["latitude_interval"][0],
                right["footprint"]["latitude_interval"][0],
            ) < min(
                left["footprint"]["latitude_interval"][1],
                right["footprint"]["latitude_interval"][1],
            )
            longitude_overlap = any(
                max(left_interval[0], right_interval[0])
                < min(left_interval[1], right_interval[1])
                for left_interval in left["footprint"]["longitude_intervals"]
                for right_interval in right["footprint"]["longitude_intervals"]
            )
            if latitude_overlap and longitude_overlap:
                overlaps.append(
                    {
                        "left_inventory": left["inventory"],
                        "left_filename": left["filename"],
                        "right_inventory": right["inventory"],
                        "right_filename": right["filename"],
                    }
                )
    return overlaps


def main() -> None:
    result = json.loads(RESULT.read_text())
    manifest = json.loads(MANIFEST.read_text())
    paths_by_inventory: dict[str, list[Path]] = {}
    for row in manifest["files"]:
        path = STUDY / row["path"]
        if path.stat().st_size != row["size"] or sha256(path) != row["sha256"]:
            raise RuntimeError(f"protected file identity mismatch: {path}")
        paths_by_inventory.setdefault(path.parent.name, []).append(path)
    targets = {}
    high_confidence_masks = {}
    spatial_records = []
    for inventory, paths in paths_by_inventory.items():
        arrays = []
        confidence_flags = []
        scl_histogram: Counter[int] = Counter()
        scl_valid_fractions = []
        cloud_fractions = []
        cloud_or_shadow_fractions = []
        for path in sorted(paths):
            with xr.open_dataset(path, engine="h5netcdf") as dataset:
                mask = np.asarray(dataset["MASK"].isel(time=0).values)
                if not np.all(np.asarray(dataset["MASK"].values) == mask):
                    raise RuntimeError(f"non-static MASK: {path}")
                post = parse_post(dataset.attrs["pre_post_dates"])
                scl = np.asarray(dataset["SCL"].isel(time=post).values)
                values, counts = np.unique(scl, return_counts=True)
                scl_histogram.update(
                    {
                        int(value): int(count)
                        for value, count in zip(values, counts)
                    }
                )
                valid = scl != 255
                scl_valid_fractions.append(float(np.mean(valid)))
                cloud_fractions.append(
                    float(np.mean(np.isin(scl[valid], [8, 9, 10])))
                    if np.any(valid)
                    else float("nan")
                )
                cloud_or_shadow_fractions.append(
                    float(np.mean(np.isin(scl[valid], [3, 8, 9, 10])))
                    if np.any(valid)
                    else float("nan")
                )
                confidence_flags.append(
                    high_confidence(dataset.attrs.get("date_confidence", 0))
                )
                spatial_records.append(
                    {
                        "inventory": inventory,
                        "filename": path.name,
                        "footprint": footprint(dataset),
                    }
                )
                arrays.append(mask.astype(np.float32))
        targets[inventory] = np.stack(arrays)
        high_confidence_masks[inventory] = np.asarray(
            confidence_flags, dtype=bool
        )
        reported_metadata = result["inventory_metadata"][inventory]
        if reported_metadata["scl_histogram"] != {
            str(code): count for code, count in sorted(scl_histogram.items())
        }:
            raise RuntimeError(f"{inventory}: SCL histogram mismatch")
        assert_close(
            float(np.mean(scl_valid_fractions)),
            reported_metadata["mean_scl_valid_fraction"],
            f"{inventory}: SCL valid fraction",
        )
        if reported_metadata["scl_unavailable_patch_count"] != int(
            np.sum(~np.isfinite(cloud_fractions))
        ):
            raise RuntimeError(f"{inventory}: unavailable SCL count mismatch")
        valid_cloud = np.isfinite(cloud_fractions)
        expected_cloud = (
            float(np.mean(np.asarray(cloud_fractions)[valid_cloud]))
            if np.any(valid_cloud)
            else None
        )
        if expected_cloud is not None:
            assert_close(
                expected_cloud,
                reported_metadata[
                    "mean_cloud_fraction_scl_8_9_10_among_valid_scl"
                ],
                f"{inventory}: cloud fraction",
            )
        valid_cloud_shadow = np.isfinite(cloud_or_shadow_fractions)
        expected_cloud_shadow = (
            float(
                np.mean(
                    np.asarray(cloud_or_shadow_fractions)[valid_cloud_shadow]
                )
            )
            if np.any(valid_cloud_shadow)
            else None
        )
        if expected_cloud_shadow is not None:
            assert_close(
                expected_cloud_shadow,
                reported_metadata[
                    "mean_cloud_or_shadow_fraction_scl_3_8_9_10_among_valid_scl"
                ],
                f"{inventory}: cloud-or-shadow fraction",
            )

    overlaps = transformed_overlaps(spatial_records)
    if overlaps != [
        {
            "left_inventory": "kyrgyzstan1",
            "left_filename": "kyrgyzstan1_s2_3307.nc",
            "right_inventory": "kyrgyzstan2",
            "right_filename": "kyrgyzstan2_s2_6204.nc",
        }
    ]:
        raise RuntimeError(f"registered overlap mismatch: {overlaps}")
    native_positive_area_overlaps = []
    for left_index, left in enumerate(spatial_records):
        for right in spatial_records[left_index + 1 :]:
            if (
                left["inventory"] == right["inventory"]
                or left["footprint"]["crs"] != right["footprint"]["crs"]
            ):
                continue
            left_bounds = left["footprint"]["native"]
            right_bounds = right["footprint"]["native"]
            if max(left_bounds[0], right_bounds[0]) < min(
                left_bounds[2], right_bounds[2]
            ) and max(left_bounds[1], right_bounds[1]) < min(
                left_bounds[3], right_bounds[3]
            ):
                native_positive_area_overlaps.append(
                    (left["filename"], right["filename"])
                )

    lookup = {}
    actual_metrics: dict[tuple[str, int, str, str], float] = {}
    verified_metric_cells = 0
    verified_high_confidence_cells = 0
    for run in result["runs"]:
        lookup[(run["arm"], run["seed"])] = run
        for inventory, reported in run["protected"].items():
            probability_path = STUDY / reported["probabilities"]
            if sha256(probability_path) != reported["probabilities_sha256"]:
                raise RuntimeError(f"probability hash mismatch: {probability_path}")
            probabilities = np.load(probability_path, allow_pickle=False)
            actual_f1 = pooled_f1(probabilities, targets[inventory])
            actual_boundary_f1 = boundary_f1(
                probabilities, targets[inventory]
            )
            assert_close(actual_f1, reported["f1"], f"{run['arm']} F1")
            assert_close(
                actual_boundary_f1,
                reported["boundary_f1"],
                f"{run['arm']} boundary F1",
            )
            actual_metrics[(run["arm"], run["seed"], inventory, "f1")] = (
                actual_f1
            )
            actual_metrics[
                (run["arm"], run["seed"], inventory, "boundary_f1")
            ] = actual_boundary_f1
            verified_metric_cells += 2
            high = high_confidence_masks[inventory]
            if np.any(high):
                high_f1 = pooled_f1(
                    probabilities[high], targets[inventory][high]
                )
                high_boundary_f1 = boundary_f1(
                    probabilities[high], targets[inventory][high]
                )
                assert_close(
                    high_f1,
                    reported["high_confidence_f1"],
                    f"{run['arm']} high-confidence F1",
                )
                assert_close(
                    high_boundary_f1,
                    reported["high_confidence_boundary_f1"],
                    f"{run['arm']} high-confidence boundary F1",
                )
                verified_high_confidence_cells += 2

    def effect(
        treatment: str, control: str, metric: str
    ) -> dict[str, float]:
        return {
            inventory: float(
                np.mean(
                    [
                        actual_metrics[(treatment, seed, inventory, metric)]
                        - actual_metrics[(control, seed, inventory, metric)]
                        for seed in SEEDS
                    ]
                )
            )
            for inventory in sorted(targets)
        }

    contrast_specs = {
        "ndvi_attention_vs_raw_edge_f1": (
            "s3_ndvi_base",
            "s3_raw_base",
            "f1",
        ),
        "ndvi_modulated_vs_plain_f1": (
            "s3_ndvi_biophysical",
            "s3_ndvi_boundary",
            "f1",
        ),
        "ndvi_modulated_vs_plain_boundary_f1": (
            "s3_ndvi_biophysical",
            "s3_ndvi_boundary",
            "boundary_f1",
        ),
    }
    contrasts = {}
    for name, specification in contrast_specs.items():
        values = effect(*specification)
        computed = summarize(values, name)
        reported = result["contrasts"][name]
        assert_close(computed["mean"], reported["mean"], f"{name} mean")
        for index in range(2):
            assert_close(
                computed["interval"][index],
                reported["interval"][index],
                f"{name} interval {index}",
            )
        contrasts[name] = computed

    control_specs = {
        "zero_control": ("s3_none_boundary", "s3_none_base"),
        "raw": ("s3_raw_boundary", "s3_raw_base"),
        "ndvi": ("s3_ndvi_boundary", "s3_ndvi_base"),
    }
    generic = {
        name: effect(treatment, control, "f1")
        for name, (treatment, control) in control_specs.items()
    }
    generic_average = {
        inventory: float(
            np.mean([generic[name][inventory] for name in generic])
        )
        for inventory in sorted(targets)
    }
    generic_summary = summarize(generic_average, "generic-average")
    reported_generic = result["generic_boundary_average"]
    assert_close(
        generic_summary["mean"], reported_generic["mean"], "generic mean"
    )
    for index in range(2):
        assert_close(
            generic_summary["interval"][index],
            reported_generic["interval"][index],
            f"generic interval {index}",
        )

    absolute_arm_means = {}
    for arm in result["absolute_performance"]:
        per_inventory = {
            inventory: float(
                np.mean(
                    [
                        actual_metrics[(arm, seed, inventory, "f1")]
                        for seed in SEEDS
                    ]
                )
            )
            for inventory in sorted(targets)
        }
        absolute_arm_means[arm] = float(np.mean(list(per_inventory.values())))
        assert_close(
            absolute_arm_means[arm],
            result["absolute_performance"][arm]["f1"]["mean"],
            f"{arm}: absolute mean F1",
        )
    conditions = {
        "ndvi_attention_vs_raw_edge_upper_ci_below_1_point": (
            contrasts["ndvi_attention_vs_raw_edge_f1"]["interval"][1] < 0.01
        ),
        "ndvi_modulated_vs_plain_upper_ci_below_1_point": (
            contrasts["ndvi_modulated_vs_plain_f1"]["interval"][1] < 0.01
        ),
        "generic_boundary_all_control_means_positive_lower_ci_above_zero_8_of_10": (
            all(np.mean(list(values.values())) > 0 for values in generic.values())
            and generic_summary["interval"][0] > 0
            and sum(value > 0 for value in generic_average.values()) >= 8
        ),
        "all_arm_absolute_mean_f1_at_least_0_10": all(
            value >= 0.10 for value in absolute_arm_means.values()
        ),
        "all_integrity_checks_pass": len(overlaps) == 0,
    }
    if conditions != result["conditions"]:
        raise RuntimeError("condition recomputation mismatch")
    payload = {
        "schema_version": 1,
        "result_sha256": sha256(RESULT),
        "protected_manifest_sha256": sha256(MANIFEST),
        "run_count": len(result["runs"]),
        "inventory_count": len(targets),
        "verified_metric_cells": verified_metric_cells,
        "verified_high_confidence_metric_cells": (
            verified_high_confidence_cells
        ),
        "verified_scl_inventory_summaries": len(targets),
        "registered_transformed_envelope_overlap_count": len(overlaps),
        "native_same_crs_positive_area_overlap_count": len(
            native_positive_area_overlaps
        ),
        "conditions": conditions,
        "all_conditions_pass": all(conditions.values()),
        "status": "verified-not-confirmed",
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
