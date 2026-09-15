#!/usr/bin/env python3
"""Run the frozen inventory-held Sen12 Sentinel-2 confirmation."""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import email.utils
import hashlib
import json
import os
import platform
import random
import sys
import urllib.request
from pathlib import Path
from typing import Any

os.environ["HRGLDD_ARRAY_ORDER"] = "BGRN"

import numpy as np
import h5netcdf
import h5py
import pyproj
import scipy
import torch
import xarray as xr
from scipy.ndimage import binary_dilation, binary_erosion
from torch.utils.data import DataLoader, TensorDataset

from revised_models import ControlledPixelLoss, ControlledS3Net


STUDY = Path(__file__).resolve().parents[2]
DATA = STUDY / "experiments/raw/external/sen12-s2"
OUTPUT = STUDY / "experiments/derived/results/sen12_s2_confirmation_v5"
CHECKPOINTS = STUDY / "experiments/derived/checkpoints/sen12_s2_confirmation_v5"
METADATA = STUDY / "research/dataset-metadata/sen12-s2-confirmation"
MEMBER_INDEX = METADATA / "member_index.json"
PROTOCOL = STUDY / "research/sen12-s2-confirmation-protocol.md"
ACCESS_AUDIT = STUDY / "research/sen12-s2-access-audit.json"
REQUIREMENTS = STUDY / "environment/requirements-sen12.txt"
AUTHORIZATION = OUTPUT / "protected_access_authorization.json"
PUBLIC_RECEIPT = METADATA / "public_authorization_receipt.json"
VALIDATION_PREDICTIONS = OUTPUT / "validation_predictions"
DATASET_REVISION = "40af2dd6b4e568edb6640d6e14dc67ebd01038a4"
PUBLIC_REPOSITORY = "akssha74/S3Net-Landslide"
PUBLIC_PROTOCOL_PREFIX = "protocols/sen12-s2-confirmation"
PUBLIC_AUTHORIZATION_PATH = (
    f"{PUBLIC_PROTOCOL_PREFIX}/experiments/derived/results/"
    "sen12_s2_confirmation_v5/protected_access_authorization.json"
)
SEEDS = (42, 43, 44)
EPOCHS = 15
BATCH_SIZE = 16
THRESHOLD = 0.5
INDIVIDUAL_VALIDATION_FLOOR = 0.20
ARM_MEAN_VALIDATION_FLOOR = 0.25
DEVELOPMENT_SPLIT_SALT = "v4-mixed"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
ARMS = {
    "s3_none_base": {"gating": "none", "loss": "base"},
    "s3_raw_base": {"gating": "raw", "loss": "base"},
    "s3_ndvi_base": {"gating": "ndvi", "loss": "base"},
    "s3_none_boundary": {"gating": "none", "loss": "boundary"},
    "s3_raw_boundary": {"gating": "raw", "loss": "boundary"},
    "s3_ndvi_boundary": {"gating": "ndvi", "loss": "boundary"},
    "s3_ndvi_biophysical": {"gating": "ndvi", "loss": "biophysical"},
}
SOURCE_ARTIFACTS = (
    "research/sen12-s2-confirmation-protocol.md",
    "research/sen12-s2-access-audit.json",
    "research/sen12-v3-fit-status.md",
    "research/sen12-v4-fit-status.md",
    "research/dataset-metadata/sen12-s2-confirmation/archive_manifest.json",
    (
        "research/dataset-metadata/sen12-s2-confirmation/"
        "s12ls_ld_s2_membership.json"
    ),
    "research/dataset-metadata/sen12-s2-confirmation/member_index.json",
    "experiments/code/prepare_sen12_s2.py",
    "experiments/code/run_sen12_s2_confirmation.py",
    "experiments/code/test_sen12_s2_confirmation.py",
    "experiments/code/analyze_sen12_v3_thresholds.py",
    "experiments/code/explore_sen12_v4.py",
    "experiments/code/explore_sen12_v4_mixed_matrix.py",
    "experiments/code/revised_models.py",
    "experiments/code/data_semantics.py",
    "experiments/derived/results/sen12_s2_confirmation/fit_decisions.json",
    "experiments/derived/results/sen12_s2_confirmation_v4/fit_decisions.json",
    "experiments/derived/results/sen12_v3_threshold_diagnostic.json",
    "experiments/derived/results/sen12_v4_development_exploration.json",
    "experiments/derived/results/sen12_v4_mixed_matrix_exploration.json",
    "experiments/logs/R055-sen12-v5-numeric-recheck.log",
    "environment/requirements-sen12.txt",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def source_artifact_hashes() -> dict[str, str]:
    return {path: sha256(STUDY / path) for path in SOURCE_ARTIFACTS}


def verify_public_sources(commit: str, expected: dict[str, str]) -> None:
    for local_path, expected_hash in expected.items():
        public_path = f"{PUBLIC_PROTOCOL_PREFIX}/{local_path}"
        url = (
            f"https://raw.githubusercontent.com/{PUBLIC_REPOSITORY}/"
            f"{commit}/{public_path}"
        )
        with urllib.request.urlopen(url) as response:
            actual_hash = hashlib.sha256(response.read()).hexdigest()
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"public source mismatch for {public_path}: "
                f"{actual_hash} != {expected_hash}"
            )


def github_commit_evidence(commit: str) -> dict[str, Any]:
    if len(commit) != 40 or any(value not in "0123456789abcdef" for value in commit):
        raise RuntimeError("public protocol commit must be a full lowercase SHA")
    commit_url = f"https://api.github.com/repos/{PUBLIC_REPOSITORY}/commits/{commit}"
    with urllib.request.urlopen(commit_url) as response:
        record = json.load(response)
        api_observed_at = email.utils.parsedate_to_datetime(
            response.headers["Date"]
        ).astimezone(dt.timezone.utc)
    if record.get("sha") != commit:
        raise RuntimeError("protocol commit is not publicly resolvable")
    push_event = None
    for page in range(1, 3):
        events_url = (
            f"https://api.github.com/repos/{PUBLIC_REPOSITORY}/events"
            f"?per_page=100&page={page}"
        )
        with urllib.request.urlopen(events_url) as response:
            events = json.load(response)
        for event in events:
            payload = event.get("payload", {})
            commit_ids = [row.get("sha") for row in payload.get("commits", [])]
            if event.get("type") == "PushEvent" and (
                payload.get("head") == commit or commit in commit_ids
            ):
                push_event = event
                break
        if push_event is not None or not events:
            break
    if push_event is not None:
        push_time = dt.datetime.fromisoformat(
            push_event["created_at"].replace("Z", "+00:00")
        )
        if push_time > api_observed_at:
            raise RuntimeError("public push time is later than API observation time")
    return {
        "commit_record": record,
        "api_observed_at": api_observed_at.isoformat().replace("+00:00", "Z"),
        "push_event_id": push_event["id"] if push_event is not None else None,
        "push_event_time": (
            push_event["created_at"] if push_event is not None else None
        ),
    }


def runtime_identity() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pyproj": pyproj.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
        "xarray": xr.__version__,
        "h5netcdf": h5netcdf.__version__,
        "h5py": h5py.__version__,
    }


def verify_runtime_identity() -> dict[str, str]:
    observed = runtime_identity()
    expected = {}
    for line in REQUIREMENTS.read_text().splitlines():
        if line and not line.startswith("#"):
            name, version = line.split("==", 1)
            expected[name] = version
    for name, version in expected.items():
        if observed.get(name) != version:
            raise RuntimeError(
                f"runtime dependency mismatch for {name}: "
                f"{observed.get(name)} != {version}"
            )
    return observed


def verify_extraction_manifest(fold: str) -> dict[str, Any]:
    path = METADATA / f"{fold}_extraction_manifest.json"
    payload = json.loads(path.read_text())
    if payload["fold"] != fold:
        raise RuntimeError(f"{fold} extraction manifest fold mismatch")
    if payload["dataset_revision"] != DATASET_REVISION:
        raise RuntimeError(f"{fold} extraction dataset revision mismatch")
    if payload["member_index_sha256"] != sha256(MEMBER_INDEX):
        raise RuntimeError(f"{fold} extraction member-index mismatch")
    if fold == "protected" and payload["authorization_sha256"] != sha256(
        AUTHORIZATION
    ):
        raise RuntimeError("protected extraction authorization mismatch")
    for row in payload["files"]:
        file_path = STUDY / row["path"]
        if file_path.stat().st_size != row["size"] or sha256(file_path) != row["sha256"]:
            raise RuntimeError(f"extracted file identity mismatch: {file_path}")
    return payload


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def parse_post(value: Any) -> int:
    if isinstance(value, str):
        value = ast.literal_eval(value)
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("pre_post_dates cannot be empty")
        value = value[0]
    if not isinstance(value, dict) or "post" not in value:
        raise ValueError("pre_post_dates must be a dict or list of dicts")
    post = int(value["post"])
    if post < 0:
        raise ValueError("post index must be non-negative")
    return post


def load_file(path: Path) -> dict[str, Any]:
    with xr.open_dataset(path, engine="h5netcdf") as dataset:
        post = parse_post(dataset.attrs["pre_post_dates"])
        if post >= dataset.sizes["time"]:
            raise RuntimeError(f"{path}: post index is outside the time axis")
        if dataset.attrs.get("satellite") != "s2":
            raise RuntimeError(f"{path}: satellite is not s2")
        channels = []
        for band in ("B02", "B03", "B04", "B08"):
            if dataset[band].dims != ("time", "x", "y"):
                raise RuntimeError(f"{path}: unexpected {band} dimensions")
            value = np.asarray(dataset[band].isel(time=post).values, dtype=np.float32)
            if (
                not np.all(np.isfinite(value))
                or float(value.min()) < 0
                or float(value.max()) > 10000
            ):
                raise RuntimeError(f"{path}: {band} violates harmonized DN bounds")
            channels.append(np.clip(value / 10000.0, 0.0, 1.0))
        image = np.stack(channels)
        if dataset["MASK"].dims != ("time", "x", "y"):
            raise RuntimeError(f"{path}: unexpected MASK dimensions")
        mask_series = np.asarray(dataset["MASK"].values)
        if mask_series.ndim != 3 or not np.all(mask_series == mask_series[0]):
            raise RuntimeError(f"{path}: MASK is not static over time")
        if not set(np.unique(mask_series)).issubset({0, 1}):
            raise RuntimeError(f"{path}: MASK is not binary")
        mask = np.asarray(mask_series[0] > 0, dtype=np.float32)
        annotated = dataset.attrs.get("annotated", False)
        if annotated not in (True, 1, "True", "true"):
            raise RuntimeError(f"{path}: file is not annotated")
        confidence = dataset.attrs.get("date_confidence", 0)
        if isinstance(confidence, str):
            confidence_values = [float(x) for x in confidence.split(",")]
        elif isinstance(confidence, (list, tuple, np.ndarray)):
            confidence_values = [float(x) for x in confidence]
        else:
            confidence_values = [float(confidence)]
        high_confidence = all(value == 1.0 for value in confidence_values)
        if dataset["SCL"].dims != ("time", "x", "y"):
            raise RuntimeError(f"{path}: unexpected SCL dimensions")
        scl = np.asarray(dataset["SCL"].isel(time=post).values)
        if not set(np.unique(scl)).issubset(set(range(12))):
            raise RuntimeError(f"{path}: unexpected SCL class")
        time_value = np.asarray(dataset["time"].values)[post]
        post_date = str(np.datetime_as_string(time_value, unit="D"))
        x_coordinates = np.asarray(dataset["x"].values, dtype=float)
        y_coordinates = np.asarray(dataset["y"].values, dtype=float)
        if (
            x_coordinates.size < 2
            or y_coordinates.size < 2
            or not np.all(np.isfinite(x_coordinates))
            or not np.all(np.isfinite(y_coordinates))
        ):
            raise RuntimeError(f"{path}: invalid spatial coordinates")
        crs = str(dataset.attrs["crs"])
        x_resolution = float(np.median(np.abs(np.diff(x_coordinates))))
        y_resolution = float(np.median(np.abs(np.diff(y_coordinates))))
        if (
            x_resolution <= 0
            or y_resolution <= 0
            or not np.allclose(
                np.abs(np.diff(x_coordinates)), x_resolution, rtol=0, atol=1e-6
            )
            or not np.allclose(
                np.abs(np.diff(y_coordinates)), y_resolution, rtol=0, atol=1e-6
            )
        ):
            raise RuntimeError(f"{path}: invalid spatial resolution")
        native_bounds = [
            float(x_coordinates.min() - x_resolution / 2),
            float(y_coordinates.min() - y_resolution / 2),
            float(x_coordinates.max() + x_resolution / 2),
            float(y_coordinates.max() + y_resolution / 2),
        ]
        transformer = pyproj.Transformer.from_crs(
            crs, "EPSG:4326", always_xy=True
        )
        left, bottom, right, top = transformer.transform_bounds(
            *native_bounds, densify_pts=64
        )
        if (
            not np.all(np.isfinite([left, bottom, right, top]))
            or not -180 <= left <= 180
            or not -180 <= right <= 180
            or not -90 <= bottom < top <= 90
        ):
            raise RuntimeError(f"{path}: invalid EPSG:4326 footprint")
        longitude_intervals = (
            [[float(left), 180.0], [-180.0, float(right)]]
            if right < left
            else [[float(left), float(right)]]
        )
        geographic_footprint = {
            "longitude_intervals": longitude_intervals,
            "latitude_interval": [float(bottom), float(top)],
        }
    return {
        "image": image,
        "mask": mask,
        "high_confidence": high_confidence,
        "post_date": post_date,
        "scl_histogram": {
            str(code): int(np.sum(scl == code)) for code in sorted(set(scl.flat))
        },
        "cloud_fraction": float(np.mean(np.isin(scl, [8, 9, 10]))),
        "cloud_or_shadow_fraction": float(np.mean(np.isin(scl, [3, 8, 9, 10]))),
        "crs": crs,
        "native_pixel_edge_bounds": native_bounds,
        "epsg4326_footprint": geographic_footprint,
    }


def load_fold(fold: str) -> dict[str, Any]:
    manifest = json.loads(
        (METADATA / f"{fold}_extraction_manifest.json").read_text()
    )
    manifest_paths = sorted(STUDY / row["path"] for row in manifest["files"])
    observed_paths = sorted((DATA / fold).rglob("*.nc"))
    if observed_paths != manifest_paths:
        raise RuntimeError(f"{fold} on-disk file set does not match its manifest")
    rows = []
    for path in manifest_paths:
        record = load_file(path)
        rows.append(
            {
                "inventory": path.parent.name,
                "filename": path.name,
                **record,
            }
        )
    if not rows:
        raise RuntimeError(f"no {fold} files")
    return {
        "inventories": [row["inventory"] for row in rows],
        "filenames": [row["filename"] for row in rows],
        "x": np.stack([row["image"] for row in rows]).astype(np.float32),
        "y": np.stack([row["mask"] for row in rows]).astype(np.float32),
        "high_confidence": np.array(
            [row["high_confidence"] for row in rows], dtype=bool
        ),
        "post_dates": [row["post_date"] for row in rows],
        "crs": [row["crs"] for row in rows],
        "native_pixel_edge_bounds": np.asarray(
            [row["native_pixel_edge_bounds"] for row in rows], dtype=np.float64
        ),
        "epsg4326_footprints": [row["epsg4326_footprint"] for row in rows],
        "scl_histograms": [row["scl_histogram"] for row in rows],
        "cloud_fractions": np.array(
            [row["cloud_fraction"] for row in rows], dtype=np.float64
        ),
        "cloud_or_shadow_fractions": np.array(
            [row["cloud_or_shadow_fraction"] for row in rows], dtype=np.float64
        ),
    }


def tensor_loader(
    x: np.ndarray, y: np.ndarray, shuffle: bool, seed: int
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        generator=generator,
    )


def infer(model: torch.nn.Module, x: np.ndarray) -> np.ndarray:
    model.eval()
    output = []
    loader = DataLoader(torch.from_numpy(x), batch_size=BATCH_SIZE, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            output.append(torch.sigmoid(model(batch.to(DEVICE))).cpu().numpy())
    return np.concatenate(output).astype(np.float32)


def pooled_f1(probabilities: np.ndarray, truth: np.ndarray) -> float:
    counts = confusion_counts(probabilities, truth)
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    return float(2 * tp / (2 * tp + fp + fn + 1e-12))


def confusion_counts(
    probabilities: np.ndarray, truth: np.ndarray
) -> dict[str, int]:
    prediction = probabilities >= THRESHOLD
    target = truth >= 0.5
    return {
        "tp": int(np.sum(prediction & target)),
        "fp": int(np.sum(prediction & ~target)),
        "fn": int(np.sum(~prediction & target)),
        "tn": int(np.sum(~prediction & ~target)),
    }


def boundary(mask: np.ndarray) -> np.ndarray:
    structure = np.ones((3, 3), dtype=bool)
    return binary_dilation(mask, structure=structure) ^ binary_erosion(
        mask, structure=structure, border_value=0
    )


def boundary_f1(
    probabilities: np.ndarray, truth: np.ndarray, empty_score: float = 1.0
) -> float:
    structure = np.ones((3, 3), dtype=bool)
    values = []
    for prediction, target in zip(probabilities >= THRESHOLD, truth >= 0.5):
        predicted_boundary = boundary(prediction)
        target_boundary = boundary(target)
        if not predicted_boundary.any() and not target_boundary.any():
            values.append(empty_score)
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


def frozen_configuration(
    arm: str, config: dict[str, str], seed: int
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "arm": arm,
        "arm_config": config,
        "seed": seed,
        "dataset_revision": DATASET_REVISION,
        "band_order": ["B02", "B03", "B04", "B08"],
        "normalization": "divide_by_10000_then_clip_0_1",
        "optimizer": {
            "name": "AdamW",
            "learning_rate": 1.5e-3,
            "weight_decay": 1e-4,
        },
        "scheduler": {
            "name": "CosineAnnealingLR",
            "epochs": EPOCHS,
            "eta_min": 1e-5,
        },
        "batch_size": BATCH_SIZE,
        "threshold": THRESHOLD,
        "augmentation": "none",
    }


def train_one(
    arm: str,
    config: dict[str, str],
    seed: int,
    train: dict[str, Any],
    validation: dict[str, Any],
) -> dict[str, Any]:
    set_seed(seed)
    model = ControlledS3Net(in_ch=4, gating_mode=config["gating"]).to(DEVICE)
    criterion = ControlledPixelLoss(mode=config["loss"]).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1.5e-3, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-5
    )
    train_loader = tensor_loader(train["x"], train["y"], True, seed)
    best_f1 = -1.0
    best_epoch = None
    best_state = None
    validation_history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y, batch_x)
            loss.backward()
            optimizer.step()
        scheduler.step()
        probabilities = infer(model, validation["x"])
        score = pooled_f1(probabilities, validation["y"])
        validation_history.append({"epoch": epoch, "f1": score})
        if score > best_f1:
            best_f1 = score
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("no checkpoint selected")
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    checkpoint = CHECKPOINTS / f"{arm}_seed{seed}.pt"
    torch.save(best_state, checkpoint)
    model.load_state_dict(best_state)
    frozen_probabilities = infer(model, validation["x"])
    frozen_f1 = pooled_f1(frozen_probabilities, validation["y"])
    if abs(frozen_f1 - best_f1) > 1e-12:
        raise RuntimeError("frozen validation prediction does not match selected F1")
    VALIDATION_PREDICTIONS.mkdir(parents=True, exist_ok=True)
    validation_prediction = VALIDATION_PREDICTIONS / f"{arm}_seed{seed}.npy"
    np.save(validation_prediction, frozen_probabilities)
    configuration = frozen_configuration(arm, config, seed)
    return {
        "arm": arm,
        "seed": seed,
        "config": config,
        "configuration_sha256": canonical_sha256(configuration),
        "configuration": configuration,
        "selected_epoch": best_epoch,
        "validation_history": validation_history,
        "validation_f1": frozen_f1,
        "validation_predictions": str(validation_prediction.relative_to(STUDY)),
        "validation_predictions_sha256": sha256(validation_prediction),
        "checkpoint": str(checkpoint.relative_to(STUDY)),
        "checkpoint_sha256": sha256(checkpoint),
    }


def development_validation_mask(filenames: list[str]) -> np.ndarray:
    return np.asarray(
        [
            int(
                hashlib.sha256(
                    f"{DEVELOPMENT_SPLIT_SALT}:{filename}".encode()
                ).hexdigest(),
                16,
            )
            % 5
            == 0
            for filename in filenames
        ],
        dtype=bool,
    )


def fit() -> None:
    if (DATA / "protected").exists() and any((DATA / "protected").rglob("*.nc")):
        raise RuntimeError("protected files already extracted before fit")
    fit_path = OUTPUT / "fit_decisions.json"
    occupied = [
        path
        for path in (fit_path, AUTHORIZATION, OUTPUT / "sen12_s2_confirmation_summary.json")
        if path.exists()
    ]
    if occupied or (
        CHECKPOINTS.exists() and any(CHECKPOINTS.iterdir())
    ) or (
        VALIDATION_PREDICTIONS.exists() and any(VALIDATION_PREDICTIONS.iterdir())
    ):
        raise RuntimeError(f"fit destination is not pristine: {occupied}")
    runtime = verify_runtime_identity()
    development_manifest = verify_extraction_manifest("development")
    development = load_fold("development")
    if set(development["inventories"]) != {"chimanimani", "dominicamaria", "china"}:
        raise RuntimeError("development inventory allocation mismatch")
    member_payload = json.loads(MEMBER_INDEX.read_text())
    expected_filenames = sorted(
        row["filename"]
        for row in member_payload["selected"]
        if row["inventory"] in {"chimanimani", "dominicamaria", "china"}
    )
    if sorted(development["filenames"]) != expected_filenames:
        raise RuntimeError("development extraction does not match member index")
    validation_mask = development_validation_mask(development["filenames"])
    train_mask = ~validation_mask
    train = {
        "x": development["x"][train_mask],
        "y": development["y"][train_mask],
    }
    validation = {
        "x": development["x"][validation_mask],
        "y": development["y"][validation_mask],
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    runs = [
        train_one(arm, config, seed, train, validation)
        for arm, config in ARMS.items()
        for seed in SEEDS
    ]
    arm_seed_mean = {
        arm: float(
            np.mean(
                [
                    row["validation_f1"]
                    for row in runs
                    if row["arm"] == arm
                ]
            )
        )
        for arm in ARMS
    }
    individual_pass = all(
        row["validation_f1"] >= INDIVIDUAL_VALIDATION_FLOOR for row in runs
    )
    arm_mean_pass = all(
        value >= ARM_MEAN_VALIDATION_FLOOR for value in arm_seed_mean.values()
    )
    inventory_array = np.asarray(development["inventories"])
    payload = {
        "schema_version": 1,
        "dataset_revision": DATASET_REVISION,
        "member_index_sha256": sha256(MEMBER_INDEX),
        "development_extraction_manifest_sha256": sha256(
            METADATA / "development_extraction_manifest.json"
        ),
        "development_file_content_identity_sha256": canonical_sha256(
            [
                {
                    "path": row["path"],
                    "size": row["size"],
                    "sha256": row["sha256"],
                }
                for row in development_manifest["files"]
            ]
        ),
        "development_filenames_sha256": canonical_sha256(expected_filenames),
        "source_artifacts": source_artifact_hashes(),
        "runtime": runtime,
        "device": str(DEVICE),
        "development_split_salt": DEVELOPMENT_SPLIT_SALT,
        "development_split_rule": (
            "sha256(salt:filename) modulo 5 equals zero for validation"
        ),
        "train_inventories": ["chimanimani", "china", "dominicamaria"],
        "validation_inventories": ["chimanimani", "china", "dominicamaria"],
        "train_count": int(np.sum(train_mask)),
        "validation_count": int(np.sum(validation_mask)),
        "validation_by_inventory": {
            inventory: int(
                np.sum(validation_mask & (inventory_array == inventory))
            )
            for inventory in sorted(set(development["inventories"]))
        },
        "individual_validation_floor": INDIVIDUAL_VALIDATION_FLOOR,
        "arm_mean_validation_floor": ARM_MEAN_VALIDATION_FLOOR,
        "arm_seed_mean_validation_f1": arm_seed_mean,
        "individual_validation_pass": individual_pass,
        "arm_mean_validation_pass": arm_mean_pass,
        "all_validation_pass": individual_pass and arm_mean_pass,
        "runs": runs,
    }
    fit_path.write_text(json.dumps(payload, indent=2) + "\n")
    (OUTPUT / "fit_decisions.sha256").write_text(f"{sha256(fit_path)}\n")
    print(
        f"fit {len(runs)} runs; all_validation_pass="
        f"{payload['all_validation_pass']}"
    )


def validate_fit_matrix(runs: list[dict[str, Any]]) -> None:
    expected_pairs = {(arm, seed) for arm in ARMS for seed in SEEDS}
    observed_pairs = {(row["arm"], row["seed"]) for row in runs}
    if len(runs) != 21 or observed_pairs != expected_pairs:
        raise RuntimeError("fit does not contain the exact 21 arm-seed pairs")


def verify_float32_prediction_reproduction(
    saved: np.ndarray, inferred: np.ndarray
) -> dict[str, Any]:
    if (
        saved.dtype != np.float32
        or inferred.dtype != np.float32
        or saved.shape != inferred.shape
    ):
        raise RuntimeError("checkpoint predictions are not matching float32 arrays")
    saved_bits = np.ascontiguousarray(saved).view(np.uint32).astype(np.int64)
    inferred_bits = (
        np.ascontiguousarray(inferred).view(np.uint32).astype(np.int64)
    )
    maximum_ulp = int(np.max(np.abs(saved_bits - inferred_bits)))
    classifications_equal = bool(
        np.array_equal(saved >= THRESHOLD, inferred >= THRESHOLD)
    )
    maximum_absolute_error = float(np.max(np.abs(inferred - saved)))
    absolute_tolerance = float(2 * np.finfo(np.float32).eps)
    relative_tolerance = 1e-6
    numerically_close = bool(
        np.allclose(
            saved,
            inferred,
            atol=absolute_tolerance,
            rtol=relative_tolerance,
        )
    )
    if not numerically_close or not classifications_equal:
        raise RuntimeError(
            "checkpoint/prediction reproduction mismatch: "
            f"max_ulp={maximum_ulp}, "
            f"maximum_absolute_error={maximum_absolute_error}, "
            f"classifications_equal={classifications_equal}"
        )
    return {
        "maximum_prediction_ulp": maximum_ulp,
        "maximum_absolute_error": maximum_absolute_error,
        "absolute_tolerance": absolute_tolerance,
        "relative_tolerance": relative_tolerance,
        "numerically_close": numerically_close,
        "threshold_classifications_identical": classifications_equal,
    }


def validate_fit_for_authorization(
    fit_payload: dict[str, Any],
) -> dict[str, Any]:
    validate_fit_matrix(fit_payload["runs"])
    development = load_fold("development")
    validation_mask = development_validation_mask(development["filenames"])
    expected_inventories = ["chimanimani", "china", "dominicamaria"]
    if fit_payload["development_split_salt"] != DEVELOPMENT_SPLIT_SALT:
        raise RuntimeError("fit development split salt changed")
    if fit_payload["train_inventories"] != expected_inventories:
        raise RuntimeError("fit training inventories changed")
    if fit_payload["validation_inventories"] != expected_inventories:
        raise RuntimeError("fit validation inventories changed")
    if fit_payload["train_count"] != int(np.sum(~validation_mask)):
        raise RuntimeError("fit training count mismatch")
    if fit_payload["validation_count"] != int(np.sum(validation_mask)):
        raise RuntimeError("fit validation count mismatch")
    inventory_array = np.asarray(development["inventories"])
    expected_validation_by_inventory = {
        inventory: int(np.sum(validation_mask & (inventory_array == inventory)))
        for inventory in expected_inventories
    }
    if (
        fit_payload["validation_by_inventory"]
        != expected_validation_by_inventory
    ):
        raise RuntimeError("fit validation inventory counts mismatch")
    if fit_payload["development_filenames_sha256"] != canonical_sha256(
        sorted(development["filenames"])
    ):
        raise RuntimeError("fit development filename identity mismatch")
    validation_x = development["x"][validation_mask]
    validation_y = development["y"][validation_mask]
    records = []
    for row in fit_payload["runs"]:
        arm = row["arm"]
        seed = row["seed"]
        expected_configuration = frozen_configuration(arm, ARMS[arm], seed)
        if row["config"] != ARMS[arm]:
            raise RuntimeError(f"{arm} seed {seed}: arm configuration mismatch")
        if row["configuration"] != expected_configuration:
            raise RuntimeError(f"{arm} seed {seed}: frozen configuration mismatch")
        if row["configuration_sha256"] != canonical_sha256(
            expected_configuration
        ):
            raise RuntimeError(f"{arm} seed {seed}: configuration hash mismatch")
        history = row["validation_history"]
        if [value["epoch"] for value in history] != list(range(1, EPOCHS + 1)):
            raise RuntimeError(f"{arm} seed {seed}: validation history mismatch")
        best_f1 = max(value["f1"] for value in history)
        first_best_epoch = next(
            value["epoch"] for value in history if value["f1"] == best_f1
        )
        if row["selected_epoch"] != first_best_epoch:
            raise RuntimeError(f"{arm} seed {seed}: selected epoch mismatch")
        prediction_path = STUDY / row["validation_predictions"]
        if sha256(prediction_path) != row["validation_predictions_sha256"]:
            raise RuntimeError(f"{arm} seed {seed}: prediction hash mismatch")
        saved_probabilities = np.load(prediction_path, allow_pickle=False)
        if (
            saved_probabilities.shape != validation_y.shape
            or not np.all(np.isfinite(saved_probabilities))
            or float(saved_probabilities.min()) < 0
            or float(saved_probabilities.max()) > 1
        ):
            raise RuntimeError(f"{arm} seed {seed}: invalid validation predictions")
        recomputed_f1 = pooled_f1(saved_probabilities, validation_y)
        if not np.isclose(recomputed_f1, row["validation_f1"], atol=1e-12, rtol=0):
            raise RuntimeError(f"{arm} seed {seed}: validation F1 mismatch")
        if not np.isclose(recomputed_f1, best_f1, atol=1e-12, rtol=0):
            raise RuntimeError(
                f"{arm} seed {seed}: checkpoint is not the best history state"
            )
        checkpoint_path = STUDY / row["checkpoint"]
        if sha256(checkpoint_path) != row["checkpoint_sha256"]:
            raise RuntimeError(f"{arm} seed {seed}: checkpoint hash mismatch")
        model = ControlledS3Net(
            in_ch=4, gating_mode=ARMS[arm]["gating"]
        ).to(DEVICE)
        state = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state)
        inferred_probabilities = infer(model, validation_x)
        reproduction = verify_float32_prediction_reproduction(
            saved_probabilities, inferred_probabilities
        )
        records.append(
            {
                "arm": arm,
                "seed": seed,
                "recomputed_validation_f1": recomputed_f1,
                **reproduction,
                "passes_individual_floor": (
                    recomputed_f1 >= INDIVIDUAL_VALIDATION_FLOOR
                ),
            }
        )
    arm_seed_mean = {
        arm: float(
            np.mean(
                [
                    row["recomputed_validation_f1"]
                    for row in records
                    if row["arm"] == arm
                ]
            )
        )
        for arm in ARMS
    }
    individual_pass = all(row["passes_individual_floor"] for row in records)
    arm_mean_pass = all(
        value >= ARM_MEAN_VALIDATION_FLOOR for value in arm_seed_mean.values()
    )
    if fit_payload["individual_validation_pass"] != individual_pass:
        raise RuntimeError("stored individual validation decision mismatch")
    if fit_payload["arm_mean_validation_pass"] != arm_mean_pass:
        raise RuntimeError("stored arm-mean validation decision mismatch")
    for arm, value in arm_seed_mean.items():
        if not np.isclose(
            value,
            fit_payload["arm_seed_mean_validation_f1"][arm],
            atol=1e-12,
            rtol=0,
        ):
            raise RuntimeError(f"{arm}: stored arm-mean F1 mismatch")
    recomputed_all_pass = individual_pass and arm_mean_pass
    if fit_payload["all_validation_pass"] != recomputed_all_pass:
        raise RuntimeError("stored aggregate validation decision mismatch")
    if not recomputed_all_pass:
        raise RuntimeError("validation floor failed; protected access forbidden")
    return {
        "validation_design": "deterministic mixed-inventory calibration split",
        "validation_inventories": expected_inventories,
        "development_split_salt": DEVELOPMENT_SPLIT_SALT,
        "validation_by_inventory": expected_validation_by_inventory,
        "validation_count": int(np.sum(validation_mask)),
        "expected_arm_seed_pairs": 21,
        "individual_validation_floor": INDIVIDUAL_VALIDATION_FLOOR,
        "arm_mean_validation_floor": ARM_MEAN_VALIDATION_FLOOR,
        "arm_seed_mean_validation_f1": arm_seed_mean,
        "individual_validation_pass": individual_pass,
        "arm_mean_validation_pass": arm_mean_pass,
        "records": records,
        "all_validation_pass": recomputed_all_pass,
    }


def authorize(public_protocol_commit: str) -> None:
    if (DATA / "protected").exists() and any((DATA / "protected").rglob("*.nc")):
        raise RuntimeError("protected files already extracted before authorization")
    if AUTHORIZATION.exists():
        raise RuntimeError("protected authorization already exists")
    fit_path = OUTPUT / "fit_decisions.json"
    fit_hash = sha256(fit_path)
    if (OUTPUT / "fit_decisions.sha256").read_text().strip() != fit_hash:
        raise RuntimeError("fit-decision sidecar hash mismatch")
    fit_payload = json.loads(fit_path.read_text())
    if not fit_payload["all_validation_pass"]:
        raise RuntimeError("validation floor failed; protected access forbidden")
    if len(fit_payload["runs"]) != len(ARMS) * len(SEEDS):
        raise RuntimeError("fit run count mismatch")
    if fit_payload["member_index_sha256"] != sha256(MEMBER_INDEX):
        raise RuntimeError("member-index changed after fit")
    if fit_payload["runtime"] != verify_runtime_identity():
        raise RuntimeError("runtime identity changed after fit")
    if fit_payload["development_extraction_manifest_sha256"] != sha256(
        METADATA / "development_extraction_manifest.json"
    ):
        raise RuntimeError("development extraction manifest changed after fit")
    verify_extraction_manifest("development")
    current_sources = source_artifact_hashes()
    if fit_payload["source_artifacts"] != current_sources:
        raise RuntimeError("source artifacts changed after fit")
    for row in fit_payload["runs"]:
        if row["configuration_sha256"] != canonical_sha256(row["configuration"]):
            raise RuntimeError("configuration identity mismatch")
        if sha256(STUDY / row["checkpoint"]) != row["checkpoint_sha256"]:
            raise RuntimeError("checkpoint identity mismatch")
        if (
            sha256(STUDY / row["validation_predictions"])
            != row["validation_predictions_sha256"]
        ):
            raise RuntimeError("validation-prediction identity mismatch")
    fit_gate_reverification = validate_fit_for_authorization(fit_payload)
    public_evidence = github_commit_evidence(public_protocol_commit)
    verify_public_sources(public_protocol_commit, current_sources)
    authorized_at = dt.datetime.now(dt.timezone.utc)
    publicly_observed_at = dt.datetime.fromisoformat(
        public_evidence["api_observed_at"].replace("Z", "+00:00")
    )
    if publicly_observed_at >= authorized_at:
        raise RuntimeError("public protocol observation does not precede authorization")
    protected = json.loads(ACCESS_AUDIT.read_text())[
        "untouched_protected_inventories"
    ]
    authorization = {
        "schema_version": 1,
        "status": "authorized",
        "authorized_at": authorized_at.isoformat().replace("+00:00", "Z"),
        "public_protocol_commit": public_protocol_commit,
        "public_protocol_commit_time": public_evidence["commit_record"]["commit"][
            "committer"
        ]["date"],
        "public_protocol_push_event_id": public_evidence["push_event_id"],
        "public_protocol_push_event_time": public_evidence["push_event_time"],
        "public_protocol_api_observed_at": public_evidence["api_observed_at"],
        "dataset_revision": DATASET_REVISION,
        "protected_inventories": protected,
        "member_index_sha256": sha256(MEMBER_INDEX),
        "fit_decisions_sha256": fit_hash,
        "development_extraction_manifest_sha256": fit_payload[
            "development_extraction_manifest_sha256"
        ],
        "development_file_content_identity_sha256": fit_payload[
            "development_file_content_identity_sha256"
        ],
        "fit_gate_reverification": fit_gate_reverification,
        "source_artifacts": current_sources,
        "checkpoints": [
            {
                "path": row["checkpoint"],
                "sha256": row["checkpoint_sha256"],
                "validation_predictions": row["validation_predictions"],
                "validation_predictions_sha256": row[
                    "validation_predictions_sha256"
                ],
                "configuration_sha256": row["configuration_sha256"],
                "arm": row["arm"],
                "seed": row["seed"],
                "validation_f1": row["validation_f1"],
            }
            for row in fit_payload["runs"]
        ],
        "threshold": THRESHOLD,
        "conditions": [
            "ndvi_attention_vs_raw_edge_upper_ci_below_1_point",
            "ndvi_modulated_vs_plain_upper_ci_below_1_point",
            "generic_boundary_all_control_means_positive_lower_ci_above_zero_8_of_10",
            "all_arm_absolute_mean_f1_at_least_0_10",
            "all_integrity_checks_pass",
        ],
    }
    AUTHORIZATION.write_text(json.dumps(authorization, indent=2) + "\n")
    print(f"authorized {len(protected)} protected inventories")


def record_public_authorization_receipt(public_authorization_commit: str) -> None:
    if PUBLIC_RECEIPT.exists():
        raise RuntimeError("public authorization receipt already exists")
    authorization = json.loads(AUTHORIZATION.read_text())
    authorization_hash = sha256(AUTHORIZATION)
    evidence = github_commit_evidence(public_authorization_commit)
    public_url = (
        f"https://raw.githubusercontent.com/{PUBLIC_REPOSITORY}/"
        f"{public_authorization_commit}/{PUBLIC_AUTHORIZATION_PATH}"
    )
    with urllib.request.urlopen(public_url) as response:
        public_hash = hashlib.sha256(response.read()).hexdigest()
    if public_hash != authorization_hash:
        raise RuntimeError("published authorization is not byte-identical")
    authorized_at = dt.datetime.fromisoformat(
        authorization["authorized_at"].replace("Z", "+00:00")
    )
    observed_at = dt.datetime.fromisoformat(
        evidence["api_observed_at"].replace("Z", "+00:00")
    )
    if observed_at <= authorized_at:
        raise RuntimeError("public authorization observation is not post-authorization")
    payload = {
        "schema_version": 1,
        "authorization_sha256": authorization_hash,
        "public_authorization_commit": public_authorization_commit,
        "public_authorization_commit_time": evidence["commit_record"]["commit"][
            "committer"
        ]["date"],
        "public_authorization_api_observed_at": evidence["api_observed_at"],
        "public_authorization_push_event_id": evidence["push_event_id"],
        "public_authorization_push_event_time": evidence["push_event_time"],
        "public_repository": f"https://github.com/{PUBLIC_REPOSITORY}",
        "status": "public-authorization-byte-identity-verified",
    }
    PUBLIC_RECEIPT.write_text(json.dumps(payload, indent=2) + "\n")
    verify_authorization_state()
    print("recorded and verified public authorization receipt")


def verify_authorization_state() -> tuple[dict[str, Any], dict[str, Any]]:
    authorization = json.loads(AUTHORIZATION.read_text())
    receipt = json.loads(PUBLIC_RECEIPT.read_text())
    if authorization.get("status") != "authorized":
        raise RuntimeError("protected access is not authorized")
    if receipt["authorization_sha256"] != sha256(AUTHORIZATION):
        raise RuntimeError("authorization receipt mismatch")
    authorization_commit = receipt["public_authorization_commit"]
    authorization_evidence = github_commit_evidence(authorization_commit)
    if authorization_evidence["push_event_id"] is not None and receipt.get(
        "public_authorization_push_event_id"
    ) not in (None, authorization_evidence["push_event_id"]):
        raise RuntimeError("public authorization push evidence mismatch")
    authorized_at = dt.datetime.fromisoformat(
        authorization["authorized_at"].replace("Z", "+00:00")
    )
    protocol_observed_at = dt.datetime.fromisoformat(
        authorization["public_protocol_api_observed_at"].replace("Z", "+00:00")
    )
    receipt_observed_at = dt.datetime.fromisoformat(
        receipt["public_authorization_api_observed_at"].replace("Z", "+00:00")
    )
    current_observed_at = dt.datetime.fromisoformat(
        authorization_evidence["api_observed_at"].replace("Z", "+00:00")
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
        f"{authorization_commit}/{PUBLIC_AUTHORIZATION_PATH}"
    )
    with urllib.request.urlopen(public_authorization_url) as response:
        public_authorization_hash = hashlib.sha256(response.read()).hexdigest()
    if public_authorization_hash != receipt["authorization_sha256"]:
        raise RuntimeError("public authorization artifact hash mismatch")
    if authorization["member_index_sha256"] != sha256(MEMBER_INDEX):
        raise RuntimeError("member index changed after authorization")
    fit_path = OUTPUT / "fit_decisions.json"
    if authorization["fit_decisions_sha256"] != sha256(fit_path):
        raise RuntimeError("fit decisions changed after authorization")
    fit_payload = json.loads(fit_path.read_text())
    if fit_payload["runtime"] != verify_runtime_identity():
        raise RuntimeError("runtime identity changed after authorization")
    if authorization["development_extraction_manifest_sha256"] != sha256(
        METADATA / "development_extraction_manifest.json"
    ):
        raise RuntimeError(
            "development extraction manifest changed after authorization"
        )
    verify_extraction_manifest("development")
    current_sources = source_artifact_hashes()
    if authorization["source_artifacts"] != current_sources:
        raise RuntimeError("source artifacts changed after authorization")
    verify_public_sources(authorization["public_protocol_commit"], current_sources)
    for row in authorization["checkpoints"]:
        if sha256(STUDY / row["path"]) != row["sha256"]:
            raise RuntimeError(f"checkpoint changed: {row['path']}")
        if (
            sha256(STUDY / row["validation_predictions"])
            != row["validation_predictions_sha256"]
        ):
            raise RuntimeError(
                f"validation predictions changed: {row['validation_predictions']}"
            )
    return authorization, receipt


def interval(values: list[float], salt: str) -> list[float]:
    rng = np.random.default_rng(
        int(hashlib.sha256(salt.encode()).hexdigest()[:16], 16)
    )
    array = np.asarray(values, dtype=float)
    draws = np.mean(
        array[rng.integers(0, len(array), size=(10000, len(array)))], axis=1
    )
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def cross_inventory_overlaps(fold: dict[str, Any]) -> list[dict[str, str]]:
    overlaps = []
    for left in range(len(fold["filenames"])):
        for right in range(left + 1, len(fold["filenames"])):
            if fold["inventories"][left] == fold["inventories"][right]:
                continue
            left_footprint = fold["epsg4326_footprints"][left]
            right_footprint = fold["epsg4326_footprints"][right]
            latitude_overlap = max(
                left_footprint["latitude_interval"][0],
                right_footprint["latitude_interval"][0],
            ) < min(
                left_footprint["latitude_interval"][1],
                right_footprint["latitude_interval"][1],
            )
            longitude_overlap = any(
                max(left_interval[0], right_interval[0])
                < min(left_interval[1], right_interval[1])
                for left_interval in left_footprint["longitude_intervals"]
                for right_interval in right_footprint["longitude_intervals"]
            )
            if longitude_overlap and latitude_overlap:
                overlaps.append(
                    {
                        "left_inventory": fold["inventories"][left],
                        "left_filename": fold["filenames"][left],
                        "right_inventory": fold["inventories"][right],
                        "right_filename": fold["filenames"][right],
                        "comparison_crs": "EPSG:4326",
                        "left_source_crs": fold["crs"][left],
                        "right_source_crs": fold["crs"][right],
                    }
                )
    return overlaps


def evaluate() -> None:
    authorization, receipt = verify_authorization_state()
    protected_manifest_path = METADATA / "protected_extraction_manifest.json"
    protected_manifest = verify_extraction_manifest("protected")
    protected_manifest_hash = sha256(protected_manifest_path)
    protected_content_identity_hash = canonical_sha256(
        [
            {
                "path": row["path"],
                "size": row["size"],
                "sha256": row["sha256"],
            }
            for row in protected_manifest["files"]
        ]
    )
    protected = load_fold("protected")
    if set(protected["inventories"]) != set(authorization["protected_inventories"]):
        raise RuntimeError("protected inventory allocation mismatch")
    member_payload = json.loads(MEMBER_INDEX.read_text())
    expected_filenames = sorted(
        row["filename"]
        for row in member_payload["selected"]
        if row["inventory"] in set(authorization["protected_inventories"])
    )
    if sorted(protected["filenames"]) != expected_filenames:
        raise RuntimeError("protected extraction does not match member index")
    fit_payload = json.loads((OUTPUT / "fit_decisions.json").read_text())
    runs = []
    probability_dir = OUTPUT / "probabilities"
    probability_dir.mkdir(parents=True, exist_ok=True)
    inventories = sorted(set(protected["inventories"]))
    inventory_metadata = {}
    for inventory in inventories:
        selected = np.array(
            [value == inventory for value in protected["inventories"]], dtype=bool
        )
        scl_histogram: dict[str, int] = {}
        for histogram in np.asarray(
            protected["scl_histograms"], dtype=object
        )[selected]:
            for code, count in histogram.items():
                scl_histogram[code] = scl_histogram.get(code, 0) + int(count)
        truth = protected["y"][selected]
        inventory_metadata[inventory] = {
            "n": int(np.sum(selected)),
            "positive_fraction": float(np.mean(truth)),
            "empty_masks": int(np.sum(~np.any(truth > 0, axis=(1, 2)))),
            "high_confidence_n": int(
                np.sum(protected["high_confidence"][selected])
            ),
            "first_post_dates": sorted(
                set(np.asarray(protected["post_dates"])[selected].tolist())
            ),
            "scl_histogram": dict(sorted(scl_histogram.items())),
            "mean_cloud_fraction_scl_8_9_10": float(
                np.mean(protected["cloud_fractions"][selected])
            ),
            "mean_cloud_or_shadow_fraction_scl_3_8_9_10": float(
                np.mean(protected["cloud_or_shadow_fractions"][selected])
            ),
        }
    for fit_row in fit_payload["runs"]:
        model = ControlledS3Net(
            in_ch=4, gating_mode=fit_row["config"]["gating"]
        ).to(DEVICE)
        state = torch.load(
            STUDY / fit_row["checkpoint"], map_location=DEVICE, weights_only=True
        )
        model.load_state_dict(state)
        probabilities = infer(model, protected["x"])
        if (
            not np.all(np.isfinite(probabilities))
            or float(probabilities.min()) < 0
            or float(probabilities.max()) > 1
        ):
            raise RuntimeError("protected probabilities are not finite probabilities")
        event_metrics = {}
        for inventory in inventories:
            selected = np.array(
                [value == inventory for value in protected["inventories"]],
                dtype=bool,
            )
            destination = (
                probability_dir
                / f"{fit_row['arm']}_seed{fit_row['seed']}_{inventory}.npy"
            )
            np.save(destination, probabilities[selected].astype(np.float32))
            truth = protected["y"][selected]
            high = protected["high_confidence"][selected]
            event_metrics[inventory] = {
                "n": int(np.sum(selected)),
                "positive_fraction": float(np.mean(truth)),
                "empty_masks": int(np.sum(~np.any(truth > 0, axis=(1, 2)))),
                "confusion_counts": confusion_counts(
                    probabilities[selected], truth
                ),
                "f1": pooled_f1(probabilities[selected], truth),
                "boundary_f1": boundary_f1(probabilities[selected], truth),
                "boundary_f1_empty_zero": boundary_f1(
                    probabilities[selected], truth, empty_score=0.0
                ),
                "high_confidence_n": int(np.sum(high)),
                "high_confidence_f1": (
                    pooled_f1(probabilities[selected][high], truth[high])
                    if np.any(high)
                    else None
                ),
                "high_confidence_boundary_f1": (
                    boundary_f1(probabilities[selected][high], truth[high])
                    if np.any(high)
                    else None
                ),
                "high_confidence_boundary_f1_empty_zero": (
                    boundary_f1(
                        probabilities[selected][high],
                        truth[high],
                        empty_score=0.0,
                    )
                    if np.any(high)
                    else None
                ),
                "probabilities": str(destination.relative_to(STUDY)),
                "probabilities_sha256": sha256(destination),
            }
        runs.append({**fit_row, "protected": event_metrics})

    lookup = {(row["arm"], row["seed"]): row for row in runs}

    def event_effect(treatment: str, control: str, metric: str) -> dict[str, float]:
        output = {}
        for inventory in inventories:
            values = []
            for seed in SEEDS:
                treatment_value = lookup[(treatment, seed)]["protected"][inventory][
                    metric
                ]
                control_value = lookup[(control, seed)]["protected"][inventory][
                    metric
                ]
                if treatment_value is not None and control_value is not None:
                    values.append(treatment_value - control_value)
            if values:
                output[inventory] = float(np.mean(values))
        return output

    def event_seed_effect(
        treatment: str, control: str, metric: str
    ) -> dict[str, dict[str, float]]:
        output = {}
        for inventory in inventories:
            values = {}
            for seed in SEEDS:
                treatment_value = lookup[(treatment, seed)]["protected"][inventory][
                    metric
                ]
                control_value = lookup[(control, seed)]["protected"][inventory][
                    metric
                ]
                if treatment_value is not None and control_value is not None:
                    values[str(seed)] = float(treatment_value - control_value)
            if values:
                output[inventory] = values
        return output

    def summarize(values: dict[str, float], name: str) -> dict[str, Any]:
        ordered = [values[inventory] for inventory in sorted(values)]
        if not ordered:
            return {
                "per_inventory": values,
                "inventory_count": 0,
                "mean": None,
                "interval": None,
            }
        return {
            "per_inventory": values,
            "inventory_count": len(ordered),
            "mean": float(np.mean(ordered)),
            "interval": interval(ordered, name),
        }

    def contrast_summary(
        treatment: str, control: str, metric: str, name: str
    ) -> dict[str, Any]:
        return {
            **summarize(event_effect(treatment, control, metric), name),
            "per_inventory_seed": event_seed_effect(treatment, control, metric),
            "treatment": treatment,
            "control": control,
            "metric": metric,
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
    summaries = {
        name: contrast_summary(*spec, name)
        for name, spec in contrast_specs.items()
    }
    control_specs = {
        control: (boundary_arm, base_arm)
        for control, base_arm, boundary_arm in (
            ("zero_control", "s3_none_base", "s3_none_boundary"),
            ("raw", "s3_raw_base", "s3_raw_boundary"),
            ("ndvi", "s3_ndvi_base", "s3_ndvi_boundary"),
        )
    }
    generic_summaries = {
        control: contrast_summary(
            boundary_arm,
            base_arm,
            "f1",
            f"generic-{control}",
        )
        for control, (boundary_arm, base_arm) in control_specs.items()
    }
    generic_average = {
        inventory: float(
            np.mean(
                [
                    generic_summaries[control]["per_inventory"][inventory]
                    for control in generic_summaries
                ]
            )
        )
        for inventory in inventories
    }
    pooled_generic = summarize(generic_average, "generic-average")

    absolute_performance = {}
    for arm in ARMS:
        absolute_performance[arm] = {}
        for metric in ("f1", "boundary_f1", "boundary_f1_empty_zero"):
            per_inventory = {
                inventory: float(
                    np.mean(
                        [
                            lookup[(arm, seed)]["protected"][inventory][metric]
                            for seed in SEEDS
                        ]
                    )
                )
                for inventory in inventories
            }
            absolute_performance[arm][metric] = summarize(
                per_inventory, f"absolute-{arm}-{metric}"
            )

    high_confidence_contrasts = {
        name: contrast_summary(
            treatment,
            control,
            "high_confidence_" + metric,
            f"high-confidence-{name}",
        )
        for name, (treatment, control, metric) in contrast_specs.items()
    }
    high_confidence_generic = {
        control: contrast_summary(
            boundary_arm,
            base_arm,
            "high_confidence_f1",
            f"high-confidence-generic-{control}",
        )
        for control, (boundary_arm, base_arm) in control_specs.items()
    }
    high_confidence_absolute = {}
    for arm in ARMS:
        high_confidence_absolute[arm] = {}
        for metric in ("high_confidence_f1", "high_confidence_boundary_f1"):
            per_inventory = {}
            for inventory in inventories:
                values = [
                    lookup[(arm, seed)]["protected"][inventory][metric]
                    for seed in SEEDS
                ]
                available = [value for value in values if value is not None]
                if available:
                    per_inventory[inventory] = float(np.mean(available))
            high_confidence_absolute[arm][metric] = summarize(
                per_inventory, f"high-confidence-absolute-{arm}-{metric}"
            )
    empty_zero_boundary_contrast = contrast_summary(
        "s3_ndvi_biophysical",
        "s3_ndvi_boundary",
        "boundary_f1_empty_zero",
        "empty-zero-ndvi-modulated-vs-plain-boundary-f1",
    )
    protected_overlaps = cross_inventory_overlaps(protected)
    development_inventories = set(
        json.loads(ACCESS_AUDIT.read_text())["development_train_inventories"]
        + json.loads(ACCESS_AUDIT.read_text())[
            "development_validation_inventories"
        ]
    )
    selected_development_files = {
        row["filename"]
        for row in member_payload["selected"]
        if row["inventory"] in development_inventories
    }
    selected_protected_files = set(expected_filenames)
    arithmetic_f1_pass = all(
        np.isclose(
            event["f1"],
            2
            * event["confusion_counts"]["tp"]
            / (
                2 * event["confusion_counts"]["tp"]
                + event["confusion_counts"]["fp"]
                + event["confusion_counts"]["fn"]
                + 1e-12
            ),
            atol=1e-12,
            rtol=0,
        )
        for run in runs
        for event in run["protected"].values()
    )
    probability_hash_pass = all(
        sha256(STUDY / event["probabilities"])
        == event["probabilities_sha256"]
        for run in runs
        for event in run["protected"].values()
    )
    registered_checks = [
        {
            "name": "public_protocol_and_authorization_byte_identity",
            "passed": (
                len(authorization["public_protocol_commit"]) == 40
                and len(receipt["public_authorization_commit"]) == 40
                and receipt["authorization_sha256"] == sha256(AUTHORIZATION)
                and authorization["source_artifacts"] == source_artifact_hashes()
            ),
            "evidence": (
                f"protocol={authorization['public_protocol_commit']}; "
                f"authorization={receipt['public_authorization_commit']}"
            ),
        },
        {
            "name": "authorized_source_fit_checkpoint_validation_identity",
            "passed": (
                authorization["source_artifacts"] == source_artifact_hashes()
                and authorization["fit_decisions_sha256"]
                == sha256(OUTPUT / "fit_decisions.json")
                and len(authorization["checkpoints"]) == 21
            ),
        },
        {
            "name": "exact_21_unique_arm_seed_configurations",
            "passed": (
                len(lookup) == 21
                and len(
                    {
                        (row["arm"], row["seed"], row["configuration_sha256"])
                        for row in runs
                    }
                )
                == 21
            ),
        },
        {
            "name": "exact_authorized_protected_inventories",
            "passed": (
                len(inventories) == 10
                and set(inventories)
                == set(authorization["protected_inventories"])
            ),
        },
        {
            "name": "exact_member_index_file_set",
            "passed": sorted(protected["filenames"]) == expected_filenames,
        },
        {
            "name": "protected_extraction_manifest_and_file_content_identity",
            "passed": (
                protected_manifest["authorization_sha256"]
                == sha256(AUTHORIZATION)
                and len(protected_manifest["files"]) == len(expected_filenames)
            ),
            "protected_extraction_manifest_sha256": protected_manifest_hash,
            "protected_file_content_identity_sha256": (
                protected_content_identity_hash
            ),
        },
        {
            "name": "development_protected_disjointness",
            "passed": (
                not development_inventories
                & set(authorization["protected_inventories"])
                and not selected_development_files & selected_protected_files
            ),
        },
        {
            "name": "cross_inventory_epsg4326_footprint_nonoverlap",
            "passed": not protected_overlaps,
            "observed_overlap_count": len(protected_overlaps),
        },
        {
            "name": "strict_semantic_loader_completed",
            "passed": len(protected["filenames"]) == len(expected_filenames),
            "files_checked": len(protected["filenames"]),
        },
        {
            "name": "probability_artifact_hashes",
            "passed": probability_hash_pass,
        },
        {
            "name": "confusion_count_f1_arithmetic",
            "passed": arithmetic_f1_pass,
        },
    ]
    all_integrity_checks_pass = all(
        check["passed"] for check in registered_checks
    )
    conditions = {
        "ndvi_attention_vs_raw_edge_upper_ci_below_1_point": (
            summaries["ndvi_attention_vs_raw_edge_f1"]["interval"][1] < 0.01
        ),
        "ndvi_modulated_vs_plain_upper_ci_below_1_point": (
            summaries["ndvi_modulated_vs_plain_f1"]["interval"][1] < 0.01
        ),
        "generic_boundary_all_control_means_positive_lower_ci_above_zero_8_of_10": (
            all(row["mean"] > 0 for row in generic_summaries.values())
            and pooled_generic["interval"][0] > 0
            and sum(value > 0 for value in generic_average.values()) >= 8
        ),
        "all_arm_absolute_mean_f1_at_least_0_10": all(
            absolute_performance[arm]["f1"]["mean"] >= 0.10 for arm in ARMS
        ),
        "all_integrity_checks_pass": all_integrity_checks_pass,
    }
    output = {
        "schema_version": 1,
        "dataset_revision": DATASET_REVISION,
        "protocol_sha256": sha256(PROTOCOL),
        "authorization_sha256": sha256(AUTHORIZATION),
        "protected_extraction_manifest_sha256": protected_manifest_hash,
        "protected_file_content_identity_sha256": (
            protected_content_identity_hash
        ),
        "public_authorization_commit": receipt["public_authorization_commit"],
        "threshold": THRESHOLD,
        "metric_units": {
            "f1_and_boundary_f1": "fraction",
            "contrasts": "fraction; multiply by 100 for percentage points",
        },
        "inventories": inventories,
        "inventory_metadata": inventory_metadata,
        "cross_inventory_spatial_overlaps": protected_overlaps,
        "integrity_checks": registered_checks,
        "runs": runs,
        "absolute_performance": absolute_performance,
        "contrasts": summaries,
        "generic_boundary_by_control": generic_summaries,
        "generic_boundary_average": pooled_generic,
        "sensitivities": {
            "high_confidence_absolute_performance": high_confidence_absolute,
            "high_confidence_contrasts": high_confidence_contrasts,
            "high_confidence_generic_boundary_by_control": high_confidence_generic,
            "empty_empty_boundary_score_zero": empty_zero_boundary_contrast,
        },
        "conditions": conditions,
        "all_conditions_pass": all(conditions.values()),
        "scope": (
            "Inventory-level confirmation on ten untouched public Sen12 "
            "Sentinel-2 inventories with authoritative named bands."
        ),
    }
    (OUTPUT / "sen12_s2_confirmation_summary.json").write_text(
        json.dumps(output, indent=2) + "\n"
    )
    print(f"evaluated {len(inventories)} inventories: {conditions}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("fit", "authorize", "receipt", "evaluate"),
        required=True,
    )
    parser.add_argument("--public-protocol-commit")
    parser.add_argument("--public-authorization-commit")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.stage == "fit":
        fit()
    elif args.stage == "authorize":
        if not args.public_protocol_commit:
            raise RuntimeError("--public-protocol-commit is required")
        authorize(args.public_protocol_commit)
    elif args.stage == "receipt":
        if not args.public_authorization_commit:
            raise RuntimeError("--public-authorization-commit is required")
        record_public_authorization_receipt(args.public_authorization_commit)
    else:
        evaluate()


if __name__ == "__main__":
    main()
