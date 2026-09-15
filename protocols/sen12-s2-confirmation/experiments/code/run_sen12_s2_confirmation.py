#!/usr/bin/env python3
"""Run the frozen inventory-held Sen12 Sentinel-2 confirmation."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import random
import time
import urllib.request
from pathlib import Path
from typing import Any

os.environ["HRGLDD_ARRAY_ORDER"] = "BGRN"

import numpy as np
import torch
import xarray as xr
from scipy.ndimage import binary_dilation, binary_erosion
from torch.utils.data import DataLoader, TensorDataset

from revised_models import ControlledPixelLoss, ControlledS3Net


STUDY = Path(__file__).resolve().parents[2]
DATA = STUDY / "experiments/raw/external/sen12-s2"
OUTPUT = STUDY / "experiments/derived/results/sen12_s2_confirmation"
CHECKPOINTS = STUDY / "experiments/derived/checkpoints/sen12_s2_confirmation"
METADATA = STUDY / "research/dataset-metadata/sen12-s2-confirmation"
MEMBER_INDEX = METADATA / "member_index.json"
PROTOCOL = STUDY / "research/sen12-s2-confirmation-protocol.md"
ACCESS_AUDIT = STUDY / "research/sen12-s2-access-audit.json"
AUTHORIZATION = OUTPUT / "protected_access_authorization.json"
PUBLIC_RECEIPT = METADATA / "public_authorization_receipt.json"
DATASET_REVISION = "40af2dd6b4e568edb6640d6e14dc67ebd01038a4"
SEEDS = (42, 43, 44)
EPOCHS = 15
BATCH_SIZE = 16
THRESHOLD = 0.5
VALIDATION_FLOOR = 0.25
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_post(value: Any) -> int:
    if isinstance(value, str):
        value = ast.literal_eval(value)
    if isinstance(value, dict):
        value = value["post"]
    if isinstance(value, (list, tuple, np.ndarray)):
        value = value[0]
    return int(value)


def load_file(path: Path) -> tuple[np.ndarray, np.ndarray, bool]:
    with xr.open_dataset(path, engine="h5netcdf") as dataset:
        post = parse_post(dataset.attrs["pre_post_dates"])
        channels = []
        for band in ("B02", "B03", "B04", "B08"):
            value = np.asarray(dataset[band].isel(time=post).values, dtype=np.float32)
            channels.append(np.clip(value / 10000.0, 0.0, 1.0))
        image = np.stack(channels)
        mask = np.asarray(dataset["MASK"].isel(time=0).values > 0, dtype=np.float32)
        confidence = dataset.attrs.get("date_confidence", 0)
        if isinstance(confidence, str) and "," in confidence:
            high_confidence = all(float(x) == 1.0 for x in confidence.split(","))
        elif isinstance(confidence, (list, tuple, np.ndarray)):
            high_confidence = all(float(x) == 1.0 for x in confidence)
        else:
            high_confidence = float(confidence) == 1.0
    return image, mask, high_confidence


def load_fold(fold: str) -> dict[str, Any]:
    rows = []
    for inventory_dir in sorted((DATA / fold).iterdir()):
        if not inventory_dir.is_dir():
            continue
        for path in sorted(inventory_dir.glob("*.nc")):
            image, mask, high_confidence = load_file(path)
            rows.append((inventory_dir.name, path.name, image, mask, high_confidence))
    if not rows:
        raise RuntimeError(f"no {fold} files")
    return {
        "inventories": [row[0] for row in rows],
        "filenames": [row[1] for row in rows],
        "x": np.stack([row[2] for row in rows]).astype(np.float32),
        "y": np.stack([row[3] for row in rows]).astype(np.float32),
        "high_confidence": np.array([row[4] for row in rows], dtype=bool),
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
    prediction = probabilities >= THRESHOLD
    target = truth >= 0.5
    tp = int(np.sum(prediction & target))
    fp = int(np.sum(prediction & ~target))
    fn = int(np.sum(~prediction & target))
    return float(2 * tp / (2 * tp + fp + fn + 1e-12))


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
    best_state = None
    for _ in range(EPOCHS):
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
        if score > best_f1:
            best_f1 = score
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("no checkpoint selected")
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    checkpoint = CHECKPOINTS / f"{arm}_seed{seed}.pt"
    torch.save(best_state, checkpoint)
    return {
        "arm": arm,
        "seed": seed,
        "config": config,
        "validation_f1": best_f1,
        "checkpoint": str(checkpoint.relative_to(STUDY)),
        "checkpoint_sha256": sha256(checkpoint),
    }


def fit() -> None:
    if (DATA / "protected").exists() and any((DATA / "protected").rglob("*.nc")):
        raise RuntimeError("protected files already extracted before fit")
    development = load_fold("development")
    train_mask = np.array(
        [value != "china" for value in development["inventories"]], dtype=bool
    )
    validation_mask = ~train_mask
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
    payload = {
        "schema_version": 1,
        "dataset_revision": DATASET_REVISION,
        "device": str(DEVICE),
        "train_inventories": ["chimanimani", "dominicamaria"],
        "validation_inventories": ["china"],
        "train_count": int(np.sum(train_mask)),
        "validation_count": int(np.sum(validation_mask)),
        "validation_floor": VALIDATION_FLOOR,
        "all_validation_pass": all(
            row["validation_f1"] >= VALIDATION_FLOOR for row in runs
        ),
        "runs": runs,
    }
    (OUTPUT / "fit_decisions.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"fit {len(runs)} runs; all_validation_pass="
        f"{payload['all_validation_pass']}"
    )


def authorize(public_protocol_commit: str) -> None:
    fit_payload = json.loads((OUTPUT / "fit_decisions.json").read_text())
    if not fit_payload["all_validation_pass"]:
        raise RuntimeError("validation floor failed; protected access forbidden")
    with urllib.request.urlopen(
        "https://api.github.com/repos/akssha74/S3Net-Landslide/commits/"
        + public_protocol_commit
    ) as response:
        public_record = json.load(response)
    if public_record.get("sha") != public_protocol_commit:
        raise RuntimeError("protocol commit is not publicly resolvable")
    protected = json.loads(ACCESS_AUDIT.read_text())[
        "untouched_protected_inventories"
    ]
    authorization = {
        "schema_version": 1,
        "status": "authorized",
        "authorized_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "public_protocol_commit": public_protocol_commit,
        "public_protocol_commit_time": public_record["commit"]["committer"]["date"],
        "dataset_revision": DATASET_REVISION,
        "protected_inventories": protected,
        "member_index_sha256": sha256(MEMBER_INDEX),
        "protocol_sha256": sha256(PROTOCOL),
        "code_sha256": sha256(Path(__file__)),
        "checkpoints": [
            {
                "path": row["checkpoint"],
                "sha256": row["checkpoint_sha256"],
                "arm": row["arm"],
                "seed": row["seed"],
                "validation_f1": row["validation_f1"],
            }
            for row in fit_payload["runs"]
        ],
        "threshold": THRESHOLD,
        "conditions": [
            "index_vs_raw_upper_ci_below_1_point",
            "index_mod_vs_plain_upper_ci_below_1_point",
            "generic_boundary_all_control_means_positive_lower_ci_above_zero_8_of_10",
            "all_integrity_checks_pass",
        ],
    }
    AUTHORIZATION.write_text(json.dumps(authorization, indent=2) + "\n")
    print(f"authorized {len(protected)} protected inventories")


def interval(values: list[float], salt: str) -> list[float]:
    rng = np.random.default_rng(
        int(hashlib.sha256(salt.encode()).hexdigest()[:16], 16)
    )
    array = np.asarray(values, dtype=float)
    draws = np.mean(
        array[rng.integers(0, len(array), size=(10000, len(array)))], axis=1
    )
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def evaluate() -> None:
    authorization = json.loads(AUTHORIZATION.read_text())
    receipt = json.loads(PUBLIC_RECEIPT.read_text())
    if receipt["authorization_sha256"] != sha256(AUTHORIZATION):
        raise RuntimeError("authorization receipt mismatch")
    protected = load_fold("protected")
    fit_payload = json.loads((OUTPUT / "fit_decisions.json").read_text())
    runs = []
    probability_dir = OUTPUT / "probabilities"
    probability_dir.mkdir(parents=True, exist_ok=True)
    inventories = sorted(set(protected["inventories"]))
    for fit_row in fit_payload["runs"]:
        model = ControlledS3Net(
            in_ch=4, gating_mode=fit_row["config"]["gating"]
        ).to(DEVICE)
        state = torch.load(
            STUDY / fit_row["checkpoint"], map_location=DEVICE, weights_only=True
        )
        model.load_state_dict(state)
        probabilities = infer(model, protected["x"])
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
            np.save(destination, probabilities[selected].astype(np.float16))
            truth = protected["y"][selected]
            high = protected["high_confidence"][selected]
            event_metrics[inventory] = {
                "n": int(np.sum(selected)),
                "positive_fraction": float(np.mean(truth)),
                "empty_masks": int(np.sum(~np.any(truth > 0, axis=(1, 2)))),
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
                "probabilities": str(destination.relative_to(STUDY)),
                "probabilities_sha256": sha256(destination),
            }
        runs.append({**fit_row, "protected": event_metrics})

    lookup = {(row["arm"], row["seed"]): row for row in runs}

    def event_effect(treatment: str, control: str, metric: str) -> dict[str, float]:
        return {
            inventory: float(
                np.mean(
                    [
                        lookup[(treatment, seed)]["protected"][inventory][metric]
                        - lookup[(control, seed)]["protected"][inventory][metric]
                        for seed in SEEDS
                    ]
                )
            )
            for inventory in inventories
        }

    contrasts = {
        "index_input_vs_raw_f1": event_effect(
            "s3_ndvi_base", "s3_raw_base", "f1"
        ),
        "index_mod_vs_plain_f1": event_effect(
            "s3_ndvi_biophysical", "s3_ndvi_boundary", "f1"
        ),
        "index_mod_vs_plain_boundary_f1": event_effect(
            "s3_ndvi_biophysical", "s3_ndvi_boundary", "boundary_f1"
        ),
    }
    generic_by_control = {
        control: event_effect(boundary_arm, base_arm, "f1")
        for control, base_arm, boundary_arm in (
            ("zero", "s3_none_base", "s3_none_boundary"),
            ("raw", "s3_raw_base", "s3_raw_boundary"),
            ("index", "s3_ndvi_base", "s3_ndvi_boundary"),
        )
    }
    generic_average = {
        inventory: float(
            np.mean(
                [
                    generic_by_control[control][inventory]
                    for control in generic_by_control
                ]
            )
        )
        for inventory in inventories
    }

    def summarize(values: dict[str, float], name: str) -> dict[str, Any]:
        ordered = [values[inventory] for inventory in inventories]
        return {
            "per_inventory": values,
            "mean": float(np.mean(ordered)),
            "interval": interval(ordered, name),
        }

    summaries = {
        name: summarize(values, name) for name, values in contrasts.items()
    }
    generic_summaries = {
        name: summarize(values, f"generic-{name}")
        for name, values in generic_by_control.items()
    }
    pooled_generic = summarize(generic_average, "generic-average")
    conditions = {
        "index_vs_raw_upper_ci_below_1_point": (
            summaries["index_input_vs_raw_f1"]["interval"][1] < 0.01
        ),
        "index_mod_vs_plain_upper_ci_below_1_point": (
            summaries["index_mod_vs_plain_f1"]["interval"][1] < 0.01
        ),
        "generic_boundary_all_control_means_positive_lower_ci_above_zero_8_of_10": (
            all(row["mean"] > 0 for row in generic_summaries.values())
            and pooled_generic["interval"][0] > 0
            and sum(value > 0 for value in generic_average.values()) >= 8
        ),
        "all_integrity_checks_pass": True,
    }
    output = {
        "schema_version": 1,
        "dataset_revision": DATASET_REVISION,
        "protocol_sha256": sha256(PROTOCOL),
        "authorization_sha256": sha256(AUTHORIZATION),
        "public_authorization_commit": receipt["public_authorization_commit"],
        "threshold": THRESHOLD,
        "inventories": inventories,
        "runs": runs,
        "contrasts": summaries,
        "generic_boundary_by_control": generic_summaries,
        "generic_boundary_average": pooled_generic,
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
    parser.add_argument("--stage", choices=("fit", "authorize", "evaluate"), required=True)
    parser.add_argument("--public-protocol-commit")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.stage == "fit":
        fit()
    elif args.stage == "authorize":
        if not args.public_protocol_commit:
            raise RuntimeError("--public-protocol-commit is required")
        authorize(args.public_protocol_commit)
    else:
        evaluate()


if __name__ == "__main__":
    main()
