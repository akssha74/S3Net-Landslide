#!/usr/bin/env python3
"""Fit and evaluate the preregistered LRD boundary-weighting confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
import torch
from torch.utils.data import DataLoader, Dataset

from revised_models import ControlledPixelLoss, ControlledS3Net
from run_cas_boundary_confirmation import binary_f1, event_metrics


SCRIPT = Path(__file__).resolve()
STUDY = SCRIPT.parents[2]
REPO = STUDY.parents[1]
DATA = STUDY / "experiments/raw/external/lrd-optical"
METADATA = STUDY / "research/dataset-metadata/lrd-prospective-confirmation"
FOLDS_PATH = METADATA / "protected_event_folds.json"
PROTOCOL_PATH = STUDY / "research/lrd-boundary-confirmation-preregistration.md"
OUTPUT = STUDY / "experiments/derived/results/lrd_boundary_confirmation"
CHECKPOINTS = STUDY / "experiments/derived/checkpoints/lrd_boundary_confirmation"
OUTPUT.mkdir(parents=True, exist_ok=True)
CHECKPOINTS.mkdir(parents=True, exist_ok=True)

PROTOCOL_COMMIT = os.environ.get("LRD_PROTOCOL_COMMIT", "")
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
LOSSES = ("base", "boundary")
CROP_SIZE = 128
CROPS_PER_EVENT = 64
CROP_SALT = "lrd-boundary-crops-v1"
EPOCHS = 15
BATCH_SIZE = 32
LEARNING_RATE = 0.0015
WEIGHT_DECAY = 0.0001
MIN_LEARNING_RATE = 0.00001
THRESHOLDS = tuple(float(value) for value in np.arange(0.05, 1.0, 0.05))
DEVICE = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def assert_protocol_pin() -> None:
    if len(PROTOCOL_COMMIT) != 40:
        raise RuntimeError("LRD_PROTOCOL_COMMIT must be a full Git SHA")
    subprocess.run(
        ["git", "-C", str(REPO), "merge-base", "--is-ancestor", PROTOCOL_COMMIT, "HEAD"],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(REPO),
            "diff",
            "--quiet",
            PROTOCOL_COMMIT,
            "--",
            *PROTOCOL_FILES,
        ],
        check=True,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def one(root: Path, pattern: str) -> Path:
    matches = list(root.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"{root} {pattern}: expected 1 file, got {len(matches)}")
    return matches[0]


def load_event(fold: str, eid: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    root = DATA / fold / eid
    blue = tifffile.imread(one(root, "*__B02.tif"))
    green = tifffile.imread(one(root, "*__B03.tif"))
    red = tifffile.imread(one(root, "*__B04.tif"))
    mask = tifffile.imread(one(root, "*__MASK.tif"))
    while mask.ndim > 2:
        mask = mask[..., 0]
    if not (blue.shape == green.shape == red.shape == mask.shape):
        raise RuntimeError(
            f"{eid}: shape mismatch B02={blue.shape} B03={green.shape} "
            f"B04={red.shape} mask={mask.shape}"
        )
    rgb = np.stack([red, green, blue], axis=-1).astype(np.float32)
    rgb = np.clip(rgb, 0, 10000) / 10000.0
    truth = mask > 0
    height, width = truth.shape
    padded_height = ((height + CROP_SIZE - 1) // CROP_SIZE) * CROP_SIZE
    padded_width = ((width + CROP_SIZE - 1) // CROP_SIZE) * CROP_SIZE
    pad = ((0, padded_height - height), (0, padded_width - width))
    rgb = np.pad(rgb, (pad[0], pad[1], (0, 0)), mode="edge")
    truth = np.pad(truth, pad, mode="constant", constant_values=False)
    coordinates = [
        (y, x)
        for y in range(0, padded_height, CROP_SIZE)
        for x in range(0, padded_width, CROP_SIZE)
    ]
    coordinates.sort(
        key=lambda row: hashlib.sha256(
            f"{CROP_SALT}:{eid}:{row[0]}:{row[1]}".encode()
        ).hexdigest()
    )
    coordinates = coordinates[:CROPS_PER_EVENT]
    images = np.stack(
        [
            rgb[y : y + CROP_SIZE, x : x + CROP_SIZE].transpose(2, 0, 1)
            for y, x in coordinates
        ]
    ).astype(np.float32)
    masks = np.stack(
        [
            truth[y : y + CROP_SIZE, x : x + CROP_SIZE]
            for y, x in coordinates
        ]
    ).astype(np.uint8)
    crop_ids = [
        f"{eid}:{y}:{x}:{height}:{width}" for y, x in coordinates
    ]
    return images, masks, crop_ids


class Crops(Dataset):
    def __init__(
        self, images: np.ndarray, masks: np.ndarray, augment: bool
    ) -> None:
        self.images = images
        self.masks = masks
        self.augment = augment

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = torch.from_numpy(self.images[index].copy())
        mask = torch.from_numpy(self.masks[index].copy()).float()
        if self.augment:
            if torch.rand(()) < 0.5:
                image = image.flip(-1)
                mask = mask.flip(-1)
            if torch.rand(()) < 0.5:
                image = image.flip(-2)
                mask = mask.flip(-2)
        return image, mask


def loader(
    images: np.ndarray,
    masks: np.ndarray,
    *,
    augment: bool,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    return DataLoader(
        Crops(images, masks, augment),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
    )


def infer(
    model: torch.nn.Module, images: np.ndarray, masks: np.ndarray
) -> np.ndarray:
    model.eval()
    output = []
    with torch.no_grad():
        for batch_images, _ in loader(
            images, masks, augment=False, shuffle=False, seed=0
        ):
            output.append(
                torch.sigmoid(model(batch_images.to(DEVICE))).cpu().numpy()
            )
    return np.concatenate(output).astype(np.float32)


def threshold(probabilities: np.ndarray, masks: np.ndarray) -> float:
    scores = [
        (binary_f1(probabilities, masks, value), value)
        for value in THRESHOLDS
    ]
    return float(max(scores, key=lambda row: (row[0], row[1]))[1])


def load_fold(fold: str, eids: list[str]) -> dict[str, tuple[Any, Any, Any]]:
    return {eid: load_event(fold, eid) for eid in eids}


def train_one(
    loss_mode: str,
    seed: int,
    train_images: np.ndarray,
    train_masks: np.ndarray,
    val_images: np.ndarray,
    val_masks: np.ndarray,
) -> dict[str, Any]:
    set_seed(seed)
    model = ControlledS3Net(in_ch=3, gating_mode="none").to(DEVICE)
    criterion = ControlledPixelLoss(mode=loss_mode)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=MIN_LEARNING_RATE
    )
    train_loader = loader(
        train_images, train_masks, augment=True, shuffle=True, seed=seed
    )
    checkpoint = CHECKPOINTS / f"{loss_mode}_seed{seed}.pt"
    best_f1 = -1.0
    best_epoch = -1
    history = []
    started = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for images, masks in train_loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), masks, images)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        val_probabilities = infer(model, val_images, val_masks)
        val_f1 = binary_f1(val_probabilities, val_masks, 0.5)
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "validation_f1_at_0_5": val_f1,
            }
        )
        print(
            f"{loss_mode} seed={seed} epoch={epoch} "
            f"loss={np.mean(losses):.6f} val_f1={val_f1:.6f}",
            flush=True,
        )
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            torch.save(model.state_dict(), checkpoint)
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
    val_probabilities = infer(model, val_images, val_masks)
    selected_threshold = threshold(val_probabilities, val_masks)
    probabilities_path = (
        OUTPUT / f"validation_probabilities_{loss_mode}_seed{seed}.npy"
    )
    np.save(probabilities_path, val_probabilities)
    return {
        "loss_mode": loss_mode,
        "seed": seed,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "best_epoch": best_epoch,
        "validation_f1_at_0_5": best_f1,
        "validation_threshold": selected_threshold,
        "checkpoint": str(checkpoint.relative_to(STUDY)),
        "validation_probabilities": str(probabilities_path.relative_to(STUDY)),
        "history": history,
        "elapsed_seconds": time.time() - started,
    }


def config(folds: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset": "Landslide Reference Data v3",
        "protocol_commit": PROTOCOL_COMMIT,
        "development_eids": folds["development_eids"],
        "validation_eids": folds["validation_eids"],
        "protected_test_eids": folds["protected_test_eids"],
        "input": "POST1 Sentinel-2 L2A B04/B03/B02",
        "crop_size": CROP_SIZE,
        "crops_per_event": CROPS_PER_EVENT,
        "crop_salt": CROP_SALT,
        "model": "ControlledS3Net(in_ch=3,gating_mode=none)",
        "losses": list(LOSSES),
        "seeds": list(SEEDS),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "thresholds": list(THRESHOLDS),
    }


def fit() -> None:
    assert_protocol_pin()
    folds = json.loads(FOLDS_PATH.read_text())
    development = load_fold("development", folds["development_eids"])
    validation = load_fold("validation", folds["validation_eids"])
    train_images = np.concatenate([row[0] for row in development.values()])
    train_masks = np.concatenate([row[1] for row in development.values()])
    val_images = np.concatenate([row[0] for row in validation.values()])
    val_masks = np.concatenate([row[1] for row in validation.values()])
    runs = [
        train_one(
            loss_mode,
            seed,
            train_images,
            train_masks,
            val_images,
            val_masks,
        )
        for loss_mode in LOSSES
        for seed in SEEDS
    ]
    configuration = config(folds)
    config_id = hashlib.sha256(
        json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    fit_payload = {
        "schema_version": 1,
        "config_id": config_id,
        "config": configuration,
        "development_crop_counts": {
            eid: len(row[2]) for eid, row in development.items()
        },
        "validation_crop_counts": {
            eid: len(row[2]) for eid, row in validation.items()
        },
        "training_records": runs,
    }
    fit_path = OUTPUT / "fit_decisions.json"
    fit_path.write_text(json.dumps(fit_payload, indent=2) + "\n")
    assert_protocol_pin()
    authorization = {
        "schema_version": 1,
        "status": "all-decisions-frozen",
        "protocol_commit": PROTOCOL_COMMIT,
        "fit_decisions": str(fit_path.relative_to(STUDY)),
        "fit_decisions_sha256": sha256(fit_path),
        "checkpoints": [
            {
                "path": run["checkpoint"],
                "sha256": sha256(STUDY / run["checkpoint"]),
                "threshold": run["validation_threshold"],
            }
            for run in runs
        ],
    }
    marker = OUTPUT / "protected_access_authorization.json"
    marker.write_text(json.dumps(authorization, indent=2) + "\n")
    print(f"Fit complete; protected authorization: {marker}")


def evaluate() -> None:
    assert_protocol_pin()
    marker = OUTPUT / "protected_access_authorization.json"
    authorization = json.loads(marker.read_text())
    if authorization["status"] != "all-decisions-frozen":
        raise RuntimeError("Protected access not authorized")
    fit_path = STUDY / authorization["fit_decisions"]
    if sha256(fit_path) != authorization["fit_decisions_sha256"]:
        raise RuntimeError("Fit-decision hash mismatch")
    fit_payload = json.loads(fit_path.read_text())
    for checkpoint in authorization["checkpoints"]:
        if sha256(STUDY / checkpoint["path"]) != checkpoint["sha256"]:
            raise RuntimeError("Checkpoint changed after authorization")
    folds = json.loads(FOLDS_PATH.read_text())
    protected = load_fold("protected", folds["protected_test_eids"])
    runs = fit_payload["training_records"]
    for run in runs:
        model = ControlledS3Net(in_ch=3, gating_mode="none").to(DEVICE)
        model.load_state_dict(
            torch.load(STUDY / run["checkpoint"], map_location=DEVICE)
        )
        run["test_events"] = {}
        for eid, (images, masks, crop_ids) in protected.items():
            probabilities = infer(model, images, masks)
            probabilities_path = (
                OUTPUT
                / f"test_probabilities_{eid}_{run['loss_mode']}_seed{run['seed']}.npy"
            )
            np.save(probabilities_path, probabilities)
            run["test_events"][eid] = {
                **event_metrics(
                    probabilities,
                    masks,
                    float(run["validation_threshold"]),
                ),
                "n_crops": len(crop_ids),
                "probabilities": str(probabilities_path.relative_to(STUDY)),
                "crop_id_sha256": hashlib.sha256(
                    "\n".join(crop_ids).encode()
                ).hexdigest(),
            }
    effects = {}
    for seed in SEEDS:
        base = next(
            row for row in runs if row["loss_mode"] == "base" and row["seed"] == seed
        )
        boundary = next(
            row
            for row in runs
            if row["loss_mode"] == "boundary" and row["seed"] == seed
        )
        per_event = {
            eid: {
                "delta_f1": (
                    boundary["test_events"][eid]["f1"]
                    - base["test_events"][eid]["f1"]
                ),
                "delta_boundary_f1": (
                    boundary["test_events"][eid]["boundary_f1"]
                    - base["test_events"][eid]["boundary_f1"]
                ),
            }
            for eid in folds["protected_test_eids"]
        }
        effects[str(seed)] = {
            "per_event": per_event,
            "event_macro_delta_f1": float(
                np.mean([row["delta_f1"] for row in per_event.values()])
            ),
            "event_macro_delta_boundary_f1": float(
                np.mean(
                    [row["delta_boundary_f1"] for row in per_event.values()]
                )
            ),
        }
    event_boundary = {
        eid: float(
            np.mean(
                [
                    effects[str(seed)]["per_event"][eid]["delta_boundary_f1"]
                    for seed in SEEDS
                ]
            )
        )
        for eid in folds["protected_test_eids"]
    }
    mean_f1 = float(
        np.mean([effects[str(seed)]["event_macro_delta_f1"] for seed in SEEDS])
    )
    mean_boundary = float(
        np.mean(
            [
                effects[str(seed)]["event_macro_delta_boundary_f1"]
                for seed in SEEDS
            ]
        )
    )
    conditions = {
        "mean_event_macro_boundary_f1_gain_ge_0_015": mean_boundary >= 0.015,
        "all_test_events_positive_boundary_effect": all(
            value > 0 for value in event_boundary.values()
        ),
        "event_macro_f1_noninferior_ge_minus_0_010": mean_f1 >= -0.010,
        "at_least_two_seeds_nonnegative_f1_effect": (
            sum(
                effects[str(seed)]["event_macro_delta_f1"] >= 0
                for seed in SEEDS
            )
            >= 2
        ),
    }
    assert_protocol_pin()
    payload = {
        **fit_payload,
        "protected_crop_counts": {
            eid: len(row[2]) for eid, row in protected.items()
        },
        "training_records": runs,
        "effects": effects,
        "event_mean_boundary_effects": event_boundary,
        "mean_event_macro_delta_f1": mean_f1,
        "mean_event_macro_delta_boundary_f1": mean_boundary,
        "pass_conditions": conditions,
        "process_condition": {
            "status": "passed",
            "protocol_commit": PROTOCOL_COMMIT,
            "fit_decisions_sha256": authorization["fit_decisions_sha256"],
            "protected_access_after_authorization": True,
            "code_protocol_folds_unchanged_through_reporting": True,
        },
        "verdict": "pass" if all(conditions.values()) else "failed-confirmation",
        "scope": (
            "Prospective event-level confirmation over six protected LRD EIDs; "
            "no population, sensor-invariant, operational, or global claim."
        ),
    }
    destination = OUTPUT / "lrd_confirmation_summary.json"
    destination.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {
                "destination": str(destination),
                "verdict": payload["verdict"],
                "mean_event_macro_delta_f1": mean_f1,
                "mean_event_macro_delta_boundary_f1": mean_boundary,
                "conditions": conditions,
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("fit", "evaluate"), required=True)
    args = parser.parse_args()
    if args.stage == "fit":
        fit()
    else:
        evaluate()


if __name__ == "__main__":
    main()
