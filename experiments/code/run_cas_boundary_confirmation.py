#!/usr/bin/env python3
"""Local-protocol event-held-out CAS confirmation of generic boundary weighting."""

from __future__ import annotations

import hashlib
import json
import random
import time
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from revised_models import ControlledPixelLoss, ControlledS3Net
from run_reviewer_remediation import boundary_f1


SCRIPT_DIR = Path(__file__).resolve().parent
STUDY_DIR = SCRIPT_DIR.parent.parent
CAS_ROOT = STUDY_DIR / "experiments/raw/external/cas"
FOLDS_PATH = (
    STUDY_DIR
    / "research/dataset-metadata/cas-boundary-confirmation/folds.json"
)
OUTPUT_ROOT = (
    STUDY_DIR
    / "experiments/derived/results/cas_boundary_confirmation"
)
CHECKPOINT_ROOT = (
    STUDY_DIR
    / "experiments/derived/checkpoints/cas_boundary_confirmation"
)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)

SEEDS = (42, 43, 44)
LOSSES = ("base", "boundary")
EPOCHS = 15
BATCH_SIZE = 32
LEARNING_RATE = 1.5e-3
WEIGHT_DECAY = 1e-4
MIN_LEARNING_RATE = 1e-5
CROP_SIZE = 128
TRAIN_TILE_LIMIT = 80
EVAL_TILE_LIMIT = 160
SELECTION_SALT = "r010-cas-boundary-crops-v1"
THRESHOLDS = tuple(float(value) for value in np.arange(0.05, 1.0, 0.05))
DEVICE = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)

EVENT_ARCHIVES = {
    "Wenchuan": "Wenchuan.zip",
    "Tiburon-Planet": "Tiburon Peninsula（planet）.zip",
    "Lombok": "Lombok.zip",
    "Palu": "palu.zip",
    "Moxi-UAV-1m": "Moxi town（UAV-1m）.zip",
    "Moxitaidi-UAV-1m": "Moxitaidi (UAV-1m).zip",
    "Hokkaido": "Hokkaido Iburi-Tobu.zip",
    "Mengdong": "Mengdong Township.zip",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def normalized_stem(path: str) -> str:
    return Path(path).stem.lower().replace("_mask", "").replace("-mask", "")


def paired_members(
    archive: ZipFile,
) -> list[tuple[str, str, str]]:
    image_members: dict[str, str] = {}
    mask_members: dict[str, str] = {}
    for name in archive.namelist():
        if name.endswith("/"):
            continue
        lowered_parts = [part.lower() for part in Path(name).parts]
        key = normalized_stem(name)
        if "img" in lowered_parts:
            image_members[key] = name
        elif "mask" in lowered_parts:
            mask_members[key] = name
    shared = sorted(set(image_members) & set(mask_members))
    if not shared:
        raise ValueError("CAS archive has no paired img/mask members")
    return [(key, image_members[key], mask_members[key]) for key in shared]


def selected_pairs(
    event: str, archive: ZipFile, tile_limit: int
) -> list[tuple[str, str, str]]:
    pairs = paired_members(archive)
    pairs.sort(
        key=lambda row: hashlib.sha256(
            f"{SELECTION_SALT}:{event}:{row[0]}".encode()
        ).hexdigest()
    )
    return pairs[:tile_limit]


def read_rgb(payload: bytes) -> np.ndarray:
    image = Image.open(BytesIO(payload)).convert("RGB")
    array = np.asarray(image, dtype=np.uint8)
    if array.shape != (512, 512, 3):
        raise ValueError(f"Expected 512x512 RGB image, got {array.shape}")
    return array


def read_mask(payload: bytes) -> np.ndarray:
    image = Image.open(BytesIO(payload))
    array = np.asarray(image)
    while array.ndim > 2:
        array = array[..., 0]
    if array.shape != (512, 512):
        raise ValueError(f"Expected 512x512 mask, got {array.shape}")
    values = set(np.unique(array).tolist())
    if not values <= {0, 1, 255}:
        raise ValueError(f"Unexpected mask values: {sorted(values)}")
    return (array > 0).astype(np.uint8)


def load_event(
    event: str, tile_limit: int
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    path = CAS_ROOT / EVENT_ARCHIVES[event]
    if not path.is_file():
        raise FileNotFoundError(path)
    images: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    crop_ids: list[str] = []
    with ZipFile(path) as archive:
        for tile_id, image_name, mask_name in selected_pairs(
            event, archive, tile_limit
        ):
            image = read_rgb(archive.read(image_name))
            mask = read_mask(archive.read(mask_name))
            for y in range(0, 512, CROP_SIZE):
                for x in range(0, 512, CROP_SIZE):
                    images.append(
                        image[y : y + CROP_SIZE, x : x + CROP_SIZE].transpose(
                            2, 0, 1
                        )
                    )
                    masks.append(mask[y : y + CROP_SIZE, x : x + CROP_SIZE])
                    crop_ids.append(f"{event}:{tile_id}:{y}:{x}")
    return (
        np.stack(images).astype(np.uint8),
        np.stack(masks).astype(np.uint8),
        crop_ids,
    )


class CropDataset(Dataset):
    def __init__(
        self, images: np.ndarray, masks: np.ndarray, augment: bool
    ) -> None:
        self.images = images
        self.masks = masks
        self.augment = augment

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = torch.from_numpy(self.images[index].copy()).float() / 255.0
        mask = torch.from_numpy(self.masks[index].copy()).float()
        if self.augment:
            if torch.rand(()) < 0.5:
                image = image.flip(-1)
                mask = mask.flip(-1)
            if torch.rand(()) < 0.5:
                image = image.flip(-2)
                mask = mask.flip(-2)
            turns = int(torch.randint(0, 4, ()).item())
            if turns:
                image = torch.rot90(image, turns, dims=(-2, -1))
                mask = torch.rot90(mask, turns, dims=(-2, -1))
        return image, mask


def data_loader(
    images: np.ndarray,
    masks: np.ndarray,
    *,
    augment: bool,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        CropDataset(images, masks, augment),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        generator=generator,
    )


def infer(
    model: torch.nn.Module, images: np.ndarray, masks: np.ndarray
) -> np.ndarray:
    model.eval()
    output = []
    loader = data_loader(
        images, masks, augment=False, shuffle=False, seed=0
    )
    with torch.no_grad():
        for batch_images, _ in loader:
            logits = model(batch_images.to(DEVICE))
            output.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(output).astype(np.float32)


def binary_f1(
    probabilities: np.ndarray, targets: np.ndarray, threshold: float
) -> float:
    prediction = probabilities >= threshold
    truth = targets >= 0.5
    tp = int(np.sum(prediction & truth))
    fp = int(np.sum(prediction & ~truth))
    fn = int(np.sum(~prediction & truth))
    return float(2 * tp / (2 * tp + fp + fn + 1e-12))


def validation_threshold(
    probabilities: np.ndarray, targets: np.ndarray
) -> float:
    scored = [
        (binary_f1(probabilities, targets, threshold), threshold)
        for threshold in THRESHOLDS
    ]
    return float(max(scored, key=lambda row: (row[0], row[1]))[1])


def train_one(
    loss_mode: str,
    seed: int,
    train_images: np.ndarray,
    train_masks: np.ndarray,
    validation_images: np.ndarray,
    validation_masks: np.ndarray,
) -> dict[str, object]:
    set_seed(seed)
    model = ControlledS3Net(in_ch=3, gating_mode="none").to(DEVICE)
    criterion = ControlledPixelLoss(mode=loss_mode)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=MIN_LEARNING_RATE
    )
    train_loader = data_loader(
        train_images, train_masks, augment=True, shuffle=True, seed=seed
    )
    best_f1 = -1.0
    best_epoch = -1
    checkpoint_path = CHECKPOINT_ROOT / f"{loss_mode}_seed{seed}.pt"
    history = []
    started = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for images, masks in train_loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, masks, images)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        validation_probabilities = infer(
            model, validation_images, validation_masks
        )
        validation_f1 = binary_f1(
            validation_probabilities, validation_masks, 0.5
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "validation_f1_at_0_5": validation_f1,
            }
        )
        print(
            f"{loss_mode} seed={seed} epoch={epoch} "
            f"loss={np.mean(losses):.6f} val_f1={validation_f1:.6f}",
            flush=True,
        )
        if validation_f1 > best_f1:
            best_f1 = validation_f1
            best_epoch = epoch
            torch.save(model.state_dict(), checkpoint_path)

    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    validation_probabilities = infer(model, validation_images, validation_masks)
    threshold = validation_threshold(
        validation_probabilities, validation_masks
    )
    np.save(
        OUTPUT_ROOT / f"validation_probabilities_{loss_mode}_seed{seed}.npy",
        validation_probabilities,
    )
    return {
        "loss_mode": loss_mode,
        "seed": seed,
        "parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "best_epoch": best_epoch,
        "validation_f1_at_0_5": best_f1,
        "validation_threshold": threshold,
        "history": history,
        "checkpoint": str(checkpoint_path.relative_to(STUDY_DIR)),
        "elapsed_seconds": time.time() - started,
    }


def event_metrics(
    probabilities: np.ndarray, targets: np.ndarray, threshold: float
) -> dict[str, float]:
    prediction = probabilities >= threshold
    truth = targets >= 0.5
    tp = int(np.sum(prediction & truth))
    fp = int(np.sum(prediction & ~truth))
    fn = int(np.sum(~prediction & truth))
    tn = int(np.sum(~prediction & ~truth))
    result = {
        "threshold": float(threshold),
        "f1": float(2 * tp / (2 * tp + fp + fn + 1e-12)),
        "iou": float(tp / (tp + fp + fn + 1e-12)),
        "precision": float(tp / (tp + fp + 1e-12)),
        "recall": float(tp / (tp + fn + 1e-12)),
        "background_fpr": float(fp / (fp + tn + 1e-12)),
        "false_positive_pixels": fp,
    }
    result["boundary_f1"] = boundary_f1(
        probabilities, targets, threshold, tolerance_pixels=1
    )
    return result


def canonical_config(folds: dict[str, object]) -> dict[str, object]:
    execution_folds = {
        key: folds[key]
        for key in (
            "salt",
            "independence_unit",
            "development_events",
            "validation_events",
            "protected_test_events",
            "hash_order",
        )
    }
    return {
        "dataset": "CAS Zenodo 10294997",
        "folds": execution_folds,
        "model": "ControlledS3Net(in_ch=3,gating_mode=none)",
        "losses": list(LOSSES),
        "seeds": list(SEEDS),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "crop_size": CROP_SIZE,
        "train_tile_limit_per_event": TRAIN_TILE_LIMIT,
        "eval_tile_limit_per_event": EVAL_TILE_LIMIT,
        "selection_salt": SELECTION_SALT,
        "threshold_grid": list(THRESHOLDS),
    }


def main() -> None:
    folds = json.loads(FOLDS_PATH.read_text(encoding="utf-8"))
    config = canonical_config(folds)
    config_text = json.dumps(config, sort_keys=True, separators=(",", ":"))
    config_id = hashlib.sha256(config_text.encode()).hexdigest()[:16]
    (OUTPUT_ROOT / "frozen_config.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "artifact_status": (
                    "post-execution record of the executed configuration; "
                    "filename retained for compatibility"
                ),
                "config_id": config_id,
                "config_identity_scope": (
                    "Execution-only fields; post-run provenance prose is "
                    "excluded from the hash."
                ),
                **config,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"device={DEVICE}; config_id={config_id}", flush=True)

    development_arrays = [
        load_event(event, TRAIN_TILE_LIMIT)
        for event in folds["development_events"]
    ]
    train_images = np.concatenate([row[0] for row in development_arrays])
    train_masks = np.concatenate([row[1] for row in development_arrays])
    validation_event = folds["validation_events"][0]
    validation_images, validation_masks, validation_ids = load_event(
        validation_event, EVAL_TILE_LIMIT
    )

    runs = []
    for loss_mode in LOSSES:
        for seed in SEEDS:
            runs.append(
                train_one(
                    loss_mode,
                    seed,
                    train_images,
                    train_masks,
                    validation_images,
                    validation_masks,
                )
            )
    # Method, weights, epochs, and thresholds are frozen before this point.
    test_cache = {
        event: load_event(event, EVAL_TILE_LIMIT)
        for event in folds["protected_test_events"]
    }
    for run in runs:
        model = ControlledS3Net(in_ch=3, gating_mode="none").to(DEVICE)
        model.load_state_dict(
            torch.load(
                STUDY_DIR / str(run["checkpoint"]), map_location=DEVICE
            )
        )
        run["test_events"] = {}
        for event, (images, masks, crop_ids) in test_cache.items():
            probabilities = infer(model, images, masks)
            probability_path = (
                OUTPUT_ROOT
                / f"test_probabilities_{event}_{run['loss_mode']}_seed{run['seed']}.npy"
            )
            np.save(probability_path, probabilities)
            run["test_events"][event] = {
                **event_metrics(
                    probabilities,
                    masks,
                    float(run["validation_threshold"]),
                ),
                "n_crops": len(images),
                "probabilities": str(
                    probability_path.relative_to(STUDY_DIR)
                ),
                "crop_id_sha256": hashlib.sha256(
                    "\n".join(crop_ids).encode()
                ).hexdigest(),
            }

    effects = {}
    for seed in SEEDS:
        base = next(
            run
            for run in runs
            if run["seed"] == seed and run["loss_mode"] == "base"
        )
        boundary = next(
            run
            for run in runs
            if run["seed"] == seed and run["loss_mode"] == "boundary"
        )
        per_event = {}
        for event in folds["protected_test_events"]:
            per_event[event] = {
                "delta_f1": (
                    boundary["test_events"][event]["f1"]
                    - base["test_events"][event]["f1"]
                ),
                "delta_boundary_f1": (
                    boundary["test_events"][event]["boundary_f1"]
                    - base["test_events"][event]["boundary_f1"]
                ),
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

    event_mean_boundary_effects = {
        event: float(
            np.mean(
                [
                    effects[str(seed)]["per_event"][event][
                        "delta_boundary_f1"
                    ]
                    for seed in SEEDS
                ]
            )
        )
        for event in folds["protected_test_events"]
    }
    mean_delta_boundary = float(
        np.mean(
            [
                effects[str(seed)]["event_macro_delta_boundary_f1"]
                for seed in SEEDS
            ]
        )
    )
    mean_delta_f1 = float(
        np.mean(
            [effects[str(seed)]["event_macro_delta_f1"] for seed in SEEDS]
        )
    )
    pass_conditions = {
        "mean_event_macro_boundary_f1_gain_ge_0_015": (
            mean_delta_boundary >= 0.015
        ),
        "all_test_events_positive_boundary_effect": all(
            value > 0 for value in event_mean_boundary_effects.values()
        ),
        "event_macro_f1_noninferior_ge_minus_0_010": mean_delta_f1 >= -0.010,
        "at_least_two_seeds_nonnegative_f1_effect": sum(
            effects[str(seed)]["event_macro_delta_f1"] >= 0 for seed in SEEDS
        )
        >= 2,
    }
    process_condition = {
        "condition": (
            "No post-access code, threshold, or reporting-rule change."
        ),
        "status": "not-independently-time-verifiable",
        "reason": (
            "The only surviving protocol was first committed after execution; "
            "run logs are retained but cannot establish a pre-run immutable lock."
        ),
        "included_in_quantitative_verdict": False,
    }
    payload = {
        "schema_version": 1,
        "config_id": config_id,
        "config_identity_scope": (
            "Execution-only fields; post-run provenance prose is excluded "
            "from the hash."
        ),
        "config": config,
        "archive_hashes": {
            event: sha256(CAS_ROOT / EVENT_ARCHIVES[event])
            for event in EVENT_ARCHIVES
        },
        "training_records": runs,
        "effects": effects,
        "event_mean_boundary_effects": event_mean_boundary_effects,
        "mean_event_macro_delta_f1": mean_delta_f1,
        "mean_event_macro_delta_boundary_f1": mean_delta_boundary,
        "pass_conditions": pass_conditions,
        "pass_conditions_scope": "four quantitative conditions",
        "process_condition": process_condition,
        "verdict": (
            "pass" if all(pass_conditions.values()) else "failed-confirmation"
        ),
        "verdict_basis": (
            "Three of four quantitative conditions failed; the separate "
            "process condition is not independently time-verifiable."
        ),
        "scope": (
            "Bounded replication on three held CAS regions; no population, "
            "sensor-invariant, operational, or global claim."
        ),
    }
    (OUTPUT_ROOT / "cas_boundary_confirmation_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "verdict": payload["verdict"],
                "mean_event_macro_delta_f1": mean_delta_f1,
                "mean_event_macro_delta_boundary_f1": mean_delta_boundary,
                "event_mean_boundary_effects": event_mean_boundary_effects,
                "pass_conditions": pass_conditions,
                "process_condition": process_condition,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
