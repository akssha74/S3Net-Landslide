#!/usr/bin/env python3
"""Record evidence and counterevidence for HR-GLDD channel-order ambiguity."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


STUDY = Path(__file__).resolve().parents[2]
DATA = STUDY / "experiments/raw/hr_gldd"
NOTEBOOK = (
    STUDY
    / "research/dataset-metadata/hrgldd-official/Unet_v1.ipynb"
)
OUTPUT = (
    STUDY
    / "research/dataset-metadata/hrgldd-official/"
    "band-order-audit.json"
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def means(images: np.ndarray, masks: np.ndarray) -> dict[str, list[float]]:
    truth = masks[..., 0] >= 0.5
    flat = images.reshape(-1, 4).astype(np.float64)
    truth_flat = truth.reshape(-1)
    return {
        "all_pixels": np.mean(flat, axis=0).tolist(),
        "landslide_pixels": np.mean(flat[truth_flat], axis=0).tolist(),
        "background_pixels": np.mean(flat[~truth_flat], axis=0).tolist(),
    }


def main() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join(
        line
        for cell in notebook["cells"]
        for line in cell.get("source", [])
    )
    display_statement = "imshow(X_train[i][:,:,:3])"
    if display_statement not in source:
        raise RuntimeError("Official direct-display statement not found")

    split_statistics = {}
    for split in ("train", "val", "test"):
        images = np.load(DATA / f"{split}X.npy", mmap_mode="r")
        masks = np.load(DATA / f"{split}Y.npy", mmap_mode="r")
        split_statistics[split] = {
            "shape": list(images.shape),
            "channel_means": means(images, masks),
        }

    payload = {
        "schema_version": 1,
        "status": "unresolved-two-candidate-order-sensitivity-required",
        "array_artifact": "HR-GLDD Zenodo record 7189381 NumPy X arrays",
        "fixed_semantics": {
            "green_index": 1,
            "nir_index": 3,
        },
        "candidate_orders": {
            "RGBN": {
                "order": ["red", "green", "blue", "nir"],
                "support": (
                    "The official notebook passes columns 0:3 directly to "
                    "matplotlib.imshow, and the paper lists selected bands as "
                    "Red, Green, Blue, NIR."
                ),
            },
            "BGRN": {
                "order": ["blue", "green", "red", "nir"],
                "support": (
                    "Native four-band PlanetScope products use Blue, Green, "
                    "Red, NIR; no released array-construction/reordering code "
                    "was found."
                ),
            },
        },
        "official_notebook": {
            "path": str(NOTEBOOK.relative_to(STUDY)),
            "sha256": digest(NOTEBOOK),
            "direct_display_statement": display_statement,
            "limitation": (
                "Display intent does not prove the physical identity of stored "
                "columns without array-construction metadata."
            ),
        },
        "dataset_paper_limitation": (
            "The paper names the four selected bands but does not explicitly "
            "bind those names to NumPy column indices."
        ),
        "split_statistics": split_statistics,
        "statistical_limitation": (
            "Channel means are descriptive and cannot identify wavelengths "
            "without land-cover or calibration references."
        ),
        "decision": (
            "Do not declare either order authoritative. Execute and report "
            "both candidate-order analyses; retain only order-robust claims."
        ),
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote unresolved band-order audit to {OUTPUT}")


if __name__ == "__main__":
    main()
