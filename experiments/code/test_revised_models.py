"""Semantic and capacity tests for the controlled reviewer-remediation arms."""

from __future__ import annotations

import torch

from revised_models import ControlledPixelLoss, ControlledS3Net


def main() -> None:
    models = {
        mode: ControlledS3Net(gating_mode=mode) for mode in ("none", "raw", "ndvi")
    }
    counts = {
        mode: sum(parameter.numel() for parameter in model.parameters())
        for mode, model in models.items()
    }
    assert len(set(counts.values())) == 1, counts

    x = torch.zeros((2, 4, 16, 16), dtype=torch.float32)
    x[:, 0] = 0.01
    x[:, 1] = 0.02
    x[:, 2] = 0.20
    x[:, 3] = 0.60
    for model in models.values():
        output = model(x)
        assert output.shape == (2, 16, 16)

    targets = torch.zeros((2, 16, 16), dtype=torch.float32)
    targets[:, 4:12, 4:12] = 1.0
    logits = torch.zeros_like(targets)
    values = {}
    for mode in ("base", "boundary", "biophysical"):
        value = ControlledPixelLoss(mode=mode)(logits, targets, x)
        assert torch.isfinite(value)
        values[mode] = float(value)
    assert values["boundary"] > values["base"]
    assert values["biophysical"] > values["base"]
    print(f"PASS: capacity matched at {next(iter(counts.values())):,} parameters")


if __name__ == "__main__":
    main()
