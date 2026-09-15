#!/usr/bin/env python3
"""Synthetic sentinel tests for the frozen Sen12 confirmation path."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import xarray as xr

import run_sen12_s2_confirmation as confirmation


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "sentinel_s2_1.nc"
        shape = (2, 4, 4)
        dataset = xr.Dataset(
            {
                "B02": (("time", "x", "y"), np.full(shape, 1000, dtype=np.int16)),
                "B03": (("time", "x", "y"), np.full(shape, 2000, dtype=np.int16)),
                "B04": (("time", "x", "y"), np.full(shape, 3000, dtype=np.int16)),
                "B08": (("time", "x", "y"), np.full(shape, 8000, dtype=np.int16)),
                "MASK": (
                    ("time", "x", "y"),
                    np.array(
                        [
                            [[1, 1, 0, 0]] * 4,
                            [[1, 1, 0, 0]] * 4,
                        ],
                        dtype=np.uint8,
                    ),
                ),
            },
            attrs={
                "pre_post_dates": "{'pre': 0, 'post': 1}",
                "date_confidence": "1.0",
            },
        )
        dataset.to_netcdf(path, engine="h5netcdf")
        image, mask, high_confidence = confirmation.load_file(path)
    assert image.shape == (4, 4, 4)
    assert np.allclose(image[:, 0, 0], [0.1, 0.2, 0.3, 0.8])
    assert mask.shape == (4, 4)
    assert high_confidence
    assert np.isclose(confirmation.pooled_f1(mask, mask), 1.0)
    first = confirmation.interval([0.1, 0.2, 0.3], "sentinel")
    second = confirmation.interval([0.1, 0.2, 0.3], "sentinel")
    assert first == second
    model = confirmation.ControlledS3Net(in_ch=4, gating_mode="ndvi")
    assert sum(parameter.numel() for parameter in model.parameters()) == 2_114_084
    print("PASS: Sen12 S2 synthetic semantics, metrics, and capacity")


if __name__ == "__main__":
    main()
