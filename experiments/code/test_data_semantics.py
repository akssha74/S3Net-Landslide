"""Sentinel tests for the load-bearing HR-GLDD channel mapping."""

from __future__ import annotations

import numpy as np
import torch

from data_semantics import BAND_ORDER, BLUE, GREEN, NIR, RED, numpy_ndvi, torch_ndvi


def main() -> None:
    assert BAND_ORDER == ("red", "green", "blue", "nir")
    assert (RED, GREEN, BLUE, NIR) == (0, 1, 2, 3)

    nhwc = np.zeros((1, 1, 1, 4), dtype=np.float32)
    nhwc[..., RED] = 0.20
    nhwc[..., GREEN] = 0.02
    nhwc[..., BLUE] = 0.01
    nhwc[..., NIR] = 0.60
    expected = (0.60 - 0.20) / (0.60 + 0.20 + 1e-6)
    np.testing.assert_allclose(numpy_ndvi(nhwc), expected, rtol=0, atol=1e-7)

    nchw = torch.from_numpy(nhwc).permute(0, 3, 1, 2)
    torch.testing.assert_close(
        torch_ndvi(nchw),
        torch.tensor([[[[expected]]]], dtype=torch.float32),
        rtol=0,
        atol=1e-7,
    )
    print("PASS: HR-GLDD arrays are Red, Green, Blue, NIR = 0, 1, 2, 3")


if __name__ == "__main__":
    main()
