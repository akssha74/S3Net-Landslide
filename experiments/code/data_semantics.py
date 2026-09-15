"""Explicit candidate HR-GLDD array semantics used by every experiment.

The released arrays have no authoritative channel-axis metadata.  The official
notebook's direct RGB display supports RGBN, while native four-band PlanetScope
supports BGRN.  Experiments must therefore name the candidate order explicitly.
"""

from __future__ import annotations

import os
from typing import Any


SUPPORTED_BAND_ORDERS = {
    "RGBN": ("red", "green", "blue", "nir"),
    "BGRN": ("blue", "green", "red", "nir"),
}
BAND_ORDER_ID = os.environ.get("HRGLDD_ARRAY_ORDER", "RGBN").upper()
if BAND_ORDER_ID not in SUPPORTED_BAND_ORDERS:
    raise ValueError(
        f"Unsupported HRGLDD_ARRAY_ORDER={BAND_ORDER_ID!r}; "
        f"choose one of {sorted(SUPPORTED_BAND_ORDERS)}"
    )
BAND_ORDER = SUPPORTED_BAND_ORDERS[BAND_ORDER_ID]
BAND_INDEX = {name: index for index, name in enumerate(BAND_ORDER)}
BLUE = BAND_INDEX["blue"]
GREEN = BAND_INDEX["green"]
RED = BAND_INDEX["red"]
NIR = BAND_INDEX["nir"]
EPSILON = 1e-6


def validate_nhwc(array: Any) -> None:
    if getattr(array, "ndim", None) != 4 or array.shape[-1] != len(BAND_ORDER):
        raise ValueError(
            f"Expected NHWC imagery with {len(BAND_ORDER)} channels "
            f"{BAND_ORDER}, got shape {getattr(array, 'shape', None)}"
        )


def validate_nchw(tensor: Any) -> None:
    if getattr(tensor, "ndim", None) != 4 or tensor.shape[1] != len(BAND_ORDER):
        raise ValueError(
            f"Expected NCHW imagery with {len(BAND_ORDER)} channels "
            f"{BAND_ORDER}, got shape {getattr(tensor, 'shape', None)}"
        )


def numpy_ndvi(array: Any) -> Any:
    validate_nhwc(array)
    red = array[..., RED]
    nir = array[..., NIR]
    return (nir - red) / (nir + red + EPSILON)


def torch_ndvi(tensor: Any) -> Any:
    validate_nchw(tensor)
    red = tensor[:, RED : RED + 1]
    nir = tensor[:, NIR : NIR + 1]
    return (nir - red) / (nir + red + EPSILON)
