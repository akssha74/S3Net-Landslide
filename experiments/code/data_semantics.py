"""Authoritative HR-GLDD channel semantics used by every experiment."""

from __future__ import annotations

from typing import Any


BAND_ORDER = ("blue", "green", "red", "nir")
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
