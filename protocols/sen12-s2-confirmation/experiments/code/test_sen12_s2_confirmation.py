#!/usr/bin/env python3
"""Synthetic sentinel tests for the frozen Sen12 confirmation path."""

from __future__ import annotations

import datetime as dt
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
import xarray as xr

import prepare_sen12_s2 as preparation
import run_sen12_s2_confirmation as confirmation
from data_semantics import NIR, RED


def main() -> None:
    runtime = confirmation.verify_runtime_identity()
    assert {
        "numpy",
        "pyproj",
        "scipy",
        "torch",
        "xarray",
        "h5netcdf",
        "h5py",
    }.issubset(runtime)
    assert (
        preparation.sha256(preparation.TASK_MEMBERSHIP)
        == preparation.TASK_MEMBERSHIP_SHA256
    )
    membership = json.loads(preparation.TASK_MEMBERSHIP.read_text())
    assert len(membership["members"]) == 4988
    assert all(set(row) == {"filename", "inventory"} for row in membership["members"])
    development, protected = preparation.inventories()
    assert not set(development) & set(protected)
    assert len(set(development + protected)) == 13
    fit_matrix = [
        {"arm": arm, "seed": seed}
        for arm in confirmation.ARMS
        for seed in confirmation.SEEDS
    ]
    confirmation.validate_fit_matrix(fit_matrix)
    try:
        confirmation.validate_fit_matrix(fit_matrix[:-1])
        raise AssertionError("incomplete fit matrix was accepted")
    except RuntimeError:
        pass
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "sentinel_s2_1.nc"
        shape = (2, 4, 4)
        dataset = xr.Dataset(
            {
                "B02": (("time", "x", "y"), np.full(shape, 1000, dtype=np.int16)),
                "B03": (("time", "x", "y"), np.full(shape, 2000, dtype=np.int16)),
                "B04": (("time", "x", "y"), np.full(shape, 3000, dtype=np.int16)),
                "B08": (("time", "x", "y"), np.full(shape, 8000, dtype=np.int16)),
                "SCL": (("time", "x", "y"), np.full(shape, 4, dtype=np.int16)),
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
            coords={
                "time": np.array(
                    ["2020-01-01", "2020-01-15"], dtype="datetime64[D]"
                )
            },
            attrs={
                "pre_post_dates": "{'pre': 0, 'post': 1}",
                "date_confidence": "1.0",
                "annotated": "True",
                "satellite": "s2",
                "crs": "EPSG:32632",
            },
        )
        dataset.to_netcdf(path, engine="h5netcdf")
        record = confirmation.load_file(path)
        image = record["image"]
        mask = record["mask"]
        assert confirmation.parse_post(
            "[{'pre': 0, 'post': 1}, {'pre': 1, 'post': 2}]"
        ) == 1
        try:
            confirmation.parse_post("[0, 1]")
            raise AssertionError("non-dict date metadata was accepted")
        except ValueError:
            pass
        unannotated = dataset.copy(deep=True)
        unannotated.attrs["annotated"] = "False"
        unannotated_path = Path(temporary) / "unannotated.nc"
        unannotated.to_netcdf(unannotated_path, engine="h5netcdf")
        try:
            confirmation.load_file(unannotated_path)
            raise AssertionError("unannotated file was accepted")
        except RuntimeError:
            pass
        dynamic_mask = dataset.copy(deep=True)
        dynamic_mask["MASK"].values[1, 0, 0] = 0
        dynamic_mask_path = Path(temporary) / "dynamic-mask.nc"
        dynamic_mask.to_netcdf(dynamic_mask_path, engine="h5netcdf")
        try:
            confirmation.load_file(dynamic_mask_path)
            raise AssertionError("dynamic MASK was accepted")
        except RuntimeError:
            pass
        invalid_dn = dataset.copy(deep=True)
        invalid_dn["B02"].values[1, 0, 0] = 10001
        invalid_dn_path = Path(temporary) / "invalid-dn.nc"
        invalid_dn.to_netcdf(invalid_dn_path, engine="h5netcdf")
        try:
            confirmation.load_file(invalid_dn_path)
            raise AssertionError("out-of-bounds DN was accepted")
        except RuntimeError:
            pass
    assert image.shape == (4, 4, 4)
    assert np.allclose(image[:, 0, 0], [0.1, 0.2, 0.3, 0.8])
    assert mask.shape == (4, 4)
    assert record["high_confidence"]
    assert record["post_date"] == "2020-01-15"
    assert record["scl_histogram"] == {"4": 16}
    assert record["cloud_fraction"] == 0.0
    assert len(record["native_pixel_edge_bounds"]) == 4
    assert set(record["epsg4326_footprint"]) == {
        "longitude_intervals",
        "latitude_interval",
    }
    assert RED == 2
    assert NIR == 3
    assert np.isclose(confirmation.pooled_f1(mask, mask), 1.0)
    first = confirmation.interval([0.1, 0.2, 0.3], "sentinel")
    second = confirmation.interval([0.1, 0.2, 0.3], "sentinel")
    assert first == second
    no_overlap = {
        "filenames": ["a.nc", "b.nc"],
        "inventories": ["a", "b"],
        "crs": ["EPSG:32632", "EPSG:32633"],
        "epsg4326_footprints": [
            {
                "longitude_intervals": [[0.0, 10.0]],
                "latitude_interval": [0.0, 10.0],
            },
            {
                "longitude_intervals": [[10.0, 20.0]],
                "latitude_interval": [0.0, 10.0],
            },
        ],
    }
    assert confirmation.cross_inventory_overlaps(no_overlap) == []
    overlapping = {
        **no_overlap,
        "epsg4326_footprints": [
            no_overlap["epsg4326_footprints"][0],
            {
                "longitude_intervals": [[9.0, 20.0]],
                "latitude_interval": [0.0, 10.0],
            },
        ],
    }
    assert len(confirmation.cross_inventory_overlaps(overlapping)) == 1
    antimeridian = {
        "filenames": ["c.nc", "d.nc"],
        "inventories": ["c", "d"],
        "crs": ["EPSG:32601", "EPSG:32660"],
        "epsg4326_footprints": [
            {
                "longitude_intervals": [[170.0, 180.0], [-180.0, -175.0]],
                "latitude_interval": [-5.0, 5.0],
            },
            {
                "longitude_intervals": [[175.0, 179.0]],
                "latitude_interval": [0.0, 10.0],
            },
        ],
    }
    assert len(confirmation.cross_inventory_overlaps(antimeridian)) == 1
    model = confirmation.ControlledS3Net(in_ch=4, gating_mode="ndvi")
    assert sum(parameter.numel() for parameter in model.parameters()) == 2_114_084
    for config in confirmation.ARMS.values():
        confirmation.set_seed(42)
        model = confirmation.ControlledS3Net(
            in_ch=4, gating_mode=config["gating"]
        ).to(confirmation.DEVICE)
        criterion = confirmation.ControlledPixelLoss(
            mode=config["loss"]
        ).to(confirmation.DEVICE)
        inputs = torch.rand(2, 4, 16, 16, device=confirmation.DEVICE)
        targets = (torch.rand(2, 16, 16, device=confirmation.DEVICE) > 0.9).float()
        loss = criterion(model(inputs), targets, inputs)
        assert torch.isfinite(loss)
        loss.backward()
    print(
        json.dumps(
            {
                "schema_version": 1,
                "test": "sen12-s2-preaccess-sentinel",
                "status": "pass",
                "executed_at": dt.datetime.now(dt.timezone.utc)
                .isoformat()
                .replace("+00:00", "Z"),
                "command": (
                    ".venv-sen12/bin/python "
                    "experiments/code/test_sen12_s2_confirmation.py"
                ),
                "runtime": runtime,
                "source_artifacts": confirmation.source_artifact_hashes(),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
