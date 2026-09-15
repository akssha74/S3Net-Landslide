#!/usr/bin/env python3
"""Group LRD EIDs by country and trigger date to detect family leakage."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


STUDY = Path(__file__).resolve().parents[2]
ROOT = STUDY / "research/dataset-metadata/lrd-prospective-confirmation"
SOURCE = ROOT / "source_metadata_audit.json"
FOLDS = ROOT / "protected_event_folds.json"
OUTPUT = ROOT / "event_family_audit.json"


def main() -> None:
    source = json.loads(SOURCE.read_text())
    folds = json.loads(FOLDS.read_text())
    fold_by_eid = {
        eid: fold
        for fold, key in (
            ("development", "development_eids"),
            ("validation", "validation_eids"),
            ("protected", "protected_test_eids"),
        )
        for eid in folds[key]
    }
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for event in source["events"]:
        family = f"{event['country']}|{event['date']}"
        groups[family].append(
            {
                "eid": event["eid"],
                "fold": fold_by_eid[event["eid"]],
                "country": event["country"],
                "region": event["region"],
                "date": event["date"],
                "trigger": event["trigger"],
            }
        )
    cross_fold = {
        family: rows
        for family, rows in groups.items()
        if len({row["fold"] for row in rows}) > 1
    }
    compromised_protected = sorted(
        row["eid"]
        for rows in cross_fold.values()
        for row in rows
        if row["fold"] == "protected"
    )
    valid_protected = sorted(
        set(folds["protected_test_eids"]) - set(compromised_protected)
    )
    payload = {
        "schema_version": 1,
        "family_rule": (
            "country + documented trigger date; EIDs sharing both are one "
            "minimum trigger family"
        ),
        "families": dict(sorted(groups.items())),
        "cross_fold_families": cross_fold,
        "compromised_protected_eids": compromised_protected,
        "valid_nonoverlapping_protected_eids": valid_protected,
        "released_protected_eid_count": len(folds["protected_test_eids"]),
        "minimum_independent_protected_trigger_count": len(valid_protected),
        "finding": (
            "PH0003 development and PH0001/PH0004 protected share the "
            "Philippines 2022-04-10 trigger family. The preregistered EID split "
            "does not establish six independent triggers."
        ),
        "decision": (
            "Reclassify the LRD run as internally precommitted but "
            "independence-compromised; report four nonoverlapping EIDs only as "
            "post-hoc sensitivity."
        ),
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"Wrote {OUTPUT}; compromised={compromised_protected}; "
        f"valid={valid_protected}"
    )


if __name__ == "__main__":
    main()
