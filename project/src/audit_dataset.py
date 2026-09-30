"""Audit category coverage and train/validation/test representation."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from src.config import MIN_PRODUCTS_PER_CATEGORY, PROCESSED_CSV


def _normalized_name(value):
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def _category_column(frame):
    names = {_normalized_name(column): column for column in frame.columns}
    for candidate in ("category", "product_category", "master_category", "type"):
        column = names.get(_normalized_name(candidate))
        if column is not None:
            return column
    raise ValueError(f"No category column found in: {list(frame.columns)}")


def _category_series(frame):
    column = _category_column(frame)
    return frame[column].fillna("").astype(str).str.strip().replace("", "Unknown")


def audit_categories(source, selected, minimum=MIN_PRODUCTS_PER_CATEGORY):
    """Return one deterministic audit row per category.

    ``source`` contains eligible rows before sampling. ``selected`` is the
    prepared catalog or split dataset. If it has a ``split`` column, split
    coverage is checked too.
    """
    if minimum < 1:
        raise ValueError("minimum must be at least 1")
    source = source.copy()
    selected = selected.copy()
    source["_audit_category"] = _category_series(source)
    selected["_audit_category"] = _category_series(selected)
    source_counts = source["_audit_category"].value_counts()
    selected_counts = selected["_audit_category"].value_counts()
    categories = sorted(set(source_counts.index) | set(selected_counts.index))
    has_splits = "split" in selected.columns
    rows = []

    for category in categories:
        source_count = int(source_counts.get(category, 0))
        selected_count = int(selected_counts.get(category, 0))
        split_counts = {name: 0 for name in ("train", "validation", "test")}
        if has_splits:
            values = selected.loc[
                selected["_audit_category"].eq(category), "split"
            ].astype(str).value_counts()
            split_counts.update({name: int(values.get(name, 0)) for name in split_counts})

        warnings = []
        severity = 0
        if selected_count == 0:
            warnings.append("missing_from_selected")
            severity = 2
        elif selected_count < minimum:
            if source_count < minimum:
                warnings.append("limited_source")
                severity = max(severity, 1)
            else:
                warnings.append("below_minimum")
                severity = 2
        if has_splits:
            for split in ("train", "validation", "test"):
                if split_counts[split] == 0:
                    warnings.append(f"no_{split}")
                    severity = max(severity, 2 if split == "train" else 1)

        rows.append(
            {
                "category": category,
                "source_count": source_count,
                "selected_count": selected_count,
                "coverage_ratio": selected_count / source_count if source_count else 0.0,
                "train_count": split_counts["train"] if has_splits else None,
                "validation_count": split_counts["validation"] if has_splits else None,
                "test_count": split_counts["test"] if has_splits else None,
                "minimum_target": int(minimum),
                "status": (
                    "critical" if severity == 2 else "warning" if severity == 1 else "ok"
                ),
                "warnings": ";".join(warnings),
            }
        )
    return pd.DataFrame(rows)


def audit_summary(audit):
    status_counts = audit["status"].value_counts().to_dict() if len(audit) else {}
    flagged = audit.loc[audit.status.ne("ok"), "category"].astype(str).tolist()
    return {
        "categories": int(len(audit)),
        "status_counts": {
            str(key): int(value) for key, value in status_counts.items()
        },
        "flagged_categories": flagged,
    }


def write_category_audit(source, selected, output_csv, minimum=MIN_PRODUCTS_PER_CATEGORY):
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    audit = audit_categories(source, selected, minimum)
    audit.to_csv(output_csv, index=False, encoding="utf-8-sig")
    output_json = output_csv.with_suffix(".json")
    payload = audit_summary(audit)
    payload.update(minimum_target=int(minimum), report_csv=str(output_csv.resolve()))
    output_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return audit, payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-csv", type=Path, default=PROCESSED_CSV)
    parser.add_argument(
        "--source-csv",
        type=Path,
        help="Optional eligible/source CSV. Defaults to the selected dataset.",
    )
    parser.add_argument("--minimum", type=int, default=MIN_PRODUCTS_PER_CATEGORY)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.minimum < 1:
        parser.error("minimum must be at least 1")
    selected = pd.read_csv(args.dataset_csv)
    source = pd.read_csv(args.source_csv) if args.source_csv else selected
    output = args.output or args.dataset_csv.with_suffix(".category_audit.csv")
    audit, summary = write_category_audit(source, selected, output, args.minimum)
    print(audit.to_string(index=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
