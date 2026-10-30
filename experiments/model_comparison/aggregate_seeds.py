"""Aggregate completed comparison CSV files across independent random seeds.

This script only reads existing experiment outputs.  It does not prepare a
catalog, fine-tune a model, generate embeddings, or rerun comparisons.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


IDENTITY_COLUMNS = (
    "comparison_group",
    "system",
    "role",
    "deployed",
    "input",
    "formula",
    "task",
    "k",
)
METRIC_COLUMNS = (
    "precision",
    "recall",
    "f1",
    "ndcg",
    "category_macro_precision",
    "category_macro_recall",
    "category_macro_f1",
    "category_macro_ndcg",
)
COUNT_COLUMNS = ("evaluated_queries", "skipped_queries", "gallery_size")


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        required=True,
        help="Result directories or comparison.csv files, one per seed.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _resolve_csv(path: Path) -> Path:
    candidate = path / "comparison.csv" if path.is_dir() else path
    if not candidate.is_file():
        raise FileNotFoundError(f"Comparison CSV not found: {candidate}")
    return candidate.resolve()


def _seed_metadata(csv_path: Path) -> dict:
    result_dir = csv_path.parent
    experiment_path = result_dir / "experiment.json"
    experiment = {}
    if experiment_path.is_file():
        experiment = json.loads(experiment_path.read_text(encoding="utf-8"))

    run_dir = experiment.get("run_dir")
    training_path = Path(run_dir) / "training.json" if run_dir else None
    seed = None
    if training_path and training_path.is_file():
        training = json.loads(training_path.read_text(encoding="utf-8"))
        seed = training.get("seed")
    if seed is None:
        seed = result_dir.name

    return {
        "seed": str(seed),
        "comparison_csv": str(csv_path),
        "experiment_json": (
            str(experiment_path.resolve()) if experiment_path.is_file() else None
        ),
        "training_json": (
            str(training_path.resolve())
            if training_path and training_path.is_file()
            else None
        ),
        "catalog_sha256": experiment.get("catalog_sha256"),
        "dataset_sha256": experiment.get("dataset_sha256"),
        "query_count": experiment.get("query_count"),
        "gallery_size": experiment.get("gallery_size"),
    }


def _key_set(frame: pd.DataFrame, columns: tuple[str, ...]) -> set[tuple]:
    return set(
        map(tuple, frame[list(columns)].astype(str).itertuples(index=False, name=None))
    )


def main():
    args = arguments()
    if len(args.inputs) < 2:
        raise ValueError("At least two completed seed results are required.")

    frames = []
    provenance = []
    reference_keys = None
    key_columns = None
    seen_seeds = set()

    for source in args.inputs:
        csv_path = _resolve_csv(source)
        metadata = _seed_metadata(csv_path)
        if metadata["seed"] in seen_seeds:
            raise ValueError(f"Duplicate seed input: {metadata['seed']}")
        seen_seeds.add(metadata["seed"])

        frame = pd.read_csv(csv_path)
        required = {"system", "task", "k", "precision", "recall", "f1", "ndcg"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")

        current_keys = tuple(c for c in IDENTITY_COLUMNS if c in frame.columns)
        if key_columns is None:
            key_columns = current_keys
        elif current_keys != key_columns:
            raise ValueError("All inputs must use the same comparison schema.")

        keys = _key_set(frame, key_columns)
        if reference_keys is None:
            reference_keys = keys
        elif keys != reference_keys:
            raise ValueError(
                "Seed comparisons do not contain identical systems/tasks/K values "
                f"(missing={len(reference_keys - keys)}, "
                f"extra={len(keys - reference_keys)})."
            )

        frame.insert(0, "seed", metadata["seed"])
        frames.append(frame)
        provenance.append(metadata)

    per_seed = pd.concat(frames, ignore_index=True)
    numeric_columns = [
        c for c in (*METRIC_COLUMNS, *COUNT_COLUMNS) if c in per_seed.columns
    ]
    grouped = per_seed.groupby(list(key_columns), dropna=False, sort=True)[
        numeric_columns
    ]
    summary = grouped.mean().add_suffix("_mean").join(
        grouped.std(ddof=1).add_suffix("_std")
    ).reset_index()
    summary.insert(len(key_columns), "seed_count", len(frames))

    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(output / "per_seed_results.csv", index=False)
    summary.to_csv(output / "mean_std_summary.csv", index=False)
    (output / "aggregation.json").write_text(
        json.dumps(
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "seed_count": len(frames),
                "aggregation": (
                    "unweighted arithmetic mean and sample standard deviation "
                    "across seeds"
                ),
                "identity_columns": list(key_columns),
                "metric_columns": numeric_columns,
                "inputs": provenance,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"Saved multi-seed aggregation to: {output}")


if __name__ == "__main__":
    main()
