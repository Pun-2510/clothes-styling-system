"""Aggregate paired Hard/Dynamic category-recovery diagnostics across seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .compare_modalities import _prepare_output


METRICS = ("recovery_at_k", "loss_at_k", "net_hit_rate_change")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if len(args.inputs) < 2:
        parser.error("Provide at least two independently sampled catalog runs.")

    frames = []
    protocols = []
    for path in args.inputs:
        directory = path if path.is_dir() else path.parent
        diagnostics = directory / "hard_dynamic_recovery.csv"
        experiment = directory / "experiment.json"
        if not diagnostics.is_file() or not experiment.is_file():
            raise FileNotFoundError(
                f"Missing recovery diagnostics or experiment record in {directory}"
            )
        settings = json.loads(experiment.read_text(encoding="utf-8"))
        protocols.append((
            settings.get("gallery_size"),
            tuple(settings.get("ks", [])),
            settings.get("category_bonus"),
            settings.get("category_confidence_floor"),
            settings.get("category_confidence_threshold"),
            settings.get("category_top_per_class"),
            settings.get("category_prototype_weight"),
            settings.get("category_relation_neighbors"),
            settings.get("related_top1_bonus_factor"),
            settings.get("related_top2_bonus_factor"),
            settings.get("unrelated_top1_bonus_factor"),
        ))
        frame = pd.read_csv(diagnostics)
        required = {"k", *METRICS}
        if not required.issubset(frame.columns):
            raise ValueError(f"Missing recovery columns in {diagnostics}")
        if frame["k"].duplicated().any():
            raise ValueError(f"Duplicate K values in {diagnostics}")
        frame.insert(0, "seed_run", directory.parent.name)
        frame.insert(1, "source", str(diagnostics.resolve()))
        frames.append(frame)

    if len(set(protocols)) != 1:
        raise ValueError("All runs must use the same gallery size, K values and category policy.")
    if any(set(frame["k"]) != set(frames[0]["k"]) for frame in frames[1:]):
        raise ValueError("All runs must contain the same K values.")
    if len({frame["seed_run"].iloc[0] for frame in frames}) != len(frames):
        raise ValueError("Inputs must come from distinct seed directories.")

    combined = pd.concat(frames, ignore_index=True)
    rows = []
    for k, group in combined.groupby("k", sort=True):
        row = {"k": int(k), "seed_count": len(group)}
        for metric in METRICS:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))
        rows.append(row)

    output = _prepare_output(args.output_dir)
    combined.to_csv(output / "per_seed_results.csv", index=False)
    pd.DataFrame(rows).to_csv(output / "mean_std_summary.csv", index=False)
    (output / "aggregation.json").write_text(
        json.dumps({
            "metric_definition": {
                "recovery_at_k": "Among eligible queries with a wrong predicted category, fraction where Dynamic retrieves at least one true-category product and Hard does not.",
                "loss_at_k": "Among eligible queries with a correct predicted category, fraction where Hard retrieves a true-category product but Dynamic does not.",
                "net_hit_rate_change": "Dynamic hit rate minus Hard hit rate over all eligible queries.",
            },
            "relevance": "Same catalog category as the query, excluding the query product.",
            "gallery_size": protocols[0][0],
            "ks": list(protocols[0][1]),
            "inputs": [str(path.resolve()) for path in args.inputs],
        }, indent=2),
        encoding="utf-8",
    )
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"Saved paired recovery summary to: {output}")


if __name__ == "__main__":
    main()
