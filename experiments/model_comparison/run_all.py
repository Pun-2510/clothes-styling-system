"""Run the modality benchmark and the existing pretrained/fine-tuned CLIP comparison."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[2]
PROJECT = REPOSITORY / "project"
DEFAULT_RUN = PROJECT / "runs" / "clip_finetune_3epochs"
DEFAULT_RESULTS = Path(__file__).resolve().parent / "results"
DEFAULT_RESNET_CHECKPOINT = REPOSITORY / "model" / "Web_Test" / "models" / "resnet_outfit.pth"


def build_matched_outputs(output):
    """Combine only task-compatible category-retrieval rows in one table."""
    import pandas as pd

    modality_csv = output / "modalities" / "comparison.csv"
    modality_category_csv = output / "modalities" / "comparison_by_category.csv"
    clip_json = output / "clip" / "comparison.json"
    if not (modality_csv.is_file() and modality_category_csv.is_file() and clip_json.is_file()):
        return []

    rows = pd.read_csv(modality_csv).to_dict(orient="records")
    category_rows = pd.read_csv(modality_category_csv).to_dict(orient="records")
    clip_report = json.loads(clip_json.read_text(encoding="utf-8"))
    ks = sorted(set(int(k) for k in clip_report["ks"]))
    for variant in ("pretrained", "finetuned"):
        metrics = clip_report["metrics"][variant]["image_to_image_category"]
        system = f"clip_{variant}_image_only"
        for k in ks:
            rows.append(
                {
                    "system": system,
                    "input": "image",
                    "task": "same-category product retrieval",
                    "k": k,
                    "precision": metrics[f"precision@{k}"],
                    "recall": metrics[f"recall@{k}"],
                    "f1": metrics[f"f1@{k}"],
                    "category_macro_precision": metrics["category_macro"][f"precision@{k}"],
                    "category_macro_recall": metrics["category_macro"][f"recall@{k}"],
                    "category_macro_f1": metrics["category_macro"][f"f1@{k}"],
                    "evaluated_queries": metrics["evaluated_queries"],
                    "skipped_queries": metrics["skipped_queries"],
                    "gallery_size": metrics["gallery_size"],
                }
            )
        for category, values in metrics["per_category"].items():
            for k in ks:
                category_rows.append(
                    {
                        "system": system,
                        "category": category,
                        "k": k,
                        "precision": values[f"precision@{k}"],
                        "recall": values[f"recall@{k}"],
                        "f1": values[f"f1@{k}"],
                        "query_count": values["query_count"],
                        "evaluated_queries": values["evaluated_queries"],
                        "skipped_queries": values["skipped_queries"],
                    }
                )

    frame = pd.DataFrame(rows)
    category_frame = pd.DataFrame(category_rows)
    frame.to_csv(output / "matched_comparison.csv", index=False)
    category_frame.to_csv(output / "matched_comparison_by_category.csv", index=False)
    (output / "matched_comparison.json").write_text(
        json.dumps(
            {
                "task": "same-category product retrieval",
                "text_policy": "Product + Description; category omitted",
                "self_match": "excluded",
                "rows": rows,
                "limitations": [
                    "The fine-tuned CLIP text result is a category-free inference ablation; its training prompts included category.",
                    "The RRF system requires both image and text, while the other rows use one input modality.",
                ],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    lines = [
        "# Matched category-retrieval comparison",
        "",
        "All rows use the same held-out products, category relevance, self-match exclusion, and K values.",
        "Text-only rows use product name and description without the category label.",
        "",
        "| System | Input | K | Precision | Recall | F1 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['system']} | {row['input']} | {int(row['k'])} | "
            f"{float(row['precision']):.4f} | {float(row['recall']):.4f} | "
            f"{float(row['f1']):.4f} |"
        )
    lines += [
        "",
        "The RRF row requires both image and text and should not be treated as a single-input model.",
        "The fine-tuned CLIP text result is an inference ablation because its training prompts included category.",
    ]
    (output / "matched_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return [
        "matched_comparison.csv",
        "matched_comparison_by_category.csv",
        "matched_comparison.json",
        "matched_summary.md",
    ]


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 5, 10])
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--resnet-source", choices=("project-trained", "imagenet"), default="project-trained")
    parser.add_argument("--resnet-checkpoint", type=Path, default=DEFAULT_RESNET_CHECKPOINT)
    parser.add_argument("--clip-catalog-csv", type=Path)
    parser.add_argument("--clip-cache-root", type=Path)
    parser.add_argument(
        "--reuse-clip-dir",
        type=Path,
        help="Reuse a completed clip/ result directory instead of rerunning CLIP cross-modal evaluation.",
    )
    parser.add_argument("--skip-modalities", action="store_true")
    parser.add_argument("--skip-clip", action="store_true")
    return parser.parse_args()


def main():
    args = arguments()
    if args.skip_modalities and args.skip_clip:
        raise ValueError("At least one comparison must be enabled.")
    if bool(args.clip_catalog_csv) != bool(args.clip_cache_root):
        raise ValueError("--clip-catalog-csv and --clip-cache-root must be provided together.")
    if args.reuse_clip_dir and args.skip_clip:
        raise ValueError("--reuse-clip-dir cannot be combined with --skip-clip.")
    if args.reuse_clip_dir and (args.clip_catalog_csv or args.clip_cache_root):
        raise ValueError("Cached CLIP embedding arguments are not used with --reuse-clip-dir.")
    output = (
        args.output_dir
        or DEFAULT_RESULTS / datetime.now().strftime("%Y%m%d_%H%M%S_full")
    ).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    commands = []
    if not args.skip_modalities:
        command = [
            sys.executable, "-m", "experiments.model_comparison.compare_modalities",
            "--run-dir", str(args.run_dir.resolve()),
            "--output-dir", str(output / "modalities"),
            "--device", args.device,
            "--batch-size", str(args.batch_size),
            "--ks", *[str(k) for k in args.ks],
            "--rrf-k", str(args.rrf_k),
            "--resnet-source", args.resnet_source,
            "--resnet-checkpoint", str(args.resnet_checkpoint.resolve()),
        ]
        commands.append({"name": "modalities", "cwd": str(REPOSITORY), "command": command})
    if not args.skip_clip and not args.reuse_clip_dir:
        command = [
            sys.executable, "-m", "comparisons.compare_clip",
            "--run-dir", str(args.run_dir.resolve()),
            "--output-dir", str(output / "clip"),
            "--device", args.device,
            "--batch-size", str(args.batch_size),
            "--ks", *[str(k) for k in args.ks],
        ]
        if args.clip_catalog_csv:
            command += [
                "--catalog-csv", str(args.clip_catalog_csv.resolve()),
                "--cache-root", str(args.clip_cache_root.resolve()),
            ]
        commands.append({"name": "clip", "cwd": str(PROJECT), "command": command})

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "run_dir": str(args.run_dir.resolve()),
        "commands": commands,
        "reused_clip_results": str(args.reuse_clip_dir.resolve()) if args.reuse_clip_dir else None,
    }
    manifest_path = output / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    try:
        if args.reuse_clip_dir:
            source = args.reuse_clip_dir.resolve()
            required = ("comparison.csv", "comparison.json", "comparison_by_category.csv")
            missing = [name for name in required if not (source / name).is_file()]
            if missing:
                raise FileNotFoundError(
                    f"Reusable CLIP result directory is missing: {', '.join(missing)}"
                )
            previous = json.loads((source / "comparison.json").read_text(encoding="utf-8"))
            if Path(previous["training_run"]).resolve() != args.run_dir.resolve():
                raise ValueError("Reusable CLIP results belong to a different training run.")
            shutil.copytree(source, output / "clip")
            print(f"Reused completed CLIP results: {source}", flush=True)
        for stage in commands:
            print(f"Running {stage['name']} comparison...", flush=True)
            subprocess.run(stage["command"], cwd=stage["cwd"], check=True)
        manifest["matched_outputs"] = build_matched_outputs(output)
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        raise
    manifest["status"] = "complete"
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"All requested comparisons are complete: {output}")


if __name__ == "__main__":
    main()
