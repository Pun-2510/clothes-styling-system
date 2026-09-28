"""Prepare an exact-sized catalog, encode both CLIPs, and compare held-out products."""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import sys

from src.config import BASE_DIR, CSV_FILE, IMAGE_DIR, CLIP_MODEL


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-products", type=int, required=True,
                        help="Exact number of valid, image-deduplicated catalog rows")
    parser.add_argument("--output-dir", type=Path, required=True, help="New experiment folder")
    parser.add_argument("--csv", type=Path, default=CSV_FILE)
    parser.add_argument("--image-dir", type=Path, default=IMAGE_DIR)
    parser.add_argument("--categories", nargs="+")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reuse-run", type=Path,
                        help="Reuse a completed CLIP run and its original held-out test split; no training")
    parser.add_argument("--model", default=CLIP_MODEL, help="Base model for NEW training only")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--trainable", choices=["full", "projections"], default="full")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 5, 10])
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--dry-run", action="store_true", help="Print commands without writing files or loading models")
    args = parser.parse_args(argv)
    if args.num_products < 2 or args.batch_size < 2 or args.epochs < 1 or any(k < 1 for k in args.ks):
        parser.error("Need num-products >= 2, batch-size >= 2, epochs >= 1 and positive ks")
    if (not math.isfinite(args.learning_rate) or args.learning_rate <= 0
            or not math.isfinite(args.weight_decay) or args.weight_decay < 0):
        parser.error("Need finite positive learning-rate and nonnegative weight-decay")
    if not (0 < args.val_fraction < 1 and 0 < args.test_fraction < 1
            and args.val_fraction + args.test_fraction < 1):
        parser.error("Validation/test fractions must be positive and sum to < 1")
    args.output_dir = args.output_dir.resolve()
    args.csv, args.image_dir = args.csv.resolve(), args.image_dir.resolve()
    if args.reuse_run:
        args.reuse_run = args.reuse_run.resolve()
    if Path(args.model).is_dir():
        args.model = str(Path(args.model).resolve())
    return args


def build_commands(args):
    root = args.output_dir
    catalog = root / "data" / "products.csv"
    run = args.reuse_run or root / "training"
    base_model = args.model
    if args.reuse_run:
        training = json.loads((run / "training.json").read_text(encoding="utf-8"))
        if training.get("status") != "complete":
            raise ValueError("--reuse-run requires completed training")
        base_model = training["base_model"]
    device = ["--device", args.device] if args.device else []

    def command(module, *options):
        return [sys.executable, "-m", module, *map(str, options)]

    prepare = command("src.prepare_dataset", "--csv", args.csv, "--image-dir", args.image_dir,
                      "--num-products", args.num_products, "--seed", args.seed, "--output", catalog)
    if args.categories:
        prepare += ["--categories", *args.categories]
    stages = [("prepare", prepare)]
    if not args.reuse_run:
        stages.append(("train", command("src.finetune_clip", "--csv", catalog, "--run-dir", run,
                      "--model", base_model, "--epochs", args.epochs, "--batch-size", args.batch_size,
                      "--learning-rate", args.learning_rate, "--weight-decay", args.weight_decay,
                      "--trainable", args.trainable, "--seed", args.seed,
                      "--val-fraction", args.val_fraction, "--test-fraction", args.test_fraction, *device)))
    for label, model in (("pretrained", base_model), ("finetuned", run / "best")):
        stages.append((f"encode_{label}", command("src.generate_clip_embeddings", "--csv", catalog,
                      "--model", model, "--output-dir", root / "embeddings" / label,
                      "--batch-size", args.batch_size, *device)))
    stages.append(("compare", command("comparisons.compare_clip", "--run-dir", run,
                  "--catalog-csv", catalog, "--cache-root", root / "embeddings",
                  "--output-dir", root, "--ks", *args.ks, *device)))
    return stages, run


def validate_reused_catalog(run, catalog_path):
    # Fail before encoding thousands of products if the catalog cannot be evaluated.
    import pandas as pd
    from comparisons.compare_clip import catalog_test_rows
    from src.clip_data import load_experiment_data
    from src.clip_runtime import model_identity, same_model_identity

    products, metadata = load_experiment_data(run)
    training = json.loads((run / "training.json").read_text(encoding="utf-8"))
    if training["dataset_sha256"] != metadata["dataset_sha256"]:
        raise ValueError("Run dataset identity does not match training")
    for expected, source in ((training["base_model_identity"], training["base_model"]),
                             (training["checkpoint_identity"], run / "best")):
        if not same_model_identity(expected, model_identity(source)):
            raise ValueError("Run model changed since training")
    test, _ = catalog_test_rows(products.loc[products.split == "test"],
                               pd.read_csv(catalog_path, dtype={"product_id": str}))
    print(f"Reusable held-out products: {len(test)} (not the full catalog).", flush=True)


def website_environment(root, run, image_dir):
    values = {"FASHION_CATALOG_DIR": root / "data",
              "FASHION_EMBEDDINGS_DIR": root / "embeddings" / "finetuned",
              "FASHION_CHECKPOINT_DIR": run / "best", "FASHION_IMAGE_DIR": image_dir}
    lines = []
    for name, path in values.items():
        value = path.resolve().as_posix()
        if any(char in value for char in ("'", "\n", "\r")):
            raise ValueError("Artifact paths cannot contain quotes or newlines for the Compose env file")
        lines.append(f"{name}='{value}'")
    return "\n".join(lines) + "\n"


def main(argv=None):
    args = parse_args(argv)
    stages, run = build_commands(args)
    for name, command in stages:
        print(f"{name}: {subprocess.list2cmdline(command)}", flush=True)
    if args.dry_run:
        return
    if not args.csv.is_file() or not args.image_dir.is_dir():
        raise FileNotFoundError("Check --csv and --image-dir before starting")
    # Never replace an existing experiment, the current website catalog, or an old checkpoint.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    log_dir = args.output_dir / "logs"
    log_dir.mkdir()
    manifest_path = args.output_dir / "pipeline.json"
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(),
                "status": "running", "mode": "reuse" if args.reuse_run else "new_training",
                "parameters": {key: str(value) if isinstance(value, Path) else value
                               for key, value in vars(args).items()},
                "stages": [{"name": name, "command": command, "status": "pending"}
                           for name, command in stages]}

    def save():
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    save()
    try:
        for stage in manifest["stages"]:
            stage["status"] = "running"
            save()
            # Child output goes straight to the log: no encoding-dependent terminal pipe.
            log_path = log_dir / f"{stage['name']}.log"
            print(f"Running {stage['name']} ... log: {log_path}", flush=True)
            with log_path.open("w", encoding="utf-8") as log:
                subprocess.run(stage["command"], cwd=BASE_DIR, stdout=log,
                               stderr=subprocess.STDOUT, check=True,
                               env=child_environment())
            if stage["name"] == "prepare" and args.reuse_run:
                validate_reused_catalog(run, args.output_dir / "data" / "products.csv")
            stage["status"] = "complete"
            save()
        report = json.loads((args.output_dir / "comparison.json").read_text(encoding="utf-8"))
        manifest["catalog_size"] = report["catalog_size"]
        manifest["evaluated_test_size"] = report["evaluated_test_size"]
        env_path = args.output_dir / "website.env"
        env_path.write_text(website_environment(args.output_dir, run, args.image_dir), encoding="utf-8")
        manifest["status"] = "complete"
        save()
        print(f"Done: {args.output_dir / 'comparison.csv'}", flush=True)
        print(f"Catalog: {report['catalog_size']}; test queries: {report['evaluated_test_size']}")
        print(f'Website: docker compose --env-file "{env_path}" up -d --build')
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error"] = str(error)
        for stage in manifest["stages"]:
            if stage["status"] == "running":
                stage["status"] = "failed"
        save()
        raise


def child_environment():
    import os
    return {**os.environ, "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1"}


if __name__ == "__main__":
    main()
