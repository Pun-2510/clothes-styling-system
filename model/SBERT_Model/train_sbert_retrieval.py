"""Fine-tune the SBERT retrieval encoder on balanced fashion categories.

This is intentionally separate from ``train.py``, which trains a BERT category
classifier rather than the SentenceTransformer used by Web_Test.
"""

import argparse
import csv
import json
import os
import random
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer import losses
from sentence_transformers.sentence_transformer.evaluation import InformationRetrievalEvaluator


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_DATASET = "ashraq/fashion-product-images-small"
CLASSES = ["Tshirts", "Shirts", "Jeans", "Trousers", "Dresses", "Jackets"]


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--max-per-class", type=int, default=200)
    parser.add_argument("--validation-ratio", type=float, default=0.20)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--margin", type=float, default=0.20)
    parser.add_argument("--max-seq-length", type=int, default=64)
    parser.add_argument("--trainable-layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu-threads", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "models" / "sbert_fashion_finetuned",
    )
    parser.add_argument("--log-dir", type=Path)
    return parser.parse_args()


def product_text(row):
    """Build text without directly copying the supervised category label."""
    name = str(row.get("productDisplayName") or "").strip()
    fields = [
        ("Color", row.get("baseColour")),
        ("Gender", row.get("gender")),
        ("Usage", row.get("usage")),
        ("Season", row.get("season")),
    ]
    details = [f"{label}: {str(value).strip()}" for label, value in fields
               if value is not None and str(value).strip()]
    return ". ".join([name, *details])


def balanced_rows(dataset_name, max_per_class, validation_ratio, seed):
    raw = load_dataset(dataset_name, split="train")
    if "image" in raw.column_names:
        raw = raw.remove_columns("image")
    grouped = defaultdict(list)
    seen = defaultdict(set)
    for row in raw:
        category = str(row.get("articleType") or "")
        if category not in CLASSES:
            continue
        text = product_text(row)
        if not text or text in seen[category]:
            continue
        seen[category].add(text)
        grouped[category].append(text)

    rng = random.Random(seed)
    train_rows, validation_rows = [], []
    counts = {}
    for label, category in enumerate(CLASSES):
        values = grouped[category]
        if len(values) < max_per_class:
            raise ValueError(
                f"Class {category} only has {len(values)} unique texts; "
                f"requested {max_per_class}."
            )
        selected = rng.sample(values, max_per_class)
        validation_count = max(4, round(len(selected) * validation_ratio))
        validation = selected[:validation_count]
        training = selected[validation_count:]
        train_rows.extend({"sentence": text, "label": label} for text in training)
        validation_rows.extend(
            {"sentence": text, "label": label, "category": category}
            for text in validation
        )
        counts[category] = {
            "selected": len(selected),
            "train": len(training),
            "validation": len(validation),
        }
    rng.shuffle(train_rows)
    rng.shuffle(validation_rows)
    return train_rows, validation_rows, counts


def retrieval_evaluator(validation_rows, seed):
    grouped = defaultdict(list)
    for row in validation_rows:
        grouped[row["category"]].append(row)
    rng = random.Random(seed + 1)
    queries, corpus, relevant = {}, {}, {}
    for category in CLASSES:
        rows = list(grouped[category])
        rng.shuffle(rows)
        query_count = max(1, len(rows) // 4)
        query_rows, corpus_rows = rows[:query_count], rows[query_count:]
        corpus_ids = set()
        for index, row in enumerate(corpus_rows):
            corpus_id = f"{category}-c-{index}"
            corpus[corpus_id] = row["sentence"]
            corpus_ids.add(corpus_id)
        for index, row in enumerate(query_rows):
            query_id = f"{category}-q-{index}"
            queries[query_id] = row["sentence"]
            relevant[query_id] = set(corpus_ids)
    evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant,
        accuracy_at_k=[1, 5, 10],
        precision_recall_at_k=[1, 5, 10],
        mrr_at_k=[10],
        ndcg_at_k=[10],
        map_at_k=[10],
        batch_size=32,
        name="fashion_validation",
        show_progress_bar=False,
        write_csv=True,
        main_score_function="cosine",
    )
    return evaluator, len(queries), len(corpus)


def serializable(value):
    if isinstance(value, dict):
        return {str(key): serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def balanced_batches(rows, batch_size, seed):
    """Yield batches containing the same number of samples from every class."""
    if batch_size % len(CLASSES):
        raise ValueError(f"batch-size must be divisible by {len(CLASSES)}")
    per_class = batch_size // len(CLASSES)
    grouped = defaultdict(list)
    for row in rows:
        grouped[int(row["label"])].append(row)
    rng = random.Random(seed)
    for values in grouped.values():
        rng.shuffle(values)
    batch_count = min(len(values) // per_class for values in grouped.values())
    for batch_index in range(batch_count):
        batch = []
        start = batch_index * per_class
        for label in range(len(CLASSES)):
            batch.extend(grouped[label][start:start + per_class])
        rng.shuffle(batch)
        yield batch


def scheduler_factor(step, warmup_steps, total_steps):
    if step < warmup_steps:
        return float(step + 1) / max(1, warmup_steps)
    remaining = total_steps - step
    return max(0.0, float(remaining) / max(1, total_steps - warmup_steps))


def configure_trainable_layers(model, trainable_layers):
    """Freeze MiniLM except for the requested number of final encoder layers."""
    transformer = model[0].auto_model
    encoder = getattr(transformer, "encoder", None)
    layers = getattr(encoder, "layer", None)
    if layers is None:
        raise TypeError("Unsupported transformer architecture: encoder.layer not found")
    if not 1 <= trainable_layers <= len(layers):
        raise ValueError(f"trainable-layers must be between 1 and {len(layers)}")
    for parameter in transformer.parameters():
        parameter.requires_grad = False
    for layer in layers[-trainable_layers:]:
        for parameter in layer.parameters():
            parameter.requires_grad = True
    trainable = sum(parameter.numel() for parameter in model.parameters()
                    if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    return trainable, total


def write_history(path, history):
    keys = sorted({key for row in history for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(history)


def main():
    args = arguments()
    if args.epochs < 1 or args.batch_size < 4 or args.max_per_class < 10:
        raise ValueError("epochs >= 1, batch-size >= 4 and max-per-class >= 10 are required")
    if args.batch_size % len(CLASSES):
        raise ValueError(f"batch-size must be divisible by {len(CLASSES)}")
    if not 0 < args.validation_ratio < 0.5:
        raise ValueError("validation-ratio must be between 0 and 0.5")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, args.cpu_threads))
    args.output_dir = args.output_dir.expanduser().resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = (args.log_dir or ROOT / "logs" / f"sbert_finetune_{timestamp}")
    log_dir = log_dir.expanduser().resolve()
    log_dir.mkdir(parents=True, exist_ok=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    print("=" * 72, flush=True)
    print("SBERT FASHION RETRIEVAL FINE-TUNING", flush=True)
    print("=" * 72, flush=True)
    print(f"Base model       : {args.model}", flush=True)
    print(f"Dataset          : {args.dataset}", flush=True)
    print(f"Classes          : {CLASSES}", flush=True)
    print(f"Epochs           : {args.epochs}", flush=True)
    print(f"Batch size       : {args.batch_size}", flush=True)
    print(f"Learning rate    : {args.learning_rate}", flush=True)
    print(f"Max sequence     : {args.max_seq_length}", flush=True)
    print(f"Trainable layers : {args.trainable_layers}", flush=True)
    print(f"Device           : {'cuda' if torch.cuda.is_available() else 'cpu'}", flush=True)
    print(f"CPU threads      : {args.cpu_threads}", flush=True)
    print(f"Output model     : {args.output_dir}", flush=True)
    print(f"Structured logs  : {log_dir}", flush=True)

    train_rows, validation_rows, counts = balanced_rows(
        args.dataset, args.max_per_class, args.validation_ratio, args.seed
    )
    evaluator, query_count, corpus_count = retrieval_evaluator(
        validation_rows, args.seed
    )
    print(f"Train samples    : {len(train_rows)}", flush=True)
    print(f"Validation rows  : {len(validation_rows)}", flush=True)
    print(f"Eval queries     : {query_count}", flush=True)
    print(f"Eval corpus      : {corpus_count}", flush=True)
    print(f"Per-class counts : {json.dumps(counts, ensure_ascii=False)}", flush=True)

    model = SentenceTransformer(args.model, device="cuda" if torch.cuda.is_available() else "cpu")
    model.max_seq_length = args.max_seq_length
    trainable_parameters, total_parameters = configure_trainable_layers(
        model, args.trainable_layers
    )
    print(
        f"Trainable params : {trainable_parameters:,}/{total_parameters:,} "
        f"({100 * trainable_parameters / total_parameters:.2f}%)",
        flush=True,
    )
    print("\nBASELINE RETRIEVAL", flush=True)
    baseline = evaluator(model, output_path=str(log_dir), epoch=0, steps=0)
    print(json.dumps(serializable(baseline), ensure_ascii=False, indent=2), flush=True)

    triplet_loss = losses.BatchAllTripletLoss(
        model,
        distance_metric=losses.BatchHardTripletLossDistanceFunction.cosine_distance,
        margin=args.margin,
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=0.01,
    )
    batches_per_epoch = len(train_rows) // args.batch_size
    total_steps = batches_per_epoch * args.epochs
    warmup_steps = max(1, round(total_steps * 0.10))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: scheduler_factor(step, warmup_steps, total_steps),
    )

    print("\nTRAINING", flush=True)
    history = []
    best_mrr = -1.0
    best_epoch = None
    best_checkpoint = args.output_dir / "best_checkpoint"
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        epoch_losses = []
        batches = list(balanced_batches(train_rows, args.batch_size, args.seed + epoch))
        for batch_index, batch in enumerate(batches, start=1):
            optimizer.zero_grad(set_to_none=True)
            features = model.preprocess([row["sentence"] for row in batch])
            features = {
                key: value.to(model.device) if torch.is_tensor(value) else value
                for key, value in features.items()
            }
            labels = torch.tensor(
                [row["label"] for row in batch], dtype=torch.long, device=model.device
            )
            loss = triplet_loss([features], labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1
            epoch_losses.append(float(loss.detach().cpu()))
            if batch_index == 1 or batch_index % 10 == 0 or batch_index == len(batches):
                print(
                    f"Epoch [{epoch}/{args.epochs}] batch {batch_index}/{len(batches)} "
                    f"loss={epoch_losses[-1]:.6f} lr={scheduler.get_last_lr()[0]:.2e}",
                    flush=True,
                )

        model.eval()
        metrics = evaluator(
            model, output_path=str(log_dir), epoch=epoch, steps=global_step
        )
        average_loss = float(np.mean(epoch_losses))
        epoch_record = {
            "epoch": epoch,
            "train_loss": average_loss,
            "learning_rate": scheduler.get_last_lr()[0],
            "epoch_seconds": round(time.perf_counter() - epoch_started, 3),
            **serializable(metrics),
        }
        history.append(epoch_record)
        mrr = float(metrics["fashion_validation_cosine_mrr@10"])
        print(
            f"Epoch [{epoch}/{args.epochs}] COMPLETE "
            f"loss={average_loss:.6f} hit@1={metrics['fashion_validation_cosine_accuracy@1']:.4f} "
            f"hit@5={metrics['fashion_validation_cosine_accuracy@5']:.4f} "
            f"mrr@10={mrr:.4f} time={epoch_record['epoch_seconds']:.3f}s",
            flush=True,
        )
        if mrr > best_mrr:
            best_mrr = mrr
            best_epoch = epoch
            model.save_pretrained(str(best_checkpoint))
            print(f"Saved best checkpoint: epoch {epoch}", flush=True)

    model = SentenceTransformer(
        str(best_checkpoint), device="cuda" if torch.cuda.is_available() else "cpu"
    )
    model.save_pretrained(str(args.output_dir))

    print("\nFINAL RETRIEVAL", flush=True)
    final_metrics = evaluator(
        model, output_path=str(log_dir), epoch=best_epoch, steps=global_step
    )
    print(json.dumps(serializable(final_metrics), ensure_ascii=False, indent=2), flush=True)

    summary = {
        "status": "complete",
        "base_model": args.model,
        "output_model": str(args.output_dir),
        "dataset": args.dataset,
        "classes": CLASSES,
        "class_counts": counts,
        "train_samples": len(train_rows),
        "validation_rows": len(validation_rows),
        "evaluation_queries": query_count,
        "evaluation_corpus": corpus_count,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "margin": args.margin,
        "max_seq_length": args.max_seq_length,
        "trainable_layers": args.trainable_layers,
        "trainable_parameters": trainable_parameters,
        "total_parameters": total_parameters,
        "seed": args.seed,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "optimizer": "AdamW",
        "loss": "BatchAllTripletLoss_cosine",
        "best_epoch": best_epoch,
        "best_mrr_at_10": best_mrr,
        "global_steps": global_step,
        "baseline": serializable(baseline),
        "final": serializable(final_metrics),
        "history": history,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    (log_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_history(log_dir / "epoch_history.csv", history)
    (log_dir / "run_config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print("\nCOMPLETE", flush=True)
    print(f"Model   : {args.output_dir}", flush=True)
    print(f"Summary : {log_dir / 'training_summary.json'}", flush=True)
    print(f"History : {log_dir / 'epoch_history.csv'}", flush=True)
    print(f"Elapsed : {summary['elapsed_seconds']:.3f} seconds", flush=True)


if __name__ == "__main__":
    main()
