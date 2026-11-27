"""Benchmark the deployed image, text, and combined API routes on matched cases.

This measures warm sequential HTTP response times; it does not measure retrieval
quality or isolated model inference time.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path


ROUTES = {
    "text_only": "/api/recommendations/text",
    "image_only": "/api/recommendations/image",
    "combined": "/api/recommendations/combined",
}


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def select_cases(catalog_csv: Path, image_dir: Path, count: int, seed: int) -> list[dict]:
    with catalog_csv.open(newline="", encoding="utf-8-sig") as handle:
        products = list(csv.DictReader(handle))
    rng = random.Random(seed)
    rng.shuffle(products)
    cases = []
    used_categories = set()
    for product in products:
        category = (product.get("category") or "").strip()
        name = (product.get("product_name") or "").strip()
        reference = (product.get("image_reference") or "").strip()
        image = image_dir / Path(reference).name
        if not category or category in used_categories or not name or not image.is_file():
            continue
        if image.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        if image.stat().st_size > 5 * 1024 * 1024:
            continue
        cases.append({
            "case_id": len(cases) + 1,
            "product_id": product.get("product_id", ""),
            "category": category,
            "text": name[:512],
            "image": image,
        })
        used_categories.add(category)
        if len(cases) == count:
            break
    if len(cases) < count:
        raise ValueError(f"Only {len(cases)} valid distinct-category cases found; requested {count}.")
    return cases


def multipart(fields: dict[str, str], image: Path) -> tuple[bytes, str]:
    boundary = "fashion-benchmark-" + uuid.uuid4().hex
    chunks = []
    for key, value in fields.items():
        chunks.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode()
        )
    mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}[image.suffix.lower()]
    chunks.append(
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="query{image.suffix.lower()}"\r\nContent-Type: {mime}\r\n\r\n'.encode()
    )
    chunks.append(image.read_bytes())
    chunks.append(f"\r\n--{boundary}--\r\n".encode())
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def request_payload(mode: str, case: dict, top_k: int) -> tuple[bytes, str]:
    if mode == "text_only":
        return json.dumps({
            "query": case["text"], "language": "en", "top_k": top_k
        }).encode(), "application/json"
    fields = {"top_k": str(top_k)}
    if mode == "image_only":
        fields["category_mode"] = "hard_category"
    else:
        fields.update({"query": case["text"], "language": "en"})
    return multipart(fields, case["image"])


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost")
    parser.add_argument("--catalog-csv", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=8)
    parser.add_argument("--pause-seconds", type=float, default=0.30)
    parser.add_argument("--timeout-seconds", type=float, default=120)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.top_k <= 10 or args.sample_size < 2 or args.repetitions < 2:
        parser.error("top-k must be 1..10; sample-size and repetitions must be >= 2")
    if args.warmup < 0 or args.pause_seconds < 0:
        parser.error("warmup and pause-seconds must be non-negative")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    cases = select_cases(args.catalog_csv, args.image_dir, args.sample_size, args.seed)
    base = args.base_url.rstrip("/")
    rows = []
    rng = random.Random(args.seed)
    # Interleave modes and cases in every round to reduce effects from time drift.
    for round_number in range(-args.warmup, args.repetitions):
        tasks = [(mode, case) for case in cases for mode in ROUTES]
        rng.shuffle(tasks)
        for mode, case in tasks:
            data, content_type = request_payload(mode, case, args.top_k)
            request = urllib.request.Request(
                base + ROUTES[mode], data=data,
                headers={"Content-Type": content_type}, method="POST",
            )
            start = time.perf_counter()
            try:
                with urllib.request.urlopen(request, timeout=args.timeout_seconds) as response:
                    response_bytes = response.read()
                    status = response.status
                    upstream = response.headers.get("X-Upstream-Addr", "")
            except urllib.error.HTTPError as error:
                raise RuntimeError(
                    f"{mode}, case {case['case_id']}: HTTP {error.code}: {error.read(400)!r}"
                ) from error
            latency_ms = (time.perf_counter() - start) * 1000
            if status != 200:
                raise RuntimeError(f"{mode}, case {case['case_id']}: HTTP {status}")
            result = json.loads(response_bytes)
            if round_number >= 0:
                server_seconds = result.get("elapsed_seconds")
                if mode == "image_only":
                    server_seconds = result.get("processing_times", {}).get("total")
                rows.append({
                    "mode": mode, "round": round_number + 1,
                    "case_id": case["case_id"], "product_id": case["product_id"],
                    "category": case["category"],
                    "latency_ms": round(latency_ms, 3),
                    "server_elapsed_ms": round(float(server_seconds) * 1000, 3) if server_seconds is not None else "",
                    "request_bytes": len(data), "response_bytes": len(response_bytes),
                    "results_count": len(result.get("items", [])), "upstream": upstream,
                })
            time.sleep(args.pause_seconds)
        print(f"Round {round_number + 1}/{args.repetitions} completed")
    summary = []
    for mode in ROUTES:
        matches = [row for row in rows if row["mode"] == mode]
        times = [row["latency_ms"] for row in matches]
        server_times = [row["server_elapsed_ms"] for row in matches if row["server_elapsed_ms"] != ""]
        summary.append({
            "mode": mode, "queries": len(cases), "runs": len(matches), "top_k": args.top_k,
            "mean_ms": round(statistics.mean(times), 2),
            "median_ms": round(statistics.median(times), 2),
            "p95_ms": round(percentile(times, 0.95), 2),
            "server_median_ms": round(statistics.median(server_times), 2) if server_times else "",
            "mean_request_kib": round(statistics.mean(row["request_bytes"] for row in matches) / 1024, 2),
            "mean_response_kib": round(statistics.mean(row["response_bytes"] for row in matches) / 1024, 2),
        })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "request_latency.csv", rows)
    write_csv(args.output_dir / "performance_summary.csv", summary)
    write_csv(args.output_dir / "benchmark_cases.csv", [{
        "case_id": case["case_id"], "product_id": case["product_id"],
        "category": case["category"], "text": case["text"],
        "image_reference": case["image"].name,
    } for case in cases])
    (args.output_dir / "benchmark_protocol.json").write_text(json.dumps({
        "base_url": base, "catalog_csv": str(args.catalog_csv.resolve()),
        "image_dir": str(args.image_dir.resolve()), "sample_size": args.sample_size,
        "seed": args.seed, "top_k": args.top_k, "warmup_rounds": args.warmup,
        "measured_rounds": args.repetitions, "pause_seconds": args.pause_seconds,
        "sequential_requests": True, "mode_order": "shuffled per round",
        "language": "en", "audience_filter": None,
        "measurement": "warm client end-to-end HTTP time through public reverse proxy",
        "server_elapsed_note": "backend-reported time; may exclude API serialization and translation",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    for row in summary:
        print(row)


if __name__ == "__main__":
    main()
