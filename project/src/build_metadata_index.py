"""Build the project-local TF-IDF catalog metadata index."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import METADATA_INDEX_DIR, PROCESSED_CSV
from src.metadata_retrieval import build_metadata_index


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog-csv", type=Path, default=PROCESSED_CSV)
    parser.add_argument("--output-dir", type=Path, default=METADATA_INDEX_DIR)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--max-df", type=float, default=0.98)
    parser.add_argument("--max-features", type=int, default=50_000)
    return parser.parse_args()


def main():
    args = arguments()
    if args.min_df < 1 or not 0.0 < args.max_df <= 1.0:
        raise ValueError("min-df must be positive and max-df must be in (0, 1].")
    metadata = build_metadata_index(
        args.catalog_csv,
        args.output_dir,
        min_df=args.min_df,
        max_df=args.max_df,
        max_features=args.max_features,
    )
    print(
        "TF-IDF metadata index ready | "
        f"rows={metadata['rows']:,} | features={metadata['features']:,} | "
        f"output={Path(args.output_dir).resolve()}"
    )


if __name__ == "__main__":
    main()
