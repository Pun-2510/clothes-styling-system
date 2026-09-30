from pathlib import Path
import argparse
import json
import sys
import re

import pandas as pd
import numpy as np
from PIL import Image, UnidentifiedImageError


# =========================================================
# IMPORT CONFIG
# =========================================================

sys.path.append(
    str(
        Path(__file__)
        .resolve()
        .parent
        .parent
    )
)

from src.clip_data import file_sha256
from src.audit_dataset import write_category_audit
from src.config import (
    IMAGE_DIR,
    CSV_FILE,
    PROCESSED_CSV,
    MAX_PRODUCTS,
    MIN_PRODUCTS_PER_CATEGORY,
    RANDOM_SEED,
)


# =========================================================
# IMAGE EXTENSIONS
# =========================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}


# =========================================================
# NORMALIZE COLUMN NAME
# =========================================================

def normalize_column_name(name):
    """
    Chuyển tên column về dạng dễ so sánh.

    Ví dụ:

    Product Display Name
    ->
    productdisplayname
    """

    return re.sub(
        r"[^a-z0-9]+",
        "",
        str(name)
        .strip()
        .lower()
    )


# =========================================================
# FIND COLUMN
# =========================================================

def find_column(
    df,
    possible_names,
    required=False
):
    """
    Tìm column dựa trên nhiều tên có thể có.
    """

    normalized = {
        normalize_column_name(column): column
        for column in df.columns
    }

    # -----------------------------------------------------
    # Exact match
    # -----------------------------------------------------

    for name in possible_names:

        key = normalize_column_name(
            name
        )

        if key in normalized:

            return normalized[key]

    # -----------------------------------------------------
    # Fuzzy match
    # -----------------------------------------------------

    for name in possible_names:

        key = normalize_column_name(
            name
        )

        for normalized_name, original_name in normalized.items():

            if (
                key in normalized_name
                or normalized_name in key
            ):

                return original_name

    # -----------------------------------------------------
    # Not found
    # -----------------------------------------------------

    if required:

        raise KeyError(
            "\nKhông tìm thấy column.\n"
            f"Đã thử: {possible_names}\n\n"
            f"Columns hiện có:\n"
            f"{list(df.columns)}"
        )

    return None


# =========================================================
# FIND IMAGE
# =========================================================

def find_image(
    image_reference, image_dir=IMAGE_DIR
):
    """
    Tìm ảnh trong data/data/
    """

    if pd.isna(
        image_reference
    ):
        return None

    value = str(
        image_reference
    ).strip()

    if not value:
        return None

    # -----------------------------------------------------
    # Trường hợp CSV có filename
    #
    # Ví dụ:
    # 10001.jpg
    # -----------------------------------------------------

    filename = Path(
        value
    ).name

    candidate = (
        image_dir / filename
    )

    if candidate.is_file():

        return candidate


    # -----------------------------------------------------
    # Trường hợp CSV chỉ có ID
    #
    # Ví dụ:
    # 10001
    # -----------------------------------------------------

    stem = Path(
        value
    ).stem

    for extension in sorted(IMAGE_EXTENSIONS):

        candidate = (
            image_dir /
            f"{stem}{extension}"
        )

        if candidate.is_file():

            return candidate


    # -----------------------------------------------------
    # Không tìm thấy
    # -----------------------------------------------------

    return None


# =========================================================
# MAIN
# =========================================================

def sample_products(products, count, seed=RANDOM_SEED, min_per_category=1):
    """Exact-size sampling with a category floor before proportional fill.

    The floor is reached for every category when the requested size permits.
    If the budget is smaller, rows are distributed as evenly as possible. A
    category with fewer source rows than the floor contributes all its rows.
    """
    if count is None:
        return products.reset_index(drop=True)
    if count < 1 or count > len(products):
        raise ValueError(f"Requested {count} products, but {len(products)} are available after filtering.")
    if min_per_category < 1:
        raise ValueError("min_per_category must be at least 1.")
    groups = list(products.groupby("category", sort=True))
    sizes = np.array([len(group) for _, group in groups])
    allocation = np.zeros(len(groups), dtype=int)
    remaining = int(count)
    rng = np.random.default_rng(seed)

    # Raise all eligible categories one level at a time up to the target. This
    # avoids hard-coding priority categories and remains exact for small runs.
    for level in range(1, int(min_per_category) + 1):
        candidates = np.flatnonzero(sizes >= level)
        if not len(candidates) or remaining == 0:
            break
        if remaining >= len(candidates):
            allocation[candidates] += 1
            remaining -= len(candidates)
        else:
            chosen = rng.permutation(candidates)[:remaining]
            allocation[chosen] += 1
            remaining = 0
            break

    # Preserve the original distribution only after the shared floor has been
    # allocated. Largest remainders make the final size exact.
    if remaining:
        capacity = sizes - allocation
        quotas = (
            capacity * (remaining / capacity.sum())
            if capacity.sum()
            else np.zeros(len(groups))
        )
        extra = np.floor(quotas).astype(int)
        allocation += extra
        missing = remaining - int(extra.sum())
        order = np.argsort(-(quotas - extra), kind="stable")
        for index in order:
            if missing == 0:
                break
            if allocation[index] < sizes[index]:
                allocation[index] += 1
                missing -= 1

    selected = []
    group_seeds = rng.integers(0, np.iinfo(np.int32).max, size=len(groups))
    for (_, group), amount, group_seed in zip(groups, allocation, group_seeds):
        if amount:
            selected.append(group.sample(n=int(amount), random_state=int(group_seed)))
    return pd.concat(selected).reset_index(drop=True)


def prepare_catalog(csv_path=CSV_FILE, image_dir=IMAGE_DIR, output_path=PROCESSED_CSV,
                    num_products=MAX_PRODUCTS, seed=RANDOM_SEED, categories=None,
                    min_per_category=MIN_PRODUCTS_PER_CATEGORY):
    csv_path, image_dir, output_path = Path(csv_path), Path(image_dir), Path(output_path)
    if num_products is not None and num_products < 1:
        raise ValueError("num_products must be positive, or None for all products.")
    if min_per_category < 1:
        raise ValueError("min_per_category must be positive.")

    print("=" * 60)
    print("PREPARE MINI FASHION DATASET")
    print("=" * 60)


    # =====================================================
    # CHECK FILES
    # =====================================================

    if not csv_path.exists():

        raise FileNotFoundError(
            f"\nKhông tìm thấy CSV:\n"
            f"{csv_path}"
        )


    if not image_dir.is_dir():

        raise FileNotFoundError(
            f"\nKhông tìm thấy thư mục ảnh:\n"
            f"{image_dir}"
        )


    # =====================================================
    # LOAD CSV
    # =====================================================

    print(
        f"\nĐang đọc:\n"
        f"{csv_path}"
    )

    df = pd.read_csv(csv_path, on_bad_lines="skip", engine="python")

    print(
        f"\nSố dòng ban đầu: "
        f"{len(df):,}"
    )

    print(
        "\nCác column:"
    )

    print(
        list(df.columns)
    )


    # =====================================================
    # FIND IMPORTANT COLUMNS
    # =====================================================

    # Column chứa tên file ảnh
    image_column = find_column(
        df,
        [
            "image",
            "image_file",
            "image_filename",
            "image_file_name",
            "filename",
            "file_name",
            "image_path",
            "id",
        ],
        required=True,
    )


    # Product name
    product_name_column = find_column(
        df,
        [
            "product_display_name",
            "display_name",
            "product_name",
            "title",
            "name",
        ],
        required=False,
    )


    # Description
    description_column = find_column(
        df,
        [
            "product_description",
            "description",
            "desc",
            "text",
        ],
        required=False,
    )


    # Category
    category_column = find_column(
        df,
        [
            "category",
            "product_category",
            "master_category",
            "type",
        ],
        required=False,
    )


    print(
        f"\nImage column: "
        f"{image_column}"
    )

    print(
        f"Product name column: "
        f"{product_name_column}"
    )

    print(
        f"Description column: "
        f"{description_column}"
    )

    print(
        f"Category column: "
        f"{category_column}"
    )


    # =====================================================
    # CREATE OUTPUT DATAFRAME
    # =====================================================

    output = pd.DataFrame()


    # -----------------------------------------------------
    # Product ID
    # -----------------------------------------------------

    output["product_id"] = [
        f"product_{i:06d}"
        for i in range(
            len(df)
        )
    ]


    # -----------------------------------------------------
    # Image reference
    # -----------------------------------------------------

    output["image_reference"] = (
        df[image_column]
        .astype(str)
    )


    # -----------------------------------------------------
    # Product name
    # -----------------------------------------------------

    if product_name_column:

        output["product_name"] = (
            df[
                product_name_column
            ]
            .fillna("")
            .astype(str)
        )

    else:

        output["product_name"] = ""


    # -----------------------------------------------------
    # Description
    # -----------------------------------------------------

    if description_column:

        output["description"] = (
            df[
                description_column
            ]
            .fillna("")
            .astype(str)
        )

    else:

        output["description"] = ""


    # -----------------------------------------------------
    # Category
    # -----------------------------------------------------

    if category_column:

        output["category"] = (
            df[
                category_column
            ]
            .fillna("")
            .astype(str)
        )

    else:

        output["category"] = ""


    # =====================================================
    # FIND IMAGE PATH
    # =====================================================

    print(
        "\nĐang tìm ảnh..."
    )

    output["category"] = output.category.str.strip().replace("", "Unknown")
    output["product_name"] = output.product_name.str.strip()
    blank_names = output.product_name.eq("")
    output.loc[blank_names, "product_name"] = output.loc[blank_names, "image_reference"]
    if categories:
        missing_categories = set(categories) - set(output.category)
        if missing_categories:
            raise ValueError(f"Unknown categories: {sorted(missing_categories)}")
        output = output.loc[output.category.isin(categories)].copy()

    output["image_path"] = (
        output[
            "image_reference"
        ]
        .apply(lambda value: find_image(value, image_dir))
    )


    # =====================================================
    # REMOVE MISSING IMAGES
    # =====================================================

    before = len(
        output
    )

    output = output[
        output["image_path"]
        .notna()
    ].copy()


    print(
        f"\nẢnh tìm thấy: "
        f"{len(output):,} / "
        f"{before:,}"
    )


    # =====================================================
    # CONVERT PATH TO STRING
    # =====================================================

    output["image_path"] = (
        output["image_path"]
        .apply(
            lambda path:
            str(path.resolve())
        )
    )


    # =====================================================
    # REMOVE DUPLICATE IMAGE
    # =====================================================

    output = (
        output
        .drop_duplicates(
            subset=[
                "image_path"
            ]
        )
    )


    # =====================================================
    # RANDOM SAMPLE
    # =====================================================

    # Deduplicate contents before sampling so N stays exact in the training split.
    hashes = []
    for path in output.image_path:
        try:
            with Image.open(path) as image:
                image.verify()
            hashes.append(file_sha256(path))
        except (OSError, ValueError, UnidentifiedImageError):
            hashes.append(None)
    output["image_sha256"] = hashes
    invalid_images = int(output.image_sha256.isna().sum())
    output = output.dropna(subset=["image_sha256"])
    before_dedup = len(output)
    output = output.drop_duplicates("image_sha256")
    duplicates = before_dedup - len(output)
    eligible = len(output)
    if not eligible:
        raise ValueError("No valid products remain after filtering.")
    eligible_products = output.copy()
    output = sample_products(
        output,
        num_products,
        seed,
        min_per_category=min_per_category,
    )


    # =====================================================
    # RESET INDEX
    # =====================================================

    output = output.reset_index(
        drop=True
    )


    # =====================================================
    # CREATE OUTPUT DIRECTORY
    # =====================================================

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    # =====================================================
    # SAVE
    # =====================================================

    output.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )

    audit_path = output_path.with_suffix(".category_audit.csv")
    category_audit, audit_metadata = write_category_audit(
        eligible_products,
        output,
        audit_path,
        minimum=min_per_category,
    )

    summary = {"source_csv": str(csv_path.resolve()), "source_csv_sha256": file_sha256(csv_path),
               "image_dir": str(image_dir.resolve()), "requested_products": num_products,
               "selected_products": len(output), "eligible_products": eligible,
               "invalid_images": invalid_images, "duplicate_images_removed": duplicates,
               "seed": seed, "categories": categories,
               "min_per_category": min_per_category,
               "category_counts": output.category.value_counts().to_dict(),
               "category_audit_csv": str(audit_path.resolve()),
               "category_audit": audit_metadata,
               "products_sha256": file_sha256(output_path)}
    output_path.with_suffix(".preparation.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


    # =====================================================
    # SHOW RESULT
    # =====================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "DATASET READY"
    )

    print(
        "=" * 60
    )

    print(
        f"Số sản phẩm sử dụng: "
        f"{len(output):,}"
    )

    print(
        f"\nSaved:\n"
        f"{output_path}"
    )

    print(
        f"Category audit: {audit_path} "
        f"({int(category_audit.status.ne('ok').sum())} categories flagged)"
    )

    print(
        "\n5 sản phẩm đầu:"
    )

    print(
        output.head(5).to_string(
            index=False
        )
    )
    return output


def main():
    parser = argparse.ArgumentParser(description="Prepare an exact-sized fashion catalog.")
    parser.add_argument("--csv", type=Path, default=CSV_FILE)
    parser.add_argument("--image-dir", type=Path, default=IMAGE_DIR)
    parser.add_argument("--output", type=Path, default=PROCESSED_CSV)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--num-products", type=int, default=MAX_PRODUCTS)
    selection.add_argument("--all", action="store_true", help="Select all eligible products")
    parser.add_argument("--categories", nargs="+", help="Exact category names; quote names containing spaces")
    parser.add_argument(
        "--min-per-category",
        type=int,
        default=MIN_PRODUCTS_PER_CATEGORY,
        help="Target minimum per category when the requested catalog size permits",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()
    if not args.all and args.num_products is not None and args.num_products < 1:
        parser.error("num-products must be positive")
    if args.min_per_category < 1:
        parser.error("min-per-category must be positive")
    prepare_catalog(args.csv, args.image_dir, args.output,
                    None if args.all else args.num_products, args.seed, args.categories,
                    args.min_per_category)


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    main()
