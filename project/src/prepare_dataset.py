from pathlib import Path
import sys
import re

import pandas as pd


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

from src.config import (
    IMAGE_DIR,
    CSV_FILE,
    PROCESSED_DIR,
    PROCESSED_CSV,
    MAX_PRODUCTS,
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
    image_reference
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
        IMAGE_DIR / filename
    )

    if candidate.exists():

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

    for extension in IMAGE_EXTENSIONS:

        candidate = (
            IMAGE_DIR /
            f"{stem}{extension}"
        )

        if candidate.exists():

            return candidate


    # -----------------------------------------------------
    # Không tìm thấy
    # -----------------------------------------------------

    return None


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 60)
    print("PREPARE MINI FASHION DATASET")
    print("=" * 60)


    # =====================================================
    # CHECK FILES
    # =====================================================

    if not CSV_FILE.exists():

        raise FileNotFoundError(
            f"\nKhông tìm thấy CSV:\n"
            f"{CSV_FILE}"
        )


    if not IMAGE_DIR.exists():

        raise FileNotFoundError(
            f"\nKhông tìm thấy thư mục ảnh:\n"
            f"{IMAGE_DIR}"
        )


    # =====================================================
    # LOAD CSV
    # =====================================================

    print(
        f"\nĐang đọc:\n"
        f"{CSV_FILE}"
    )

    df = pd.read_csv(CSV_FILE, on_bad_lines="skip", engine="python")

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

    output["image_path"] = (
        output[
            "image_reference"
        ]
        .apply(find_image)
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

    if (
        MAX_PRODUCTS is not None
        and len(output)
        > MAX_PRODUCTS
    ):

        def sample_group(group):

            # Lấy category từ group.name — luôn tồn tại
            # bất kể pandas có giữ cột category trong group hay không
            category = group.name

            # Tỉ lệ của category này trong toàn bộ dataset
            fraction = len(group) / len(output)

            # Số lượng cần lấy, tối thiểu 1 để không bị mất hẳn category
            n_samples = max(
                1,
                round(fraction * MAX_PRODUCTS)
            )

            n_samples = min(
                n_samples,
                len(group)
            )

            sampled = group.sample(
                n=n_samples,
                random_state=RANDOM_SEED
            )

            sampled["category"] = category

            return sampled

        output = (
            output
            .groupby("category", group_keys=False)
            .apply(sample_group)
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

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    # =====================================================
    # SAVE
    # =====================================================

    output.to_csv(
        PROCESSED_CSV,
        index=False,
        encoding="utf-8-sig"
    )


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
        f"{PROCESSED_CSV}"
    )

    print(
        "\n5 sản phẩm đầu:"
    )

    print(
        output.head(5).to_string(
            index=False
        )
    )


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    main()