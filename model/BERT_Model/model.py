# ============================================================
# SBERT FASHION PRODUCT SIMILARITY SEARCH
# ============================================================
#
# Pipeline:
#
# User Query
#      |
#      v
# Normalize Vietnamese
#      |
#      +----> Hard Category
#      |
#      +----> Soft Category
#      |
#      +----> Hard Brand
#      |
#      +----> Soft Brand
#      |
#      v
# Build Search Query
#      |
#      v
# SBERT Embedding
#      |
#      v
# Filter Dataset
#      |
#      +---- Category
#      |
#      +---- Brand (if specified)
#      |
#      v
# Cosine Similarity
#      |
#      v
# Remove Duplicate Products
#      |
#      v
# TOP K Similar Products
#
# ============================================================

import os
import re
import json

import numpy as np
import pandas as pd
import torch

from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz, process


# ============================================================
# CONFIG
# ============================================================

from config import (
    SBERT_MODEL,
    DATASET_NAME,
    TOP_K,
    EMBEDDING_PATH,
    MIN_SIMILARITY,
    MAX_SAMPLES,
    EMBEDDING_DTYPE
)


# ============================================================
# DEVICE
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# CATEGORY
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CATEGORY_PATH = os.path.join(
    BASE_DIR,
    "categories.json"
)

with open(
    CATEGORY_PATH,
    "r",
    encoding="utf-8"
) as f:
    CATEGORIES = json.load(f)

VALID_CATEGORIES = set(
    CATEGORIES.values()
)

CATEGORY_ALIASES = {

    # --------------------------------------------------------
    # T-Shirt
    # --------------------------------------------------------

    "áo thun": "Tshirts",
    "ao thun": "Tshirts",

    "áo phông": "Tshirts",
    "ao phong": "Tshirts",

    "áo tshirt": "Tshirts",
    "ao tshirt": "Tshirts",

    "tshirt": "Tshirts",
    "t-shirt": "Tshirts",

    # --------------------------------------------------------
    # Shirt
    # --------------------------------------------------------

    "áo sơ mi": "Shirts",
    "ao so mi": "Shirts",

    "sơ mi": "Shirts",
    "so mi": "Shirts",

    "shirt": "Shirts",
    "shirts": "Shirts",

    # --------------------------------------------------------
    # Jeans
    # --------------------------------------------------------

    "quần jean": "Jeans",
    "quan jean": "Jeans",

    "quần jeans": "Jeans",
    "quan jeans": "Jeans",

    "jeans": "Jeans",
    "jean": "Jeans",

    # --------------------------------------------------------
    # Pants
    # --------------------------------------------------------

    "quần dài": "Trousers",
    "quan dai": "Trousers",

    "quần": "Trousers",
    "quan": "Trousers",

    "pants": "Trousers",
    "trousers": "Trousers",

    # --------------------------------------------------------
    # Shorts
    # --------------------------------------------------------

    "quần short": "Shorts",
    "quan short": "Shorts",

    "quần shorts": "Shorts",
    "quan shorts": "Shorts",

    "short": "Shorts",
    "shorts": "Shorts",

    # --------------------------------------------------------
    # Dress
    # --------------------------------------------------------

    "váy": "Dresses",
    "vay": "Dresses",

    "đầm": "Dresses",
    "dam": "Dresses",

    "dress": "Dresses",
    "dresses": "Dresses",

    # --------------------------------------------------------
    # Skirt
    # --------------------------------------------------------

    "chân váy": "Skirts",
    "chan vay": "Skirts",

    "skirt": "Skirts",
    "skirts": "Skirts",

    # --------------------------------------------------------
    # Jacket
    # --------------------------------------------------------

    "áo khoác": "Jackets",
    "ao khoac": "Jackets",

    "jacket": "Jackets",
    "jackets": "Jackets",

    # --------------------------------------------------------
    # Hoodie
    # --------------------------------------------------------

    "áo hoodie": "Sweatshirts",
    "ao hoodie": "Sweatshirts",

    "hoodie": "Sweatshirts",

    # --------------------------------------------------------
    # Sports Shoes
    # --------------------------------------------------------

    "giày thể thao": "Sports Shoes",
    "giay the thao": "Sports Shoes",

    "giày sneaker": "Sports Shoes",
    "giay sneaker": "Sports Shoes",

    "sneaker": "Sports Shoes",
    "sneakers": "Sports Shoes",

    # --------------------------------------------------------
    # Heels
    # --------------------------------------------------------

    "giày cao gót": "Heels",
    "giay cao got": "Heels",

    "cao gót": "Heels",
    "cao got": "Heels",

    "giày cao got": "Heels",

    "heels": "Heels",
    "high heel": "Heels",
    "high heels": "Heels",

    # --------------------------------------------------------
    # Boots
    # --------------------------------------------------------

    "giày boot": "Boots",
    "giay boot": "Boots",

    "boot": "Boots",
    "boots": "Boots",

    # --------------------------------------------------------
    # Sandals
    # --------------------------------------------------------

    "dép": "Sandals",
    "dep": "Sandals",

    "sandal": "Sandals",
    "sandals": "Sandals",

    # --------------------------------------------------------
    # Shoes
    # --------------------------------------------------------

    "giày": "Shoes",
    "giay": "Shoes",

    "shoes": "Shoes",
    "shoe": "Shoes",
}


# ============================================================
# BRAND ALIASES
# ============================================================

BRAND_ALIASES = {

    # Adidas
    "adidas": "Adidas",
    "adias": "Adidas",
    "addidas": "Adidas",
    "adidass": "Adidas",
    "adid": "Adidas",

    # Nike
    "nike": "Nike",
    "nik": "Nike",
    "ni ke": "Nike",

    # Puma
    "puma": "Puma",

    # Reebok
    "reebok": "Reebok",
    "reebok": "Reebok",

    # Fila
    "fila": "Fila",

    # New Balance
    "new balance": "New Balance",
    "newbalance": "New Balance",

    # Converse
    "converse": "Converse",

    # Vans
    "vans": "Vans",

    # Zara
    "zara": "Zara",

    # H&M
    "h&m": "H&M",
    "hm": "H&M",

    # Uniqlo
    "uniqlo": "Uniqlo",

    # Levi's
    "levis": "Levi's",
    "levi": "Levi's",
    "levi's": "Levi's",

    # Luxury
    "gucci": "Gucci",
    "prada": "Prada",
    "chanel": "Chanel",
    "dior": "Dior",

    "louis vuitton": "Louis Vuitton",

    "balenciaga": "Balenciaga",
    "versace": "Versace",
    "burberry": "Burberry",

    # Other
    "under armour": "Under Armour",
    "underarmor": "Under Armour",

    "asics": "ASICS",

    "skechers": "Skechers",

    "timberland": "Timberland",

    "lacoste": "Lacoste",

    "ralph lauren": "Ralph Lauren",

    "tommy hilfiger": "Tommy Hilfiger",

    "calvin klein": "Calvin Klein",

    "gap": "GAP",

    "forever 21": "Forever 21",

    "mango": "Mango",

    "pull&bear": "Pull&Bear",

    "bershka": "Bershka",

    "stradivarius": "Stradivarius",

    "supreme": "Supreme",
}


# ============================================================
# NORMALIZE
# ============================================================

def normalize_text(text):

    if text is None:
        return ""

    text = str(text).strip().lower()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 60)
print("LOADING PRODUCT DATASET")
print("=" * 60)

print("Dataset:", DATASET_NAME)

# ------------------------------------------------------------
# Nếu bạn đã export CSV
# ------------------------------------------------------------

if os.path.isfile(DATASET_NAME):

    df = pd.read_csv(
        DATASET_NAME
    )

else:

    # --------------------------------------------------------
    # HuggingFace dataset
    # --------------------------------------------------------

    from datasets import load_dataset

    dataset = load_dataset(
        DATASET_NAME
    )

    df = dataset["train"].to_pandas()


print(
    "Products:",
    len(df)
)

print()


# ============================================================
# CHECK COLUMNS
# ============================================================

required_columns = [
    "productDisplayName",
    "articleType"
]

for column in required_columns:

    if column not in df.columns:

        raise ValueError(
            f"Dataset thiếu column: {column}"
        )


# ============================================================
# BRAND COLUMN
# ============================================================

if "brandName" not in df.columns:

    if "brand" in df.columns:

        df["brandName"] = df["brand"]

    else:

        df["brandName"] = None


# ============================================================
# CLEAN DATA
# ============================================================

df["productDisplayName"] = (
    df["productDisplayName"]
    .fillna("")
    .astype(str)
    .str.strip()
)

df["articleType"] = (
    df["articleType"]
    .fillna("")
    .astype(str)
    .str.strip()
)

df["brandName"] = (
    df["brandName"]
    .fillna("")
    .astype(str)
    .str.strip()
)


# ============================================================
# REMOVE EMPTY PRODUCTS
# ============================================================

df = df[
    df["productDisplayName"] != ""
].copy()


# ============================================================
# REMOVE DUPLICATE PRODUCTS
# ============================================================

before = len(df)

df = df.drop_duplicates(
    subset=[
        "productDisplayName"
    ],
    keep="first"
).reset_index(
    drop=True
)

after = len(df)

print(
    "Removed duplicates:",
    before - after
)

print(
    "Unique products:",
    after
)

print()

# ============================================================
# LIMIT SAMPLES
# ============================================================

if MAX_SAMPLES is not None:

    df = df.head(MAX_SAMPLES).copy()

print(
    "Products used for embedding:",
    len(df)
)

print()

# ============================================================
# NORMALIZED COLUMNS
# ============================================================

df["_name_normalized"] = (
    df["productDisplayName"]
    .apply(normalize_text)
)

df["_category_normalized"] = (
    df["articleType"]
    .apply(normalize_text)
)

df["_brand_normalized"] = (
    df["brandName"]
    .apply(normalize_text)
)


# ============================================================
# HARD CATEGORY
# ============================================================

def hard_match_category(text):

    text = normalize_text(text)

    rules = sorted(
        CATEGORY_ALIASES.items(),
        key=lambda x: len(x[0]),
        reverse=True
    )

    for keyword, category in rules:

        # Chỉ sử dụng category tồn tại trong categories.json
        if category not in VALID_CATEGORIES:
            continue

        pattern = (
            r"(?<!\w)"
            + re.escape(keyword)
            + r"(?!\w)"
        )

        if re.search(
            pattern,
            text
        ):

            return {
                "category": category,
                "matched": keyword,
                "score": 100,
                "source": "hard"
            }

    return None


# ============================================================
# SOFT CATEGORY
# ============================================================

def soft_match_category(
    text,
    threshold=82
):

    text = normalize_text(text)

    if not text:
        return None

    valid_aliases = [
        keyword
        for keyword, category
        in CATEGORY_ALIASES.items()
        if category in VALID_CATEGORIES
    ]

    result = process.extractOne(
        text,
        valid_aliases,
        scorer=fuzz.partial_ratio
    )

    if result is None:
        return None

    matched, score, _ = result

    if score < threshold:
        return None

    category = CATEGORY_ALIASES[
        matched
    ]

    return {
        "category": category,
        "matched": matched,
        "score": score,
        "source": "soft"
    }


# ============================================================
# HARD BRAND
# ============================================================

def hard_match_brand(text):

    text = normalize_text(text)

    aliases = sorted(
        BRAND_ALIASES.keys(),
        key=len,
        reverse=True
    )

    for alias in aliases:

        pattern = (
            r"(?<!\w)"
            + re.escape(alias)
            + r"(?!\w)"
        )

        if re.search(
            pattern,
            text
        ):

            return {
                "brand":
                    BRAND_ALIASES[alias],

                "matched":
                    alias,

                "score":
                    100,

                "source":
                    "hard"
            }

    return None


# ============================================================
# SOFT BRAND
# ============================================================

def soft_match_brand(
    text,
    threshold=78
):

    text = normalize_text(text)

    words = text.split()

    canonical_brands = list(
        set(
            BRAND_ALIASES.values()
        )
    )

    best_brand = None
    best_score = 0

    for word in words:

        result = process.extractOne(
            word,
            canonical_brands,
            scorer=fuzz.ratio
        )

        if result is None:
            continue

        brand, score, _ = result

        if score > best_score:

            best_brand = brand
            best_score = score

    if (
        best_brand is not None
        and best_score >= threshold
    ):

        return {
            "brand":
                best_brand,

            "score":
                best_score,

            "source":
                "soft"
        }

    return None


# ============================================================
# DETECT CATEGORY
# ============================================================

def detect_category(text):

    # ========================================================
    # 1. HARD
    # ========================================================

    result = hard_match_category(
        text
    )

    if result is not None:

        return result


    # ========================================================
    # 2. SOFT
    # ========================================================

    result = soft_match_category(
        text
    )

    if result is not None:

        return result


    # ========================================================
    # 3. NONE
    # ========================================================

    return {
        "category": None,
        "matched": None,
        "score": 0,
        "source": None
    }


# ============================================================
# DETECT BRAND
# ============================================================

def detect_brand(text):

    # ========================================================
    # 1. HARD
    # ========================================================

    result = hard_match_brand(
        text
    )

    if result is not None:

        return result


    # ========================================================
    # 2. SOFT
    # ========================================================

    result = soft_match_brand(
        text
    )

    if result is not None:

        return result


    # ========================================================
    # 3. NONE
    # ========================================================

    return {
        "brand": None,
        "matched": None,
        "score": 0,
        "source": None
    }


# ============================================================
# LOAD SBERT
# ============================================================

print("=" * 60)
print("LOADING SBERT")
print("=" * 60)

print(
    "Model:",
    SBERT_MODEL
)

sbert = SentenceTransformer(
    SBERT_MODEL,
    device=device
)

print(
    "SBERT loaded."
)

print()


# ============================================================
# BUILD PRODUCT TEXT
# ============================================================

def build_product_text(row):

    name = str(
        row["productDisplayName"]
    ).strip()

    category = str(
        row["articleType"]
    ).strip()

    brand = str(
        row["brandName"]
    ).strip()

    parts = [
        name
    ]

    if category:
        parts.append(
            category
        )

    if brand:
        parts.append(
            brand
        )

    return " ".join(
        parts
    )


print("=" * 60)
print("BUILDING PRODUCT TEXT")
print("=" * 60)

product_texts = [
    build_product_text(row)
    for _, row in df.iterrows()
]

print(
    "Texts:",
    len(product_texts)
)

print()


# ============================================================
# LOAD / CREATE EMBEDDINGS
# ============================================================

if (
    EMBEDDING_PATH
    and os.path.exists(EMBEDDING_PATH)
):

    print("=" * 60)
    print("LOADING EMBEDDINGS")
    print("=" * 60)

    embeddings = np.load(
        EMBEDDING_PATH
    )

    print(
        "Embedding shape:",
        embeddings.shape
    )

else:

    print("=" * 60)
    print("CREATING EMBEDDINGS")
    print("=" * 60)

    embeddings = sbert.encode(
        product_texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    embeddings = embeddings.astype(
        EMBEDDING_DTYPE
    )

    if EMBEDDING_PATH:

        os.makedirs(
            os.path.dirname(
                EMBEDDING_PATH
            ),
            exist_ok=True
        )

        np.save(
            EMBEDDING_PATH,
            embeddings
        )

        print(
            "Saved embeddings:",
            EMBEDDING_PATH
        )

print()


# ============================================================
# ENSURE NORMALIZED
# ============================================================

norms = np.linalg.norm(
    embeddings,
    axis=1,
    keepdims=True
)

norms[
    norms == 0
] = 1

embeddings = (
    embeddings
    / norms
)


# ============================================================
# QUERY TEXT
# ============================================================

def build_query_text(
    text,
    category=None,
    brand=None
):

    query = normalize_text(
        text
    )

    parts = [
        query
    ]

    if category:

        parts.append(
            category
        )

    if brand:

        parts.append(
            brand
        )

    return " ".join(
        parts
    )


# ============================================================
# SIMILARITY SEARCH
# ============================================================

def search_products(
    query,
    top_k=TOP_K
):

    query = str(
        query
    ).strip()

    if not query:

        return {
            "query": "",
            "category": None,
            "category_confidence": 0,
            "category_source": None,
            "brand": None,
            "brand_confidence": 0,
            "brand_source": None,
            "results": []
        }


    # ========================================================
    # CATEGORY
    # ========================================================

    category_result = detect_category(
        query
    )

    category = category_result[
        "category"
    ]

    category_score = category_result[
        "score"
    ]

    category_source = category_result[
        "source"
    ]


    # ========================================================
    # BRAND
    # ========================================================

    brand_result = detect_brand(
        query
    )

    brand = brand_result[
        "brand"
    ]

    brand_score = brand_result[
        "score"
    ]

    brand_source = brand_result[
        "source"
    ]


    # ========================================================
    # BUILD QUERY
    # ========================================================

    query_text = build_query_text(
        query,
        category,
        brand
    )


    # ========================================================
    # QUERY EMBEDDING
    # ========================================================

    query_embedding = sbert.encode(
        query_text,
        convert_to_numpy=True,
        normalize_embeddings=True
    )


    # ========================================================
    # COSINE SIMILARITY
    # ========================================================

    similarities = (
        embeddings
        @ query_embedding
    )


    # ========================================================
    # CREATE SEARCH DATAFRAME
    # ========================================================

    search_df = df.copy()

    search_df[
        "_similarity"
    ] = similarities


    # ========================================================
    # CATEGORY FILTER
    # ========================================================

    if category is not None:

        category_mask = (
            search_df[
                "_category_normalized"
            ]
            ==
            normalize_text(
                category
            )
        )

        category_df = (
            search_df[
                category_mask
            ]
            .copy()
        )

    else:

        category_df = (
            search_df.copy()
        )


    # ========================================================
    # BRAND FILTER
    # ========================================================

    if brand is not None:

        brand_mask = (
            category_df[
                "_brand_normalized"
            ]
            ==
            normalize_text(
                brand
            )
        )

        brand_df = (
            category_df[
                brand_mask
            ]
            .copy()
        )

        if len(brand_df) > 0:

            filtered_df = brand_df

        else:

            filtered_df = category_df

    else:

        filtered_df = category_df


    # ========================================================
    # FALLBACK
    # ========================================================

    if len(filtered_df) == 0:

        filtered_df = search_df


    # ========================================================
    # MIN SIMILARITY
    # ========================================================

    if (
        MIN_SIMILARITY is not None
        and len(filtered_df) > 0
    ):

        similarity_df = (
            filtered_df[
                filtered_df[
                    "_similarity"
                ]
                >= MIN_SIMILARITY
            ]
            .copy()
        )

        # Không để kết quả rỗng chỉ vì threshold
        if len(similarity_df) > 0:

            filtered_df = similarity_df


    # ========================================================
    # SORT
    # ========================================================

    filtered_df = (
        filtered_df
        .sort_values(
            "_similarity",
            ascending=False
        )
    )


    # ========================================================
    # REMOVE DUPLICATE PRODUCT
    # ========================================================

    filtered_df = (
        filtered_df
        .drop_duplicates(
            subset=[
                "productDisplayName"
            ],
            keep="first"
        )
    )


    # ========================================================
    # TOP K
    # ========================================================

    filtered_df = (
        filtered_df
        .head(top_k)
    )


    # ========================================================
    # FORMAT RESULT
    # ========================================================

    results = []

    for _, row in filtered_df.iterrows():

        results.append({

            "product":
                row[
                    "productDisplayName"
                ],

            "category":
                row[
                    "articleType"
                ],

            "brand":
                row[
                    "brandName"
                ]
                if row[
                    "brandName"
                ]
                else None,

            "similarity":
                round(
                    float(
                        row[
                            "_similarity"
                        ]
                    ),
                    4
                )
        })


    # ========================================================
    # RETURN
    # ========================================================

    return {

        "query":
            query,

        "category":
            category,

        "category_confidence":
            round(
                category_score / 100,
                4
            ),

        "category_source":
            category_source,

        "brand":
            brand,

        "brand_confidence":
            round(
                brand_score / 100,
                4
            ),

        "brand_source":
            brand_source,

        "results":
            results
    }


# ============================================================
# PRINT RESULT
# ============================================================

def print_result(
    result
):

    print()

    print("=" * 60)
    print("RESULT")
    print("=" * 60)

    print(
        "Query:",
        result["query"]
    )

    print(
        "Category:",
        result["category"]
    )

    print(
        "Category confidence:",
        result["category_confidence"]
    )

    print(
        "Category source:",
        result["category_source"]
    )

    print(
        "Brand:",
        result["brand"]
    )

    print(
        "Brand confidence:",
        result["brand_confidence"]
    )

    print(
        "Brand source:",
        result["brand_source"]
    )

    print()

    print(
        "SIMILAR PRODUCTS"
    )

    print("-" * 60)

    if not result["results"]:

        print(
            "Không tìm thấy sản phẩm."
        )

        return


    for index, item in enumerate(
        result["results"],
        start=1
    ):

        print()

        print(
            f"{index}. "
            f"{item['product']}"
        )

        print(
            f"   Category: "
            f"{item['category']}"
        )

        print(
            f"   Brand: "
            f"{item['brand']}"
        )

        print(
            f"   Similarity: "
            f"{item['similarity']:.4f}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print(
        "FASHION PRODUCT SIMILARITY SEARCH"
    )
    print("=" * 60)

    print()

    print("Ví dụ:")

    print(
        "  giày adidas"
    )

    print(
        "  giày thể thao nike"
    )

    print(
        "  áo thun"
    )

    print(
        "  quần jean levi's"
    )

    print(
        "  giày cao gót"
    )

    print()

    print(
        "Nhập 'exit' để thoát."
    )

    print()


    while True:

        try:

            query = input(
                "Nhập sản phẩm: "
            ).strip()

        except KeyboardInterrupt:

            print()
            print(
                "Thoát."
            )

            break


        if query.lower() == "exit":

            print(
                "Thoát chương trình."
            )

            break


        if not query:

            print(
                "Vui lòng nhập sản phẩm."
            )

            continue


        try:

            result = search_products(
                query
            )

            print_result(
                result
            )

        except Exception as e:

            print()

            print(
                "SEARCH ERROR:"
            )

            print(
                str(e)
            )

            print()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()
