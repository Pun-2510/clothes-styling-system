"""Read-only catalog browsing endpoints independent of model inference."""

import csv
from collections import defaultdict
from functools import lru_cache

from fastapi import APIRouter, HTTPException, Query

from src.audience import AUDIENCE_ORDER, infer_audience
from src.config import PROCESSED_CSV


router = APIRouter(prefix="/catalog", tags=["Catalog"])


def _serialize_product(row: dict[str, str]) -> dict[str, object]:
    """Return public catalog fields without exposing local filesystem paths."""
    product = {
        "product_id": row.get("product_id", "").strip(),
        "product_name": row.get("product_name", "").strip(),
        "description": row.get("description", "").strip(),
        "category": row.get("category", "").strip(),
        "image_reference": row.get("image_reference", "").strip(),
    }
    inferred = infer_audience(
        product["product_name"],
        product["description"],
        product["category"],
    )
    product.update({
        "audience": inferred.audience,
        "age_group": inferred.age_group,
        "gender_group": inferred.gender_group,
        "audience_source": inferred.source,
        "audience_confidence": inferred.confidence,
        "audience_evidence": list(inferred.evidence),
    })
    return product


@lru_cache(maxsize=2)
def _load_catalog(modified_at: int):
    del modified_at
    with PROCESSED_CSV.open(encoding="utf-8-sig", newline="") as handle:
        products = [_serialize_product(row) for row in csv.DictReader(handle)]

    by_category = defaultdict(list)
    by_audience = defaultdict(list)
    by_id = {}
    for product in products:
        by_category[product["category"]].append(product)
        by_audience[product["audience"]].append(product)
        by_id[product["product_id"]] = product
    return products, by_category, by_audience, by_id


def _catalog():
    try:
        return _load_catalog(PROCESSED_CSV.stat().st_mtime_ns)
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=503,
            detail="Processed catalog is not available.",
        ) from error


@router.get("/categories")
def list_categories():
    _, by_category, _, _ = _catalog()
    categories = [
        {
            "name": name,
            "count": len(items),
            "image_reference": next(
                (item["image_reference"] for item in items if item["image_reference"]),
                "",
            ),
        }
        for name, items in by_category.items()
        if name
    ]
    categories.sort(key=lambda item: (-item["count"], item["name"].casefold()))
    return {"categories": categories}


@router.get("/audiences")
def list_audiences():
    """List metadata-derived catalog audiences and representative images."""
    _, _, by_audience, _ = _catalog()
    audiences = []
    for name in AUDIENCE_ORDER:
        items = by_audience.get(name, [])
        audiences.append({
            "name": name,
            "count": len(items),
            "image_reference": next(
                (item["image_reference"] for item in items if item["image_reference"]),
                "",
            ),
        })
    return {"audiences": audiences}


@router.get("/products")
def list_products(
    category: str | None = None,
    audience: str | None = None,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=24, ge=1, le=48),
):
    products, by_category, by_audience, _ = _catalog()
    if category is not None:
        if category not in by_category:
            raise HTTPException(status_code=404, detail="Category not found.")
        products = by_category[category]
    if audience is not None:
        audience = audience.casefold().strip()
        if audience not in AUDIENCE_ORDER:
            raise HTTPException(status_code=404, detail="Audience not found.")
        if category is None:
            products = by_audience[audience]
        else:
            products = [item for item in products if item["audience"] == audience]

    total = len(products)
    start = (page - 1) * page_size
    return {
        "items": products[start:start + page_size],
        "category": category,
        "audience": audience,
        "page": page,
        "page_size": page_size,
        "total": total,
        "total_pages": (total + page_size - 1) // page_size,
    }


@router.get("/products/{product_id}")
def get_product(product_id: str):
    _, by_category, _, by_id = _catalog()
    product = by_id.get(product_id)
    if product is None:
        raise HTTPException(status_code=404, detail="Product not found.")

    related = [
        item for item in by_category[product["category"]]
        if item["product_id"] != product_id
    ][:5]
    return {"product": product, "related": related}
