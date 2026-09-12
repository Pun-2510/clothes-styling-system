from pathlib import Path

from PIL import Image

from config import (
    TOP_K,
    MAX_IMAGES,
    FILTER_BY_CATEGORY,
    RESULTS_DIR
)

from image_search import (
    DEVICE,
    load_resnet_model,
    load_fashion_dataset,
    get_transform,
    create_image_embeddings,
    search_by_image
)


def clear_results():

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    for file in RESULTS_DIR.glob("*.jpg"):
        file.unlink()


def save_results(results):

    clear_results()

    for result in results:

        image = result["image"]

        if image is None:
            continue

        image = image.convert("RGB")

        filename = (
            RESULTS_DIR /
            f"{result['rank']:02d}.jpg"
        )

        image.save(filename)

        print(
            f"Saved: {filename}"
        )


def main():

    print("=" * 60)
    print("RESNET IMAGE SEARCH")
    print("=" * 60)

    print(
        f"Device: {DEVICE}"
    )

    # -----------------------------------------------------
    # Input
    # -----------------------------------------------------

    query_path = input(
        "\nNhập đường dẫn ảnh cần tìm: "
    ).strip()

    query_file = Path(query_path)

    if not query_file.exists():

        print(
            "Không tìm thấy file ảnh!"
        )

        return

    try:

        query_image = Image.open(
            query_file
        ).convert("RGB")

    except Exception as e:

        print(
            f"Không thể đọc ảnh: {e}"
        )

        return

    # -----------------------------------------------------
    # Load model
    # -----------------------------------------------------

    print(
        "\nLoading trained ResNet18..."
    )

    model, resnet_config = load_resnet_model(
        model_type="trained"
    )

    # -----------------------------------------------------
    # Transform
    # -----------------------------------------------------

    transform = get_transform(
        resnet_config
    )

    # -----------------------------------------------------
    # Dataset
    # -----------------------------------------------------

    products = load_fashion_dataset(
        max_images=MAX_IMAGES
    )

    # -----------------------------------------------------
    # Catalog embedding
    # -----------------------------------------------------

    image_embeddings = create_image_embeddings(
        model,
        products,
        transform
    )

    # -----------------------------------------------------
    # Search
    # -----------------------------------------------------

    search_result = search_by_image(
        query_image=query_image,
        model=model,
        products=products,
        image_embeddings=image_embeddings,
        transform=transform,
        classes=resnet_config.SELECTED_CLASSES,
        top_k=TOP_K,
        filter_category=FILTER_BY_CATEGORY
    )

    # -----------------------------------------------------
    # Output
    # -----------------------------------------------------

    print("\n")
    print("=" * 60)
    print("SEARCH RESULTS")
    print("=" * 60)

    print(
        f"Category: "
        f"{search_result['query_category']}"
    )

    print(
        f"Confidence: "
        f"{search_result['confidence']:.4f}"
    )

    print("\nTop products:\n")

    for result in search_result["results"]:

        print(
            f"{result['rank']}. "
            f"{result['product']}"
        )

        print(
            f"   Category: "
            f"{result['category']}"
        )

        print(
            f"   Brand: "
            f"{result['brand']}"
        )

        print(
            f"   Similarity: "
            f"{result['similarity']:.4f}"
        )

        print()

    # -----------------------------------------------------
    # Save images
    # -----------------------------------------------------

    save_results(
        search_result["results"]
    )

    print(
        f"Images saved to: {RESULTS_DIR}"
    )


if __name__ == "__main__":
    main()