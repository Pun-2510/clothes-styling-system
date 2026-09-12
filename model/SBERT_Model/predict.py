from model import search_products

def main():

    print("=" * 60)
    print("FASHION PRODUCT SIMILARITY SEARCH")
    print("=" * 60)

    print()
    print("Ví dụ:")
    print("  giày adidas")
    print("  giày thể thao nike")
    print("  áo thun")
    print("  quần jean levi's")
    print("  giày cao gót")
    print()
    print("Nhập 'exit' để thoát.")
    print()

    while True:

        query = input(
            "Nhập sản phẩm: "
        ).strip()

        if query.lower() == "exit":
            break

        if not query:
            continue

        try:

            result = search_products(
                query,
                top_k=10
            )

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
                "Brand source:",
                result["brand_source"]
            )

            print()
            print("SIMILAR PRODUCTS")
            print("-" * 60)

            if not result["results"]:

                print(
                    "Không tìm thấy sản phẩm."
                )

                continue

            for i, item in enumerate(
                result["results"],
                start=1
            ):

                print(
                    f"{i}. {item['product']}"
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

                print()

        except Exception as e:

            print()
            print(
                "ERROR:",
                str(e)
            )
            print()

if __name__ == "__main__":
    main()