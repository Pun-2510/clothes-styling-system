import requests
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
from ddgs import DDGS

# ==============================
# CATEGORY -> QUERY
# ==============================

def category_to_query(category):

    mapping = {

        "Tshirts":
            "t shirt clothing product",

        "Shirts":
            "shirt clothing product",

        "Dresses":
            "dress clothing product",

        "Jeans":
            "jeans clothing product",

        "Trousers":
            "trousers pants clothing product",

        "Shorts":
            "shorts clothing product",

        "Skirts":
            "skirt clothing product",

        "Jackets":
            "jacket clothing product",

        "Sweaters":
            "sweater clothing product",

        "Sweatshirts":
            "sweatshirt clothing product",

        "Sarees":
            "saree traditional clothing",

        "Kurtas":
            "kurta traditional clothing",

        "Kurtis":
            "kurti traditional clothing",

        "Sandals":
            "sandals shoes product",

        "Sports Shoes":
            "sports shoes product",

        "Casual Shoes":
            "casual shoes product",

        "Heels":
            "high heels shoes product"
    }

    return mapping.get(
        category,
        category + " fashion product"
    )


# ==============================
# SEARCH IMAGES
# ==============================

def search_images(
    query,
    max_images=6
):

    images = []

    try:

        with DDGS() as ddgs:

            results = ddgs.images(
                query,
                max_results=max_images
            )

            for result in results:

                url = result.get(
                    "image"
                )

                if url:

                    images.append(url)

    except Exception as e:

        print(
            "Lỗi tìm ảnh:",
            e
        )

    return images


# ==============================
# DISPLAY
# ==============================

def show_images(
    image_urls,
    category
):

    if not image_urls:

        print(
            "Không tìm thấy hình ảnh."
        )

        return


    plt.figure(
        figsize=(15, 8)
    )

    count = 0


    for url in image_urls:

        try:

            response = requests.get(
                url,
                timeout=10,
                headers={
                    "User-Agent":
                    "Mozilla/5.0"
                }
            )

            image = Image.open(
                BytesIO(
                    response.content
                )
            ).convert("RGB")


            count += 1

            plt.subplot(
                2,
                3,
                count
            )

            plt.imshow(image)

            plt.axis("off")

            plt.title(category)


        except Exception:

            continue


    plt.tight_layout()

    plt.show()