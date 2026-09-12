import sys

from config import (
    BERT_DIR,
    TOP_K
)


# ============================================================
# ADD BERT MODEL PATH
# ============================================================

if str(BERT_DIR) not in sys.path:

    sys.path.insert(
        0,
        str(BERT_DIR)
    )


# ============================================================
# IMPORT SBERT SEARCH
# ============================================================

import sys
import importlib.util
from pathlib import Path


# ============================================================
# LOAD SBERT MODEL
# ============================================================

CURRENT_DIR = Path(__file__).resolve().parent

SBERT_DIR = CURRENT_DIR.parent / "SBERT_Model"

SBERT_CONFIG_PATH = SBERT_DIR / "config.py"
SBERT_MODEL_PATH = SBERT_DIR / "model.py"


def load_sbert_model():

    # Load SBERT config
    config_spec = importlib.util.spec_from_file_location(
        "sbert_config",
        SBERT_CONFIG_PATH
    )

    sbert_config = importlib.util.module_from_spec(
        config_spec
    )

    config_spec.loader.exec_module(
        sbert_config
    )

    # Temporarily make SBERT config available
    # as module name "config"
    old_config = sys.modules.get("config")

    sys.modules["config"] = sbert_config

    try:

        model_spec = importlib.util.spec_from_file_location(
            "sbert_model",
            SBERT_MODEL_PATH
        )

        sbert_model = importlib.util.module_from_spec(
            model_spec
        )

        model_spec.loader.exec_module(
            sbert_model
        )

    finally:

        if old_config is not None:
            sys.modules["config"] = old_config

        else:
            sys.modules.pop(
                "config",
                None
            )

    return sbert_model


sbert_model = load_sbert_model()

search_products = sbert_model.search_products

# ============================================================
# IMPORT RESNET
# ============================================================

from image_search import (

    load_resnet_model,

    create_image_embeddings,

    search_by_image

)


# ============================================================
# LOAD RESNET DATA
# ============================================================

print(
    "Initializing ResNet..."
)

RESNET_MODEL, RESNET_CLASSES = (
    load_resnet_model()
)


print(
    "Creating image database..."
)

IMAGE_EMBEDDINGS, IMAGE_PRODUCTS = (
    create_image_embeddings()
)


# ============================================================
# TEXT SEARCH
# ============================================================

def text_search(

    query,

    top_k=TOP_K
):

    result = search_products(

        query,

        top_k=top_k

    )

    return result


# ============================================================
# IMAGE SEARCH
# ============================================================

def image_search(

    image,

    top_k=TOP_K
):

    results = search_by_image(

        image=image,

        image_embeddings=IMAGE_EMBEDDINGS,

        products=IMAGE_PRODUCTS,

        model=RESNET_MODEL,

        top_k=top_k

    )

    return results


# ============================================================
# MULTIMODAL SEARCH
# ============================================================

def multimodal_search(

    text=None,

    image=None,

    top_k=TOP_K
):

    result = {

        "text_results": None,

        "image_results": None

    }


    # --------------------------------------------------------

    # TEXT SEARCH

    # --------------------------------------------------------

    if text:

        print(
            "Searching by text..."
        )

        result[
            "text_results"
        ] = text_search(

            query=text,

            top_k=top_k

        )


    # --------------------------------------------------------

    # IMAGE SEARCH

    # --------------------------------------------------------

    if image is not None:

        print(
            "Searching by image..."
        )

        result[
            "image_results"
        ] = image_search(

            image=image,

            top_k=top_k

        )


    # --------------------------------------------------------

    # NO INPUT

    # --------------------------------------------------------

    if (

        not text

        and

        image is None

    ):

        raise ValueError(

            "Phải nhập text hoặc image."

        )


    return result