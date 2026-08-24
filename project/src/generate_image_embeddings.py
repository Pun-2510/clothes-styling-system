from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

from PIL import Image
from tqdm import tqdm

from transformers import (
    AutoProcessor,
    CLIPModel
)


# =========================================================
# IMPORT CONFIG
# =========================================================

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

from src.config import (
    CLIP_MODEL,
    PROCESSED_CSV,
    EMBEDDINGS_FILE,
    EMBEDDING_DIR,
    EMBEDDING_BATCH_SIZE,
)


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 60)
    print("GENERATE CLIP IMAGE EMBEDDINGS")
    print("=" * 60)

    # -----------------------------------------------------
    # Device
    # -----------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    # -----------------------------------------------------
    # Load CLIP
    # -----------------------------------------------------

    print(
        "\nĐang load CLIP..."
    )

    processor = (
        AutoProcessor.from_pretrained(
            CLIP_MODEL
        )
    )

    model = (
        CLIPModel.from_pretrained(
            CLIP_MODEL
        )
        .to(device)
    )

    model.eval()

    # -----------------------------------------------------
    # Load processed dataset
    # -----------------------------------------------------

    df = pd.read_csv(
        PROCESSED_CSV
    )

    if df.empty:

        raise RuntimeError(
            "products.csv đang rỗng."
        )

    print(
        f"\nSố sản phẩm: "
        f"{len(df):,}"
    )

    # -----------------------------------------------------
    # Generate embeddings
    # -----------------------------------------------------

    all_embeddings = []

    for start in tqdm(
        range(
            0,
            len(df),
            EMBEDDING_BATCH_SIZE
        ),
        desc="Generating embeddings"
    ):

        batch_df = df.iloc[
            start:
            start + EMBEDDING_BATCH_SIZE
        ]

        images = []

        valid_positions = []

        # -------------------------------------------------
        # Load images
        # -------------------------------------------------

        for local_index, image_path in enumerate(
            batch_df["image_path"]
        ):

            try:

                image = (
                    Image.open(
                        image_path
                    )
                    .convert("RGB")
                )

                images.append(
                    image
                )

                valid_positions.append(
                    local_index
                )

            except Exception as error:

                print(
                    f"\nKhông thể đọc: "
                    f"{image_path}"
                )

                print(
                    f"Lỗi: {error}"
                )

        # -------------------------------------------------
        # Prepare batch result
        # -------------------------------------------------

        batch_embeddings = np.zeros(
            (
                len(batch_df),
                512
            ),
            dtype=np.float32
        )

        # -------------------------------------------------
        # CLIP
        # -------------------------------------------------

        if images:

            inputs = processor(
                images=images,
                return_tensors="pt",
                padding=True,
            )

            inputs = {
                key: value.to(device)
                for key, value
                in inputs.items()
            }

            with torch.inference_mode():

                # CLIP vision encoder
                vision_outputs = model.vision_model(
                    pixel_values=inputs["pixel_values"],
                    return_dict=True
                )

                # Lấy pooled image representation
                pooled_output = vision_outputs.pooler_output

                # Chiếu sang CLIP embedding space
                features = model.visual_projection(
                    pooled_output
                )


            # -------------------------------------------------
            # Normalize
            # -------------------------------------------------

            features = (
                features
                /
                features.norm(
                    dim=-1,
                    keepdim=True
                )
            )

            features = (
                features
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            # -------------------------------------------------
            # Put embedding back into original positions
            # -------------------------------------------------

            for (
                feature_index,
                local_index
            ) in enumerate(
                valid_positions
            ):

                batch_embeddings[
                    local_index
                ] = features[
                    feature_index
                ]

        # -----------------------------------------------------
        # Add batch
        # -----------------------------------------------------

        all_embeddings.append(
            batch_embeddings
        )

    # ---------------------------------------------------------
    # Combine batches
    # ---------------------------------------------------------

    embeddings = np.concatenate(
        all_embeddings,
        axis=0
    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    EMBEDDING_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    np.save(
        EMBEDDINGS_FILE,
        embeddings
    )

    # ---------------------------------------------------------
    # Result
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("EMBEDDING COMPLETE")
    print("=" * 60)

    print(
        f"Embedding shape: "
        f"{embeddings.shape}"
    )

    print(
        f"Saved to:\n"
        f"{EMBEDDINGS_FILE}"
    )


if __name__ == "__main__":
    main()