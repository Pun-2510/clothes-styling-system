from pathlib import Path
import sys
import time
import uuid

import numpy as np
import pandas as pd
import torch

from PIL import Image

from transformers import (
    AutoProcessor,
    CLIPModel
)


# =========================================================
# IMPORT CONFIG
# =========================================================

sys.path.append(
    str(
        Path(__file__)
        .resolve()
        .parent.parent
    )
)

from src.config import (
    CLIP_MODEL,
    PROCESSED_CSV,
    EMBEDDINGS_FILE,
    TOP_K,
    CATEGORY_NEIGHBORS,
    CATEGORY_BONUS,
)

from src.similarity import (
    cosine_similarity,
    top_k_indices,
)

from src.logger import (
    log_info,
    log_warning,
    log_error,
    log_exception,
)


# =========================================================
# RECOMMENDER
# =========================================================

class FashionRecommender:

    def __init__(self):

        log_info(
            "Initializing Fashion Recommender"
        )

        # -------------------------------------------------
        # Device
        # -------------------------------------------------

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        log_info(
            f"Device={self.device}"
        )

        # -------------------------------------------------
        # Load CLIP
        # -------------------------------------------------

        log_info(
            "Loading CLIP model"
        )

        self.processor = (
            AutoProcessor.from_pretrained(
                CLIP_MODEL
            )
        )

        self.model = (
            CLIPModel.from_pretrained(
                CLIP_MODEL
            )
            .to(self.device)
        )

        self.model.eval()

        # -------------------------------------------------
        # Load products
        # -------------------------------------------------

        self.products = pd.read_csv(
            PROCESSED_CSV
        )

        # -------------------------------------------------
        # Load embeddings
        # -------------------------------------------------

        self.embeddings = np.load(
            EMBEDDINGS_FILE
        )

        # -------------------------------------------------
        # Check consistency
        # -------------------------------------------------

        if (
            len(self.products)
            != len(self.embeddings)
        ):

            raise RuntimeError(
                "Số dòng products.csv "
                "khác số embedding."
            )

        # -------------------------------------------------
        # Load categories dynamically
        # -------------------------------------------------

        self.categories = sorted(
            self.products[
                "category"
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        log_info(
            f"Loaded products="
            f"{len(self.products):,}"
        )

        log_info(
            f"Loaded categories="
            f"{len(self.categories)}"
        )

        log_info(
            "Recommender initialized successfully"
        )

    # =====================================================
    # ENCODE IMAGE
    # =====================================================

    def encode_image(
        self,
        image
    ):

        if not isinstance(
            image,
            Image.Image
        ):

            image = Image.open(
                image
            )

        image = image.convert(
            "RGB"
        )

        inputs = self.processor(
            images=image,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(
                self.device
            )
            for key, value
            in inputs.items()
        }

        with torch.inference_mode():

            vision_outputs = (
                self.model.vision_model(
                    pixel_values=(
                        inputs[
                            "pixel_values"
                        ]
                    ),
                    return_dict=True
                )
            )

            pooled_output = (
                vision_outputs.pooler_output
            )

            features = (
                self.model.visual_projection(
                    pooled_output
                )
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

        return (
            features
            .detach()
            .cpu()
            .numpy()[0]
            .astype(np.float32)
        )

    # =====================================================
    # PREDICT CATEGORY FROM VISUAL NEIGHBORS
    # =====================================================

    def predict_category(
        self,
        query_embedding,
        request_id
    ):

        # -------------------------------------------------
        # Similarity against entire catalog
        # -------------------------------------------------

        scores = cosine_similarity(
            query_embedding,
            self.embeddings
        )

        # -------------------------------------------------
        # Get nearest visual neighbors
        # -------------------------------------------------

        neighbor_k = min(
            CATEGORY_NEIGHBORS,
            len(scores)
        )

        neighbor_indices = (
            top_k_indices(
                scores,
                k=neighbor_k
            )
        )

        neighbors = (
            self.products
            .iloc[neighbor_indices]
            .copy()
        )

        neighbors["similarity"] = (
            scores[neighbor_indices]
        )

        # -------------------------------------------------
        # Category weighted voting
        # -------------------------------------------------

        category_scores = {}

        for _, row in neighbors.iterrows():

            category = str(
                row["category"]
            )

            similarity = float(
                row["similarity"]
            )

            # Similarity càng cao
            # càng có trọng số lớn
            weight = max(
                similarity,
                0.0
            ) ** 4

            category_scores[
                category
            ] = (
                category_scores.get(
                    category,
                    0.0
                )
                +
                weight
            )

        if not category_scores:

            raise RuntimeError(
                "Không thể xác định category."
            )

        # -------------------------------------------------
        # Sort categories
        # -------------------------------------------------

        ranked_categories = sorted(
            category_scores.items(),
            key=lambda item: item[1],
            reverse=True
        )

        predicted_category = (
            ranked_categories[0][0]
        )

        total_score = sum(
            category_scores.values()
        )

        confidence = (
            category_scores[
                predicted_category
            ]
            /
            total_score
            if total_score > 0
            else 0
        )

        # -------------------------------------------------
        # Log category
        # -------------------------------------------------

        log_info(
            f"request={request_id} | "
            f"CATEGORY | "
            f"predicted={predicted_category} | "
            f"confidence={confidence:.4f}"
        )

        # -------------------------------------------------
        # Log top categories
        # -------------------------------------------------

        for rank, (
            category,
            category_score
        ) in enumerate(
            ranked_categories[:5],
            start=1
        ):

            log_info(
                f"request={request_id} | "
                f"CATEGORY_CANDIDATE | "
                f"rank={rank} | "
                f"category={category} | "
                f"score={category_score:.6f}"
            )

        return (
            predicted_category,
            confidence,
            scores
        )

    # =====================================================
    # RECOMMEND
    # =====================================================

    # =====================================================
    # METHOD 1: NO CATEGORY
    # =====================================================

    def recommend_no_category(
        self,
        query_embedding,
        top_k=TOP_K
    ):

        start_time = time.perf_counter()

        visual_scores = cosine_similarity(
            query_embedding,
            self.embeddings
        )

        actual_k = min(
            top_k,
            len(visual_scores)
        )

        top_indices = top_k_indices(
            visual_scores,
            k=actual_k
        )

        results = (
            self.products
            .iloc[top_indices]
            .copy()
            .reset_index(drop=True)
        )

        results["visual_similarity"] = (
            visual_scores[top_indices]
        )

        results["similarity"] = (
            visual_scores[top_indices]
        )

        elapsed = (
            time.perf_counter()
            - start_time
        )

        return results, elapsed


    # =====================================================
    # METHOD 2: HARD CATEGORY
    # =====================================================

    def recommend_hard_category(
        self,
        query_embedding,
        predicted_category,
        top_k=TOP_K
    ):

        start_time = time.perf_counter()

        category_mask = (
            self.products["category"]
            .astype(str)
            .eq(predicted_category)
            .to_numpy()
        )

        category_indices = np.where(
            category_mask
        )[0]

        # Không có sản phẩm cùng category
        if len(category_indices) == 0:

            return (
                self.products.iloc[[]].copy(),
                time.perf_counter() - start_time
            )

        category_embeddings = (
            self.embeddings[
                category_indices
            ]
        )

        category_scores = cosine_similarity(
            query_embedding,
            category_embeddings
        )

        actual_k = min(
            top_k,
            len(category_scores)
        )

        local_indices = top_k_indices(
            category_scores,
            k=actual_k
        )

        global_indices = (
            category_indices[
                local_indices
            ]
        )

        results = (
            self.products
            .iloc[global_indices]
            .copy()
            .reset_index(drop=True)
        )

        results["visual_similarity"] = (
            category_scores[local_indices]
        )

        results["similarity"] = (
            category_scores[local_indices]
        )

        elapsed = (
            time.perf_counter()
            - start_time
        )

        return results, elapsed


    # =====================================================
    # METHOD 3: SOFT CATEGORY CONSTRAINT
    # =====================================================

    def recommend_soft_category(
        self,
        query_embedding,
        predicted_category,
        top_k=TOP_K
    ):

        start_time = time.perf_counter()

        # -------------------------------------------------
        # Visual similarity với TOÀN BỘ catalog
        # -------------------------------------------------

        visual_scores = cosine_similarity(
            query_embedding,
            self.embeddings
        )

        # -------------------------------------------------
        # Category bonus
        # -------------------------------------------------

        category_scores = (
            self.products["category"]
            .astype(str)
            .eq(predicted_category)
            .astype(np.float32)
            .to_numpy()
        )

        # -------------------------------------------------
        # Final score
        # -------------------------------------------------

        final_scores = (
            visual_scores
            + CATEGORY_BONUS * category_scores
        )

        # -------------------------------------------------
        # Top-K
        # -------------------------------------------------

        actual_k = min(
            top_k,
            len(final_scores)
        )

        top_indices = top_k_indices(
            final_scores,
            k=actual_k
        )

        results = (
            self.products
            .iloc[top_indices]
            .copy()
            .reset_index(drop=True)
        )

        results["visual_similarity"] = (
            visual_scores[top_indices]
        )

        results["category_bonus"] = (
            CATEGORY_BONUS
            * category_scores[top_indices]
        )

        results["similarity"] = (
            final_scores[top_indices]
        )

        elapsed = (
            time.perf_counter()
            - start_time
        )

        return results, elapsed


    # =====================================================
    # RUN ALL METHODS
    # =====================================================

    def compare_methods(
        self,
        image,
        top_k=TOP_K,
        request_id=None
    ):

        if request_id is None:

            request_id = (
                uuid.uuid4()
                .hex[:8]
            )

        total_start = (
            time.perf_counter()
        )

        # -------------------------------------------------
        # Encode image
        # -------------------------------------------------

        query_embedding = (
            self.encode_image(image)
        )

        # -------------------------------------------------
        # Predict category
        # -------------------------------------------------

        (
            predicted_category,
            confidence,
            _
        ) = self.predict_category(
            query_embedding,
            request_id
        )

        # -------------------------------------------------
        # Method 1
        # -------------------------------------------------

        no_category, time_no_category = (
            self.recommend_no_category(
                query_embedding,
                top_k
            )
        )

        # -------------------------------------------------
        # Method 2
        # -------------------------------------------------

        hard_category, time_hard_category = (
            self.recommend_hard_category(
                query_embedding,
                predicted_category,
                top_k
            )
        )

        # -------------------------------------------------
        # Method 3
        # -------------------------------------------------

        soft_category, time_soft_category = (
            self.recommend_soft_category(
                query_embedding,
                predicted_category,
                top_k
            )
        )

        # -------------------------------------------------
        # Total time
        # -------------------------------------------------

        total_time = (
            time.perf_counter()
            - total_start
        )

        log_info(
            f"request={request_id} | "
            f"COMPARISON_COMPLETE | "
            f"category={predicted_category} | "
            f"confidence={confidence:.4f} | "
            f"no_category={time_no_category:.4f}s | "
            f"hard_category={time_hard_category:.4f}s | "
            f"soft_category={time_soft_category:.4f}s | "
            f"total={total_time:.4f}s"
        )

        return {
            "no_category": no_category,
            "hard_category": hard_category,
            "soft_category": soft_category,

            "predicted_category":
                predicted_category,

            "category_confidence":
                confidence,

            "time_no_category":
                time_no_category,

            "time_hard_category":
                time_hard_category,

            "time_soft_category":
                time_soft_category,

            "total_time":
                total_time,
        }


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    print(
        "recommend.py là module "
        "xử lý recommendation."
    )

    print(
        "Chạy app bằng:"
    )

    print(
        "streamlit run app.py"
    )