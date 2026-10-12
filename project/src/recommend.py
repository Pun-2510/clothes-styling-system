from pathlib import Path
import re
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
    ACTIVE_CLIP_MODEL,
    PROCESSED_CSV,
    EMBEDDINGS_FILE,
    TEXT_EMBEDDINGS_FILE,
    TOP_K,
    CATEGORY_TOP_PER_CLASS,
    CATEGORY_CONFIDENCE_THRESHOLD,
    CATEGORY_BONUS,
    TEXT_RETRIEVAL_IMAGE_WEIGHT,
    COMBINED_IMAGE_WEIGHT,
    COMBINED_CATEGORY_WEIGHT,
    COMBINED_ATTRIBUTE_WEIGHT,
)

from src.clip_runtime import model_identity, validate_embedding_metadata

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


ATTRIBUTE_RULES = (
    ("black", (r"\bblack\b",), (r"\bblack\b",)),
    ("white", (r"\bwhite\b",), (r"\bwhite\b",)),
    ("red", (r"\bred\b",), (r"\bred\b",)),
    ("blue", (r"\bblue\b",), (r"\bblue\b",)),
    ("green", (r"\bgreen\b",), (r"\bgreen\b",)),
    ("yellow", (r"\byellow\b",), (r"\byellow\b",)),
    ("pink", (r"\bpink\b",), (r"\bpink\b",)),
    ("orange", (r"\borange\b",), (r"\borange\b",)),
    ("purple", (r"\bpurple\b",), (r"\bpurple\b",)),
    ("brown", (r"\bbrown\b",), (r"\bbrown\b",)),
    ("grey", (r"\b(?:grey|gray)\b",), (r"\b(?:grey|gray)\b",)),
    ("navy", (r"\bnavy\b",), (r"\bnavy\b",)),
    ("maroon", (r"\bmaroon\b",), (r"\bmaroon\b",)),
    ("beige", (r"\bbeige\b",), (r"\bbeige\b",)),
    ("gold", (r"\b(?:gold|golden)\b",), (r"\b(?:gold|golden)\b",)),
    ("silver", (r"\bsilver\b",), (r"\bsilver\b",)),
    (
        "patterned",
        (r"\b(?:patterned|printed|graphic|motif)\b",),
        (r"\b(?:patterned|printed|graphic|pattern|striped|check|checked|plaid|floral|polka)\b",),
    ),
    ("striped", (r"\b(?:stripe|striped|stripes)\b",), (r"\b(?:stripe|striped|stripes)\b",)),
    ("checked", (r"\b(?:check|checked|checkered|plaid)\b",), (r"\b(?:check|checked|checkered|plaid)\b",)),
    ("floral", (r"\b(?:floral|flower|flowers)\b",), (r"\b(?:floral|flower|flowers)\b",)),
)


def rank_normalize_scores(scores):
    """Convert one query's scores to tie-aware percentile ranks in 0..1."""
    values = np.asarray(scores, dtype=np.float32)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("scores must be a non-empty one-dimensional array")
    if len(values) == 1:
        return np.ones(1, dtype=np.float32)
    ranks = pd.Series(values).rank(method="average").to_numpy(dtype=np.float32)
    return (ranks - 1.0) / float(len(values) - 1)


def extract_query_attributes(text):
    """Return explicit fashion attributes that should override the image."""
    query = str(text).lower()
    return [
        name
        for name, query_patterns, _ in ATTRIBUTE_RULES
        if any(re.search(pattern, query) for pattern in query_patterns)
    ]


# =========================================================
# RECOMMENDER
# =========================================================

class FashionRecommender:

    def __init__(self, model_source=None, embedding_dir=None):

        model_source = str(model_source or ACTIVE_CLIP_MODEL)
        image_file = Path(embedding_dir) / "image_embeddings.npy" if embedding_dir is not None else EMBEDDINGS_FILE
        text_file = Path(embedding_dir) / "text_embeddings.npy" if embedding_dir is not None else TEXT_EMBEDDINGS_FILE
        identity = model_identity(model_source)
        for path in (image_file, text_file):
            if path.exists():
                validate_embedding_metadata(path, identity, PROCESSED_CSV,
                                            allow_legacy=model_source == CLIP_MODEL)

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
                model_source
            )
        )

        self.model = (
            CLIPModel.from_pretrained(
                model_source
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

        searchable_columns = [
            column
            for column in ("product_name", "description", "category")
            if column in self.products.columns
        ]
        self.product_search_text = (
            self.products[searchable_columns]
            .fillna("")
            .astype(str)
            .agg(" ".join, axis=1)
            .str.lower()
        )

        # -------------------------------------------------
        # Load embeddings
        # -------------------------------------------------

        self.embeddings = np.load(
            image_file
        )

        # Optional so image search still works before offline text encoding.
        self.text_embeddings = (
            np.load(text_file)
            if text_file.exists()
            else None
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

        if (
            self.text_embeddings is not None
            and len(self.products) != len(self.text_embeddings)
        ):
            raise RuntimeError(
                "Số dòng products.csv khác số text embedding. "
                "Hãy chạy lại: python -m src.generate_clip_embeddings "
                "--modality text"
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

        self.category_indices = {
            str(category): np.asarray(indices, dtype=np.int64)
            for category, indices in self.products.groupby(
                self.products["category"].astype(str), sort=True
            ).groups.items()
        }

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
    # ENCODE AND SEARCH TEXT
    # =====================================================

    def encode_text(self, text):
        """Encode one text query into the normalized CLIP space."""
        text = str(text).strip()
        if not text:
            raise ValueError("Vui lòng nhập nội dung cần tìm kiếm.")

        max_length = self.model.config.text_config.max_position_embeddings
        inputs = self.processor(
            text=[text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            text_outputs = self.model.text_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                return_dict=True,
            )
            features = self.model.text_projection(text_outputs.pooler_output)
            features = features / features.norm(
                dim=-1,
                keepdim=True,
            ).clamp_min(1e-12)

        return features.detach().cpu().numpy()[0].astype(np.float32)

    @property
    def has_text_embeddings(self):
        return self.text_embeddings is not None

    def recommend_by_text(self, text, top_k=TOP_K, image_weight=0.5):
        """Search with combined text-to-image and text-to-text scores."""
        start_time = time.perf_counter()
        query_embedding = self.encode_text(text)
        cross_modal_scores = cosine_similarity(query_embedding, self.embeddings)

        image_weight = float(np.clip(image_weight, 0.0, 1.0))
        if self.text_embeddings is not None:
            text_scores = cosine_similarity(query_embedding, self.text_embeddings)
            final_scores = (
                image_weight * cross_modal_scores
                + (1.0 - image_weight) * text_scores
            )
        else:
            text_scores = None
            final_scores = cross_modal_scores

        actual_k = min(int(top_k), len(final_scores))
        top_indices = top_k_indices(final_scores, k=actual_k)
        results = self.products.iloc[top_indices].copy().reset_index(drop=True)
        results["cross_modal_similarity"] = cross_modal_scores[top_indices]
        if text_scores is not None:
            results["text_similarity"] = text_scores[top_indices]
        results["similarity"] = final_scores[top_indices]

        elapsed = time.perf_counter() - start_time
        log_info(
            f"TEXT_SEARCH | query={str(text)[:80]!r} | top_k={actual_k} | "
            f"image_weight={image_weight:.2f} | elapsed={elapsed:.4f}s"
        )
        return results, elapsed, query_embedding

    @staticmethod
    def _normalize_words(value):
        return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()

    @classmethod
    def _category_aliases(cls, category):
        normalized = cls._normalize_words(category)
        aliases = {normalized}
        words = normalized.split()

        if words:
            last = words[-1]
            if last.endswith("ies") and len(last) > 3:
                singular = last[:-3] + "y"
            elif last.endswith("ses") and len(last) > 3:
                singular = last[:-2]
            elif last.endswith("s") and len(last) > 2:
                singular = last[:-1]
            else:
                singular = last
            aliases.add(" ".join(words[:-1] + [singular]))

        if "tshirts" in normalized:
            aliases.add(normalized.replace("tshirts", "t shirts"))
            aliases.add(normalized.replace("tshirts", "t shirt"))
        if normalized == "tshirts":
            aliases.update({"tee", "tees"})

        return {alias for alias in aliases if alias}

    def infer_category_from_text(self, text):
        """Find an explicitly named catalog category without a classifier."""
        normalized_query = self._normalize_words(text)
        matches = []

        for category in self.categories:
            for alias in self._category_aliases(category):
                if re.search(rf"\b{re.escape(alias)}\b", normalized_query):
                    matches.append((len(alias), str(category)))

        return max(matches)[1] if matches else None

    @classmethod
    def categories_are_related(cls, first, second):
        """Treat a broad text category as compatible with an image subtype."""
        left = cls._normalize_words(first).replace(" ", "")
        right = cls._normalize_words(second).replace(" ", "")
        return left == right or left.endswith(right) or right.endswith(left)

    def attribute_match_scores(self, attributes):
        """Score how many requested attributes occur in catalog metadata."""
        if not attributes:
            return np.zeros(len(self.products), dtype=np.float32)

        search_text = getattr(self, "product_search_text", None)
        if search_text is None:
            columns = [
                column
                for column in ("product_name", "description", "category")
                if column in self.products.columns
            ]
            search_text = (
                self.products[columns]
                .fillna("")
                .astype(str)
                .agg(" ".join, axis=1)
                .str.lower()
            )

        rule_map = {
            name: candidate_patterns
            for name, _, candidate_patterns in ATTRIBUTE_RULES
        }
        score = np.zeros(len(self.products), dtype=np.float32)

        for attribute in attributes:
            matched = np.zeros(len(self.products), dtype=bool)
            for pattern in rule_map[attribute]:
                matched |= search_text.str.contains(
                    pattern,
                    regex=True,
                    na=False,
                ).to_numpy()
            score += matched.astype(np.float32)

        return score / float(len(attributes))

    # =====================================================
    # ENCODE AND SEARCH WITH IMAGE + TEXT
    # =====================================================

    def recommend_combined(
        self,
        image,
        text,
        top_k=TOP_K,
        image_weight=COMBINED_IMAGE_WEIGHT,
        text_image_weight=TEXT_RETRIEVAL_IMAGE_WEIGHT,
        request_id=None,
    ):
        """Apply category-aware, attribute-aware image and text fusion."""
        if request_id is None:
            request_id = uuid.uuid4().hex[:8]

        total_start = time.perf_counter()
        image_query = self.encode_image(image)
        predicted_category, category_confidence, visual_scores = (
            self.predict_category(image_query, request_id)
        )

        text_category = self.infer_category_from_text(text)
        if text_category is not None:
            if (
                category_confidence >= CATEGORY_CONFIDENCE_THRESHOLD
                and self.categories_are_related(
                    predicted_category,
                    text_category,
                )
            ):
                applied_category = predicted_category
                category_source = "text_image"
            else:
                applied_category = text_category
                category_source = "text"
            contextual_text = str(text).strip()
        elif category_confidence >= CATEGORY_CONFIDENCE_THRESHOLD:
            applied_category = predicted_category
            category_source = "image"
            contextual_text = (
                f"{str(text).strip()}. Product category: {applied_category}."
            )
        else:
            applied_category = None
            category_source = None
            contextual_text = str(text).strip()

        text_query = self.encode_text(contextual_text)
        text_to_image_scores = cosine_similarity(text_query, self.embeddings)

        normalized_visual_scores = rank_normalize_scores(visual_scores)
        normalized_text_to_image = rank_normalize_scores(
            text_to_image_scores
        )

        text_image_weight = float(np.clip(text_image_weight, 0.0, 1.0))
        if self.text_embeddings is not None:
            text_to_text_scores = cosine_similarity(
                text_query,
                self.text_embeddings,
            )
            normalized_text_to_text = rank_normalize_scores(
                text_to_text_scores
            )
            normalized_text_scores = (
                text_image_weight * normalized_text_to_image
                + (1.0 - text_image_weight) * normalized_text_to_text
            )
        else:
            text_to_text_scores = None
            normalized_text_scores = normalized_text_to_image

        image_weight = float(np.clip(image_weight, 0.0, 1.0))
        text_weight = 1.0 - image_weight
        final_scores = (
            image_weight * normalized_visual_scores
            + text_weight * normalized_text_scores
        )

        attributes = extract_query_attributes(text)
        attribute_scores = self.attribute_match_scores(attributes)
        if attributes:
            final_scores = (
                (1.0 - COMBINED_ATTRIBUTE_WEIGHT) * final_scores
                + COMBINED_ATTRIBUTE_WEIGHT * attribute_scores
            )

        if applied_category is not None:
            category_scores = (
                self.products["category"]
                .astype(str)
                .eq(applied_category)
                .astype(np.float32)
                .to_numpy()
            )
        else:
            category_scores = np.zeros(len(self.products), dtype=np.float32)

        # An image-inferred category is only a soft cue. An explicitly named
        # text category is safe to use as the candidate pool and lets a phrase
        # such as "red cap" override the blue colour in the reference image.
        if category_source == "image":
            final_scores = (
                (1.0 - COMBINED_CATEGORY_WEIGHT) * final_scores
                + COMBINED_CATEGORY_WEIGHT * category_scores
            )

        eligible_indices = np.arange(len(final_scores))
        if category_source in {"text", "text_image"}:
            category_indices = np.flatnonzero(category_scores)
            if len(category_indices) > 0:
                eligible_indices = category_indices

        actual_k = min(int(top_k), len(eligible_indices))
        local_indices = top_k_indices(
            final_scores[eligible_indices],
            k=actual_k,
        )
        top_indices = eligible_indices[local_indices]
        results = self.products.iloc[top_indices].copy().reset_index(drop=True)
        results["raw_visual_similarity"] = visual_scores[top_indices]
        results["visual_similarity"] = normalized_visual_scores[top_indices]
        results["text_to_image_similarity"] = text_to_image_scores[top_indices]
        if text_to_text_scores is not None:
            results["text_to_text_similarity"] = text_to_text_scores[top_indices]
        results["text_similarity"] = normalized_text_scores[top_indices]
        results["attribute_match"] = attribute_scores[top_indices]
        results["category_match"] = category_scores[top_indices]
        results["similarity"] = final_scores[top_indices]

        elapsed = time.perf_counter() - total_start
        log_info(
            f"request={request_id} | COMBINED_SEARCH_COMPLETE | "
            f"query={str(text)[:80]!r} | context={contextual_text[:100]!r} | "
            f"category={applied_category or 'not_used'} | "
            f"category_source={category_source or 'not_used'} | "
            f"attributes={attributes or 'not_used'} | top_k={actual_k} | "
            f"image_weight={image_weight:.2f} | text_weight={text_weight:.2f} | "
            f"elapsed={elapsed:.4f}s"
        )

        return {
            "results": results,
            "image_weight": image_weight,
            "text_weight": text_weight,
            "applied_category": applied_category,
            "category_source": category_source,
            "detected_attributes": attributes,
            "elapsed": elapsed,
        }

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

        # Every category contributes at most the same number of its strongest
        # similarities. Large categories therefore cannot win only by count.
        category_scores = {}

        for category, indices in self.category_indices.items():

            category_similarities = (
                np.maximum(scores[indices], 0.0)
                ** 4
            )

            sample_size = min(
                CATEGORY_TOP_PER_CLASS,
                len(category_similarities)
            )

            if sample_size == 0:
                continue

            strongest = np.partition(
                category_similarities,
                len(category_similarities) - sample_size
            )[-sample_size:]

            category_scores[category] = float(
                strongest.mean()
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

        top_score = float(
            ranked_categories[0][1]
        )

        second_score = (
            float(ranked_categories[1][1])
            if len(ranked_categories) > 1
            else 0.0
        )

        # Relative top-two confidence. This remains a heuristic score rather
        # than a calibrated classification probability.
        confidence = (
            top_score
            /
            (top_score + second_score)
            if top_score + second_score > 0
            else 0.0
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
    # RUN ONE IMAGE RECOMMENDATION METHOD
    # =====================================================

    def recommend_by_image(
        self,
        image,
        category_mode="no_category",
        top_k=TOP_K,
        request_id=None
    ):

        if request_id is None:

            request_id = (
                uuid.uuid4()
                .hex[:8]
            )

        requested_category_mode = (
            str(category_mode)
            .strip()
            .lower()
        )

        category_mode = requested_category_mode

        valid_modes = {
            "no_category",
            "hard_category",
            "soft_category",
        }

        if category_mode not in valid_modes:

            raise ValueError(
                "category_mode phải là một trong: "
                "no_category, hard_category, soft_category."
            )

        total_start = (
            time.perf_counter()
        )

        query_embedding = (
            self.encode_image(image)
        )

        predicted_category = None
        confidence = None

        if category_mode == "no_category":

            results, ranking_time = (
                self.recommend_no_category(
                    query_embedding,
                    top_k
                )
            )

        else:

            (
                predicted_category,
                confidence,
                _
            ) = self.predict_category(
                query_embedding,
                request_id
            )

            if category_mode == "hard_category":

                results, ranking_time = (
                    self.recommend_hard_category(
                        query_embedding,
                        predicted_category,
                        top_k
                    )
                )

            else:

                if confidence < CATEGORY_CONFIDENCE_THRESHOLD:

                    log_info(
                        f"request={request_id} | "
                        "CATEGORY_FALLBACK | "
                        f"predicted={predicted_category} | "
                        f"confidence={confidence:.4f} | "
                        f"threshold={CATEGORY_CONFIDENCE_THRESHOLD:.4f} | "
                        "applied=no_category"
                    )

                    results, ranking_time = (
                        self.recommend_no_category(
                            query_embedding,
                            top_k
                        )
                    )

                    category_mode = "no_category"
                    predicted_category = None
                    confidence = None

                else:

                    results, ranking_time = (
                        self.recommend_soft_category(
                            query_embedding,
                            predicted_category,
                            top_k
                        )
                    )

        total_time = (
            time.perf_counter()
            - total_start
        )

        confidence_log = (
            f"{confidence:.4f}"
            if confidence is not None
            else "not_used"
        )

        log_info(
            f"request={request_id} | "
            f"IMAGE_RECOMMENDATION_COMPLETE | "
            f"requested_mode={requested_category_mode} | "
            f"mode={category_mode} | "
            f"category={predicted_category or 'not_used'} | "
            f"confidence={confidence_log} | "
            f"ranking={ranking_time:.4f}s | "
            f"total={total_time:.4f}s"
        )

        return {
            "category_mode": category_mode,
            "predicted_category": predicted_category,
            "category_confidence": confidence,
            "results": results,
            "ranking_time": ranking_time,
            "total_time": total_time,
        }


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
