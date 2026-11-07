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
    ACTIVE_CLIP_MODEL,
    PROCESSED_CSV,
    EMBEDDINGS_FILE,
    TEXT_EMBEDDINGS_FILE,
    TOP_K,
    CATEGORY_TOP_PER_CLASS,
    CATEGORY_PROTOTYPE_WEIGHT,
    CATEGORY_RELATION_NEIGHBORS,
    RELATED_TOP1_BONUS_FACTOR,
    RELATED_TOP2_BONUS_FACTOR,
    UNRELATED_TOP1_BONUS_FACTOR,
    CATEGORY_CONFIDENCE_FLOOR,
    CATEGORY_CONFIDENCE_THRESHOLD,
    CATEGORY_BONUS,
    TEXT_RETRIEVAL_IMAGE_WEIGHT,
    COMBINED_IMAGE_WEIGHT,
    COMBINED_CATEGORY_WEIGHT,
    COMBINED_ATTRIBUTE_WEIGHT,
)

from src.clip_runtime import model_identity, validate_embedding_metadata

from src.category_ranking import (
    apply_confidence_weighted_category_bonus,
    category_confidence_strength,
    build_mutual_category_neighbors,
    categories_are_mutual_neighbors,
    combine_category_evidence,
    rank_category_score_mapping,
    rank_category_candidates,
)

from src.query_signals import (
    ATTRIBUTE_RULES,
    categories_are_related,
    category_aliases,
    explicit_categories_from_text,
    extract_query_attributes,
    fuzzy_category_suggestions,
    infer_category_from_text,
    is_broad_query,
    normalize_words,
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

        # Used only as user-facing guidance for intentionally broad queries.
        # Ranking still runs against the complete catalog.
        self.popular_categories = (
            self.products["category"]
            .dropna()
            .astype(str)
            .value_counts()
            .head(5)
            .index
            .tolist()
        )

        self.category_indices = {
            str(category): np.asarray(indices, dtype=np.int64)
            for category, indices in self.products.groupby(
                self.products["category"].astype(str), sort=True
            ).groups.items()
        }

        # A short text prototype supplies semantic category evidence. This is
        # especially useful for cropped real-world photos whose closest
        # catalog neighbours can be dominated by pose, skin, or background.
        self.category_prototype_embeddings = (
            self._encode_category_prototypes()
        )
        self.category_relation_neighbors = build_mutual_category_neighbors(
            self.categories,
            self.category_prototype_embeddings,
            CATEGORY_RELATION_NEIGHBORS,
        )
        relation_count = sum(
            len(values)
            for values in self.category_relation_neighbors.values()
        ) // 2

        log_info(
            f"Loaded products="
            f"{len(self.products):,}"
        )

        log_info(
            f"Loaded categories="
            f"{len(self.categories)}"
        )

        log_info(
            "Category prototype relation graph | "
            f"mutual_neighbors={CATEGORY_RELATION_NEIGHBORS} | "
            f"relations={relation_count}"
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

    def _encode_category_prototypes(self):
        """Encode one normalized CLIP text prototype for every category."""
        prompts = [
            f"a catalog product photo of {category}"
            for category in self.categories
        ]
        if not prompts:
            raise RuntimeError("Cannot encode category prototypes without categories.")

        max_length = self.model.config.text_config.max_position_embeddings
        inputs = self.processor(
            text=prompts,
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

        return features.detach().cpu().numpy().astype(np.float32)

    @property
    def has_text_embeddings(self):
        return self.text_embeddings is not None

    def recommend_by_text(
        self,
        text,
        top_k=TOP_K,
        image_weight=TEXT_RETRIEVAL_IMAGE_WEIGHT,
        return_guidance=False,
    ):
        """Search with text scores and honor an explicitly named category.

        Category matching is conservative: it only restricts the candidate
        pool when a catalog category (or its singular alias) is present in the
        query.  Queries without an explicit category keep using the complete
        catalog.
        """
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

        explicit_categories = explicit_categories_from_text(
            text,
            self.categories,
        )
        applied_category = (
            explicit_categories[0]
            if len(explicit_categories) == 1
            else None
        )
        spelling_suggestions = (
            []
            if explicit_categories
            else fuzzy_category_suggestions(text, self.categories)
        )
        eligible_indices = np.arange(len(final_scores))
        if applied_category is not None:
            category_matches = (
                self.products["category"]
                .astype(str)
                .eq(applied_category)
                .to_numpy()
            )
            category_indices = np.flatnonzero(category_matches)
            # A stale or externally supplied category list must never produce
            # an empty response; fall back to unrestricted retrieval instead.
            if len(category_indices) > 0:
                eligible_indices = category_indices
            else:
                applied_category = None

        actual_k = min(int(top_k), len(eligible_indices))
        local_indices = top_k_indices(
            final_scores[eligible_indices],
            k=actual_k,
        )
        top_indices = eligible_indices[local_indices]
        results = self.products.iloc[top_indices].copy().reset_index(drop=True)
        results["cross_modal_similarity"] = cross_modal_scores[top_indices]
        if text_scores is not None:
            results["text_similarity"] = text_scores[top_indices]
        results["similarity"] = final_scores[top_indices]

        if applied_category is not None:
            guidance_status = "category_applied"
            suggestions = []
        elif len(explicit_categories) > 1:
            guidance_status = "multiple_categories"
            suggestions = [
                {
                    "category": category,
                    "corrected_query": category,
                    "confidence": None,
                }
                for category in explicit_categories
            ]
        elif spelling_suggestions:
            guidance_status = "spelling_suggestion"
            suggestions = spelling_suggestions
        elif is_broad_query(text):
            guidance_status = "broad_query"
            suggestions = [
                {
                    "category": category,
                    "corrected_query": category,
                    "confidence": None,
                }
                for category in getattr(
                    self,
                    "popular_categories",
                    list(self.categories)[:5],
                )
            ]
        else:
            guidance_status = "unrestricted"
            suggestions = []

        guidance = {
            "status": guidance_status,
            "applied_category": applied_category,
            "suggestions": suggestions,
        }

        elapsed = time.perf_counter() - start_time
        log_info(
            f"TEXT_SEARCH | query={str(text)[:80]!r} | top_k={actual_k} | "
            f"category={applied_category or 'not_used'} | "
            f"candidate_count={len(eligible_indices)} | "
            f"image_weight={image_weight:.2f} | elapsed={elapsed:.4f}s"
        )
        if return_guidance:
            return results, elapsed, query_embedding, guidance
        return results, elapsed, query_embedding

    @staticmethod
    def _normalize_words(value):
        return normalize_words(value)

    @classmethod
    def _category_aliases(cls, category):
        return category_aliases(category)

    def infer_category_from_text(self, text):
        """Find an explicitly named catalog category without a classifier."""
        return infer_category_from_text(text, self.categories)

    @classmethod
    def categories_are_related(cls, first, second):
        """Treat a broad text category as compatible with an image subtype."""
        return categories_are_related(first, second)

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
        image_category_strength = category_confidence_strength(
            category_confidence,
            CATEGORY_CONFIDENCE_FLOOR,
            CATEGORY_CONFIDENCE_THRESHOLD,
        )

        explicit_text_categories = explicit_categories_from_text(
            text,
            self.categories,
        )
        text_category = (
            explicit_text_categories[0]
            if len(explicit_text_categories) == 1
            else None
        )
        category_factors = {}
        candidate_categories = []
        category_route = "none"

        # An explicit text category is the clearest user instruction.  It
        # therefore takes precedence over the image prediction and is handled
        # as a hard candidate constraint below.  Image evidence is only used
        # when text does not name a category; in that case the same dynamic
        # Top-2 policy as image-only retrieval is applied to the fused score.
        if text_category is not None:
            applied_category = text_category
            category_source = "text"
            category_route = "explicit_text"
            contextual_text = str(text).strip()
        elif len(explicit_text_categories) > 1:
            applied_category = None
            category_source = None
            category_route = "multiple_text_categories"
            contextual_text = str(text).strip()
        elif image_category_strength > 0.0:
            category_evidence = self._rank_category_evidence(
                image_query,
                visual_scores,
            )
            leading_categories = [
                category
                for category, _ in category_evidence["ranked"][:2]
            ]
            applied_category = leading_categories[0]
            category_source = "image"

            if category_confidence >= CATEGORY_CONFIDENCE_THRESHOLD:
                category_factors = {applied_category: 1.0}
                category_route = "confident_top1"
            else:
                is_related_pair = (
                    len(leading_categories) == 2
                    and categories_are_mutual_neighbors(
                        leading_categories[0],
                        leading_categories[1],
                        self.category_relation_neighbors,
                    )
                )
                if is_related_pair:
                    candidate_categories = leading_categories
                    category_factors = {
                        leading_categories[0]: RELATED_TOP1_BONUS_FACTOR,
                        leading_categories[1]: RELATED_TOP2_BONUS_FACTOR,
                    }
                    category_route = "ambiguous_related_top2"
                else:
                    category_factors = {
                        applied_category: UNRELATED_TOP1_BONUS_FACTOR,
                    }
                    category_route = "ambiguous_unrelated_top1"
            contextual_text = str(text).strip()
        else:
            applied_category = None
            category_source = None
            category_route = "low_confidence"
            contextual_text = str(text).strip()

        text_query = self.encode_text(contextual_text)
        text_to_image_scores = cosine_similarity(text_query, self.embeddings)

        text_image_weight = float(np.clip(text_image_weight, 0.0, 1.0))
        if self.text_embeddings is not None:
            text_to_text_scores = cosine_similarity(
                text_query,
                self.text_embeddings,
            )
            text_scores = (
                text_image_weight * text_to_image_scores
                + (1.0 - text_image_weight) * text_to_text_scores
            )
        else:
            text_to_text_scores = None
            text_scores = text_to_image_scores

        image_weight = float(np.clip(image_weight, 0.0, 1.0))
        text_weight = 1.0 - image_weight
        final_scores = (
            image_weight * visual_scores
            + text_weight * text_scores
        )

        attributes = extract_query_attributes(text)
        attribute_scores = self.attribute_match_scores(attributes)
        if attributes:
            final_scores = (
                (1.0 - COMBINED_ATTRIBUTE_WEIGHT) * final_scores
                + COMBINED_ATTRIBUTE_WEIGHT * attribute_scores
            )

        if category_source == "text":
            category_scores = (
                self.products["category"]
                .astype(str)
                .eq(applied_category)
                .astype(np.float32)
                .to_numpy()
            )
        elif category_source == "image":
            category_scores = (
                self.products["category"]
                .astype(str)
                .map(category_factors)
                .fillna(0.0)
                .astype(np.float32)
                .to_numpy()
            )
        else:
            category_scores = np.zeros(len(self.products), dtype=np.float32)

        # An image-inferred category is only a soft cue. An explicitly named
        # text category is safe to use as the candidate pool and lets a phrase
        # such as "red cap" override the blue colour in the reference image.
        if category_source == "image":
            final_scores, category_strength, effective_category_bonus = (
                apply_confidence_weighted_category_bonus(
                    final_scores,
                    category_scores,
                    category_confidence,
                    COMBINED_CATEGORY_WEIGHT,
                    CATEGORY_CONFIDENCE_FLOOR,
                    CATEGORY_CONFIDENCE_THRESHOLD,
                )
            )
        elif category_source == "text":
            category_strength = 1.0
            effective_category_bonus = 0.0
        else:
            category_strength = 0.0
            effective_category_bonus = 0.0

        eligible_indices = np.arange(len(final_scores))
        if category_source == "text":
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
        results["visual_similarity"] = visual_scores[top_indices]
        results["text_to_image_similarity"] = text_to_image_scores[top_indices]
        if text_to_text_scores is not None:
            results["text_to_text_similarity"] = text_to_text_scores[top_indices]
        results["text_similarity"] = text_scores[top_indices]
        results["attribute_match"] = attribute_scores[top_indices]
        results["category_match"] = category_scores[top_indices]
        results["similarity"] = final_scores[top_indices]

        if len(explicit_text_categories) > 1:
            guidance = {
                "status": "multiple_categories",
                "applied_category": None,
                "suggestions": [
                    {
                        "category": category,
                        "corrected_query": category,
                        "confidence": None,
                    }
                    for category in explicit_text_categories
                ],
            }
        elif category_source == "text":
            guidance = {
                "status": "category_applied",
                "applied_category": applied_category,
                "suggestions": [],
            }
        else:
            guidance = {
                "status": "unrestricted",
                "applied_category": None,
                "suggestions": [],
            }

        elapsed = time.perf_counter() - total_start
        log_info(
            f"request={request_id} | COMBINED_SEARCH_COMPLETE | "
            f"query={str(text)[:80]!r} | context={contextual_text[:100]!r} | "
            f"category={applied_category or 'not_used'} | "
            f"category_source={category_source or 'not_used'} | "
            f"category_route={category_route} | "
            f"candidate_categories={candidate_categories or 'not_used'} | "
            f"category_strength={category_strength:.4f} | "
            f"category_bonus={effective_category_bonus:.4f} | "
            f"attributes={attributes or 'not_used'} | top_k={actual_k} | "
            "fusion=raw_cosine | "
            f"image_weight={image_weight:.2f} | text_weight={text_weight:.2f} | "
            f"elapsed={elapsed:.4f}s"
        )

        return {
            "results": results,
            "image_weight": image_weight,
            "text_weight": text_weight,
            "applied_category": applied_category,
            "category_source": category_source,
            "category_route": category_route,
            "candidate_categories": candidate_categories,
            "category_strength": category_strength,
            "detected_attributes": attributes,
            "guidance": guidance,
            "elapsed": elapsed,
        }

    # =====================================================
    # PREDICT CATEGORY FROM NEIGHBORS AND CLIP PROTOTYPES
    # =====================================================

    def _rank_category_evidence(self, query_embedding, scores):
        """Return the combined category ranking and its evidence details."""
        visual_ranked_categories, visual_confidence = rank_category_candidates(
            scores,
            self.category_indices,
            CATEGORY_TOP_PER_CLASS,
        )

        prototype_values = np.asarray(
            self.category_prototype_embeddings @ query_embedding,
            dtype=np.float32,
        )
        prototype_scores = dict(zip(self.categories, prototype_values))
        prototype_ranked_categories, prototype_confidence = (
            rank_category_score_mapping(prototype_scores)
        )
        ranked_categories, combined_confidence = combine_category_evidence(
            visual_ranked_categories,
            prototype_scores,
            CATEGORY_PROTOTYPE_WEIGHT,
        )

        predicted_category = ranked_categories[0][0]
        if predicted_category == prototype_ranked_categories[0][0]:
            confidence = prototype_confidence
            confidence_source = "prototype"
        elif predicted_category == visual_ranked_categories[0][0]:
            confidence = visual_confidence
            confidence_source = "visual_neighbors"
        else:
            confidence = combined_confidence
            confidence_source = "combined"

        return {
            "ranked": ranked_categories,
            "confidence": confidence,
            "confidence_source": confidence_source,
            "visual_ranked": visual_ranked_categories,
            "prototype_ranked": prototype_ranked_categories,
            "prototype_scores": prototype_scores,
        }

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

        evidence = self._rank_category_evidence(query_embedding, scores)
        ranked_categories = evidence["ranked"]
        visual_ranked_categories = evidence["visual_ranked"]
        prototype_ranked_categories = evidence["prototype_ranked"]
        prototype_scores = evidence["prototype_scores"]
        predicted_category = ranked_categories[0][0]
        confidence = evidence["confidence"]
        confidence_source = evidence["confidence_source"]

        # -------------------------------------------------
        # Log category
        # -------------------------------------------------

        log_info(
            f"request={request_id} | "
            f"CATEGORY | "
            f"predicted={predicted_category} | "
            f"confidence={confidence:.4f} | "
            f"confidence_source={confidence_source} | "
            f"visual_top={visual_ranked_categories[0][0]} | "
            f"prototype_top={prototype_ranked_categories[0][0]}"
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
                f"combined_score={category_score:.6f} | "
                f"visual_score={dict(visual_ranked_categories).get(category, 0.0):.6f} | "
                f"prototype_score={prototype_scores.get(category, 0.0):.6f}"
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

    def recommend_weighted_categories(
        self,
        query_embedding,
        category_confidence,
        category_factors,
        top_k=TOP_K,
    ):
        """Apply weighted category evidence while retaining the full catalog."""
        start_time = time.perf_counter()
        category_factors = {
            str(category): float(factor)
            for category, factor in category_factors.items()
            if float(factor) > 0.0
        }
        visual_scores = cosine_similarity(query_embedding, self.embeddings)
        product_categories = self.products["category"].astype(str)
        category_weights = (
            product_categories
            .map(category_factors)
            .fillna(0.0)
            .astype(np.float32)
            .to_numpy()
        )
        final_scores, _, effective_category_bonus = (
            apply_confidence_weighted_category_bonus(
                visual_scores,
                category_weights,
                category_confidence,
                CATEGORY_BONUS,
                CATEGORY_CONFIDENCE_FLOOR,
                CATEGORY_CONFIDENCE_THRESHOLD,
            )
        )
        actual_k = min(int(top_k), len(final_scores))
        top_indices = top_k_indices(final_scores, k=actual_k)
        results = self.products.iloc[top_indices].copy().reset_index(drop=True)
        results["visual_similarity"] = visual_scores[top_indices]
        results["category_bonus"] = (
            effective_category_bonus * category_weights[top_indices]
        )
        results["similarity"] = final_scores[top_indices]
        return results, time.perf_counter() - start_time


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
        category_confidence,
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

        final_scores, category_strength, effective_category_bonus = (
            apply_confidence_weighted_category_bonus(
                visual_scores,
                category_scores,
                category_confidence,
                CATEGORY_BONUS,
                CATEGORY_CONFIDENCE_FLOOR,
                CATEGORY_CONFIDENCE_THRESHOLD,
            )
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
            effective_category_bonus
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
        request_id=None,
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
        candidate_categories = []

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
                visual_scores,
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

                category_strength = category_confidence_strength(
                    confidence,
                    CATEGORY_CONFIDENCE_FLOOR,
                    CATEGORY_CONFIDENCE_THRESHOLD,
                )

                if category_strength <= 0.0:

                    log_info(
                        f"request={request_id} | "
                        "CATEGORY_FALLBACK | "
                        f"predicted={predicted_category} | "
                        f"confidence={confidence:.4f} | "
                        f"floor={CATEGORY_CONFIDENCE_FLOOR:.4f} | "
                        "applied=no_category"
                    )

                    results, ranking_time = (
                        self.recommend_no_category(
                            query_embedding,
                            top_k
                        )
                    )

                    category_mode = "no_category"
                else:
                    is_ambiguous_category = (
                        confidence < CATEGORY_CONFIDENCE_THRESHOLD
                    )
                    if is_ambiguous_category:
                        category_evidence = self._rank_category_evidence(
                            query_embedding,
                            visual_scores,
                        )
                        leading_categories = [
                            category
                            for category, _ in category_evidence["ranked"][
                                :2
                            ]
                        ]
                        is_related_pair = (
                            len(leading_categories) == 2
                            and categories_are_mutual_neighbors(
                                leading_categories[0],
                                leading_categories[1],
                                self.category_relation_neighbors,
                            )
                        )
                        if is_related_pair:
                            candidate_categories = leading_categories
                            results, ranking_time = (
                                self.recommend_weighted_categories(
                                    query_embedding,
                                    confidence,
                                    {
                                        leading_categories[0]: RELATED_TOP1_BONUS_FACTOR,
                                        leading_categories[1]: RELATED_TOP2_BONUS_FACTOR,
                                    },
                                    top_k,
                                )
                            )
                            log_info(
                                f"request={request_id} | "
                                "RELATED_CATEGORY_BONUS | "
                                f"categories={candidate_categories} | "
                                f"top1_factor={RELATED_TOP1_BONUS_FACTOR:.4f} | "
                                f"top2_factor={RELATED_TOP2_BONUS_FACTOR:.4f} | "
                                f"confidence={confidence:.4f}"
                            )
                        else:
                            results, ranking_time = (
                                self.recommend_weighted_categories(
                                    query_embedding,
                                    confidence,
                                    {
                                        leading_categories[0]: UNRELATED_TOP1_BONUS_FACTOR,
                                    },
                                    top_k,
                                )
                            )
                            log_info(
                                f"request={request_id} | "
                                "AMBIGUOUS_UNRELATED_WEAK_BONUS | "
                                f"categories={leading_categories} | "
                                f"top1_factor={UNRELATED_TOP1_BONUS_FACTOR:.4f} | "
                                f"confidence={confidence:.4f} | "
                                f"applied={leading_categories[0]}"
                            )
                    else:
                        results, ranking_time = (
                            self.recommend_soft_category(
                                query_embedding,
                                predicted_category,
                                confidence,
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
            f"candidate_categories={candidate_categories or 'not_used'} | "
            f"confidence={confidence_log} | "
            f"category_strength={category_confidence_strength(confidence, CATEGORY_CONFIDENCE_FLOOR, CATEGORY_CONFIDENCE_THRESHOLD) if confidence is not None else 0.0:.4f} | "
            f"ranking={ranking_time:.4f}s | "
            f"total={total_time:.4f}s"
        )

        return {
            "category_mode": category_mode,
            "predicted_category": predicted_category,
            "category_confidence": confidence,
            "candidate_categories": candidate_categories,
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
                confidence,
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
