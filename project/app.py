import time
import uuid

from pathlib import Path

import streamlit as st

from PIL import Image

from src.recommend import (
    FashionRecommender
)

from src.config import (
    TOP_K,
    CATEGORY_BONUS
)

from src.evaluation import (
    calculate_system_metrics
)

from src.logger import (
    log_info,
    log_exception
)

# =========================================================
# DISPLAY RESULTS
# =========================================================

def display_results(
    results,
    method_name
):

    if results is None or results.empty:

        st.warning(
            f"{method_name}: "
            "Không tìm thấy sản phẩm."
        )

        return

    # =====================================================
    # COLUMNS
    # =====================================================

    number_of_columns = min(
        5,
        len(results)
    )

    columns = st.columns(
        number_of_columns
    )

    # =====================================================
    # PRODUCTS
    # =====================================================

    for index, (_, row) in enumerate(
        results.iterrows()
    ):

        with columns[
            index % number_of_columns
        ]:

            # -------------------------------------------------
            # IMAGE
            # -------------------------------------------------

            image_path = Path(
                row["image_path"]
            )

            if image_path.exists():

                st.image(
                    str(image_path),
                    width="stretch"
                )

            else:

                st.warning(
                    "Không tìm thấy ảnh."
                )

            # -------------------------------------------------
            # RANK
            # -------------------------------------------------

            st.markdown(
                f"### #{index + 1}"
            )

            # -------------------------------------------------
            # VISUAL SIMILARITY
            # -------------------------------------------------

            if (
                "visual_similarity"
                in row.index
            ):

                visual_similarity = float(
                    row[
                        "visual_similarity"
                    ]
                )

                st.write(
                    "Visual Similarity: "
                    f"**{visual_similarity:.4f}**"
                )

            # -------------------------------------------------
            # CATEGORY BONUS
            # -------------------------------------------------

            if (
                "category_bonus"
                in row.index
            ):

                category_bonus = float(
                    row[
                        "category_bonus"
                    ]
                )

                st.write(
                    "Category Bonus: "
                    f"**{category_bonus:.4f}**"
                )

            # -------------------------------------------------
            # FINAL SCORE
            # -------------------------------------------------

            if (
                "similarity"
                in row.index
            ):

                final_score = float(
                    row[
                        "similarity"
                    ]
                )

                st.write(
                    "Final Score: "
                    f"**{final_score:.4f}**"
                )

            # -------------------------------------------------
            # PRODUCT NAME
            # -------------------------------------------------

            product_name = str(
                row.get(
                    "product_name",
                    ""
                )
            )

            if (
                product_name
                and product_name != "nan"
            ):

                st.write(
                    product_name
                )

            # -------------------------------------------------
            # CATEGORY
            # -------------------------------------------------

            category = str(
                row.get(
                    "category",
                    ""
                )
            )

            if (
                category
                and category != "nan"
            ):

                st.caption(
                    f"Category: "
                    f"{category}"
                )

            # -------------------------------------------------
            # DESCRIPTION
            # -------------------------------------------------

            description = str(
                row.get(
                    "description",
                    ""
                )
            )

            if (
                description
                and description != "nan"
            ):

                st.caption(
                    description[:180]
                )


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Fashion Recommendation",
    layout="wide"
)


# =========================================================
# TITLE
# =========================================================

st.title(
    "Fashion Product Recommendation"
)

st.caption(
    "Comparison: No Category vs Hard Category vs Soft Category"
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_recommender():

    return FashionRecommender()


try:

    recommender = load_recommender()

except Exception as error:

    st.error(
        "Không thể khởi tạo model.\n\n"
        f"Lỗi: {error}"
    )

    log_exception(
        f"MODEL_INIT_FAILED | "
        f"error={error}"
    )

    st.stop()


# =========================================================
# FILE UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "Upload fashion product image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp"
    ]
)


# =========================================================
# PROCESS IMAGE
# =========================================================

if uploaded_file is not None:

    # -----------------------------------------------------
    # Request ID
    # -----------------------------------------------------

    request_id = (
        uuid.uuid4()
        .hex[:8]
    )

    upload_start = (
        time.perf_counter()
    )

    log_info(
        f"request={request_id} | "
        f"IMAGE_UPLOAD | "
        f"filename={uploaded_file.name} | "
        f"size={uploaded_file.size}"
    )

    try:

        # =================================================
        # OPEN IMAGE
        # =================================================

        input_image = (
            Image.open(
                uploaded_file
            )
            .convert("RGB")
        )

        # =================================================
        # DISPLAY INPUT
        # =================================================

        st.subheader(
            "Input Image"
        )

        st.image(
            input_image,
            width=320
        )

        # =================================================
        # TOP-K
        # =================================================

        top_k = st.slider(
            "Number of recommendations",
            min_value=1,
            max_value=10,
            value=TOP_K
        )

        # =================================================
        # RUN COMPARISON
        # =================================================

        if st.button(
            "Compare 3 Methods",
            type="primary"
        ):

            log_info(
                f"request={request_id} | "
                f"COMPARISON_STARTED | "
                f"top_k={top_k}"
            )

            with st.spinner(
                "Đang chạy 3 phương pháp..."
            ):

                comparison = (
                    recommender.compare_methods(
                        input_image,
                        top_k=top_k,
                        request_id=request_id
                    )
                )

            # =================================================
            # GET RESULTS
            # =================================================

            no_category = comparison[
                "no_category"
            ]

            hard_category = comparison[
                "hard_category"
            ]

            soft_category = comparison[
                "soft_category"
            ]

            predicted_category = (
                comparison[
                    "predicted_category"
                ]
            )

            category_confidence = (
                comparison[
                    "category_confidence"
                ]
            )

            time_no_category = (
                comparison[
                    "time_no_category"
                ]
            )

            time_hard_category = (
                comparison[
                    "time_hard_category"
                ]
            )

            time_soft_category = (
                comparison[
                    "time_soft_category"
                ]
            )


            # =================================================
            # CATEGORY PREDICTION
            # =================================================

            st.subheader(
                "Predicted Category"
            )

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Category",
                    predicted_category
                )

            with col2:

                st.metric(
                    "Confidence",
                    f"{category_confidence:.2%}"
                )


            # =================================================
            # EVALUATION
            # =================================================

            st.divider()

            st.subheader(
                "Method Evaluation"
            )

            # -------------------------------------------------
            # Calculate metrics
            # -------------------------------------------------

            metrics_no_category = (
                calculate_system_metrics(
                    no_category,
                    predicted_category
                )
            )

            metrics_hard_category = (
                calculate_system_metrics(
                    hard_category,
                    predicted_category
                )
            )

            metrics_soft_category = (
                calculate_system_metrics(
                    soft_category,
                    predicted_category
                )
            )


            # =================================================
            # METRIC TABLE
            # =================================================

            evaluation_data = {

                "Method": [
                    "No Category",
                    "Hard Category",
                    "Soft Category"
                ],

                "Avg Visual Similarity": [
                    metrics_no_category[
                        "avg_similarity"
                    ],

                    metrics_hard_category[
                        "avg_similarity"
                    ],

                    metrics_soft_category[
                        "avg_similarity"
                    ]
                ],

                "Category Match": [
                    metrics_no_category[
                        "category_match"
                    ],

                    metrics_hard_category[
                        "category_match"
                    ],

                    metrics_soft_category[
                        "category_match"
                    ]
                ],

                "Processing Time (s)": [
                    time_no_category,
                    time_hard_category,
                    time_soft_category
                ]
            }


            st.dataframe(
                evaluation_data,
                use_container_width=True,
                hide_index=True
            )


            # =================================================
            # SOFT CONSTRAINT EXPLANATION
            # =================================================

            st.info(
                f"""
                **Soft Category Constraint**

                Category không loại bỏ sản phẩm.

                Sản phẩm cùng category với category dự đoán
                được cộng thêm:

                **+{CATEGORY_BONUS:.2f}**

                Công thức:

                **Final Score = Visual Similarity + Category Bonus**

                Sản phẩm khác category vẫn có thể được
                recommendation nếu Visual Similarity đủ cao.
                """
            )


            # =================================================
            # METHOD 1
            # =================================================

            st.divider()

            st.header(
                "1. No Category — Visual Similarity Only"
            )

            st.caption(
                "Baseline: không sử dụng category."
            )

            display_results(
                no_category,
                "No Category"
            )


            # =================================================
            # METHOD 2
            # =================================================

            st.divider()

            st.header(
                "2. Hard Category — Category Filtering"
            )

            st.caption(
                "Chỉ giữ lại các sản phẩm thuộc "
                "category được dự đoán."
            )

            display_results(
                hard_category,
                "Hard Category"
            )


            # =================================================
            # METHOD 3
            # =================================================

            st.divider()

            st.header(
                "3. Soft Category — Category Bonus"
            )

            st.caption(
                f"Không loại bỏ category khác. "
                f"Cùng category được cộng +{CATEGORY_BONUS:.2f}."
            )

            display_results(
                soft_category,
                "Soft Category"
            )


            # =================================================
            # SUMMARY
            # =================================================

            st.divider()

            st.header(
                "Comparison Summary"
            )

            summary_columns = st.columns(3)


            # -------------------------------------------------
            # No Category
            # -------------------------------------------------

            with summary_columns[0]:

                st.subheader(
                    "No Category"
                )

                st.metric(
                    "Avg Visual Similarity",
                    f"{
                        metrics_no_category[
                            'avg_similarity'
                        ]:.4f}"
                )

                st.metric(
                    "Category Match",
                    f"{
                        metrics_no_category[
                            'category_match'
                        ]:.2%}"
                )

                st.metric(
                    "Time",
                    f"{time_no_category:.4f}s"
                )


            # -------------------------------------------------
            # Hard Category
            # -------------------------------------------------

            with summary_columns[1]:

                st.subheader(
                    "Hard Category"
                )

                st.metric(
                    "Avg Visual Similarity",
                    f"{
                        metrics_hard_category[
                            'avg_similarity'
                        ]:.4f}"
                )

                st.metric(
                    "Category Match",
                    f"{
                        metrics_hard_category[
                            'category_match'
                        ]:.2%}"
                )

                st.metric(
                    "Time",
                    f"{time_hard_category:.4f}s"
                )


            # -------------------------------------------------
            # Soft Category
            # -------------------------------------------------

            with summary_columns[2]:

                st.subheader(
                    "Soft Category"
                )

                st.metric(
                    "Avg Visual Similarity",
                    f"{
                        metrics_soft_category[
                            'avg_similarity'
                        ]:.4f}"
                )

                st.metric(
                    "Category Match",
                    f"{
                        metrics_soft_category[
                            'category_match'
                        ]:.2%}"
                )

                st.metric(
                    "Time",
                    f"{time_soft_category:.4f}s"
                )


            # =================================================
            # COMPLETE LOG
            # =================================================

            elapsed = (
                time.perf_counter()
                -
                upload_start
            )

            log_info(
                f"request={request_id} | "
                f"UI_COMPARISON_COMPLETE | "
                f"processing_time={elapsed:.4f}s"
            )


    except Exception as error:

        log_exception(
            f"request={request_id} | "
            f"UI_FAILED | "
            f"error={error}"
        )

        st.error(
            "Recommendation failed: "
            f"{error}"
        )


# =========================================================
# DISPLAY RESULTS FUNCTION
# =========================================================

def display_results(
    results,
    method_name
):

    if results is None or results.empty:

        st.warning(
            f"{method_name}: "
            "Không tìm thấy sản phẩm."
        )

        return


    # =====================================================
    # COLUMNS
    # =====================================================

    number_of_columns = min(
        5,
        len(results)
    )

    columns = st.columns(
        number_of_columns
    )


    # =====================================================
    # PRODUCTS
    # =====================================================

    for index, (_, row) in enumerate(
        results.iterrows()
    ):

        with columns[
            index % number_of_columns
        ]:

            # -------------------------------------------------
            # IMAGE
            # -------------------------------------------------

            image_path = Path(
                row["image_path"]
            )

            if image_path.exists():

                st.image(
                    str(image_path),
                    width="stretch"
                )

            else:

                st.warning(
                    "Không tìm thấy ảnh."
                )


            # -------------------------------------------------
            # RANK
            # -------------------------------------------------

            st.markdown(
                f"### #{index + 1}"
            )


            # -------------------------------------------------
            # VISUAL SIMILARITY
            # -------------------------------------------------

            if (
                "visual_similarity"
                in row.index
            ):

                visual_similarity = float(
                    row[
                        "visual_similarity"
                    ]
                )

                st.write(
                    "Visual Similarity: "
                    f"**{visual_similarity:.4f}**"
                )


            # -------------------------------------------------
            # CATEGORY BONUS
            # -------------------------------------------------

            if (
                "category_bonus"
                in row.index
            ):

                category_bonus = float(
                    row[
                        "category_bonus"
                    ]
                )

                st.write(
                    "Category Bonus: "
                    f"**{category_bonus:.4f}**"
                )


            # -------------------------------------------------
            # FINAL SCORE
            # -------------------------------------------------

            if (
                "similarity"
                in row.index
            ):

                final_score = float(
                    row[
                        "similarity"
                    ]
                )

                st.write(
                    "Final Score: "
                    f"**{final_score:.4f}**"
                )


            # -------------------------------------------------
            # PRODUCT NAME
            # -------------------------------------------------

            product_name = str(
                row.get(
                    "product_name",
                    ""
                )
            )

            if (
                product_name
                and product_name != "nan"
            ):

                st.write(
                    product_name
                )


            # -------------------------------------------------
            # CATEGORY
            # -------------------------------------------------

            category = str(
                row.get(
                    "category",
                    ""
                )
            )

            if (
                category
                and category != "nan"
            ):

                st.caption(
                    f"Category: "
                    f"{category}"
                )


            # -------------------------------------------------
            # DESCRIPTION
            # -------------------------------------------------

            description = str(
                row.get(
                    "description",
                    ""
                )
            )

            if (
                description
                and description != "nan"
            ):

                st.caption(
                    description[:180]
                )


# =========================================================
# EXPLANATION
# =========================================================

st.divider()

with st.expander(
    "How does the system work?"
):

    st.markdown(
        f"""
        ### Offline preprocessing

        Catalog Images  
        ↓  
        CLIP Image Encoder  
        ↓  
        Image Embeddings  
        ↓  
        `image_embeddings.npy`


        ### Online recommendation

        Uploaded Image  
        ↓  
        CLIP Image Encoder  
        ↓  
        Query Embedding  
        ↓  
        Visual Similarity  
        ↓  
        Predict Product Category  
        ↓  
        Compare 3 Recommendation Methods


        ### Method 1 — No Category

        Query  
        ↓  
        Cosine Similarity  
        ↓  
        Top-K


        ### Method 2 — Hard Category

        Query  
        ↓  
        Predict Category  
        ↓  
        Remove other categories  
        ↓  
        Cosine Similarity  
        ↓  
        Top-K


        ### Method 3 — Soft Category

        Query  
        ↓  
        Visual Similarity  
        ↓  
        Category Bonus  
        ↓  
        Final Score  
        ↓  
        Top-K


        **Soft Constraint:**

        Category khác **không bị loại bỏ**.

        Category trùng được cộng:

        **+{CATEGORY_BONUS:.2f}**
        """
    )