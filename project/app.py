import uuid
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from src.config import CATEGORY_BONUS, TOP_K
from src.evaluation import calculate_system_metrics
from src.logger import log_exception, log_info
from src.recommend import FashionRecommender
from src.translator import VietnameseEnglishTranslator


st.set_page_config(page_title="Fashion Recommendation", page_icon="👕", layout="wide")


@st.cache_resource
def load_recommender():
    return FashionRecommender()


@st.cache_resource
def load_translator():
    return VietnameseEnglishTranslator()


def display_results(results, method_name):
    if results is None or results.empty:
        st.warning(f"{method_name}: Không tìm thấy sản phẩm.")
        return

    columns = st.columns(min(5, len(results)))
    score_labels = {
        "visual_similarity": "Visual similarity",
        "cross_modal_similarity": "Text → image",
        "text_similarity": "Text → text",
        "category_bonus": "Category bonus",
        "similarity": "Final score",
    }

    for index, (_, row) in enumerate(results.iterrows()):
        with columns[index % len(columns)]:
            image_path = Path(str(row["image_path"]))
            if image_path.exists():
                st.image(str(image_path), width="stretch")
            else:
                st.warning("Không tìm thấy ảnh.")

            st.markdown(f"#### #{index + 1}")
            product_name = str(row.get("product_name", ""))
            if product_name and product_name != "nan":
                st.write(product_name)

            category = str(row.get("category", ""))
            if category and category != "nan":
                st.caption(f"Category: {category}")

            for column, label in score_labels.items():
                if column in row.index:
                    st.write(f"{label}: **{float(row[column]):.4f}**")

            description = str(row.get("description", ""))
            if description and description != "nan":
                with st.expander("Mô tả"):
                    st.caption(description[:500])


def render_image_evaluation(comparison):
    predicted_category = comparison["predicted_category"]
    confidence = comparison["category_confidence"]
    result_keys = ("no_category", "hard_category", "soft_category")
    labels = ("No category", "Hard category", "Soft category")
    time_keys = ("time_no_category", "time_hard_category", "time_soft_category")

    metric_columns = st.columns(2)
    metric_columns[0].metric("Predicted category", predicted_category)
    metric_columns[1].metric("Confidence", f"{confidence:.2%}")

    rows = []
    for key, label, time_key in zip(result_keys, labels, time_keys):
        metrics = calculate_system_metrics(comparison[key], predicted_category)
        rows.append(
            {
                "Method": label,
                "Avg visual similarity": metrics["avg_similarity"],
                "Category match": metrics["category_match"],
                "Processing time (s)": comparison[time_key],
            }
        )
    st.dataframe(rows, width="stretch", hide_index=True)

    method_tabs = st.tabs(labels)
    captions = (
        "Baseline: chỉ dùng visual similarity.",
        "Chỉ giữ sản phẩm thuộc category được dự đoán.",
        f"Không loại category khác; category trùng được cộng +{CATEGORY_BONUS:.2f}.",
    )
    for tab, key, label, caption in zip(method_tabs, result_keys, labels, captions):
        with tab:
            st.caption(caption)
            display_results(comparison[key], label)


def render_image_search(recommender):
    st.subheader("Tìm sản phẩm bằng ảnh")
    uploaded_file = st.file_uploader(
        "Tải lên ảnh sản phẩm thời trang",
        type=["jpg", "jpeg", "png", "webp"],
        key="image_query",
    )
    top_k = st.slider("Số kết quả", 1, 10, TOP_K, key="image_top_k")

    if uploaded_file is None:
        st.info("Tải một ảnh lên để bắt đầu tìm kiếm.")
        return

    try:
        input_image = Image.open(uploaded_file).convert("RGB")
        st.image(input_image, width=320, caption="Ảnh truy vấn")
    except Exception as error:
        st.error(f"Không thể đọc ảnh: {error}")
        return

    if st.button("So sánh 3 phương pháp", type="primary", key="run_image"):
        request_id = uuid.uuid4().hex[:8]
        log_info(
            f"request={request_id} | IMAGE_SEARCH_STARTED | "
            f"filename={uploaded_file.name} | top_k={top_k}"
        )
        try:
            with st.spinner("Đang mã hóa ảnh và tìm sản phẩm..."):
                comparison = recommender.compare_methods(
                    input_image,
                    top_k=top_k,
                    request_id=request_id,
                )
            render_image_evaluation(comparison)
        except Exception as error:
            log_exception(f"request={request_id} | IMAGE_SEARCH_FAILED | error={error}")
            st.error(f"Tìm kiếm thất bại: {error}")


def render_text_search(recommender):
    st.subheader("Tìm sản phẩm bằng văn bản")
    st.caption(
        "CLIP mã hóa câu truy vấn thành vector 512 chiều rồi so khớp với ảnh "
        "và nội dung của sản phẩm. CLIP thường hiểu truy vấn tiếng Anh tốt hơn."
    )
    query_language = st.radio(
        "Ngôn ngữ truy vấn",
        options=["Tiếng Việt", "English"],
        horizontal=True,
        key="text_query_language",
    )
    query = st.text_area(
        "Mô tả sản phẩm bạn muốn tìm",
        placeholder=(
            "Ví dụ: váy nữ màu đỏ có họa tiết hoa"
            if query_language == "Tiếng Việt"
            else "Example: a red floral summer dress for women"
        ),
        key="text_query",
    )
    control_columns = st.columns(2)
    top_k = control_columns[0].slider(
        "Số kết quả", 1, 10, TOP_K, key="text_top_k"
    )

    if recommender.has_text_embeddings:
        image_weight = control_columns[1].slider(
            "Trọng số text → image",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Phần trọng số còn lại được dành cho text → product text.",
        )
    else:
        image_weight = 1.0
        st.warning(
            "Chưa có text_embeddings.npy. Demo vẫn tìm text → image; chạy "
            "`python -m src.generate_text_embeddings` để bật điểm text → text."
        )

    if st.button("Mã hóa văn bản và tìm kiếm", type="primary", key="run_text"):
        if not query.strip():
            st.warning("Vui lòng nhập mô tả sản phẩm.")
            return

        try:
            search_query = query.strip()
            if query_language == "Tiếng Việt":
                with st.spinner("Đang dịch truy vấn tiếng Việt sang tiếng Anh..."):
                    search_query = load_translator().translate(search_query)
                st.success(f"Bản dịch dùng cho CLIP: **{search_query}**")

            with st.spinner("Đang mã hóa văn bản và tìm sản phẩm..."):
                results, elapsed, embedding = recommender.recommend_by_text(
                    search_query,
                    top_k=top_k,
                    image_weight=image_weight,
                )

            info_columns = st.columns(3)
            info_columns[0].metric("Embedding dimension", embedding.shape[0])
            info_columns[1].metric("L2 norm", f"{np.linalg.norm(embedding):.4f}")
            info_columns[2].metric("Thời gian", f"{elapsed:.3f}s")

            with st.expander("Xem một phần vector text embedding"):
                st.code(np.array2string(embedding[:16], precision=6), language="text")
            display_results(results, "Text search")
        except Exception as error:
            log_exception(f"TEXT_SEARCH_FAILED | error={error}")
            st.error(f"Tìm kiếm thất bại: {error}")


st.title("Fashion Product Recommendation")
st.caption("CLIP multimodal search: image encoding + text encoding")

try:
    recommender = load_recommender()
except Exception as error:
    log_exception(f"MODEL_INIT_FAILED | error={error}")
    st.error(f"Không thể khởi tạo model: {error}")
    st.stop()

image_tab, text_tab = st.tabs(["🖼️ Image search", "✍️ Text search"])
with image_tab:
    render_image_search(recommender)
with text_tab:
    render_text_search(recommender)

st.divider()
with st.expander("Hệ thống hoạt động như thế nào?"):
    st.markdown(
        f"""
        **Image search:** ảnh tải lên → CLIP image encoder → cosine similarity →
        dự đoán category → so sánh No / Hard / Soft category.

        **Text search:** câu mô tả → CLIP text encoder → kết hợp điểm text→image
        và text→product text → Top-K sản phẩm.

        **Soft category:** `final score = visual similarity + {CATEGORY_BONUS:.2f}`
        nếu category trùng với category dự đoán.
        """
    )
