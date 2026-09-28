"""Standalone Streamlit UI for complete-outfit recommendation."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from outfit_recommendation.core import ClipQueryEncoder, OutfitIndex


load_dotenv(PROJECT_ROOT / ".env")


def project_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


DEFAULT_ARTIFACT_DIR = project_path(
    os.environ.get(
        "OUTFIT_ARTIFACT_DIR", "./outfit_recommendation/artifacts"
    )
)

st.set_page_config(
    page_title="Polyvore Outfit Recommendation",
    page_icon="👗",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def load_index(artifact_dir: str) -> OutfitIndex:
    return OutfitIndex(artifact_dir)


@st.cache_resource(show_spinner=False)
def load_encoder(model_source: str) -> ClipQueryEncoder:
    return ClipQueryEncoder(model_source)


def render_results(index: OutfitIndex, results: list[dict]) -> None:
    if not results:
        st.warning("Không tìm thấy outfit phù hợp trong index hiện tại.")
        return

    for rank, result in enumerate(results, start=1):
        st.subheader(
            f"Outfit #{rank} · ID {result['outfit_id']} · "
            f"anchor score {result['score']:.4f}"
        )
        items = result["items"]
        columns = st.columns(min(len(items), 6))
        for position, item in enumerate(items):
            with columns[position % len(columns)]:
                image_path = index.image_path(item)
                if image_path.exists():
                    st.image(str(image_path), width="stretch")
                else:
                    st.warning("Thiếu ảnh")
                if str(item["item_id"]) == result["matched_item_id"]:
                    st.markdown("**Món khớp với truy vấn**")
                st.caption(item.get("semantic_category", "unknown"))
                st.write(item.get("title") or f"Item {item['item_id']}")
        st.divider()


st.title("Complete Outfit Recommendation")
st.caption(
    "Prototype độc lập: truy vấn được dùng để tìm món neo, sau đó hệ thống trả về "
    "các outfit hoàn chỉnh đã được phối trong Polyvore."
)

with st.sidebar:
    st.header("Cấu hình")
    artifact_dir = st.text_input("Artifact directory", str(DEFAULT_ARTIFACT_DIR))
    top_k = st.slider("Số outfit", min_value=1, max_value=5, value=3)
    min_items = st.slider("Số món tối thiểu/outfit", min_value=2, max_value=6, value=3)
    st.info(
        "Đây là retrieval baseline theo outfit có thật, chưa phải compatibility model "
        "được huấn luyện bằng negative outfits."
    )

try:
    index = load_index(str(project_path(artifact_dir)))
except Exception as error:
    st.error(str(error))
    st.code(
        ".\\.venv\\Scripts\\python.exe -m outfit_recommendation.prepare_index "
        "--dataset-dir .\\datasets\\polyvore-outfits --max-outfits 1000",
        language="powershell",
    )
    st.stop()

summary = st.columns(3)
summary[0].metric("Indexed outfits", len(index.outfits))
summary[1].metric("Indexed items", len(index.items))
summary[2].metric("Embedding dimension", index.embedding_dimension)

image_tab, text_tab = st.tabs(["🖼️ Từ hình ảnh", "✍️ Từ mô tả"])

with image_tab:
    uploaded = st.file_uploader(
        "Tải ảnh một món thời trang làm điểm bắt đầu",
        type=("jpg", "jpeg", "png", "webp"),
    )
    query_image = None
    if uploaded is not None:
        try:
            query_image = Image.open(uploaded).convert("RGB")
            st.image(query_image, width=320, caption="Ảnh truy vấn")
        except Exception as error:
            st.error(f"Không đọc được ảnh: {error}")

    if st.button("Tạo outfit từ ảnh", type="primary", disabled=query_image is None):
        try:
            with st.spinner("Đang nạp CLIP và xếp hạng outfit..."):
                encoder = load_encoder(index.model_source)
                embedding = encoder.encode_image(query_image)
                image_results = index.rank(embedding, top_k=top_k, min_items=min_items)
            render_results(index, image_results)
        except Exception as error:
            st.error(f"Không thể tạo outfit: {error}")

with text_tab:
    query = st.text_area(
        "Mô tả món đồ hoặc phong cách mong muốn (nên dùng tiếng Anh)",
        placeholder="Example: a casual black jacket for autumn",
    )
    if st.button("Tạo outfit từ mô tả", type="primary"):
        if not query.strip():
            st.warning("Vui lòng nhập mô tả.")
        else:
            try:
                with st.spinner("Đang nạp CLIP và xếp hạng outfit..."):
                    encoder = load_encoder(index.model_source)
                    embedding = encoder.encode_text(query)
                    text_results = index.rank(
                        embedding, top_k=top_k, min_items=min_items
                    )
                render_results(index, text_results)
            except Exception as error:
                st.error(f"Không thể tạo outfit: {error}")

with st.expander("Prototype này hoạt động như thế nào?"):
    st.markdown(
        """
1. CLIP mã hóa ảnh hoặc mô tả đầu vào.
2. Hệ thống xác định món neo phù hợp trong index Polyvore.
3. Thay vì trả về các món tương tự, hệ thống lấy toàn bộ outfit có chứa món neo.
4. Các outfit được xếp hạng theo độ khớp của món neo với truy vấn.

Giai đoạn tiếp theo có thể huấn luyện compatibility model bằng positive/negative
outfits để hệ thống tự phối các món chưa từng nằm chung trong một set.
"""
    )
