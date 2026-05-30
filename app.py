"""
app.py
======
Giao diện trực quan (Streamlit) cho đồ án Nhận dạng Phương tiện Giao thông.
Cung cấp chức năng:
  - Tải ảnh lên (nhiều ảnh, quản lý bằng single horizontal row).
  - Áp dụng các bộ lọc tiền xử lý không gian (tùy chỉnh tham số trực tiếp).
  - So sánh trực quan ảnh gốc và ảnh đã lọc.
  - Đưa ảnh vào mô hình CNN đã huấn luyện để nhận diện.
"""

import sys
import os
from pathlib import Path

# Thêm thư mục src vào path để import các module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import base64
import io

import cv2
import numpy as np
from PIL import Image
import torch
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd

from data_prep import PIPELINE_CHAINS, apply_filter, resize_with_padding
import json
from dataset import get_transforms
from model import load_for_inference, SUPPORTED_MODELS

# ─────────────────────────────────────────────────────────────────────────────
# CẤU HÌNH GIAO DIỆN
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Nhận Dạng Phương Tiện Giao Thông",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Nhận Dạng Phương Tiện Giao Thông")
st.markdown("""
Hệ thống sử dụng các bộ lọc **Miền Không Gian (Spatial Domain Filters)** để tiền xử lý \
và mạng **CNN Transfer Learning** để phân loại phương tiện.
""")

# ─────────────────────────────────────────────────────────────────────────────
# HÀM TIỆN ÍCH
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model(model_name: str, checkpoint_path: str, num_classes: int):
    """Load model một lần duy nhất nhờ cơ chế cache của Streamlit."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(checkpoint_path):
        return None, device
    try:
        model = load_for_inference(checkpoint_path, num_classes=num_classes, device=device)
        return model, device
    except Exception as e:
        st.error(f"Lỗi khi load model: {e}")
        return None, device

@st.cache_data
def get_class_names(raw_dir: str = "data/raw"):
    """Tự động lấy tên lớp từ thư mục data/raw (nếu có)."""
    if os.path.exists(raw_dir):
        classes = sorted([d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))])
        if classes:
            return classes
    # Fallback mặc định
    return ["Bike", "Bus", "Car", "Motorbike", "Truck"]

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR - CẤU HÌNH
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Cấu Hình Hệ Thống")

    st.subheader("1. Mô hình AI")
    model_name = st.selectbox("Chọn Kiến trúc CNN:", SUPPORTED_MODELS, index=0)

    # Tìm kiếm checkpoint tốt nhất tương ứng
    default_ckpt = f"models/{model_name}_best.pth"
    checkpoint_path = st.text_input("Đường dẫn Checkpoint (.pth):", value=default_ckpt)

    class_names = get_class_names()
    num_classes = len(class_names)

    st.subheader("2. Chuỗi Tiền Xử Lý (Pipeline)")

    # --- ĐỌC CONFIG.JSON TỰ ĐỘNG ---
    config_path = checkpoint_path.replace(".pth", "_config.json")
    auto_pipeline = "baseline_v1"  # Mặc định
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                auto_pipeline = cfg.get("pipeline", "baseline_v1")
        except Exception:
            pass

    st.info(f"🔒 **Pipeline tự động khóa:**\n`{auto_pipeline}`\n\n(Dựa trên file cấu hình của trọng số)")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN AREA - IMAGE MANAGER + PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

THUMB_SIZE = 80  # px – kích thước thumbnail trong manager row


def image_to_base64(pil_img: Image.Image, max_size: int = THUMB_SIZE) -> str:
    """Chuyển PIL Image thành JPEG base64 (2× resolution để ảnh sắc nét)."""
    thumb = pil_img.copy()
    thumb.thumbnail((max_size * 2, max_size * 2), Image.LANCZOS)
    buf = io.BytesIO()
    thumb.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()


# ── Khởi tạo session state ────────────────────────────────────────────────────
if "uploaded_images" not in st.session_state:
    st.session_state["uploaded_images"] = []
if "selected_image_index" not in st.session_state:
    st.session_state["selected_image_index"] = -1

# ── Xử lý events từ JS qua query params ──────────────────────────────────────
# Luồng: JS đổi URL (location.search) → Streamlit re-run → đọc params → clear → re-run
_qp = st.query_params.to_dict()
_action = _qp.get("img_action", "")

if _action == "select":
    _i = int(_qp.get("img_idx", -1))
    if 0 <= _i < len(st.session_state["uploaded_images"]):
        st.session_state["selected_image_index"] = _i
    st.query_params.clear()
    st.rerun()

elif _action == "delete":
    _i = int(_qp.get("img_idx", -1))
    _lst = st.session_state["uploaded_images"]
    if 0 <= _i < len(_lst):
        _lst.pop(_i)
        _s = st.session_state["selected_image_index"]
        if len(_lst) == 0:
            st.session_state["selected_image_index"] = -1
        elif _s >= len(_lst):
            st.session_state["selected_image_index"] = len(_lst) - 1
        elif _s == _i and _i > 0:
            st.session_state["selected_image_index"] = _i - 1
    st.query_params.clear()
    st.rerun()

# ── File uploader ẩn – bị đẩy ra ngoài viewport, JS click để mở dialog ───────
# pointer-events: none → không bị click nhầm; JS tự click input[type=file] trực tiếp
st.markdown("""
<style>
div[data-testid="stFileUploader"] {
    position: fixed !important;
    top: -600px !important;
    left: -600px !important;
    width: 1px !important;
    height: 1px !important;
    overflow: hidden !important;
    opacity: 0 !important;
    pointer-events: none !important;
}
</style>
""", unsafe_allow_html=True)

_new_files = st.file_uploader(
    "Upload",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True,
    label_visibility="collapsed",
    key="img_uploader",
)

# Đồng bộ file mới vào session_state["uploaded_images"]
if _new_files:
    _exist_map = {img["name"]: img["size"] for img in st.session_state["uploaded_images"]}
    _added = False
    for _uf in _new_files:
        _raw = _uf.read()
        _uf.seek(0)
        # Bỏ qua nếu đã có ảnh cùng tên + kích thước (không thay đổi)
        if _uf.name in _exist_map and _exist_map[_uf.name] == len(_raw):
            continue
        _bytes = np.asarray(bytearray(_raw), dtype=np.uint8)
        _bgr = cv2.imdecode(_bytes, cv2.IMREAD_COLOR)
        if _bgr is None:
            continue
        _pil = Image.fromarray(cv2.cvtColor(_bgr, cv2.COLOR_BGR2RGB))
        _entry = {
            "name": _uf.name,
            "size": len(_raw),
            "bgr":  _bgr,
            "pil":  _pil,
            "b64":  image_to_base64(_pil, THUMB_SIZE),
        }
        # Thay thế nếu trùng tên (phiên bản mới hơn)
        _replaced = False
        for _j, _existing in enumerate(st.session_state["uploaded_images"]):
            if _existing["name"] == _uf.name:
                st.session_state["uploaded_images"][_j] = _entry
                _replaced = True
                break
        if not _replaced:
            st.session_state["uploaded_images"].append(_entry)
        _added = True

    if _added:
        # Tự động chọn ảnh vừa thêm (ảnh cuối cùng trong danh sách)
        st.session_state["selected_image_index"] = len(st.session_state["uploaded_images"]) - 1
        st.rerun()

# ── Render Single Horizontal Row ──────────────────────────────────────────────
_images  = st.session_state["uploaded_images"]
_sel_idx = st.session_state["selected_image_index"]
_S       = THUMB_SIZE

# Xây dựng HTML cho từng thumbnail
_thumbs_html = ""
for _i, _d in enumerate(_images):
    _cls = "selected" if _i == _sel_idx else ""
    _name_escaped = _d["name"].replace('"', "&quot;")
    _thumbs_html += f"""
    <div class="thumb-card {_cls}"
         onclick="handleSelect(event,{_i})"
         title="{_name_escaped}">
      <img src="data:image/jpeg;base64,{_d['b64']}"
           alt="{_name_escaped}" draggable="false"/>
      <div class="del-btn"
           onclick="handleDelete(event,{_i})"
           title="Xóa ảnh này">&#x2715;</div>
    </div>"""

_hint = (
    "" if _images
    else '<span class="hint-text">← Bấm + để thêm ảnh</span>'
)

_manager_html = f"""<!DOCTYPE html>
<html lang="vi"><head><meta charset="utf-8">
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{
  font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
  background:transparent;overflow:hidden;
}}

/* ── Container chính ── */
.row{{
  display:flex;
  flex-direction:row;
  overflow-x:auto;
  gap:12px;
  align-items:center;
  padding:10px 16px;
  border:1.5px solid #dde3ef;
  border-radius:14px;
  background:linear-gradient(135deg,#f8f9fc 0%,#eef1f8 100%);
  min-height:{_S + 24}px;
  width:100%;
}}
.row::-webkit-scrollbar{{height:5px}}
.row::-webkit-scrollbar-track{{background:#eaecf4;border-radius:4px}}
.row::-webkit-scrollbar-thumb{{background:#c0c6dc;border-radius:4px}}

/* ── Nút "+" thêm ảnh ── */
.add-btn{{
  flex:0 0 {_S}px;
  width:{_S}px;height:{_S}px;
  border:2px dashed #b4bdd4;
  border-radius:12px;
  display:flex;align-items:center;justify-content:center;
  cursor:pointer;
  font-size:32px;font-weight:200;
  color:#b4bdd4;background:#fff;
  transition:all .18s ease;
  user-select:none;flex-shrink:0;
}}
.add-btn:hover{{
  border-color:#1a73e8;color:#1a73e8;
  background:#e8f0fe;transform:scale(1.06);
}}
.add-btn:active{{transform:scale(.95)}}

/* ── Thumbnail card ── */
.thumb-card{{
  flex:0 0 {_S}px;
  position:relative;
  width:{_S}px;height:{_S}px;
  border-radius:12px;
  border:2.5px solid #dde3ef;
  cursor:pointer;flex-shrink:0;overflow:visible;
  transition:border-color .18s,box-shadow .18s,transform .18s;
  box-shadow:0 1px 4px rgba(0,0,0,.09);
}}
.thumb-card:hover{{
  border-color:#4a90d9;
  box-shadow:0 4px 14px rgba(74,144,217,.30);
  transform:scale(1.07);
}}
.thumb-card.selected{{
  border-color:#1a73e8;
  box-shadow:0 0 0 3.5px rgba(26,115,232,.28),
             0 4px 14px rgba(26,115,232,.24);
}}
.thumb-card img{{
  width:100%;height:100%;
  object-fit:cover;
  border-radius:9px;
  display:block;pointer-events:none;
}}

/* ── Nút xóa ✕ ── */
.del-btn{{
  position:absolute;top:-8px;right:-8px;
  width:22px;height:22px;
  background:#ef4444;border-radius:50%;
  display:flex;align-items:center;justify-content:center;
  font-size:12px;font-weight:800;
  color:#fff;cursor:pointer;z-index:10;
  box-shadow:0 2px 6px rgba(239,68,68,.45);
  transition:background .15s,transform .15s;
  user-select:none;line-height:1;
}}
.del-btn:hover{{background:#dc2626;transform:scale(1.22)}}

.hint-text{{
  font-size:13px;color:#b4bdd4;
  font-style:italic;padding-left:4px;white-space:nowrap;
}}
</style></head><body>
<div class="row">
  <div class="add-btn" onclick="triggerUpload()" title="Thêm ảnh mới">+</div>
  {_thumbs_html}
  {_hint}
</div>
<script>
/* Trigger file input ẩn trong parent Streamlit frame */
function triggerUpload() {{
  try {{
    var fi = window.parent.document.querySelectorAll('input[type="file"]');
    if (fi.length > 0) fi[0].click();
  }} catch(e) {{ console.warn('Upload trigger:', e); }}
}}

/* Chọn ảnh → đổi query param → Streamlit re-run */
function handleSelect(e, idx) {{
  if (e.target.classList.contains('del-btn')) return;
  window.parent.location.search = '?img_action=select&img_idx=' + idx;
}}

/* Xóa ảnh → confirm → đổi query param */
function handleDelete(e, idx) {{
  e.stopPropagation(); e.preventDefault();
  if (!confirm('Xóa ảnh này khỏi danh sách?')) return;
  window.parent.location.search = '?img_action=delete&img_idx=' + idx;
}}
</script>
</body></html>"""

components.html(_manager_html, height=_S + 42, scrolling=False)

# ── Khu vực xử lý & hiển thị kết quả ────────────────────────────────────────
if _images and 0 <= _sel_idx < len(_images):
    _chosen = _images[_sel_idx]
    _bgr    = _chosen["bgr"]
    _rgb    = np.array(_chosen["pil"])

    st.markdown(f"---\n**Đang xem:** `{_chosen['name']}`")

    _c1, _c2 = st.columns(2)
    with _c1:
        st.subheader("Ảnh gốc")
        st.image(_rgb, use_container_width=True)
        st.caption(f"Kích thước ban đầu: {_rgb.shape[1]}×{_rgb.shape[0]}")

    # Tiền xử lý: Resize & Pad + áp lọc pipeline
    _cur = resize_with_padding(_bgr, 224)
    for _step in PIPELINE_CHAINS.get(auto_pipeline, []):
        _cur = apply_filter(_cur, _step["name"], **_step.get("kwargs", {}))
    _filtered = cv2.cvtColor(_cur, cv2.COLOR_BGR2RGB)

    with _c2:
        st.subheader("Sau khi Tiền Xử Lý")
        st.image(_filtered, use_container_width=True)
        st.caption(f"Kích thước 224×224 | Pipeline: `{auto_pipeline}`")

    st.write("---")

    # AI Inference
    st.subheader("Kết quả Dự Đoán")
    _model, _device = load_model(model_name, checkpoint_path, num_classes)

    if _model is None:
        st.warning(f"Không tìm thấy file trọng số tại `{checkpoint_path}`.")
        st.info(
            "Vui lòng chạy `python src/train.py` để huấn luyện mô hình trước, "
            "hoặc chỉnh sửa đường dẫn ở thanh bên trái."
        )
    else:
        with st.spinner("Đang phân tích ảnh..."):
            _pil = Image.fromarray(_filtered)
            _tfm = get_transforms("test", img_size=224)
            _t   = _tfm(_pil).unsqueeze(0).to(_device)

            with torch.no_grad():
                _out  = _model(_t)
                _prob = torch.nn.functional.softmax(_out, dim=1)[0]

            _vals   = _prob.cpu().numpy() * 100
            _bidx   = int(np.argmax(_vals))
            _bclass = class_names[_bidx]
            _bconf  = _vals[_bidx]

            st.success(f"**Kết quả:** {_bclass} &nbsp;·&nbsp; Độ tin cậy: **{_bconf:.1f}%**")

            _df = pd.DataFrame({
                "Loại phương tiện": class_names,
                "Tỷ lệ (%)": _vals,
            }).set_index("Loại phương tiện")
            st.bar_chart(_df, height=300)

    with st.expander("Xem cấu hình Pipeline đang áp dụng"):
        st.json(PIPELINE_CHAINS.get(auto_pipeline, []))

else:
    # Màn hình chờ – chưa có ảnh hoặc chưa chọn ảnh
    st.info("Bấm **＋** để tải ảnh xe cộ lên và bắt đầu nhận diện.")
    st.markdown("""
    ### Hướng dẫn sử dụng:
    1. Chuẩn bị ảnh các loại xe (Bike, Bus, Car, Motorbike, Truck).
    2. Nếu bạn chưa train mô hình, hãy mở Terminal và chạy: `python src/train.py`
    3. Click **＋** để thêm ảnh · Click thumbnail để xem kết quả · Click **✕** để xóa ảnh.
    """)
