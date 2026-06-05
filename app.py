"""
app.py
======
Giao diện Web Demo cho đồ án
Nhận dạng Phương tiện Giao thông (Vehicle Type Recognition).

Tính năng:
    - Upload ảnh và nhận kết quả nhận dạng real-time.
    - Chọn mô hình (ResNet-50 / ViT-B/16) ngay trên sidebar.
    - Hiển thị Top-5 predictions dưới dạng biểu đồ cột màu sắc.
    - Xem lịch sử huấn luyện (Loss / Accuracy curves) từ JSON.
    - Thông tin về dự án và bộ dataset Vehicle-10.

Cách chạy:
    streamlit run app.py

Yêu cầu:
    pip install streamlit torch torchvision pillow plotly
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import v2

# ── Thêm project root vào path ────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# HẰNG SỐ & CẤU HÌNH
# ─────────────────────────────────────────────────────────────────────────────

CLASS_NAMES: List[str] = [
    "bicycle", "boat", "bus", "car", "helicopter",
    "minibus", "motorcycle", "taxi", "train", "truck",
]

CLASS_VI: Dict[str, str] = {
    "bicycle":    "Xe đạp",
    "boat":       "Thuyền / Tàu",
    "bus":        "Xe buýt",
    "car":        "Ô tô",
    "helicopter": "Trực thăng",
    "minibus":    "Xe khách nhỏ",
    "motorcycle": "Xe mô tô",
    "taxi":       "Taxi",
    "train":      "Tàu hỏa",
    "truck":      "Xe tải",
}

CLASS_EMOJI: Dict[str, str] = {
    "bicycle":    "🚲",
    "boat":       "⛵",
    "bus":        "🚌",
    "car":        "🚗",
    "helicopter": "🚁",
    "minibus":    "🚐",
    "motorcycle": "🏍️",
    "taxi":       "🚕",
    "train":      "🚂",
    "truck":      "🚛",
}

# Màu gradient cho từng class (Plotly)
CLASS_COLORS: Dict[str, str] = {
    "bicycle":    "#4ade80",
    "boat":       "#38bdf8",
    "bus":        "#f97316",
    "car":        "#a78bfa",
    "helicopter": "#f43f5e",
    "minibus":    "#fb923c",
    "motorcycle": "#facc15",
    "taxi":       "#fbbf24",
    "train":      "#60a5fa",
    "truck":      "#34d399",
}

MODEL_DISPLAY: Dict[str, str] = {
    "resnet50": "ResNet-50  (~24.5M params)",
    "vit":      "ViT-B/16   (~86M params)",
}

MODEL_FULL_NAME: Dict[str, str] = {
    "resnet50": "resnet50",
    "vit":      "vit_base_patch16_224",
}

CHECKPOINT_DIR = _ROOT / "models"
OUTPUT_DIR     = _ROOT / "outputs"
NUM_CLASSES    = 10

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG — Phải gọi TRƯỚC mọi lệnh st khác
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title = "Vehicle Type Recognition",
    page_icon  = "🚗",
    layout     = "wide",
    initial_sidebar_state = "expanded",
    menu_items = {
        "About": "Đồ án Nhận dạng Phương tiện Giao thông — Deep Learning @ UTE",
    },
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS — Dark Premium Theme
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Root Variables ── */
:root {
    --bg-primary:     #0f1117;
    --bg-card:        #1a1d27;
    --bg-card2:       #1e2130;
    --accent:         #6366f1;
    --accent-light:   #818cf8;
    --accent-glow:    rgba(99,102,241,0.25);
    --success:        #10b981;
    --warning:        #f59e0b;
    --danger:         #ef4444;
    --text-primary:   #f1f5f9;
    --text-secondary: #94a3b8;
    --border:         rgba(255,255,255,0.08);
    --radius:         16px;
    --radius-sm:      10px;
}

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Main container ── */
.block-container {
    padding: 1.5rem 2rem 3rem 2rem !important;
    max-width: 1400px !important;
}

/* ── Hero Banner ── */
.hero-banner {
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 40%, #1e3a5f 100%);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: var(--radius);
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 20px 60px rgba(0,0,0,0.4);
}
.hero-banner::before {
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(ellipse at 70% 50%, rgba(99,102,241,0.15) 0%, transparent 60%);
    pointer-events: none;
}
.hero-title {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #e0e7ff, #818cf8, #6366f1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.4rem 0;
    line-height: 1.2;
}
.hero-subtitle {
    color: #a5b4fc;
    font-size: 1.05rem;
    font-weight: 400;
    margin: 0;
}
.hero-badge {
    display: inline-block;
    background: rgba(99,102,241,0.2);
    border: 1px solid rgba(99,102,241,0.4);
    color: #a5b4fc;
    border-radius: 999px;
    padding: 0.25rem 0.85rem;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}

/* ── Glass Card ── */
.glass-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.5rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(10px);
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
}
.glass-card:hover {
    border-color: rgba(99,102,241,0.3);
    box-shadow: 0 8px 32px rgba(99,102,241,0.1);
}

/* ── Section Header ── */
.section-header {
    display: flex; align-items: center; gap: 0.6rem;
    font-size: 1.1rem; font-weight: 700;
    color: var(--text-primary);
    margin: 0 0 1.2rem 0;
    padding-bottom: 0.7rem;
    border-bottom: 1px solid var(--border);
}
.section-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--accent);
    box-shadow: 0 0 8px var(--accent);
    flex-shrink: 0;
}

/* ── Prediction Result Card ── */
.pred-card {
    background: linear-gradient(135deg, rgba(99,102,241,0.12), rgba(129,140,248,0.06));
    border: 1.5px solid rgba(99,102,241,0.4);
    border-radius: var(--radius);
    padding: 1.8rem 2rem;
    text-align: center;
    animation: fadeInUp 0.4s ease;
}
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0);    }
}
.pred-emoji { font-size: 4rem; display: block; margin-bottom: 0.6rem; }
.pred-label {
    font-size: 2rem; font-weight: 800;
    background: linear-gradient(135deg, #e0e7ff, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}
.pred-vi { color: var(--text-secondary); font-size: 1rem; margin-bottom: 1rem; }
.pred-confidence {
    font-size: 2.8rem; font-weight: 800;
    color: #10b981;
    line-height: 1;
}
.pred-conf-label { color: var(--text-secondary); font-size: 0.85rem; margin-top: 0.3rem; }
.confidence-bar-bg {
    background: rgba(255,255,255,0.07);
    border-radius: 999px;
    height: 8px; margin-top: 1rem;
    overflow: hidden;
}
.confidence-bar-fill {
    height: 100%; border-radius: 999px;
    background: linear-gradient(90deg, #6366f1, #10b981);
    transition: width 0.6s ease;
}

/* ── Metric Chip ── */
.metric-row { display: flex; gap: 0.8rem; flex-wrap: wrap; margin-bottom: 1.2rem; }
.metric-chip {
    background: rgba(99,102,241,0.1);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: var(--radius-sm);
    padding: 0.55rem 1rem;
    display: flex; flex-direction: column; align-items: center;
    min-width: 90px;
}
.metric-chip .value { font-size: 1.3rem; font-weight: 700; color: var(--accent-light); }
.metric-chip .label { font-size: 0.72rem; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.05em; }

/* ── Upload Zone ── */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.03) !important;
    border: 2px dashed rgba(99,102,241,0.35) !important;
    border-radius: var(--radius) !important;
    transition: all 0.3s ease !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(99,102,241,0.7) !important;
    background: rgba(99,102,241,0.05) !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #13151f !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .block-container { padding: 1.2rem 1rem !important; }

/* ── Streamlit Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #4f46e5) !important;
    color: white !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    font-weight: 600 !important;
    padding: 0.6rem 1.4rem !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 16px rgba(99,102,241,0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(99,102,241,0.45) !important;
}

/* ── Selectbox / Radio ── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stRadio"] label {
    color: var(--text-primary) !important;
}

/* ── Info Box ── */
.info-box {
    background: rgba(59,130,246,0.08);
    border-left: 3px solid #3b82f6;
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
    font-size: 0.88rem;
    color: #93c5fd;
}
.warn-box {
    background: rgba(245,158,11,0.08);
    border-left: 3px solid #f59e0b;
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
    font-size: 0.88rem;
    color: #fcd34d;
}

/* ── Tab styling ── */
[data-testid="stTabs"] [data-baseweb="tab"] {
    font-weight: 600 !important;
    color: var(--text-secondary) !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    color: var(--accent-light) !important;
    border-bottom-color: var(--accent) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.4); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HELPER — TRANSFORM ẢNH
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_inference_transform() -> v2.Compose:
    """Transform chuẩn hoá ảnh đầu vào cho inference."""
    return v2.Compose([
        v2.Resize((224, 224), interpolation=v2.InterpolationMode.BICUBIC),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# HELPER — NẠP MÔ HÌNH (cached theo tên file checkpoint)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model_cached(checkpoint_path: str, model_key: str) -> Optional[torch.nn.Module]:
    """
    Nạp mô hình từ file .pth và cache lại (tránh reload mỗi lần predict).

    Args:
        checkpoint_path : Đường dẫn tuyệt đối tới file .pth.
        model_key       : "resnet50" hoặc "vit".

    Returns:
        nn.Module ở chế độ eval(), hoặc None nếu lỗi.
    """
    try:
        from src.model import load_for_inference
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model  = load_for_inference(
            checkpoint_path = checkpoint_path,
            num_classes     = NUM_CLASSES,
            device          = device,
        )
        return model
    except Exception as exc:
        st.error(f"❌ Lỗi nạp mô hình: {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# HELPER — DỰ ĐOÁN
# ─────────────────────────────────────────────────────────────────────────────

def predict(
    model:     torch.nn.Module,
    image:     Image.Image,
    top_k:     int = 5,
) -> Tuple[List[str], List[float], float]:
    """
    Chạy inference và trả về Top-K predictions.

    Args:
        model  : Mô hình đã nạp.
        image  : PIL Image RGB.
        top_k  : Số lượng class trả về.

    Returns:
        Tuple (class_names, confidences, inference_ms):
            class_names  : List tên class theo thứ tự confidence giảm dần.
            confidences  : List xác suất (0.0 → 1.0) tương ứng.
            inference_ms : Thời gian inference (milliseconds).
    """
    device    = next(model.parameters()).device
    transform = get_inference_transform()

    # Chuẩn bị tensor
    img_rgb = image.convert("RGB")
    tensor  = transform(img_rgb).unsqueeze(0).to(device)   # (1, 3, 224, 224)

    # Inference
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(tensor)                             # (1, num_classes)
        probs  = F.softmax(logits, dim=1).squeeze(0)      # (num_classes,)
    inference_ms = (time.perf_counter() - t0) * 1000

    # Top-K
    top_probs, top_idxs = probs.topk(min(top_k, NUM_CLASSES))
    top_classes = [CLASS_NAMES[i] for i in top_idxs.cpu().tolist()]
    top_conf    = top_probs.cpu().tolist()

    return top_classes, top_conf, inference_ms


# ─────────────────────────────────────────────────────────────────────────────
# HELPER — ĐỌC LỊCH SỬ TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def load_history(model_key: str) -> Optional[List[Dict]]:
    """Đọc file history_<model>.json từ thư mục outputs/."""
    path = OUTPUT_DIR / f"history_{model_key}.json"
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# COMPONENT — SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar() -> Tuple[str, int]:
    """
    Render sidebar và trả về (model_key, top_k).
    """
    with st.sidebar:
        # ── Logo / Title ──────────────────────────────────────────────────
        st.markdown("""
        <div style='text-align:center; padding: 0.5rem 0 1.5rem 0;'>
            <div style='font-size:3rem;'>🚗</div>
            <div style='font-size:1.1rem; font-weight:800;
                        background: linear-gradient(135deg,#818cf8,#6366f1);
                        -webkit-background-clip:text;
                        -webkit-text-fill-color:transparent;'>
                VehicleVision
            </div>
            <div style='font-size:0.75rem; color:#64748b; margin-top:0.2rem;'>
                Deep Learning Demo v2.0
            </div>
        </div>
        <hr style='border-color:rgba(255,255,255,0.07); margin-bottom:1.2rem;'>
        """, unsafe_allow_html=True)

        # ── Chọn mô hình ──────────────────────────────────────────────────
        st.markdown("**🤖 Chọn Mô Hình**")
        model_key = st.selectbox(
            label     = "Architecture",
            options   = list(MODEL_DISPLAY.keys()),
            format_func = lambda k: MODEL_DISPLAY[k],
            label_visibility = "collapsed",
        )

        # Trạng thái checkpoint
        ckpt_path = CHECKPOINT_DIR / f"{MODEL_FULL_NAME[model_key]}_best.pth"
        if ckpt_path.exists():
            size_mb = ckpt_path.stat().st_size / 1e6
            st.markdown(
                f'<div class="info-box">✅ Checkpoint: <b>{ckpt_path.name}</b><br>'
                f'Kích thước: {size_mb:.1f} MB</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="warn-box">⚠️ Chưa có checkpoint<br>'
                f'<code>{ckpt_path.name}</code><br>'
                f'Hãy chạy <code>train.py</code> trước.</div>',
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Top-K ─────────────────────────────────────────────────────────
        st.markdown("**🎯 Hiển thị Top-K**")
        top_k = st.slider(
            "Số lượng dự đoán", min_value=3, max_value=10,
            value=5, label_visibility="collapsed",
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Device info ───────────────────────────────────────────────────
        st.markdown("**⚡ Thiết bị**")
        device_str = "CUDA 🟢" if torch.cuda.is_available() else "CPU 🔵"
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram     = torch.cuda.get_device_properties(0).total_memory / 1e9
            st.markdown(
                f'<div class="info-box">{device_str}<br>'
                f'<b>{gpu_name}</b><br>{vram:.1f} GB VRAM</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="info-box">{device_str}<br>Không có GPU khả dụng.</div>',
                unsafe_allow_html=True,
            )

        # ── About ─────────────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style='color:#475569; font-size:0.78rem; text-align:center; line-height:1.6;'>
            Đồ án Nhận dạng Phương tiện<br>
            <b>Vehicle-10 Dataset</b> — 10 lớp<br>
            ResNet-50 &amp; ViT-B/16<br>
            <span style='color:#6366f1;'>Deep Learning @ UTE</span>
        </div>
        """, unsafe_allow_html=True)

    return model_key, top_k


# ─────────────────────────────────────────────────────────────────────────────
# COMPONENT — HERO BANNER
# ─────────────────────────────────────────────────────────────────────────────

def render_hero() -> None:
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-badge">🎓 Đồ Án Deep Learning</div>
        <h1 class="hero-title">Vehicle Type Recognition</h1>
        <p class="hero-subtitle">
            Nhận dạng 10 loại phương tiện giao thông với ResNet-50 &amp; Vision Transformer (ViT-B/16).<br>
            Upload ảnh để xem mô hình phân loại real-time.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# COMPONENT — TAB PREDICT
# ─────────────────────────────────────────────────────────────────────────────

def render_predict_tab(model_key: str, top_k: int) -> None:
    """Tab dự đoán chính: upload ảnh → inference → hiển thị kết quả."""
    import plotly.graph_objects as go

    col_upload, col_result = st.columns([1, 1], gap="large")

    # ── Cột trái: Upload ─────────────────────────────────────────────────────
    with col_upload:
        st.markdown("""
        <div class="section-header">
            <div class="section-dot"></div> Upload Ảnh
        </div>
        """, unsafe_allow_html=True)

        uploaded = st.file_uploader(
            label       = "Kéo thả hoặc click để chọn ảnh",
            type        = ["jpg", "jpeg", "png", "bmp", "webp"],
            label_visibility = "visible",
            key         = "uploader",
        )

        if uploaded:
            image = Image.open(uploaded).convert("RGB")
            st.image(
                image,
                caption = f"📷 {uploaded.name}  ({image.width}×{image.height}px)",
                use_container_width = True,
            )
            st.markdown(
                f'<div class="info-box">📁 File: <b>{uploaded.name}</b> '
                f'({uploaded.size/1024:.1f} KB)</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown("""
            <div style='background:rgba(255,255,255,0.02); border:2px dashed rgba(99,102,241,0.2);
                        border-radius:12px; padding:3rem 2rem; text-align:center; color:#475569;'>
                <div style='font-size:3rem; margin-bottom:0.8rem;'>📸</div>
                <div style='font-weight:600; margin-bottom:0.4rem; color:#64748b;'>
                    Chưa có ảnh
                </div>
                <div style='font-size:0.85rem;'>
                    Hỗ trợ: JPG, PNG, BMP, WEBP
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Cột phải: Kết quả ────────────────────────────────────────────────────
    with col_result:
        st.markdown("""
        <div class="section-header">
            <div class="section-dot"></div> Kết Quả Dự Đoán
        </div>
        """, unsafe_allow_html=True)

        if not uploaded:
            st.markdown("""
            <div style='background:rgba(255,255,255,0.02); border-radius:12px;
                        padding:3rem 2rem; text-align:center; color:#475569; height:100%;'>
                <div style='font-size:3rem; margin-bottom:0.8rem;'>🔍</div>
                <div style='color:#64748b; font-size:0.9rem;'>
                    Upload ảnh bên trái để xem kết quả nhận dạng
                </div>
            </div>
            """, unsafe_allow_html=True)
            return

        # Nạp mô hình
        ckpt_path = CHECKPOINT_DIR / f"{MODEL_FULL_NAME[model_key]}_best.pth"
        if not ckpt_path.exists():
            st.markdown(
                f'<div class="warn-box">⚠️ Không tìm thấy checkpoint:<br>'
                f'<code>{ckpt_path}</code><br>'
                f'Hãy huấn luyện mô hình trước bằng <code>python src/train.py</code>.</div>',
                unsafe_allow_html=True,
            )
            return

        with st.spinner("⚡ Đang nạp mô hình..."):
            model = load_model_cached(str(ckpt_path), model_key)

        if model is None:
            return

        # Inference
        with st.spinner("🔍 Đang phân tích ảnh..."):
            classes, confs, ms = predict(model, image, top_k=top_k)

        top_class = classes[0]
        top_conf  = confs[0]
        emoji     = CLASS_EMOJI.get(top_class, "🚗")
        vi_name   = CLASS_VI.get(top_class, top_class)

        # ── Prediction Card ───────────────────────────────────────────────
        st.markdown(f"""
        <div class="pred-card">
            <span class="pred-emoji">{emoji}</span>
            <div class="pred-label">{top_class.upper()}</div>
            <div class="pred-vi">{vi_name}</div>
            <div class="pred-confidence">{top_conf*100:.1f}%</div>
            <div class="pred-conf-label">Độ tự tin (Confidence)</div>
            <div class="confidence-bar-bg">
                <div class="confidence-bar-fill" style="width:{top_conf*100:.1f}%"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Metric chips ──────────────────────────────────────────────────
        st.markdown(f"""
        <div class="metric-row" style="margin-top:1rem;">
            <div class="metric-chip">
                <div class="value">{ms:.0f}ms</div>
                <div class="label">Inference</div>
            </div>
            <div class="metric-chip">
                <div class="value">Top-{top_k}</div>
                <div class="label">Predictions</div>
            </div>
            <div class="metric-chip">
                <div class="value">{MODEL_DISPLAY[model_key].split()[0]}</div>
                <div class="label">Model</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Top-K Bar Chart (Plotly) ───────────────────────────────────────
        colors_list = [CLASS_COLORS.get(c, "#6366f1") for c in classes]
        labels_disp = [
            f"{CLASS_EMOJI.get(c,'')}{c}<br>{CLASS_VI.get(c,'')}"
            for c in classes
        ]

        fig = go.Figure(go.Bar(
            x              = [v * 100 for v in confs],
            y              = labels_disp,
            orientation    = "h",
            marker_color   = colors_list,
            marker_line_width = 0,
            text           = [f"{v*100:.1f}%" for v in confs],
            textposition   = "outside",
            textfont       = dict(color="#e2e8f0", size=13, family="Inter"),
        ))

        fig.update_layout(
            height          = max(280, top_k * 52),
            margin          = dict(l=10, r=60, t=10, b=10),
            paper_bgcolor   = "rgba(0,0,0,0)",
            plot_bgcolor    = "rgba(0,0,0,0)",
            xaxis = dict(
                range      = [0, 105],
                showgrid   = False,
                showticklabels = False,
                zeroline   = False,
            ),
            yaxis = dict(
                autorange  = "reversed",
                showgrid   = False,
                tickfont   = dict(color="#94a3b8", size=12, family="Inter"),
            ),
            showlegend = False,
        )

        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ─────────────────────────────────────────────────────────────────────────────
# COMPONENT — TAB TRAINING HISTORY
# ─────────────────────────────────────────────────────────────────────────────

def render_history_tab(model_key: str) -> None:
    """Tab lịch sử huấn luyện: đọc JSON và vẽ biểu đồ Loss / Accuracy."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    st.markdown("""
    <div class="section-header">
        <div class="section-dot"></div> Lịch Sử Huấn Luyện
    </div>
    """, unsafe_allow_html=True)

    history = load_history(model_key)

    if history is None:
        history_path = OUTPUT_DIR / f"history_{model_key}.json"
        st.markdown(
            f'<div class="warn-box">⚠️ Chưa tìm thấy file lịch sử huấn luyện:<br>'
            f'<code>{history_path}</code><br>'
            f'Hãy chạy <code>python src/train.py --model {model_key}</code> để tạo.</div>',
            unsafe_allow_html=True,
        )
        return

    epochs     = [r["epoch"]      for r in history]
    train_loss = [r["train_loss"] for r in history]
    val_loss   = [r["val_loss"]   for r in history]
    val_acc    = [r.get("val_acc", 0) * 100 for r in history]
    phases     = [r.get("phase", "") for r in history]

    # Tìm điểm chuyển phase
    phase_change_epoch: Optional[int] = None
    for i in range(1, len(phases)):
        if phases[i] != phases[i - 1]:
            phase_change_epoch = epochs[i]
            break

    # Best epoch
    best_idx    = val_loss.index(min(val_loss))
    best_epoch  = epochs[best_idx]
    best_vl     = val_loss[best_idx]
    best_acc    = val_acc[best_idx]

    # ── Summary chips ─────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-chip">
            <div class="value">{len(history)}</div>
            <div class="label">Epochs</div>
        </div>
        <div class="metric-chip">
            <div class="value">{best_epoch}</div>
            <div class="label">Best Epoch</div>
        </div>
        <div class="metric-chip">
            <div class="value">{best_vl:.4f}</div>
            <div class="label">Best Val Loss</div>
        </div>
        <div class="metric-chip">
            <div class="value">{best_acc:.1f}%</div>
            <div class="label">Best Val Acc</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Biểu đồ Loss + Accuracy ───────────────────────────────────────────
    fig = make_subplots(
        rows       = 1,
        cols       = 2,
        subplot_titles = ("Loss (Train vs Validation)", "Validation Accuracy (%)"),
        horizontal_spacing = 0.08,
    )

    # Loss
    fig.add_trace(go.Scatter(
        x=epochs, y=train_loss, name="Train Loss",
        line=dict(color="#6366f1", width=2.5),
        mode="lines+markers",
        marker=dict(size=4),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=epochs, y=val_loss, name="Val Loss",
        line=dict(color="#f43f5e", width=2.5),
        mode="lines+markers",
        marker=dict(size=4),
    ), row=1, col=1)

    # Best point marker
    fig.add_trace(go.Scatter(
        x=[best_epoch], y=[best_vl],
        name="Best Val Loss",
        mode="markers",
        marker=dict(color="#10b981", size=12, symbol="star",
                    line=dict(color="white", width=1.5)),
    ), row=1, col=1)

    # Accuracy
    fig.add_trace(go.Scatter(
        x=epochs, y=val_acc, name="Val Acc (%)",
        line=dict(color="#10b981", width=2.5),
        fill="tozeroy",
        fillcolor="rgba(16,185,129,0.08)",
        mode="lines+markers",
        marker=dict(size=4),
    ), row=1, col=2)

    # Đường phase change
    if phase_change_epoch:
        for col_idx in [1, 2]:
            fig.add_vline(
                x=phase_change_epoch - 0.5,
                line=dict(color="#f59e0b", dash="dash", width=1.5),
                annotation_text="Phase 2",
                annotation_font_color="#f59e0b",
                annotation_font_size=11,
                row=1, col=col_idx,
            )

    fig.update_layout(
        height         = 380,
        paper_bgcolor  = "rgba(0,0,0,0)",
        plot_bgcolor   = "rgba(0,0,0,0)",
        font           = dict(family="Inter", color="#94a3b8", size=12),
        legend         = dict(
            bgcolor      = "rgba(26,29,39,0.9)",
            bordercolor  = "rgba(255,255,255,0.1)",
            borderwidth  = 1,
            font         = dict(color="#e2e8f0", size=11),
        ),
        margin = dict(l=10, r=10, t=40, b=10),
    )
    fig.update_xaxes(
        gridcolor = "rgba(255,255,255,0.05)",
        tickfont  = dict(color="#64748b"),
        title_font = dict(color="#94a3b8"),
        title_text = "Epoch",
    )
    fig.update_yaxes(
        gridcolor = "rgba(255,255,255,0.05)",
        tickfont  = dict(color="#64748b"),
    )
    for annotation in fig.layout.annotations:
        annotation.font.color = "#94a3b8"

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ── Bảng lịch sử ─────────────────────────────────────────────────────
    with st.expander("📋 Xem bảng lịch sử chi tiết", expanded=False):
        import pandas as pd
        df = pd.DataFrame(history)
        df["val_acc"] = (df["val_acc"] * 100).round(2)
        df.columns = [c.replace("_", " ").title() for c in df.columns]
        st.dataframe(
            df,
            use_container_width = True,
            hide_index          = True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# COMPONENT — TAB DATASET INFO
# ─────────────────────────────────────────────────────────────────────────────

def render_dataset_tab() -> None:
    """Tab thông tin bộ dữ liệu Vehicle-10."""
    import plotly.graph_objects as go

    st.markdown("""
    <div class="section-header">
        <div class="section-dot"></div> Bộ Dữ Liệu Vehicle-10
    </div>
    """, unsafe_allow_html=True)

    # Dataset stats
    dataset_info = {
        "bicycle":    (1296, 144, 162),
        "boat":       (7117, 790, 890),
        "bus":        (3252, 361, 406),
        "car":        (6832, 759, 854),
        "helicopter": (534,  59,  67),
        "minibus":    (1181, 131, 148),
        "motorcycle": (3550, 394, 444),
        "taxi":       (726,  81,  91),
        "train":      (1346, 150, 169),
        "truck":      (2971, 330, 371),
    }

    # Metric tổng
    total_train = sum(v[0] for v in dataset_info.values())
    total_valid = sum(v[1] for v in dataset_info.values())
    total_test  = sum(v[2] for v in dataset_info.values())

    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-chip">
            <div class="value">10</div>
            <div class="label">Lớp / Classes</div>
        </div>
        <div class="metric-chip">
            <div class="value">{total_train:,}</div>
            <div class="label">Train Images</div>
        </div>
        <div class="metric-chip">
            <div class="value">{total_valid:,}</div>
            <div class="label">Valid Images</div>
        </div>
        <div class="metric-chip">
            <div class="value">{total_test:,}</div>
            <div class="label">Test Images</div>
        </div>
        <div class="metric-chip">
            <div class="value">224px</div>
            <div class="label">Image Size</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Biểu đồ phân phối
    classes_list = list(dataset_info.keys())
    train_counts = [dataset_info[c][0] for c in classes_list]
    colors_bar   = [CLASS_COLORS.get(c, "#6366f1") for c in classes_list]
    labels_chart = [f"{CLASS_EMOJI.get(c,'')} {c}" for c in classes_list]

    fig = go.Figure(go.Bar(
        x            = labels_chart,
        y            = train_counts,
        marker_color = colors_bar,
        marker_line_width = 0,
        text         = train_counts,
        textposition = "outside",
        textfont     = dict(color="#94a3b8", size=11),
    ))
    fig.update_layout(
        title          = dict(text="Phân phối số lượng ảnh Train theo lớp",
                              font=dict(color="#e2e8f0", size=14)),
        height         = 340,
        paper_bgcolor  = "rgba(0,0,0,0)",
        plot_bgcolor   = "rgba(0,0,0,0)",
        margin         = dict(l=10, r=10, t=50, b=10),
        xaxis          = dict(tickfont=dict(color="#94a3b8", size=12), showgrid=False),
        yaxis          = dict(gridcolor="rgba(255,255,255,0.05)",
                              tickfont=dict(color="#64748b")),
        showlegend     = False,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Bảng chi tiết class
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("**📊 Chi tiết từng lớp**")
        for cls, (tr, va, te) in dataset_info.items():
            emoji = CLASS_EMOJI.get(cls, "")
            vi    = CLASS_VI.get(cls, cls)
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;"
                f"align-items:center;padding:0.35rem 0;border-bottom:1px solid rgba(255,255,255,0.05);'>"
                f"<span>{emoji} <b style='color:#e2e8f0;'>{cls}</b>"
                f" <span style='color:#64748b;font-size:0.8rem;'>({vi})</span></span>"
                f"<span style='color:#818cf8;font-weight:600;'>{tr:,}</span></div>",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("**🔧 Kỹ thuật xử lý dữ liệu**")

        techniques = [
            ("Split First Protocol", "Chia 85%-5%-10% TRƯỚC khi augment → chống Data Leakage"),
            ("Class Balancing", "Nhân bản class thiểu số lên ~7,117 ảnh (class boat)"),
            ("Offline Augment ×4", "4 pipelines: Base / Night / Rain / Sun"),
            ("MixUp (α=0.2)", "Trộn tuyến tính 2 ảnh → soft-labels, giảm overfitting"),
            ("CutMix (α=1.0)", "Cắt-dán vùng giữa 2 ảnh → học toàn cấu trúc xe"),
            ("Normalize ImageNet", "mean=[.485,.456,.406], std=[.229,.224,.225]"),
        ]
        for name, desc in techniques:
            st.markdown(
                f"<div style='margin-bottom:0.7rem;'>"
                f"<div style='color:#818cf8;font-weight:600;font-size:0.9rem;'>✦ {name}</div>"
                f"<div style='color:#64748b;font-size:0.82rem;margin-top:0.2rem;'>{desc}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# COMPONENT — TAB MODEL INFO
# ─────────────────────────────────────────────────────────────────────────────

def render_model_tab(model_key: str) -> None:
    """Tab thông tin kiến trúc mô hình và chiến lược huấn luyện."""
    st.markdown("""
    <div class="section-header">
        <div class="section-dot"></div> Kiến Trúc & Chiến Lược Huấn Luyện
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        # ResNet-50
        active = "border-color:rgba(99,102,241,0.6);" if model_key == "resnet50" else ""
        st.markdown(f"""
        <div class="glass-card" style="{active}">
            <div style="font-size:1.4rem;font-weight:800;color:#818cf8;margin-bottom:0.5rem;">
                🏗️ ResNet-50
            </div>
            <div class="metric-row">
                <div class="metric-chip"><div class="value">24.5M</div><div class="label">Params</div></div>
                <div class="metric-chip"><div class="value">CNN</div><div class="label">Type</div></div>
                <div class="metric-chip"><div class="value">2048</div><div class="label">Feature dim</div></div>
            </div>
            <div style="color:#64748b;font-size:0.85rem;line-height:1.7;">
                <b style="color:#94a3b8;">Kiến trúc:</b> 4 nhóm Residual Block với Skip Connections.<br>
                <b style="color:#94a3b8;">Head:</b> Linear(2048 → 10) thay thế FC gốc.<br>
                <b style="color:#94a3b8;">Phase 1:</b> Freeze backbone, chỉ train FC.<br>
                <b style="color:#94a3b8;">Phase 2:</b> Mở <code>layer4</code> + avgpool + FC.<br>
                <b style="color:#94a3b8;">Ưu điểm:</b> Ổn định, hội tụ nhanh, ít VRAM.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # ViT-B/16
        active = "border-color:rgba(99,102,241,0.6);" if model_key == "vit" else ""
        st.markdown(f"""
        <div class="glass-card" style="{active}">
            <div style="font-size:1.4rem;font-weight:800;color:#818cf8;margin-bottom:0.5rem;">
                🤖 ViT-B/16
            </div>
            <div class="metric-row">
                <div class="metric-chip"><div class="value">86M</div><div class="label">Params</div></div>
                <div class="metric-chip"><div class="value">Transformer</div><div class="label">Type</div></div>
                <div class="metric-chip"><div class="value">768</div><div class="label">Feature dim</div></div>
            </div>
            <div style="color:#64748b;font-size:0.85rem;line-height:1.7;">
                <b style="color:#94a3b8;">Kiến trúc:</b> 12 Encoder blocks, patch 16×16.<br>
                <b style="color:#94a3b8;">Head:</b> Linear(768 → 10) tại <code>heads.head</code>.<br>
                <b style="color:#94a3b8;">Phase 1:</b> Freeze toàn bộ, chỉ train heads.<br>
                <b style="color:#94a3b8;">Phase 2:</b> Mở 4 block encoder cuối + LayerNorm.<br>
                <b style="color:#94a3b8;">Ưu điểm:</b> Self-Attention toàn cục, hiểu ngữ cảnh.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Fine-tuning timeline
    st.markdown("""
    <div class="glass-card" style="margin-top:0.5rem;">
        <div style="font-weight:700;color:#e2e8f0;margin-bottom:1rem;">
            📅 Multi-Phase Fine-Tuning Timeline
        </div>
        <div style="display:flex;gap:0;border-radius:10px;overflow:hidden;">
            <div style="flex:5;background:linear-gradient(135deg,#3730a3,#4f46e5);
                        padding:1rem 1.2rem;">
                <div style="font-weight:700;color:#e0e7ff;font-size:0.95rem;">
                    Phase 1 — head_only
                </div>
                <div style="color:#a5b4fc;font-size:0.82rem;margin-top:0.3rem;">
                    Epoch 1 → 5<br>
                    Optimizer: AdamW (head params only)<br>
                    LR Head: <code style="color:#c7d2fe;">1e-3</code><br>
                    Backbone: ❄️ Frozen
                </div>
            </div>
            <div style="width:3px;background:#f59e0b;"></div>
            <div style="flex:8;background:linear-gradient(135deg,#064e3b,#065f46);
                        padding:1rem 1.2rem;">
                <div style="font-weight:700;color:#d1fae5;font-size:0.95rem;">
                    Phase 2 — partial
                </div>
                <div style="color:#6ee7b7;font-size:0.82rem;margin-top:0.3rem;">
                    Epoch 6 → end<br>
                    Optimizer: AdamW re-created (2 param groups)<br>
                    LR Head: <code style="color:#a7f3d0;">1e-3</code> &nbsp;|&nbsp;
                    LR Backbone: <code style="color:#a7f3d0;">1e-5</code><br>
                    EarlyStopping: patience epochs
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Sidebar ────────────────────────────────────────────────────────────
    model_key, top_k = render_sidebar()

    # ── Hero ───────────────────────────────────────────────────────────────
    render_hero()

    # ── Tabs ───────────────────────────────────────────────────────────────
    tab_predict, tab_history, tab_dataset, tab_model = st.tabs([
        "🔍  Nhận Dạng",
        "📈  Lịch Sử Training",
        "📊  Dataset",
        "🏗️  Mô Hình",
    ])

    with tab_predict:
        render_predict_tab(model_key, top_k)

    with tab_history:
        render_history_tab(model_key)

    with tab_dataset:
        render_dataset_tab()

    with tab_model:
        render_model_tab(model_key)

    # ── Footer ─────────────────────────────────────────────────────────────
    st.markdown("""
    <div style='text-align:center; padding: 2rem 0 1rem 0;
                color:#334155; font-size:0.8rem; border-top:1px solid rgba(255,255,255,0.05);
                margin-top:2rem;'>
        🚗 <b>Vehicle Type Recognition</b> · Deep Learning @ UTE ·
        Built with PyTorch & Streamlit
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
