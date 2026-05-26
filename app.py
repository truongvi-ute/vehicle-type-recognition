"""
app.py
======
Giao diện trực quan (Streamlit) cho đồ án Nhận dạng Phương tiện Giao thông.
Cung cấp chức năng:
  - Tải ảnh lên.
  - Áp dụng các bộ lọc tiền xử lý không gian (tùy chỉnh tham số trực tiếp).
  - So sánh trực quan ảnh gốc và ảnh đã lọc.
  - Đưa ảnh vào mô hình CNN đã huấn luyện để nhận diện.
"""

import sys
import os
from pathlib import Path

# Thêm thư mục src vào path để import các module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import cv2
import numpy as np
from PIL import Image
import torch
import streamlit as st
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
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Nhận Dạng Phương Tiện Giao Thông")
st.markdown("""
Hệ thống sử dụng các bộ lọc **Miền Không Gian (Spatial Domain Filters)** để tiền xử lý 
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
    auto_pipeline = "baseline_v1" # Mặc định
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                auto_pipeline = cfg.get("pipeline", "baseline_v1")
        except Exception:
            pass
            
    st.info(f"🔒 **Pipeline tự động khóa:**\n`{auto_pipeline}`\n\n(Dựa trên file cấu hình của trọng số)")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN AREA - UPLOAD & PROCESSING
# ─────────────────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("Tải ảnh xe cộ lên (JPG, PNG, WEBP)...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    # 1. Đọc ảnh
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if img_bgr is None:
        st.error("Lỗi: Không thể đọc ảnh. Vui lòng thử file khác.")
    else:
        st.write("---")
        # 2. Xử lý & Hiển thị
        col1, col2 = st.columns(2)
        
        # Ảnh gốc (RGB)
        img_rgb_original = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        with col1:
            st.subheader("Ảnh gốc")
            st.image(img_rgb_original, use_container_width=True)
            st.caption(f"Kích thước ban đầu: {img_rgb_original.shape[1]}x{img_rgb_original.shape[0]}")
            
        # Tiền xử lý: Resize & Pad + Lọc
        current_img = resize_with_padding(img_bgr, 224)
        
        chain = PIPELINE_CHAINS.get(auto_pipeline, [])
        for step in chain:
            filter_name = step["name"]
            kwargs = step.get("kwargs", {})
            current_img = apply_filter(current_img, filter_name, **kwargs)
            
        filtered_rgb = cv2.cvtColor(current_img, cv2.COLOR_BGR2RGB)
        
        with col2:
            st.subheader("Sau khi Tiền Xử Lý")
            st.image(filtered_rgb, use_container_width=True)
            st.caption(f"Kích thước 224x224 | Pipeline: `{auto_pipeline}`")
            
        st.write("---")
        
        # 3. Chạy AI nhận diện
        st.subheader("Kết quả Dự Đoán")
        
        model, device = load_model(model_name, checkpoint_path, num_classes)
        
        if model is None:
            st.warning(f"hông tìm thấy file trọng số tại `{checkpoint_path}`.")
            st.info("Vui lòng chạy `python src/train.py` để huấn luyện mô hình trước, hoặc chỉnh sửa đường dẫn ở thanh bên trái.")
        else:
            with st.spinner("Đang phân tích ảnh..."):
                # Chuyển numpy array thành PIL Image để đưa vào transforms
                pil_img = Image.fromarray(filtered_rgb)
                
                # Áp dụng các bước normalize cho tập Test (Resize -> ToTensor -> Normalize)
                transform = get_transforms("test", img_size=224)
                input_tensor = transform(pil_img).unsqueeze(0).to(device)
                
                # Inference
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                    
                # Lấy kết quả
                prob_values = probabilities.cpu().numpy() * 100
                best_idx = np.argmax(prob_values)
                best_class = class_names[best_idx]
                best_conf = prob_values[best_idx]
                
                st.success(f"**Kết quả:** {best_class} (Độ tin cậy: {best_conf:.1f}%)")
                
                # Vẽ biểu đồ ngang
                df = pd.DataFrame({
                    "Loại phương tiện": class_names,
                    "Tỷ lệ (%)": prob_values
                }).set_index("Loại phương tiện")
                
                st.bar_chart(df, height=300)
                
        # Hiển thị cấu hình pipeline
        with st.expander("Xem cấu hình Pipeline đang áp dụng"):
            st.json(PIPELINE_CHAINS.get(auto_pipeline, []))
else:
    # Màn hình chờ
    st.info("Vui lòng tải một bức ảnh lên để bắt đầu.")
    st.markdown("""
    ### Hướng dẫn sử dụng:
    1. Chuẩn bị ảnh các loại xe (Bike, Bus, Car, Motorbike, Truck).
    2. Nếu bạn chưa train mô hình, hãy mở Terminal và chạy: `python src/train.py`
    3. Trải nghiệm các bộ lọc xử lý ảnh ở menu bên trái.
    """)
