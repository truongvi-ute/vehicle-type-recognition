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
import html
import io

import cv2
import numpy as np
from PIL import Image
import torch
import streamlit as st

try:
    from st_click_detector import click_detector
except ImportError:
    click_detector = None

from data_prep import PIPELINE_CHAINS, apply_filter, resize_with_padding
import json
from dataset import get_transforms
from model import load_for_inference, SUPPORTED_MODELS

BASE_DIR = Path(__file__).resolve().parent

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

st.markdown("""
<style>
:root {
    --app-bg: #eef5fb;
    --panel-bg: rgba(255, 255, 255, 0.86);
    --panel-strong: rgba(255, 255, 255, 0.96);
    --line: rgba(35, 72, 108, 0.14);
    --text: #082238;
    --muted: #48647f;
    --primary: #155dfc;
    --primary-soft: #d9e7ff;
    --accent: #00a887;
    --accent-soft: #d7f8ef;
    --warm: #ffb45c;
    --shadow: 0 22px 60px rgba(8, 34, 56, 0.11);
}

.stApp {
    background:
        linear-gradient(115deg, rgba(255, 180, 92, 0.24) 0%, transparent 24%),
        linear-gradient(155deg, rgba(0, 168, 135, 0.20) 8%, transparent 42%),
        linear-gradient(35deg, rgba(21, 93, 252, 0.20) 0%, transparent 48%),
        conic-gradient(from 210deg at 72% 18%, rgba(255,255,255,0.95), rgba(213,232,255,0.88), rgba(219,250,241,0.72), rgba(255,239,220,0.70), rgba(255,255,255,0.95)),
        linear-gradient(180deg, #f8fbff 0%, var(--app-bg) 100%);
}

.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    z-index: 0;
    pointer-events: none;
    background:
        linear-gradient(rgba(255,255,255,0.42) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.32) 1px, transparent 1px);
    background-size: 44px 44px;
    mask-image: linear-gradient(180deg, rgba(0,0,0,0.55), transparent 62%);
}

.stApp > header,
.stApp [data-testid="stAppViewContainer"],
.stApp section[data-testid="stSidebar"] {
    position: relative;
    z-index: 1;
}

html, body, [class*="css"] {
    color: var(--text);
    font-family: "Segoe UI", "Trebuchet MS", sans-serif;
}

.block-container {
    padding-top: 5.25rem;
    padding-bottom: 2rem;
    padding-left: 5rem;
    padding-right: 5rem;
    max-width: 1500px;
}

section[data-testid="stSidebar"] {
    background:
        radial-gradient(circle at 18% 4%, rgba(0, 168, 135, 0.22), transparent 28%),
        radial-gradient(circle at 88% 18%, rgba(21, 93, 252, 0.28), transparent 32%),
        linear-gradient(180deg, #0b1d31 0%, #102944 48%, #132f4b 100%);
    border-right: 1px solid rgba(177, 218, 255, 0.16);
    box-shadow: 18px 0 46px rgba(8, 34, 56, 0.14);
}

section[data-testid="stSidebar"]::before {
    content: "";
    position: absolute;
    inset: 0;
    pointer-events: none;
    background:
        linear-gradient(rgba(255,255,255,0.055) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.04) 1px, transparent 1px);
    background-size: 34px 34px;
    mask-image: linear-gradient(180deg, rgba(0,0,0,0.82), transparent 78%);
}

section[data-testid="stSidebar"]::after {
    content: "";
    position: absolute;
    left: 16px;
    right: 16px;
    bottom: 18px;
    height: 92px;
    pointer-events: none;
    border-radius: 28px;
    background:
        radial-gradient(circle at 20% 30%, rgba(0, 168, 135, .28), transparent 34%),
        radial-gradient(circle at 82% 18%, rgba(255, 180, 92, .22), transparent 36%);
    filter: blur(18px);
    opacity: .72;
}

section[data-testid="stSidebar"] * {
    color: #f4f8fc !important;
}

section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
    padding-top: 1.25rem;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {
    color: #f8fbff !important;
    letter-spacing: -0.02em;
    text-shadow: 0 10px 28px rgba(0,0,0,.24);
}

section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] {
    color: rgba(233, 245, 255, 0.88) !important;
    font-weight: 700 !important;
}

section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
    gap: .85rem;
}

section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div,
section[data-testid="stSidebar"] .stTextInput input {
    min-height: 44px;
    background:
        linear-gradient(135deg, rgba(255,255,255,0.14), rgba(255,255,255,0.06)) !important;
    border: 1px solid rgba(190, 223, 255, 0.20) !important;
    border-radius: 16px !important;
    box-shadow:
        inset 0 1px 0 rgba(255,255,255,.10),
        0 12px 26px rgba(4, 15, 29, .20);
    transition: border-color .18s ease, box-shadow .18s ease, background .18s ease;
}

section[data-testid="stSidebar"] .stTextInput input {
    background:
        linear-gradient(135deg, rgba(7, 20, 35, .72), rgba(19, 44, 70, .72)) !important;
    font-weight: 800 !important;
}

section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div:hover,
section[data-testid="stSidebar"] .stTextInput input:hover,
section[data-testid="stSidebar"] .stTextInput input:focus {
    border-color: rgba(109, 167, 255, 0.56) !important;
    box-shadow:
        0 0 0 4px rgba(21, 93, 252, .12),
        0 16px 34px rgba(4, 15, 29, .25);
}

section[data-testid="stSidebar"] [data-testid="stAlert"] {
    position: relative;
    overflow: hidden;
    padding: 1rem !important;
    border-radius: 20px !important;
    background:
        radial-gradient(circle at 16% 18%, rgba(255,255,255,.20), transparent 34%),
        linear-gradient(145deg, rgba(43, 116, 178, .84) 0%, rgba(25, 85, 137, .78) 54%, rgba(18, 68, 111, .84) 100%) !important;
    border: 1px solid rgba(158, 210, 255, 0.24) !important;
    box-shadow:
        inset 0 1px 0 rgba(255,255,255,.18),
        0 18px 42px rgba(3, 15, 28, 0.22);
}

section[data-testid="stSidebar"] [data-testid="stAlert"]::before {
    content: "";
    position: absolute;
    top: -36px;
    right: -30px;
    width: 94px;
    height: 94px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(0, 255, 209, .22), transparent 68%);
}

section[data-testid="stSidebar"] [data-testid="stAlert"]::after {
    content: "";
    position: absolute;
    inset: auto 16px 10px 16px;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,.30), transparent);
}

h1 {
    width: min(100%, 1450px);
    font-weight: 900 !important;
    letter-spacing: 0;
    color: #102235 !important;
    margin: 0 0 0.45rem 0 !important;
    padding: 0 1.1rem 0 1.1rem;
    text-shadow: 0 1px 0 rgba(255,255,255,0.8);
}

h2, h3 {
    color: #16314d !important;
    letter-spacing: 0;
}

[data-testid="stMarkdownContainer"] p {
    color: #38546f;
    line-height: 1.7;
}

.block-container > div:nth-child(2) [data-testid="stMarkdownContainer"] {
    width: min(100%, 1450px);
    padding: 0 1.1rem 1.35rem 1.1rem;
    margin-bottom: 1.6rem;
    border-bottom: 1px solid rgba(8,34,56,0.08);
}

.block-container > div:nth-child(2) [data-testid="stMarkdownContainer"] p {
    max-width: 900px;
    color: #304d69;
    font-weight: 600;
}

[data-testid="stMarkdownContainer"] li {
    color: #2d4863 !important;
    font-weight: 600;
}

[data-testid="stMarkdownContainer"] code {
    color: #d7ffe9 !important;
    background: #0c2c2d !important;
    border: 1px solid rgba(0, 168, 135, 0.28);
    border-radius: 6px;
}

[data-testid="stAlert"] [data-testid="stMarkdownContainer"] p,
[data-testid="stAlert"] [data-testid="stMarkdownContainer"] li {
    color: #2f506f !important;
}

section[data-testid="stSidebar"] [data-testid="stAlert"] [data-testid="stMarkdownContainer"] p,
section[data-testid="stSidebar"] [data-testid="stAlert"] [data-testid="stMarkdownContainer"] code {
    color: #f7fbff !important;
}

section[data-testid="stSidebar"] [data-testid="stAlert"] [data-testid="stMarkdownContainer"] {
    position: relative;
    z-index: 1;
}

section[data-testid="stSidebar"] [data-testid="stAlert"] [data-testid="stMarkdownContainer"] p {
    text-align: left !important;
    line-height: 1.65 !important;
}

section[data-testid="stSidebar"] [data-testid="stAlert"] code {
    display: inline-flex;
    align-items: center;
    padding: 3px 8px !important;
    border-radius: 999px !important;
    color: #dbfff2 !important;
    background: rgba(5, 43, 44, .62) !important;
    border: 1px solid rgba(91, 255, 207, .20) !important;
    box-shadow: 0 8px 18px rgba(0, 0, 0, .14);
}

section[data-testid="stSidebar"] button[kind="header"] {
    border-radius: 999px !important;
    background: rgba(255,255,255,.08) !important;
    border: 1px solid rgba(255,255,255,.10) !important;
    box-shadow: 0 10px 22px rgba(0,0,0,.16);
}

section[data-testid="stSidebar"] [data-baseweb="select"] svg {
    color: rgba(233, 245, 255, .88) !important;
}

div[data-testid="stImage"] img {
    border-radius: 18px;
    border: 1px solid rgba(24, 49, 74, 0.10);
    box-shadow: var(--shadow);
}

.vision-lab-panel {
    margin: 1.15rem 0 1.35rem;
    padding: 1.15rem;
    border-radius: 28px;
    border: 1px solid rgba(32, 70, 104, .13);
    background:
        radial-gradient(circle at 14% 8%, rgba(255,255,255,.92), transparent 32%),
        radial-gradient(circle at 86% 12%, rgba(0,168,135,.12), transparent 32%),
        linear-gradient(135deg, rgba(255,255,255,.78), rgba(232,242,255,.64) 52%, rgba(226,250,242,.58));
    box-shadow: 0 24px 62px rgba(8,34,56,.11), inset 0 1px 0 rgba(255,255,255,.82);
    backdrop-filter: blur(14px);
}

.vision-lab-header {
    display: flex;
    justify-content: space-between;
    gap: 1rem;
    align-items: center;
    margin-bottom: 1rem;
}

.vision-kicker {
    color: #0b6b70;
    font-size: 12px;
    font-weight: 900;
    letter-spacing: .14em;
    text-transform: uppercase;
}

.vision-title {
    color: #102235;
    font-size: 24px;
    line-height: 1.1;
    font-weight: 900;
    letter-spacing: -.03em;
    margin-top: .25rem;
}

.vision-subtitle {
    color: #46637f;
    font-size: 13px;
    font-weight: 700;
    margin-top: .45rem;
}

.pipeline-pill {
    display: inline-flex;
    align-items: center;
    gap: .45rem;
    padding: .55rem .8rem;
    border-radius: 999px;
    color: #e9fff7;
    background: linear-gradient(135deg, #0d3a4b, #0e6d69);
    border: 1px solid rgba(148, 255, 225, .22);
    box-shadow: 0 14px 30px rgba(6, 49, 60, .20);
    font-size: 12px;
    font-weight: 900;
    white-space: nowrap;
}

.pipeline-pill span {
    color: rgba(233,255,247,.72);
    font-weight: 800;
}

.vision-card {
    position: relative;
    padding: .85rem;
    border-radius: 24px;
    border: 1px solid rgba(23, 58, 91, .12);
    background: rgba(255,255,255,.64);
    box-shadow: 0 18px 44px rgba(8,34,56,.10), inset 0 1px 0 rgba(255,255,255,.74);
    min-height: 100%;
}

.vision-lab-panel + div[data-testid="stHorizontalBlock"] > div:has(.vision-card-title),
div[data-testid="stHorizontalBlock"] > div:has(.vision-card-title) {
    position: relative;
    padding: 1.35rem .95rem .95rem !important;
    border-radius: 24px;
    border: 1px solid rgba(23, 58, 91, .12);
    background: rgba(255,255,255,.64);
    box-shadow: 0 18px 44px rgba(8,34,56,.10), inset 0 1px 0 rgba(255,255,255,.74);
    min-height: 100%;
}

div[data-testid="stHorizontalBlock"] > div:has(.vision-card-title)::before {
    content: "";
    position: absolute;
    inset: 11px 14px auto 14px;
    height: 2px;
    border-radius: 999px;
    background: linear-gradient(90deg, #155dfc, #00a887, transparent);
    opacity: .52;
}

div[data-testid="stHorizontalBlock"] > div:has(.vision-card.processed)::before {
    background: linear-gradient(90deg, #00a887, #ffb45c, transparent);
}

div[data-testid="stHorizontalBlock"] > div:has(.vision-card-title) div[data-testid="stImage"] {
    margin-top: .15rem;
}

div[data-testid="stHorizontalBlock"] > div:has(.vision-card-title) div[data-testid="stImage"] img {
    width: 100%;
    max-height: 520px;
    object-fit: contain;
    background: #05080c;
}

.vision-card::before {
    content: "";
    position: absolute;
    inset: 11px 14px auto 14px;
    height: 2px;
    border-radius: 999px;
    background: linear-gradient(90deg, #155dfc, #00a887, transparent);
    opacity: .52;
}

.vision-card.processed::before {
    background: linear-gradient(90deg, #00a887, #ffb45c, transparent);
}

.vision-card-title {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: .6rem;
    padding: .2rem .15rem .95rem;
    color: #152f4c;
    font-size: 19px;
    font-weight: 900;
    letter-spacing: -.02em;
    line-height: 1.1;
    min-width: 0;
}

.vision-card-title > span:first-child {
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.vision-card,
.vision-card.original,
.vision-card.processed {
    display: contents;
}

.vision-badge {
    flex: 0 0 auto;
    padding: .32rem .6rem;
    border-radius: 999px;
    color: #0e3850;
    background: rgba(216, 235, 255, .86);
    border: 1px solid rgba(21, 93, 252, .14);
    font-size: 10px;
    font-weight: 900;
    white-space: nowrap;
    max-width: 138px;
    overflow: hidden;
    text-overflow: ellipsis;
}

.vision-card.processed .vision-badge {
    color: #0a514a;
    background: rgba(219, 250, 241, .90);
    border-color: rgba(0, 168, 135, .18);
}

.vision-meta-row {
    display: flex;
    flex-wrap: wrap;
    gap: .5rem;
    margin-top: .9rem;
    padding-bottom: .1rem;
}

.vision-meta {
    display: inline-flex;
    align-items: center;
    gap: .35rem;
    padding: .42rem .65rem;
    border-radius: 999px;
    color: #30516f;
    background: rgba(255,255,255,.72);
    border: 1px solid rgba(32, 70, 104, .10);
    font-size: 12px;
    font-weight: 850;
}

.vision-meta strong {
    color: #102235;
}

.result-dashboard {
    margin: 1.15rem 0 1.35rem;
    padding: 1.1rem;
    border-radius: 28px;
    border: 1px solid rgba(32, 70, 104, .13);
    background:
        radial-gradient(circle at 12% 8%, rgba(255,255,255,.94), transparent 32%),
        radial-gradient(circle at 88% 10%, rgba(21,93,252,.12), transparent 30%),
        linear-gradient(135deg, rgba(255,255,255,.80), rgba(232,242,255,.66) 52%, rgba(226,250,242,.58));
    box-shadow: 0 24px 62px rgba(8,34,56,.11), inset 0 1px 0 rgba(255,255,255,.82);
    backdrop-filter: blur(14px);
}

.result-dashboard-head {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 1rem;
    margin-bottom: 1rem;
}

.result-kicker {
    color: #0b6b70;
    font-size: 12px;
    font-weight: 900;
    letter-spacing: .14em;
    text-transform: uppercase;
}

.result-title {
    color: #102235;
    font-size: 24px;
    line-height: 1.1;
    font-weight: 900;
    letter-spacing: -.03em;
    margin-top: .25rem;
}

.result-status-pill {
    display: inline-flex;
    align-items: center;
    padding: .55rem .8rem;
    border-radius: 999px;
    color: #eafff8;
    background: linear-gradient(135deg, #0d3a4b, #0e6d69);
    border: 1px solid rgba(148,255,225,.22);
    box-shadow: 0 14px 30px rgba(6,49,60,.20);
    font-size: 12px;
    font-weight: 900;
    white-space: nowrap;
}

.result-hero,
.leaderboard-card {
    height: 100%;
    padding: 1rem;
    border-radius: 24px;
    border: 1px solid rgba(23, 58, 91, .12);
    background: rgba(255,255,255,.66);
    box-shadow: 0 18px 44px rgba(8,34,56,.10), inset 0 1px 0 rgba(255,255,255,.74);
}

.result-label {
    color: #48647f;
    font-size: 12px;
    font-weight: 900;
    letter-spacing: .10em;
    text-transform: uppercase;
}

.result-class {
    margin-top: .45rem;
    color: #102235;
    font-size: 38px;
    line-height: 1;
    font-weight: 950;
    letter-spacing: -.04em;
    text-transform: capitalize;
}

.confidence-meter {
    margin-top: 1rem;
    padding: .8rem;
    border-radius: 20px;
    background: linear-gradient(135deg, rgba(15,45,69,.94), rgba(14,109,105,.88));
    color: #ecfff8;
    box-shadow: 0 16px 36px rgba(8,34,56,.18);
}

.confidence-row {
    display: flex;
    justify-content: space-between;
    gap: 1rem;
    align-items: baseline;
    font-weight: 900;
}

.confidence-value {
    font-size: 34px;
    letter-spacing: -.04em;
}

.confidence-track {
    height: 12px;
    margin-top: .7rem;
    border-radius: 999px;
    background: rgba(255,255,255,.18);
    overflow: hidden;
}

.confidence-fill {
    height: 100%;
    border-radius: inherit;
    background: linear-gradient(90deg, #65d6ff, #5bffcf, #ffde8a);
    box-shadow: 0 0 22px rgba(91,255,207,.38);
}

.result-meta-row {
    display: flex;
    flex-wrap: wrap;
    gap: .55rem;
    margin-top: .9rem;
}

.result-meta {
    padding: .45rem .68rem;
    border-radius: 999px;
    color: #30516f;
    background: rgba(255,255,255,.74);
    border: 1px solid rgba(32,70,104,.10);
    font-size: 12px;
    font-weight: 850;
}

.leaderboard-title {
    color: #102235;
    font-size: 18px;
    font-weight: 950;
    letter-spacing: -.02em;
    margin-bottom: .8rem;
}

.rank-row {
    display: grid;
    grid-template-columns: 28px minmax(86px, 132px) 1fr 52px;
    align-items: center;
    gap: .65rem;
    padding: .48rem .15rem;
}

.rank-num {
    width: 26px;
    height: 26px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    border-radius: 999px;
    color: #48647f;
    background: rgba(255,255,255,.76);
    border: 1px solid rgba(32,70,104,.10);
    font-size: 11px;
    font-weight: 950;
}

.rank-name {
    color: #1b3854;
    font-size: 13px;
    font-weight: 900;
    text-transform: capitalize;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.rank-track {
    height: 10px;
    border-radius: 999px;
    background: rgba(48, 81, 111, .12);
    overflow: hidden;
}

.rank-fill {
    height: 100%;
    min-width: 2px;
    border-radius: inherit;
    background: linear-gradient(90deg, #6db8ff, #5bffcf);
}

.rank-value {
    color: #30516f;
    font-size: 12px;
    font-weight: 950;
    text-align: right;
}

.rank-row.top {
    margin-bottom: .2rem;
    padding: .55rem .45rem;
    border-radius: 16px;
    background: rgba(219,250,241,.72);
    border: 1px solid rgba(0,168,135,.14);
}

.rank-row.top .rank-num {
    color: #eafff8;
    background: linear-gradient(135deg, #0d3a4b, #0e6d69);
}

.rank-row.top .rank-name,
.rank-row.top .rank-value {
    color: #0d3a4b;
}

.probability-chart-card {
    margin-top: 1rem;
    padding: 1rem 1rem .85rem;
    border-radius: 24px;
    border: 1px solid rgba(23, 58, 91, .12);
    background:
        radial-gradient(circle at 10% 8%, rgba(255,255,255,.86), transparent 28%),
        linear-gradient(135deg, rgba(255,255,255,.72), rgba(232,242,255,.58));
    box-shadow: 0 18px 44px rgba(8,34,56,.09), inset 0 1px 0 rgba(255,255,255,.76);
}

.chart-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    margin-bottom: .8rem;
}

.chart-title {
    color: #102235;
    font-size: 18px;
    font-weight: 950;
    letter-spacing: -.02em;
}

.chart-note {
    color: #48647f;
    font-size: 12px;
    font-weight: 850;
}

.prob-chart {
    position: relative;
    display: grid;
    grid-template-columns: repeat(var(--bar-count), minmax(48px, 1fr));
    gap: .75rem;
    align-items: end;
    min-height: 270px;
    padding: 1.2rem .55rem .2rem;
    border-radius: 20px;
    background:
        linear-gradient(rgba(48,81,111,.10) 1px, transparent 1px),
        linear-gradient(180deg, rgba(255,255,255,.64), rgba(255,255,255,.28));
    background-size: 100% 25%;
    border: 1px solid rgba(32,70,104,.09);
    overflow-x: auto;
}

.prob-bar-item {
    min-width: 48px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: end;
    gap: .45rem;
}

.prob-bar-track {
    position: relative;
    width: 100%;
    max-width: 72px;
    height: 190px;
    display: flex;
    align-items: end;
    justify-content: center;
}

.prob-bar {
    position: relative;
    width: 72%;
    min-height: 3px;
    border-radius: 16px 16px 8px 8px;
    background: linear-gradient(180deg, #8bcaff 0%, #62b6ff 54%, #4d8dff 100%);
    box-shadow: 0 12px 24px rgba(77,141,255,.20);
}

.prob-bar.top {
    background: linear-gradient(180deg, #ffdc8b 0%, #5bffcf 45%, #00a887 100%);
    box-shadow: 0 12px 28px rgba(0,168,135,.28), 0 0 22px rgba(91,255,207,.28);
}

.prob-bar-value {
    position: absolute;
    top: -24px;
    left: 50%;
    transform: translateX(-50%);
    color: #102235;
    font-size: 11px;
    font-weight: 950;
    white-space: nowrap;
}

.prob-label {
    max-width: 84px;
    color: #30516f;
    font-size: 11px;
    font-weight: 900;
    text-align: center;
    text-transform: capitalize;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

div[data-testid="stExpander"] {
    border: 1px solid var(--line) !important;
    border-radius: 18px !important;
    background: var(--panel-strong);
    box-shadow: var(--shadow);
}

div[data-testid="stAlert"] {
    border-radius: 18px !important;
    border: 1px solid var(--line) !important;
    background: rgba(242, 248, 255, 0.80) !important;
    box-shadow: 0 16px 38px rgba(15, 35, 55, 0.06);
}

div[data-testid="stFileUploader"] {
    width: 96px !important;
    min-width: 96px !important;
    margin-top: 0 !important;
}

div[data-testid="stFileUploader"] > label {
    display: none !important;
}

div[data-testid="stFileUploader"] section,
div[data-testid="stFileUploaderDropzone"] {
    position: relative !important;
    min-height: 118px !important;
    height: 118px !important;
    width: 96px !important;
    padding: 0 !important;
    border-radius: 18px !important;
    border: 2px dashed rgba(21, 93, 252, 0.32) !important;
    background:
        radial-gradient(circle at 28% 20%, rgba(255,255,255,0.96), rgba(255,255,255,0.22) 34%, transparent 35%),
        linear-gradient(135deg, rgba(255,255,255,0.94) 0%, rgba(226,239,255,0.92) 58%, rgba(220,251,243,0.92) 100%) !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    cursor: pointer;
    overflow: hidden !important;
    box-shadow: inset 0 1px 0 rgba(255,255,255,.86), 0 12px 28px rgba(15, 35, 55, .07);
    transition: all .18s ease;
    margin: 0 !important;
}

div[data-testid="stFileUploader"] section:hover,
div[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #155dfc !important;
    box-shadow: 0 12px 28px rgba(21,93,252,.18);
    transform: translateY(-2px);
}

div[data-testid="stFileUploader"] section::before,
div[data-testid="stFileUploaderDropzone"]::before {
    content: "+";
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #155dfc;
    font-size: 34px;
    font-weight: 200;
    line-height: 1;
    pointer-events: none;
    z-index: 1;
}

div[data-testid="stFileUploader"] section p,
div[data-testid="stFileUploader"] section small,
div[data-testid="stFileUploader"] section span,
div[data-testid="stFileUploader"] section svg,
div[data-testid="stFileUploader"] section [data-testid="stMarkdownContainer"],
div[data-testid="stFileUploaderDropzone"] p,
div[data-testid="stFileUploaderDropzone"] small,
div[data-testid="stFileUploaderDropzone"] span,
div[data-testid="stFileUploaderDropzone"] svg,
div[data-testid="stFileUploaderDropzone"] [data-testid="stMarkdownContainer"] {
    display: none !important;
}

div[data-testid="stFileUploader"] section button,
div[data-testid="stFileUploaderDropzone"] button {
    position: absolute !important;
    inset: 0 !important;
    width: 100% !important;
    height: 100% !important;
    min-height: 100% !important;
    padding: 0 !important;
    margin: 0 !important;
    border: 0 !important;
    opacity: 0 !important;
    cursor: pointer !important;
    z-index: 2 !important;
}

div[data-testid="stFileUploaderFile"] {
    display: none !important;
}

div[data-testid="stFileUploader"] section + div,
div[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"],
div[data-testid="stFileUploader"] [data-testid="stFileUploaderFileData"],
div[data-testid="stFileUploader"] [data-testid="stFileUploaderFileName"],
div[data-testid="stFileUploader"] [data-testid="stFileUploaderFileSize"] {
    display: none !important;
}

div[data-testid="stFileUploader"] + div,
div[data-testid="stFileUploader"] small {
    display: none !important;
}

div[data-testid="stHorizontalBlock"]:has(div[data-testid="stFileUploader"]):has(iframe) {
    align-items: stretch !important;
    gap: 0 !important;
    padding: 12px 14px;
    border-radius: 26px;
    background:
        radial-gradient(circle at 8% 10%, rgba(255,255,255,.98), transparent 34%),
        linear-gradient(120deg, rgba(255,255,255,.90) 0%, rgba(228,240,255,.82) 48%, rgba(224,250,241,.76) 100%);
    border: 1px solid rgba(28,55,84,.10);
    box-shadow: 0 18px 42px rgba(15,35,55,.08), inset 0 1px 0 rgba(255,255,255,.86);
    overflow: hidden;
}

div[data-testid="stHorizontalBlock"]:has(div[data-testid="stFileUploader"]):has(iframe) > div {
    display: flex !important;
    align-items: center !important;
}

div[data-testid="stIFrame"],
div[data-testid="stIFrame"] iframe,
iframe[title*="st_click_detector"],
iframe[title*="click_detector"] {
    background: transparent !important;
    border: 0 !important;
    border-radius: 0 !important;
}

div[data-testid="stIFrame"] iframe,
iframe[title*="st_click_detector"],
iframe[title*="click_detector"] {
    min-height: 118px !important;
    height: 118px !important;
}

div[data-testid="stElementContainer"]:has(iframe) {
    overflow: hidden !important;
    border-radius: 22px !important;
    background: transparent !important;
    box-shadow: none !important;
}

.thumb-strip {
    display: grid;
    grid-template-columns: 96px minmax(0, 1fr);
    gap: 14px;
    align-items: stretch;
    padding: 16px 18px;
    border: 1px solid rgba(28, 55, 84, 0.14);
    border-radius: 24px;
    background:
        radial-gradient(circle at 12% 18%, rgba(255,255,255,0.98), transparent 32%),
        linear-gradient(120deg, rgba(255,255,255,0.78) 0%, rgba(229,240,255,0.72) 48%, rgba(224,250,241,0.68) 100%);
    box-shadow: 0 18px 42px rgba(15, 35, 55, 0.08), inset 0 1px 0 rgba(255,255,255,0.78);
    margin-bottom: 1.1rem;
    backdrop-filter: blur(12px);
}

.native-gallery {
    display: flex;
    align-items: stretch;
    gap: 14px;
    padding: 14px 18px;
    border: 1px solid rgba(28, 55, 84, 0.14);
    border-radius: 20px;
    background:
        linear-gradient(120deg, rgba(255,255,255,0.98) 0%, rgba(229,240,255,0.90) 48%, rgba(224,250,241,0.78) 100%);
    box-shadow: 0 18px 42px rgba(15, 35, 55, 0.08);
    margin-bottom: 1rem;
    overflow-x: auto;
}

.native-gallery::-webkit-scrollbar {
    height: 6px;
}

.native-gallery::-webkit-scrollbar-track {
    background: #deebf7;
    border-radius: 999px;
}

.native-gallery::-webkit-scrollbar-thumb {
    background: #aebed4;
    border-radius: 999px;
}

.upload-trigger-note {
    flex: 0 0 88px;
    min-height: 128px;
    width: 88px;
    border: 2px dashed rgba(31, 94, 255, 0.35);
    border-radius: 16px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #155dfc;
    background: linear-gradient(135deg,#ffffff 0%,#e6f0ff 60%,#dcfbf2 100%);
    font-size: 32px;
    font-weight: 200;
    text-decoration: none;
    cursor: pointer;
    transition: all .18s ease;
}

.upload-trigger-note:hover {
    border-color: #155dfc;
    box-shadow: 0 12px 28px rgba(21,93,252,.18);
    transform: translateY(-2px);
}

.gallery-hint {
    color: #7d92aa;
    font-size: 13px;
    font-style: italic;
    align-self: center;
}

.thumb-wrap {
    flex: 0 0 116px;
    position: relative;
    width: 116px;
}

.thumb-card {
    display: block;
    border-radius: 16px;
    border: 1px solid rgba(20, 48, 76, 0.12);
    background: rgba(255,255,255,.80);
    padding: 8px;
    box-shadow: 0 10px 22px rgba(15,35,55,.09);
    color: inherit;
    text-decoration: none;
    transition: border-color .18s, box-shadow .18s, transform .18s, background .18s;
}

.thumb-card:hover {
    border-color: #4d8dff;
    background: rgba(255,255,255,.96);
    box-shadow: 0 16px 30px rgba(77,141,255,.22);
    transform: translateY(-2px);
}

.thumb-card.selected {
    border-color: #155dfc;
    background: linear-gradient(180deg,rgba(255,255,255,.98),rgba(232,241,255,.98));
    box-shadow:
        0 0 0 4px rgba(21,93,252,.16),
        0 0 0 7px rgba(0,168,135,.10),
        0 16px 30px rgba(21,93,252,.22);
}

.native-thumb-img img {
    width: 100%;
    height: 76px;
    object-fit: cover;
    border-radius: 12px;
    display: block;
    border: 1px solid rgba(20,48,76,.08);
}

.thumb-action-row .stButton > button {
    width: 28px !important;
    min-width: 28px !important;
    height: 28px !important;
    min-height: 28px !important;
    padding: 0 !important;
    border-radius: 999px !important;
    background: linear-gradient(135deg,#ff5b66 0%,#d92d38 100%) !important;
    color: #fff !important;
    border: 0 !important;
    box-shadow: 0 8px 18px rgba(217,45,56,.28);
    font-size: 14px !important;
    font-weight: 900 !important;
}

.thumb-pick-button .stButton > button {
    width: 100% !important;
    min-height: 0 !important;
    height: 0 !important;
    padding: 0 !important;
    border: 0 !important;
    background: transparent !important;
    color: transparent !important;
    box-shadow: none !important;
}

.active-badge {
    display: inline-block;
    position: absolute;
    left: 8px;
    top: -10px;
    z-index: 4;
    padding: 4px 8px;
    border-radius: 999px;
    background: #102235;
    color: #ffffff;
    font-size: 10px;
    font-weight: 800;
    box-shadow: 0 8px 16px rgba(16,34,53,.20);
    pointer-events: none;
}

.thumb-name-native {
    color: #253f5c;
    font-size: 11px;
    font-weight: 700;
    text-align: center;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    margin: 6px 0 7px;
}

.del-btn {
    position: absolute;
    top: -9px;
    right: -9px;
    width: 24px;
    height: 24px;
    border-radius: 50%;
    background: linear-gradient(135deg,#ff5b66 0%,#d92d38 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    color: #fff;
    font-size: 12px;
    font-weight: 800;
    text-decoration: none;
    box-shadow: 0 8px 18px rgba(217,45,56,.34);
    z-index: 6;
}

.del-btn:hover {
    background: linear-gradient(135deg,#ff6a73 0%,#c81e2b 100%);
    transform: scale(1.12);
}

div[data-testid="stHorizontalBlock"] > div:has(> div[data-testid="stImage"]) {
    background: var(--panel-bg);
    border: 1px solid var(--line);
    border-radius: 22px;
    padding: 1rem;
    box-shadow: var(--shadow);
}

hr {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(24,49,74,0.18), transparent);
    margin: 1.4rem 0 !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HÀM TIỆN ÍCH
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model(model_name: str, checkpoint_path: str, num_classes: int):
    """Load model một lần duy nhất nhờ cơ chế cache của Streamlit."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved_checkpoint_path = resolve_project_path(checkpoint_path)
    if not resolved_checkpoint_path.exists():
        return None, device
    try:
        model = load_for_inference(str(resolved_checkpoint_path), num_classes=num_classes, device=device)
        return model, device
    except Exception as e:
        st.error(f"Lỗi khi load model: {e}")
        return None, device


def resolve_project_path(path_value: str) -> Path:
    """Resolve relative paths from the project folder instead of the terminal cwd."""
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return BASE_DIR / path


@st.cache_data
def get_class_names(raw_dir: str = "data/raw"):
    """Tự động lấy tên lớp từ thư mục data/raw (nếu có)."""
    raw_path = resolve_project_path(raw_dir)
    if raw_path.exists():
        classes = sorted([d.name for d in raw_path.iterdir() if d.is_dir()])
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
    config_path = str(resolve_project_path(checkpoint_path)).replace(".pth", "_config.json")
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
if "selected_image_name" not in st.session_state:
    st.session_state["selected_image_name"] = ""
if "gallery_version" not in st.session_state:
    st.session_state["gallery_version"] = 0
if "last_gallery_click" not in st.session_state:
    st.session_state["last_gallery_click"] = ""
if "uploader_version" not in st.session_state:
    st.session_state["uploader_version"] = 0

_upload_col, _thumb_col = st.columns([0.08, 0.92], gap="small")
with _upload_col:
    _new_files = st.file_uploader(
        "Upload",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key=f"img_uploader_{st.session_state['uploader_version']}",
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
        st.session_state["selected_image_name"] = st.session_state["uploaded_images"][-1]["name"]
        st.session_state["gallery_version"] += 1
        st.session_state["uploader_version"] += 1
        st.rerun()

# ── Render Single Horizontal Row ──────────────────────────────────────────────
_images  = st.session_state["uploaded_images"]
_sel_idx = st.session_state["selected_image_index"]
_S       = THUMB_SIZE

if _images:
    _names = [_img["name"] for _img in _images]
    if st.session_state["selected_image_name"] in _names:
        _sel_idx = _names.index(st.session_state["selected_image_name"])
        st.session_state["selected_image_index"] = _sel_idx
    elif 0 <= _sel_idx < len(_images):
        st.session_state["selected_image_name"] = _images[_sel_idx]["name"]
    else:
        _sel_idx = 0
        st.session_state["selected_image_index"] = 0
        st.session_state["selected_image_name"] = _images[0]["name"]
else:
    st.session_state["selected_image_index"] = -1
    st.session_state["selected_image_name"] = ""

with _thumb_col:
    if _images and click_detector is not None:
        _version = st.session_state["gallery_version"]
        _gallery_items = []
        for _i, _img in enumerate(_images):
            _safe_name = html.escape(_img["name"], quote=True)
            _short_name = _img["name"] if len(_img["name"]) <= 18 else f"{_img['name'][:15]}..."
            _safe_short_name = html.escape(_short_name, quote=True)
            _selected_class = " selected" if _i == _sel_idx else ""
            _active_badge = '<span class="active-badge">Đang xem</span>' if _i == _sel_idx else ""
            _gallery_items.append(
                f"""
                <div class="thumb-wrap">
                    <a id="select:{_i}:{_version}" class="thumb-card{_selected_class}" href="#">
                        {_active_badge}
                        <img src="data:image/jpeg;base64,{_img['b64']}" alt="{_safe_name}" draggable="false" />
                        <span class="thumb-name-native">{_safe_short_name}</span>
                    </a>
                    <a id="delete:{_i}:{_version}" class="del-btn" href="#">×</a>
                </div>
                """
            )

        _gallery_html = f"""
        <style>
            * {{
                box-sizing: border-box;
            }}
            html, body, #root, .st-click-detector, .stClickDetector, [data-testid="stClickDetector"] {{
                margin: 0;
                padding: 0;
                min-height: 118px;
                height: 118px;
                width: 100%;
                background: transparent !important;
                font-family: "Segoe UI", "Trebuchet MS", sans-serif;
                overflow: hidden;
            }}
            body::before {{
                content: "";
                position: fixed;
                inset: 0;
                z-index: 0;
                background:
                    radial-gradient(circle at 5% 15%, rgba(255,255,255,.96), transparent 34%),
                    linear-gradient(120deg, rgba(255,255,255,.92) 0%, rgba(226,239,255,.82) 48%, rgba(224,250,241,.78) 100%);
                pointer-events: none;
            }}
            body > * {{
                position: relative;
                z-index: 1;
            }}
            .native-gallery {{
                position: fixed;
                inset: 0;
                display: flex;
                align-items: center;
                gap: 14px;
                min-height: 118px;
                height: 118px;
                width: 100%;
                padding: 10px 18px 10px 18px;
                border: 0;
                border-radius: 20px;
                background:
                    radial-gradient(circle at 5% 15%, rgba(255,255,255,.96), transparent 34%),
                    linear-gradient(120deg, rgba(255,255,255,.92) 0%, rgba(226,239,255,.82) 48%, rgba(224,250,241,.78) 100%);
                box-shadow: none;
                overflow-x: auto;
                overflow-y: hidden;
            }}
            .native-gallery::-webkit-scrollbar {{
                height: 6px;
            }}
            .native-gallery::-webkit-scrollbar-track {{
                background: #deebf7;
                border-radius: 999px;
            }}
            .native-gallery::-webkit-scrollbar-thumb {{
                background: #aebed4;
                border-radius: 999px;
            }}
            .thumb-wrap {{
                flex: 0 0 104px;
                position: relative;
                width: 104px;
                padding: 0 3px 0;
                height: 98px;
            }}
            .thumb-card {{
                position: relative;
                display: block;
                height: 98px;
                border-radius: 15px;
                border: 1px solid rgba(20, 48, 76, 0.12);
                background: rgba(255, 255, 255, 0.88);
                padding: 7px;
                box-sizing: border-box;
                box-shadow: 0 10px 22px rgba(15, 35, 55, 0.09);
                color: inherit;
                text-decoration: none;
                transition: border-color .18s ease, box-shadow .18s ease, transform .18s ease, background .18s ease;
            }}
            .thumb-card:hover {{
                border-color: #4d8dff;
                background: rgba(255, 255, 255, 0.97);
                box-shadow: 0 16px 30px rgba(77, 141, 255, 0.22);
                transform: translateY(-2px);
            }}
            .thumb-card.selected {{
                border-color: #155dfc;
                background: linear-gradient(180deg, rgba(255,255,255,.98), rgba(232,241,255,.98));
                box-shadow:
                    0 0 0 4px rgba(21, 93, 252, .16),
                    0 0 0 7px rgba(0, 168, 135, .10),
                    0 16px 30px rgba(21, 93, 252, .22);
            }}
            .thumb-card img {{
                width: 100%;
                height: 65px;
                object-fit: cover;
                border-radius: 11px;
                display: block;
                border: 1px solid rgba(20, 48, 76, .08);
                pointer-events: none;
            }}
            .thumb-name-native {{
                display: block;
                color: #253f5c;
                font-size: 11px;
                font-weight: 750;
                text-align: center;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
                margin-top: 5px;
                pointer-events: none;
            }}
            .active-badge {{
                display: inline-block;
                position: absolute;
                left: 8px;
                top: -7px;
                z-index: 4;
                padding: 4px 8px;
                border-radius: 999px;
                background: #102235;
                color: #ffffff;
                font-size: 10px;
                font-weight: 800;
                box-shadow: 0 8px 16px rgba(16,34,53,.20);
                pointer-events: none;
            }}
            .del-btn {{
                position: absolute;
                top: -8px;
                right: -3px;
                width: 24px;
                height: 24px;
                border-radius: 50%;
                background: linear-gradient(135deg, #ff5b66 0%, #d92d38 100%);
                display: flex;
                align-items: center;
                justify-content: center;
                color: #fff;
                font-size: 13px;
                font-weight: 900;
                line-height: 1;
                text-decoration: none;
                box-shadow: 0 8px 18px rgba(217, 45, 56, .34);
                z-index: 6;
                transition: transform .15s ease, filter .15s ease;
            }}
            .del-btn:hover {{
                transform: scale(1.12);
                filter: saturate(1.12);
            }}
        </style>
        <div class="native-gallery">
            {''.join(_gallery_items)}
        </div>
        """
        _clicked = click_detector(
            _gallery_html,
            key=f"thumb_gallery_{_version}_{len(_images)}_{_sel_idx}",
        )
        if _clicked and _clicked != st.session_state["last_gallery_click"]:
            st.session_state["last_gallery_click"] = _clicked
            _parts = _clicked.split(":")
            if len(_parts) >= 2:
                _action = _parts[0]
                try:
                    _clicked_idx = int(_parts[1])
                except ValueError:
                    _clicked_idx = -1

                if _action == "select" and 0 <= _clicked_idx < len(_images):
                    st.session_state["selected_image_index"] = _clicked_idx
                    st.session_state["selected_image_name"] = _images[_clicked_idx]["name"]
                    st.rerun()

                if _action == "delete" and 0 <= _clicked_idx < len(_images):
                    _images.pop(_clicked_idx)
                    if not _images:
                        st.session_state["selected_image_index"] = -1
                        st.session_state["selected_image_name"] = ""
                    else:
                        _next_idx = min(_clicked_idx, len(_images) - 1)
                        st.session_state["selected_image_index"] = _next_idx
                        st.session_state["selected_image_name"] = _images[_next_idx]["name"]
                    st.session_state["gallery_version"] += 1
                    st.rerun()
    elif _images:
        st.warning("Thiếu package `st-click-detector`. Sau khi cài dependency, thumbnail sẽ chọn ảnh trực tiếp mà không đổi URL.")
    else:
        st.markdown('<div class="gallery-hint">← Bấm + để thêm ảnh</div>', unsafe_allow_html=True)

_sel_idx = st.session_state["selected_image_index"]

# ── Khu vực xử lý & hiển thị kết quả ────────────────────────────────────────
if _images and 0 <= _sel_idx < len(_images):
    _chosen = _images[_sel_idx]
    _bgr    = _chosen["bgr"]
    _rgb    = np.array(_chosen["pil"])

    # Tiền xử lý: Resize & Pad + áp lọc pipeline
    _cur = resize_with_padding(_bgr, 224)
    for _step in PIPELINE_CHAINS.get(auto_pipeline, []):
        _cur = apply_filter(_cur, _step["name"], **_step.get("kwargs", {}))
    _filtered = cv2.cvtColor(_cur, cv2.COLOR_BGR2RGB)

    st.markdown(
        f"""
        <section class="vision-lab-panel">
            <div class="vision-lab-header">
                <div>
                    <div class="vision-kicker">Vision Lab Compare</div>
                    <div class="vision-title">Ảnh Trước & Sau Tiền Xử Lý</div>
                    <div class="vision-subtitle">Đang xem: <strong>{html.escape(_chosen['name'])}</strong></div>
                </div>
                <div class="pipeline-pill"><span>Pipeline</span>{html.escape(auto_pipeline)}</div>
            </div>
        """,
        unsafe_allow_html=True,
    )

    _c1, _c2 = st.columns(2)
    with _c1:
        st.markdown(
            '<div class="vision-card original"><div class="vision-card-title"><span>Ảnh gốc</span><span class="vision-badge">Original</span></div>',
            unsafe_allow_html=True,
        )
        st.image(_rgb, use_container_width=True)
        st.markdown(
            f"""
            <div class="vision-meta-row">
                <span class="vision-meta">Kích thước <strong>{_rgb.shape[1]}×{_rgb.shape[0]}</strong></span>
                <span class="vision-meta">Nguồn <strong>Upload</strong></span>
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with _c2:
        st.markdown(
            '<div class="vision-card processed"><div class="vision-card-title"><span>Sau tiền xử lý</span><span class="vision-badge">CNN ready</span></div>',
            unsafe_allow_html=True,
        )
        st.image(_filtered, use_container_width=True)
        st.markdown(
            f"""
            <div class="vision-meta-row">
                <span class="vision-meta">Kích thước <strong>224×224</strong></span>
                <span class="vision-meta">Bộ lọc <strong>{len(PIPELINE_CHAINS.get(auto_pipeline, []))} bước</strong></span>
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</section>", unsafe_allow_html=True)

    st.write("---")

    # AI Inference
    _model, _device = load_model(model_name, checkpoint_path, num_classes)

    if _model is None:
        st.subheader("Kết quả Dự Đoán")
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

            _sorted = sorted(
                enumerate(zip(class_names, _vals)),
                key=lambda item: item[1][1],
                reverse=True,
            )
            _confidence_label = "High confidence" if _bconf >= 80 else "Medium confidence" if _bconf >= 50 else "Low confidence"
            _rank_rows = []
            for _rank, (_class_idx, (_class_name, _score)) in enumerate(_sorted, start=1):
                _row_class = "rank-row top" if _rank == 1 else "rank-row"
                _safe_class = html.escape(str(_class_name))
                _rank_rows.append(
                    f"""
                    <div class="{_row_class}">
                        <span class="rank-num">{_rank}</span>
                        <span class="rank-name">{_safe_class}</span>
                        <span class="rank-track"><span class="rank-fill" style="width: {_score:.2f}%"></span></span>
                        <span class="rank-value">{_score:.1f}%</span>
                    </div>
                    """
                )
            _chart_bars = []
            for _i, (_class_name, _score) in enumerate(zip(class_names, _vals)):
                _safe_class = html.escape(str(_class_name))
                _bar_class = "prob-bar top" if _i == _bidx else "prob-bar"
                _height = max(float(_score), 1.2)
                _chart_bars.append(
                    f"""
                    <div class="prob-bar-item">
                        <div class="prob-bar-track">
                            <div class="{_bar_class}" style="height: {_height:.2f}%">
                                <span class="prob-bar-value">{_score:.1f}%</span>
                            </div>
                        </div>
                        <div class="prob-label" title="{_safe_class}">{_safe_class}</div>
                    </div>
                    """
                )

            st.markdown(
                f"""
                <section class="result-dashboard">
                    <div class="result-dashboard-head">
                        <div>
                            <div class="result-kicker">AI Classification Result</div>
                            <div class="result-title">Kết quả Dự Đoán</div>
                        </div>
                        <div class="result-status-pill">{_confidence_label}</div>
                    </div>
                """,
                unsafe_allow_html=True,
            )

            _r1, _r2 = st.columns([0.38, 0.62])
            with _r1:
                st.markdown(
                    f"""
                    <div class="result-hero">
                        <div class="result-label">Phương tiện nhận diện</div>
                        <div class="result-class">{html.escape(str(_bclass))}</div>
                        <div class="confidence-meter">
                            <div class="confidence-row">
                                <span>Độ tin cậy</span>
                                <span class="confidence-value">{_bconf:.1f}%</span>
                            </div>
                            <div class="confidence-track">
                                <div class="confidence-fill" style="width: {_bconf:.2f}%"></div>
                            </div>
                        </div>
                        <div class="result-meta-row">
                            <span class="result-meta">Model: {html.escape(model_name)}</span>
                            <span class="result-meta">Classes: {len(class_names)}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with _r2:
                st.markdown(
                    f"""
                    <div class="leaderboard-card">
                        <div class="leaderboard-title">Leaderboard xác suất</div>
                        {''.join(_rank_rows)}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            _probability_chart_html = f"""
                <div class="probability-chart-card">
                    <div class="chart-head">
                        <div class="chart-title">Biểu đồ xác suất theo lớp</div>
                        <div class="chart-note">Cột nổi bật là dự đoán cao nhất</div>
                    </div>
                    <div class="prob-chart" style="--bar-count: {len(class_names)};">
                        {''.join(_chart_bars)}
                    </div>
                </div>
                """
            if hasattr(st, "html"):
                st.html(_probability_chart_html)
            else:
                st.markdown(_probability_chart_html, unsafe_allow_html=True)

            st.markdown("</section>", unsafe_allow_html=True)

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
