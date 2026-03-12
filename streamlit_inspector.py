import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import sys

# src ディレクトリをパスに追加して ScratchDetector をインポート可能にする
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
from src.scratch_detector import ScratchDetector

st.set_page_config(page_title="Multi-Scale Scratch Inspector", layout="wide")

st.title("🔍 Multi-Scale Scratch Inspector")
st.markdown("""
人間の視覚生理機構をモデル化した多重解像度アルゴリズムです。
複数のブロックサイズを統合することで、微細な傷から大きなムラまで同時に浮き彫りにします。
""")

# --- サイドバー: 設定 ---
st.sidebar.header("📁 Data Selection")
upload_file = st.sidebar.file_uploader("Upload Image", type=["jpg", "jpeg", "png", "bmp"])

# サンプル画像のリスト取得 (dataディレクトリ)
data_dir = "data"
sample_files = []
if os.path.exists(data_dir):
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                sample_files.append(os.path.join(root, f))

selected_sample = st.sidebar.selectbox("Or Select Sample", ["None"] + sorted(sample_files))

st.sidebar.header("⚙️ Parameters")
# 複数スケールの選択
available_block_sizes = [4, 8, 12, 16, 20, 24, 32, 48, 64]
selected_block_sizes = st.sidebar.multiselect(
    "Select Block Sizes (Multi-Scale)",
    available_block_sizes,
    default=[8, 16]
)

sigma = st.sidebar.slider("Pre-blur Sigma", min_value=0.0, max_value=5.0, value=1.2, step=0.1)
threshold = st.sidebar.slider("Noise Threshold (0-255)", min_value=0, max_value=255, value=30, step=1)

# --- 画像の読み込み ---
input_image = None
image_name = ""

if upload_file is not None:
    image = Image.open(upload_file)
    input_image = np.array(image)
    image_name = upload_file.name
elif selected_sample != "None":
    input_image = cv2.imread(selected_sample)
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    image_name = os.path.basename(selected_sample)

# --- 処理と表示 ---
if input_image is not None:
    if not selected_block_sizes:
        st.warning("Please select at least one block size.")
    else:
        st.subheader(f"Processing: {image_name}")
        
        # 処理の実行
        with st.spinner(f"Applying Multi-Scale Scratch {selected_block_sizes}..."):
            detector = ScratchDetector(block_sizes=selected_block_sizes, pre_blur_sigma=sigma)
            raw_pop_out = detector.process(input_image)
            
            # 閾値処理の適用
            pop_out = raw_pop_out.copy()
            pop_out[pop_out < threshold] = 0

        # カラム表示 (2列に変更)
        col1, col2 = st.columns(2)
        
        # 共通のグレースケール元画像
        if len(input_image.shape) == 3:
            gray_input = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
        else:
            gray_input = input_image

        with col1:
            st.write("### Original (Gray)")
            st.image(gray_input, width=gray_input.shape[1])
            
        with col2:
            st.write(f"### Pop-out Heatmap (>{threshold})")
            # 1. 検出結果をカラーマップ(JET)に変換
            heatmap = cv2.applyColorMap(pop_out, cv2.COLORMAP_JET)
            heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            
            # 2. 背景（値が0の部分）を黒のままにするためのマスク処理
            # 閾値以下の部分は真っ黒(0,0,0)にする
            heatmap_rgb[pop_out == 0] = [0, 0, 0]
            
            st.image(heatmap_rgb, width=heatmap_rgb.shape[1])

        with st.expander("Details & Statistics"):
            st.write(f"Image Size: {input_image.shape[1]}x{input_image.shape[0]}")
            st.write(f"Applied Scales: {selected_block_sizes}")
            st.write(f"Intensity Range: {np.min(pop_out)} - {np.max(pop_out)}")
            
            # 輝度分布
            hist_values = np.histogram(pop_out, bins=256, range=(0, 256))[0]
            st.line_chart(hist_values)

else:
    st.info("Please upload an image or select a sample from the sidebar.")

st.sidebar.markdown("---")
st.sidebar.write("Multi-Scale approach detects various sizes of anomalies.")
