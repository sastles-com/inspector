import cv2
import numpy as np
import pytest
import os
import sys

# src ディレクトリをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.kizki_detector import KizkiDetector

def create_synthetic_scratch_image(width=256, height=256, scratch_intensity=60):
    """
    テスト用の合成画像を生成する。
    背景に強めのノイズを加え、薄い「傷（線）」を1本引く。
    """
    # 背景ノイズ (平均100, 標準偏差15)
    img = np.random.normal(100, 15, (height, width)).astype(np.uint8)
    
    # 傷（線）を描画
    start_point = (50, 50)
    end_point = (200, 200)
    # 背景平均(100)より明るい輝度(160)
    cv2.line(img, start_point, end_point, 160, 1)
    
    return img, (start_point, end_point)

def test_kizki_detection_logic():
    # 1. 合成画像の作成
    img, (start, end) = create_synthetic_scratch_image()
    cv2.imwrite("tests/test_input_original.png", img)
    
    # 2. 検出器の初期化と適用 (新しい累積方式)
    detector = KizkiDetector(block_size=16, pre_blur_sigma=1.2)
    pop_out = detector.process(img)
    cv2.imwrite("tests/test_output_popout.png", pop_out)
    
    # 3. 検証: ポップアウト画像の傷の部分が、背景よりも明るくなっているか
    mid_point = (125, 125)
    bg_point = (10, 10)
    
    assert pop_out[mid_point[1], mid_point[0]] > pop_out[bg_point[1], bg_point[0]], \
        f"Scratch({pop_out[mid_point[1], mid_point[0]]}) should be brighter than background({pop_out[bg_point[1], bg_point[0]]})"

if __name__ == "__main__":
    # 簡易的な動作確認用
    img, _ = create_synthetic_scratch_image()
    detector = KizkiDetector(block_size=16)
    res = detector.process(img)
    cv2.imwrite("tests/test_result_popout_new.png", res)
    print("Test finished. Check tests/test_result_popout_new.png")
