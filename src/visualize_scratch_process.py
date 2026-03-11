import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import sys

# src ディレクトリをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.scratch_detector import ScratchDetector

def create_synthetic_scratch_image(width=256, height=256, scratch_intensity=60):
    img = np.random.normal(100, 15, (height, width)).astype(np.uint8)
    start_point = (50, 50)
    end_point = (200, 200)
    cv2.line(img, start_point, end_point, 160, 1)
    return img

def generate_scratch_animation(output_path="tests/scratch_process.gif"):
    img = create_synthetic_scratch_image()
    height, width = img.shape
    bs = 16
    pre_blur_sigma = 1.2
    
    # 前処理
    gray = cv2.GaussianBlur(img, (0, 0), pre_blur_sigma)
    accumulator = np.zeros((height, width), dtype=np.float32)
    global_mean = np.mean(gray)
    
    frames = []
    
    # 全16x16 = 256ステップだが、dy(16回)ごとにフレームを保存
    for dy in range(bs):
        for dx in range(bs):
            sub_w = ((width - dx) // bs) * bs
            sub_h = ((height - dy) // bs) * bs
            if sub_w <= 0 or sub_h <= 0: continue
            
            roi = gray[dy:dy+sub_h, dx:dx+sub_w]
            low_res = cv2.resize(roi, (sub_w // bs, sub_h // bs), interpolation=cv2.INTER_AREA)
            diff = low_res.astype(np.float32) - global_mean
            diff[diff < 0] = 0
            upsampled = cv2.resize(diff, (sub_w, sub_h), interpolation=cv2.INTER_NEAREST)
            accumulator[dy:dy+sub_h, dx:dx+sub_w] += upsampled
            
        # 1行分(dx 16回分)終わったらフレーム作成
        # 正規化して画像化
        max_val = np.max(accumulator)
        if max_val > 0:
            norm_acc = (accumulator / max_val * 255).astype(np.uint8)
        else:
            norm_acc = np.zeros_like(img, dtype=np.uint8)
            
        # PIL画像に変換して、キャプションを追加
        frame_pil = Image.fromarray(norm_acc).convert("RGB")
        draw = ImageDraw.Draw(frame_pil)
        draw.text((10, 10), f"Step: dy={dy}/15", fill=(255, 255, 0))
        frames.append(frame_pil)

    # 最後のフレームを少し長く表示するために数枚追加
    for _ in range(10):
        frames.append(frames[-1])
        
    # GIFとして保存
    frames[0].save(output_path, save_all=True, append_images=frames[1:], duration=200, loop=0)
    print(f"Animation saved to {output_path}")

if __name__ == "__main__":
    generate_scratch_animation()
