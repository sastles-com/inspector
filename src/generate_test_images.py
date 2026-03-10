import cv2
import numpy as np
import os

def create_base_image(width=512, height=512, bg_mean=100, bg_std=15):
    """背景ノイズを持つベース画像を生成"""
    return np.random.normal(bg_mean, bg_std, (height, width)).astype(np.uint8)

def add_dot_scratch(img, x, y, radius=1, intensity=60, is_bright=True):
    """点状の傷（スポット）を追加"""
    val = int(img[y, x]) + intensity if is_bright else int(img[y, x]) - intensity
    val = np.clip(val, 0, 255)
    cv2.circle(img, (int(x), int(y)), int(radius), int(val), -1) # -1 で塗りつぶし

def add_line_scratch(img, start, end, thickness=1, intensity=50):
    """線状の傷を追加"""
    color = np.clip(100 + intensity, 0, 255)
    cv2.line(img, start, end, int(color), thickness)

def generate_samples():
    output_dir = "data/synthetic"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 点状の傷 (Dots)
    img_dots = create_base_image()
    for _ in range(20):
        rx, ry = np.random.randint(50, 460, 2)
        r_size = np.random.randint(1, 3)
        r_int = np.random.randint(40, 80)
        # 明るい点と暗い点を混ぜる
        add_dot_scratch(img_dots, rx, ry, radius=r_size, intensity=r_int, is_bright=(np.random.rand() > 0.5))
    cv2.imwrite(os.path.join(output_dir, "dot_scratches.png"), img_dots)

    # 2. 混合 (Mixed: Lines and Dots)
    img_mixed = create_base_image()
    add_line_scratch(img_mixed, (100, 100), (400, 400), thickness=1, intensity=60)
    add_line_scratch(img_mixed, (400, 100), (100, 400), thickness=2, intensity=-40) # 暗い線
    for _ in range(10):
        rx, ry = np.random.randint(50, 460, 2)
        add_dot_scratch(img_mixed, rx, ry, radius=2, intensity=70)
    cv2.imwrite(os.path.join(output_dir, "mixed_scratches.png"), img_mixed)

    # 3. 非常に薄い点 (Faint Dots)
    img_faint = create_base_image(bg_std=10)
    for i in range(5):
        # 背景平均100に対して、わずか20〜30の差分（ノイズに紛れやすい）
        add_dot_scratch(img_faint, 100 + i*80, 256, radius=1, intensity=25)
    cv2.imwrite(os.path.join(output_dir, "faint_dots.png"), img_faint)

    print(f"Synthetic images generated in {output_dir}")

if __name__ == "__main__":
    generate_samples()
