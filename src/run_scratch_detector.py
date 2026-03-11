import cv2
import numpy as np
import argparse
import os
import sys

# src ディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from scratch_detector import ScratchDetector

def main():
    parser = argparse.ArgumentParser(description="Scratch Detector for general images")
    parser.add_argument("input", help="Path to the input image file")
    parser.add_argument("--output", "-o", help="Path to save the result image", default=None)
    parser.add_argument("--block-size", "-b", type=int, default=16, help="Block size for the algorithm (default: 16)")
    parser.add_argument("--sigma", "-s", type=float, default=1.2, help="Pre-blur sigma (default: 1.2)")
    parser.add_argument("--show", action="store_true", help="Show the result using OpenCV window")

    args = parser.parse_args()

    # 画像の読み込み
    if not os.path.exists(args.input):
        print(f"Error: File not found: {args.input}")
        return

    img = cv2.imread(args.input)
    if img is None:
        print(f"Error: Could not decode image: {args.input}")
        return

    print(f"Processing: {args.input} (Size: {img.shape[1]}x{img.shape[0]})")
    
    # 検出器の初期化と実行
    detector = ScratchDetector(block_size=args.block_size, pre_blur_sigma=args.sigma)
    pop_out = detector.process(img)

    # 結果の保存
    if args.output:
        output_path = args.output
    else:
        # デフォルトの出力名: {original_name}_scratch.png
        base, _ = os.path.splitext(args.input)
        output_path = f"{base}_scratch.png"

    cv2.imwrite(output_path, pop_out)
    print(f"Result saved to: {output_path}")

    # 結果の表示 (オプション)
    if args.show:
        # 比較用に横に並べる
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        combined = np.hstack((gray, pop_out))
        cv2.imshow("Original (Gray) vs Scratch Pop-out", combined)
        print("Press any key to close the window.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
