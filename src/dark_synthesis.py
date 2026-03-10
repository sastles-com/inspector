import cv2
import numpy as np
import argparse
import os
import glob

def dark_synthesis(image_paths):
    """
    複数の画像を比較暗(Darken)で合成する。
    各画素について、全画像の中の最小値を採用する。
    """
    if not image_paths:
        return None

    # 最初の画像をベースとして読み込む
    base_img = cv2.imread(image_paths[0])
    if base_img is None:
        print(f"Error: Could not read {image_paths[0]}")
        return None
    
    # 合成用のバッファ（初期値は最初の画像）
    # 比較暗なので、最小値を保持していく
    result = base_img.astype(np.uint8)

    print(f"Starting dark synthesis of {len(image_paths)} images...")
    
    for i in range(1, len(image_paths)):
        next_img = cv2.imread(image_paths[i])
        if next_img is None:
            print(f"Warning: Could not read {image_paths[i]}, skipping.")
            continue
        
        # サイズが異なる場合はリサイズ（ベースに合わせる）
        if next_img.shape != result.shape:
            next_img = cv2.resize(next_img, (result.shape[1], result.shape[0]))
            
        # 各画素の最小値をとる (比較暗合成)
        result = np.minimum(result, next_img)
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1} images...")

    return result

def main():
    parser = argparse.ArgumentParser(description="Dark Synthesis (Minimum Blend) Tool")
    parser.add_argument("input", help="Directory containing images or list of image files", nargs="+")
    parser.add_argument("--output", "-o", default="dark_synthesized.png", help="Output filename (default: dark_synthesized.png)")
    parser.add_argument("--ext", default="jpg", help="Image extension to look for if a directory is provided (default: jpg)")

    args = parser.parse_args()

    # 入力がディレクトリかファイルリストかを判定
    image_files = []
    for item in args.input:
        if os.path.isdir(item):
            # ディレクトリ内の指定拡張子のファイルを検索
            pattern = os.path.join(item, f"*.{args.ext}")
            image_files.extend(sorted(glob.glob(pattern)))
        elif os.path.isfile(item):
            image_files.append(item)

    if not image_files:
        print("No images found to process.")
        return

    # 合成実行
    final_image = dark_synthesis(image_files)

    if final_image is not None:
        cv2.imwrite(args.output, final_image)
        print(f"Successfully saved synthesized image to: {args.output}")

if __name__ == "__main__":
    main()
