import cv2
import numpy as np

class ScratchDetector:
    """
    傷のポップアウト検出アルゴリズムの実装クラス。
    人間の視覚生理機構（周辺視・固視微動）をモデル化し、
    低解像度化と位相シフトによる統合を行うことで、背景から傷をポップアウト（浮き彫り）させる。
    
    多重解像度版: 複数のブロックサイズで並列に処理し、それらを統合することで
    様々なスケールの傷（細い傷〜太い傷）を同時に検出する。
    """

    def __init__(self, block_sizes=None, pre_blur_sigma=1.2, block_size=None):
        """
        :param block_sizes: 格子（ブロック）のサイズのリスト。例: [8, 16, 32]
        :param block_size: 単一のブロックサイズ (block_sizes の互換)
        :param pre_blur_sigma: 高周波ノイズ抑制のための前段Gaussianぼかしのsigma。
        """
        if block_size is not None:
            resolved_block_sizes = block_size
        elif block_sizes is None:
            resolved_block_sizes = [16]
        else:
            resolved_block_sizes = block_sizes

        if isinstance(resolved_block_sizes, int):
            self.block_sizes = [resolved_block_sizes]
        else:
            self.block_sizes = resolved_block_sizes
        self.pre_blur_sigma = pre_blur_sigma

    def process(self, image):
        """
        入力画像に対して多重解像度アルゴリズムを適用し、ポップアウト画像を返す。
        :param image: 入力画像 (グレースケール推奨)
        :return: ポップアウト画像 (np.uint8)
        """
        # グレースケールに変換
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 前処理: 高周波ノイズの抑制
        if self.pre_blur_sigma > 0:
            gray = cv2.GaussianBlur(gray, (0, 0), self.pre_blur_sigma)

        height, width = gray.shape
        global_mean = np.mean(gray)
        
        # 最終的な統合用のバッファ
        final_accumulator = np.zeros((height, width), dtype=np.float32)
        
        for bs in self.block_sizes:
            # 各スケールごとの累積バッファ
            scale_acc = np.zeros((height, width), dtype=np.float32)
            
            # 位相シフトによる多角的サンプリング
            for dy in range(bs):
                for dx in range(bs):
                    sub_w = ((width - dx) // bs) * bs
                    sub_h = ((height - dy) // bs) * bs
                    if sub_w <= 0 or sub_h <= 0: continue
                    
                    roi = gray[dy:dy+sub_h, dx:dx+sub_w]
                    
                    # ブロック内の平均値を計算 (周辺視の模倣)
                    low_res = cv2.resize(roi, (sub_w // bs, sub_h // bs), interpolation=cv2.INTER_AREA)
                    
                    # 差分抽出 (白線も黒線も絶対値で捉える)
                    diff = np.abs(low_res.astype(np.float32) - global_mean)
                    
                    # 元のサイズに拡大（最近傍補間）
                    upsampled = cv2.resize(diff, (sub_w, sub_h), interpolation=cv2.INTER_NEAREST)
                    
                    # 累積
                    scale_acc[dy:dy+sub_h, dx:dx+sub_w] += upsampled
            
            # スケールごとの寄与を平均化 (サンプリング回数 bs*bs で割る)
            final_accumulator += (scale_acc / (bs * bs))

        # 正規化とコントラスト調整
        max_val = np.max(final_accumulator)
        if max_val > 0:
            result = (final_accumulator / max_val * 255).astype(np.uint8)
        else:
            result = np.zeros_like(gray, dtype=np.uint8)
            
        return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1])
        if img is not None:
            # 複数スケールで実行
            detector = ScratchDetector(block_sizes=[8, 16, 32])
            res = detector.process(img)
            cv2.imwrite("scratch_multi_result.png", res)
            print("Saved multi-scale result to scratch_multi_result.png")
