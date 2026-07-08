"""領域分割（スーパーピクセル）の前段処理.

画像を色が近い小領域にまとめてから色分析にかけるための前処理。小さな鮮やかな
領域（花など）を「面」として1単位にし、にじみ（アンチエイリアス）ノイズを均す。

確認用に次の3つを返せる:
  - labels:      各画素の領域ID（2D配列）
  - mean_color_image:  各領域をその平均色で塗った画像（＝実際に色分析にかける入力）
  - boundary_overlay:  元画像に領域境界を描いた画像

軽量なスーパーピクセル（SLIC / Felzenszwalb）を使う。学習不要・CPUで動く。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image
from skimage.color import label2rgb
from skimage.segmentation import felzenszwalb, mark_boundaries, slic

METHOD_SLIC = "slic"
METHOD_FELZENSZWALB = "felzenszwalb"


@dataclass
class SegmentationResult:
    labels: np.ndarray  # (H, W) 領域ID
    mean_color_image: Image.Image  # 各領域＝平均色で塗った画像（色分析の入力）
    boundary_overlay: Image.Image  # 元画像＋領域境界
    n_regions: int


def _resized(image: Image.Image, max_side: int) -> Image.Image:
    """長辺が max_side を超えないように縮小（速度のため）."""
    rgb = image.convert("RGB")
    w, h = rgb.size
    if max_side and max(w, h) > max_side:
        scale = max_side / max(w, h)
        rgb = rgb.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
    return rgb


def segment(
    image: Image.Image,
    method: str = METHOD_SLIC,
    n_segments: int = 400,
    compactness: float = 10.0,
    scale: float = 200.0,
    sigma: float = 0.8,
    min_size: int = 50,
    max_side: int = 900,
) -> SegmentationResult:
    """画像を領域分割し、確認用の画像もまとめて返す."""
    rgb = _resized(image, max_side)
    arr = np.asarray(rgb)

    if method == METHOD_FELZENSZWALB:
        labels = felzenszwalb(arr, scale=scale, sigma=sigma, min_size=min_size)
    else:
        labels = slic(arr, n_segments=n_segments, compactness=compactness, start_label=0)

    # 各領域をその平均色で塗った画像（＝色分析にかける入力そのもの）
    avg = label2rgb(labels, arr, kind="avg", bg_label=-1)
    mean_img = Image.fromarray(np.clip(avg, 0, 255).astype(np.uint8))

    # 元画像に境界線を重ねた確認画像
    ov = mark_boundaries(arr.astype(np.float64) / 255.0, labels, color=(1.0, 1.0, 0.0))
    overlay_img = Image.fromarray((np.clip(ov, 0, 1) * 255).astype(np.uint8))

    return SegmentationResult(
        labels=labels,
        mean_color_image=mean_img,
        boundary_overlay=overlay_img,
        n_regions=int(labels.max()) + 1,
    )
