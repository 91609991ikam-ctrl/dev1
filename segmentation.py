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
from scipy import ndimage
from skimage.color import label2rgb
from skimage.segmentation import felzenszwalb, mark_boundaries, quickshift, slic

from color_naming import flatten_on_white

METHOD_SLIC = "slic"
METHOD_QUICKSHIFT = "quickshift"
METHOD_FELZENSZWALB = "felzenszwalb"

REGION_MEAN = "mean"
REGION_MEDIAN = "median"


def _region_median_image(arr: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """各領域を「チャンネルごとの中央値」で塗った画像を返す（外れ値に頑健）."""
    idx = np.unique(labels)
    lut = np.zeros((int(labels.max()) + 1, 3), dtype=np.float64)
    for ch in range(3):
        meds = ndimage.labeled_comprehension(
            arr[..., ch].astype(np.float64), labels, idx, np.median, np.float64, 0.0
        )
        lut[idx, ch] = meds
    return lut[labels]


@dataclass
class SegmentationResult:
    labels: np.ndarray  # (H, W) 領域ID
    mean_color_image: Image.Image  # 各領域＝平均色で塗った画像（色分析の入力）
    boundary_overlay: Image.Image  # 元画像＋領域境界
    n_regions: int


def _resized(image: Image.Image, max_side: int) -> Image.Image:
    """長辺が max_side を超えないように縮小（速度のため）. 透過は白背景へ合成."""
    rgb = flatten_on_white(image)
    w, h = rgb.size
    if max_side and max(w, h) > max_side:
        scale = max_side / max(w, h)
        rgb = rgb.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
    return rgb


def segment(
    image: Image.Image,
    method: str = METHOD_SLIC,
    n_segments: int = 400,
    compactness: float = 5.0,
    scale: float = 200.0,
    sigma: float = 0.8,
    min_size: int = 50,
    max_dist: float = 10.0,
    kernel_size: float = 5.0,
    ratio: float = 0.8,
    max_side: int = 900,
    region_agg: str = REGION_MEAN,
) -> SegmentationResult:
    """画像を領域分割し、確認用の画像もまとめて返す.

    method:
      - slic:         格子ベース。compactness を下げるほど色の境界に沿う。
      - quickshift:   モード探索。内容に密着し小領域を保持しやすい（やや遅い）。
      - felzenszwalb: グラフベース。scale で粒度調整。
    region_agg:
      - mean:   各領域を平均色で代表（既定）。
      - median: 各領域をチャンネル中央値で代表（縁の混色など外れ値に頑健）。
    """
    # quickshift は重いので入力をやや小さめにする
    side = 640 if method == METHOD_QUICKSHIFT else max_side
    rgb = _resized(image, side)
    arr = np.asarray(rgb)

    if method == METHOD_FELZENSZWALB:
        labels = felzenszwalb(arr, scale=scale, sigma=sigma, min_size=min_size)
    elif method == METHOD_QUICKSHIFT:
        labels = quickshift(
            arr, ratio=ratio, kernel_size=kernel_size, max_dist=max_dist, sigma=0
        )
    else:
        labels = slic(arr, n_segments=n_segments, compactness=compactness, start_label=0)

    # 各領域を代表色（平均 or 中央値）で塗った画像（＝色分析にかける入力そのもの）
    if region_agg == REGION_MEDIAN:
        rep = _region_median_image(arr, labels)
    else:
        rep = label2rgb(labels, arr, kind="avg", bg_label=-1)
    rep = np.clip(rep, 0, 255).astype(np.uint8)
    # 透過画像は元のアルファを保持（透明部分＝背景を下流で除外できるように）
    has_alpha = image.mode in ("RGBA", "LA") or (
        image.mode == "P" and "transparency" in image.info
    )
    if has_alpha:
        a = image.convert("RGBA").split()[-1].resize(rgb.size, Image.BILINEAR)
        mean_img = Image.fromarray(np.dstack([rep, np.asarray(a)]), "RGBA")
    else:
        mean_img = Image.fromarray(rep)

    # 元画像に境界線を重ねた確認画像
    ov = mark_boundaries(arr.astype(np.float64) / 255.0, labels, color=(1.0, 1.0, 0.0))
    overlay_img = Image.fromarray((np.clip(ov, 0, 1) * 255).astype(np.uint8))

    return SegmentationResult(
        labels=labels,
        mean_color_image=mean_img,
        boundary_overlay=overlay_img,
        n_regions=int(labels.max()) + 1,
    )
