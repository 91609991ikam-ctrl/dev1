"""色の近似（基本色名への分類）ロジック.

抽出された色（クラスタ中心など）を、少数の「基本色名」に近似する。
近さの判定は CIELAB 空間での色差 ΔE（ΔE76 = Lab 上のユークリッド距離）で行う。
これにより、知覚的に同じ系統の色（微妙に違う複数の赤など）を 1 つの色名に
まとめられる。
"""

from __future__ import annotations

import numpy as np


# 基本色名のアンカー（代表色, sRGB 0-255）。
# 「なぜこの代表色か」を後で詰めやすいよう、ここの表を編集するだけで調整できる。
# (日本語名, 英語名, (R, G, B))
BASIC_COLORS: list[tuple[str, str, tuple[int, int, int]]] = [
    ("赤", "red", (225, 30, 30)),
    ("橙", "orange", (240, 130, 20)),
    ("黄", "yellow", (240, 220, 40)),
    ("緑", "green", (40, 160, 60)),
    ("青", "blue", (40, 70, 200)),
    ("紫", "purple", (130, 60, 160)),
    ("桃", "pink", (240, 160, 180)),
    ("白", "white", (245, 245, 245)),
    ("灰", "gray", (140, 140, 140)),
    ("黒", "black", (30, 30, 30)),
]

BASIC_NAMES_JA = [c[0] for c in BASIC_COLORS]
BASIC_NAMES_EN = [c[1] for c in BASIC_COLORS]
_ANCHOR_RGB = np.array([c[2] for c in BASIC_COLORS], dtype=np.float64)

# 無彩色（白・灰・黒）の色名。アクセントカラーから除外する判定に使う。
ACHROMATIC_NAMES_JA = {"白", "灰", "黒"}


def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB(0-255) を CIELAB(D65) に変換する.

    入力 shape (..., 3) に対して同 shape の Lab を返す。
    """
    arr = np.asarray(rgb, dtype=np.float64) / 255.0

    # sRGB ガンマを解除して線形 RGB へ
    mask = arr > 0.04045
    linear = np.where(mask, ((arr + 0.055) / 1.055) ** 2.4, arr / 12.92)

    # 線形 sRGB -> XYZ (D65)
    m = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ]
    )
    xyz = linear @ m.T

    # D65 白色点で正規化
    white = np.array([0.95047, 1.00000, 1.08883])
    xyz = xyz / white

    # XYZ -> Lab
    eps = 216.0 / 24389.0
    kappa = 24389.0 / 27.0
    f = np.where(xyz > eps, np.cbrt(xyz), (kappa * xyz + 16.0) / 116.0)

    fx, fy, fz = f[..., 0], f[..., 1], f[..., 2]
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return np.stack([L, a, b], axis=-1)


_ANCHOR_LAB = srgb_to_lab(_ANCHOR_RGB)


def nearest_basic_index(rgb: np.ndarray) -> np.ndarray:
    """各色に最も近い基本色名のインデックスを返す（ΔE76）.

    入力 shape (N, 3) -> 出力 shape (N,)。
    """
    lab = srgb_to_lab(np.asarray(rgb, dtype=np.float64).reshape(-1, 3))
    # 各色 × 各アンカー の Lab 距離
    dist = np.linalg.norm(lab[:, None, :] - _ANCHOR_LAB[None, :, :], axis=2)
    return np.argmin(dist, axis=1)


def nearest_real_pixel(target_rgb: np.ndarray, candidate_pixels: np.ndarray) -> np.ndarray:
    """target の色に Lab 色差で最も近い「実在画素」を candidate から選ぶ.

    target_rgb: shape (3,) の代表色（平均色など。実在しないことがある）
    candidate_pixels: shape (M, 3) の実画素群
    戻り値: shape (3,) の実在画素色（float）
    """
    target_lab = srgb_to_lab(np.asarray(target_rgb, dtype=np.float64).reshape(1, 3))
    cand_lab = srgb_to_lab(np.asarray(candidate_pixels, dtype=np.float64))
    dist = np.linalg.norm(cand_lab - target_lab, axis=1)
    idx = int(np.argmin(dist))
    return np.asarray(candidate_pixels[idx], dtype=np.float64)


# --- 人間の色名データ（XKCD）による分類 -----------------------------------
# color_names_data は build 時に生成した実行時軽量データ（matplotlib 非依存）。
from color_names_data import RGB as _CN_RGB, TERM as _CN_TERM  # noqa: E402

_CN_LAB = srgb_to_lab(np.array(_CN_RGB, dtype=np.float64))
_CN_TERM = np.array(_CN_TERM, dtype=int)


def classify_basic_index(rgb: np.ndarray, k_neighbors: int = 7) -> np.ndarray:
    """人間の色名データの k 近傍多数決で基本色名インデックスを返す（ΔE76）.

    無彩色⇔有彩色の境界を人の色名分布から決めるため、暗い/くすんだ有彩色が
    無彩色に誤判定されにくい。入力 (N,3) -> 出力 (N,).
    """
    lab = srgb_to_lab(np.asarray(rgb, dtype=np.float64).reshape(-1, 3))
    kk = min(k_neighbors, _CN_LAB.shape[0])
    dist = np.linalg.norm(lab[:, None, :] - _CN_LAB[None, :, :], axis=2)  # (N, M)
    idx = np.argpartition(dist, kk - 1, axis=1)[:, :kk]  # (N, kk)
    neigh = _CN_TERM[idx]
    out = np.empty(len(lab), dtype=int)
    for i in range(len(lab)):
        vals, cnts = np.unique(neigh[i], return_counts=True)
        out[i] = int(vals[int(cnts.argmax())])
    return out
