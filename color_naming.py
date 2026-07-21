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
    ("茶", "brown", (130, 80, 50)),  # 地味・くすんだ色の受け皿
    ("白", "white", (245, 245, 245)),
    ("灰", "gray", (140, 140, 140)),
    ("黒", "black", (30, 30, 30)),
]

BASIC_NAMES_JA = [c[0] for c in BASIC_COLORS]
BASIC_NAMES_EN = [c[1] for c in BASIC_COLORS]
_ANCHOR_RGB = np.array([c[2] for c in BASIC_COLORS], dtype=np.float64)

# 無彩色（白・灰・黒）の色名。明度ベースの振り分けに使う。
ACHROMATIC_NAMES_JA = {"白", "灰", "黒"}
# 地味・くすんだ色（茶）。無彩色とあわせてアクセントから除外する。
MUTED_NAMES_JA = {"茶"}
# アクセント対象外の色名（無彩色＋地味色）。
NON_ACCENT_NAMES_JA = ACHROMATIC_NAMES_JA | MUTED_NAMES_JA


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


def nearest_basic_index_lab(lab: np.ndarray) -> np.ndarray:
    """Lab から最も近い基本色名アンカーのインデックスを返す（ΔE76）."""
    lab = np.asarray(lab, dtype=np.float64).reshape(-1, 3)
    dist = np.linalg.norm(lab[:, None, :] - _ANCHOR_LAB[None, :, :], axis=2)
    return np.argmin(dist, axis=1)


def nearest_basic_index(rgb: np.ndarray) -> np.ndarray:
    """各色に最も近い基本色名のインデックスを返す（ΔE76）.

    入力 shape (N, 3) -> 出力 shape (N,)。
    """
    lab = srgb_to_lab(np.asarray(rgb, dtype=np.float64).reshape(-1, 3))
    return nearest_basic_index_lab(lab)


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

# 有彩色の参照点だけ（白・灰・黒を除く）。淡色でも色相名を付けるために使う。
_IW = BASIC_NAMES_JA.index("白")
_IG = BASIC_NAMES_JA.index("灰")
_IB = BASIC_NAMES_JA.index("黒")
_CHROMATIC_MASK = ~np.isin(_CN_TERM, [_IW, _IG, _IB])
_CN_LAB_CH = _CN_LAB[_CHROMATIC_MASK]
_CN_TERM_CH = _CN_TERM[_CHROMATIC_MASK]

# 無彩色とみなす彩度フロア（C*）。これ未満は明度で 白/灰/黒 に振り分ける。
NEUTRAL_CHROMA_FLOOR = 8.0

# ユーザー補正（人手のラベル）。近い色（ΔE < _CORR_RADIUS）はこの判定を最優先する。
# (HEX, 基本色名) を足すだけで局所的に命名を直せる。
USER_CORRECTIONS: list[tuple[str, str]] = [
    ("#B9AFC9", "灰"),
    ("#7B5C60", "茶"),
    ("#A0675C", "赤"),
]
_CORR_RADIUS = 15.0  # この ΔE 以内なら補正ラベルを採用


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


_CORR_LAB = (
    srgb_to_lab(np.array([_hex_to_rgb(h) for h, _ in USER_CORRECTIONS], dtype=np.float64))
    if USER_CORRECTIONS
    else np.empty((0, 3))
)
_CORR_TERM = np.array(
    [BASIC_NAMES_JA.index(name) for _, name in USER_CORRECTIONS], dtype=int
)


def _lab_chroma(lab: np.ndarray) -> np.ndarray:
    """Lab 配列 (...,3) の彩度 C=√(a²+b²) を返す."""
    return np.sqrt(lab[..., 1] ** 2 + lab[..., 2] ** 2)


def _achromatic_by_lightness(L: np.ndarray) -> np.ndarray:
    """無彩色（低彩度）を明度 L* で 白/灰/黒 に振り分ける."""
    out = np.full(L.shape, _IG, dtype=int)
    out[L >= 78] = _IW
    out[L <= 28] = _IB
    return out


def _knn_vote(lab: np.ndarray, ref_lab: np.ndarray, ref_term: np.ndarray, kk: int) -> np.ndarray:
    """距離重み付き k 近傍投票（近い点ほど強く効く）.

    単純多数決だとデータの多いクラス（緑など）に偏るため、近傍を距離の逆数で
    重み付けして票を集計する。入力 lab (N,3) -> 出力 (N,).
    """
    kk = min(kk, ref_lab.shape[0])
    dist = np.linalg.norm(lab[:, None, :] - ref_lab[None, :, :], axis=2)  # (N, M)
    idx = np.argpartition(dist, kk - 1, axis=1)[:, :kk]  # (N, kk)
    n_terms = int(ref_term.max()) + 1
    out = np.empty(len(lab), dtype=int)
    for i in range(len(lab)):
        nb = idx[i]
        w = 1.0 / (dist[i, nb] + 1e-6)
        score = np.zeros(n_terms)
        np.add.at(score, ref_term[nb], w)
        out[i] = int(np.argmax(score))
    return out


def classify_basic_index_lab(
    lab: np.ndarray, k_neighbors: int = 7, chroma_floor: float = NEUTRAL_CHROMA_FLOOR
) -> np.ndarray:
    """Lab から基本色名インデックスを返す（淡色対応の2段階）.

    - 彩度 C* < chroma_floor → 無彩色とみなし明度で 白/灰/黒
    - それ以外 → 有彩色の参照点だけで k 近傍投票（淡くても色相名が付く）
    入力 (N,3) Lab -> 出力 (N,).
    """
    lab = np.asarray(lab, dtype=np.float64).reshape(-1, 3)
    out = np.empty(len(lab), dtype=int)

    # 1) ユーザー補正: 近い色（ΔE < 半径）はその手動ラベルを最優先
    handled = np.zeros(len(lab), dtype=bool)
    if len(_CORR_LAB):
        cd = np.linalg.norm(lab[:, None, :] - _CORR_LAB[None, :, :], axis=2)  # (N, C)
        nearest = np.argmin(cd, axis=1)
        within = cd[np.arange(len(lab)), nearest] < _CORR_RADIUS
        out[within] = _CORR_TERM[nearest[within]]
        handled = within

    # 2) 残りは 彩度フロア（無彩色は明度で）→ 有彩色は k 近傍投票
    rest = ~handled
    if rest.any():
        sub = lab[rest]
        chroma = _lab_chroma(sub)
        achrom = chroma < chroma_floor
        res = np.empty(len(sub), dtype=int)
        if achrom.any():
            res[achrom] = _achromatic_by_lightness(sub[achrom, 0])
        if (~achrom).any():
            res[~achrom] = _knn_vote(sub[~achrom], _CN_LAB_CH, _CN_TERM_CH, k_neighbors)
        out[rest] = res
    return out


def classify_basic_index(rgb: np.ndarray, k_neighbors: int = 7) -> np.ndarray:
    """RGB から基本色名インデックスを返す（淡色対応）. 入力 (N,3) -> 出力 (N,)."""
    lab = srgb_to_lab(np.asarray(rgb, dtype=np.float64).reshape(-1, 3))
    return classify_basic_index_lab(lab, k_neighbors=k_neighbors)
