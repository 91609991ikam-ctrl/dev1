"""カラーパレット抽出のコアロジック.

画像のピクセル色を K-means（教師なし学習：クラスタリング）で減色し、
各色の割合を求めて「アクセントカラー」を定義する。
さらに、そのアクセントカラーが抽出できる最小クラスタ数を二分探索で探す。

大学時代の研究プロトタイプの再現:
  1. 画像を入力
  2. クラスタリングで色を減らす（初期クラスタ数=16）
  3. クラスタの要素数から割合を算出し、降順に並べ替え
  4. 5〜10% 付近のクラスタを「アクセントカラー」と定義
  5. アクセントカラーが抽出できる最小クラスタ数を、
     様々なクラスタ数でクラスタリングして探す（二分探索）
  6. 最小クラスタ数でパレットを作成し、アクセントカラー・16進数・割合を示す
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning


# クラスタリングの再現性を保つための既定シード
DEFAULT_SEED = 42
# 「アクセントカラー」とみなす割合の既定範囲（5〜10%）
DEFAULT_ACCENT_LOW = 0.05
DEFAULT_ACCENT_HIGH = 0.10
# 探索するクラスタ数の既定範囲
DEFAULT_K_MIN = 2
DEFAULT_K_MAX = 16


@dataclass
class ColorEntry:
    """パレット中の 1 色を表す."""

    rgb: tuple[int, int, int]
    proportion: float  # 0.0〜1.0
    is_accent: bool

    @property
    def hex(self) -> str:
        r, g, b = self.rgb
        return f"#{r:02X}{g:02X}{b:02X}"

    @property
    def percent(self) -> float:
        return self.proportion * 100.0


@dataclass
class Palette:
    """あるクラスタ数 k で作成したパレット（割合の降順）."""

    k: int
    colors: list[ColorEntry]

    @property
    def has_accent(self) -> bool:
        return any(c.is_accent for c in self.colors)

    @property
    def accent_colors(self) -> list[ColorEntry]:
        return [c for c in self.colors if c.is_accent]


@dataclass
class SearchResult:
    """最小クラスタ数の二分探索の結果."""

    min_k: int | None  # アクセントが出た最小クラスタ数（出なければ None）
    final_palette: Palette | None
    base_palette: Palette  # 初期クラスタ数（k_max）で作ったパレット
    trace: list[tuple[int, bool]] = field(default_factory=list)  # 探索した (k, アクセント有無)
    palettes: dict[int, Palette] = field(default_factory=dict)  # k -> Palette のキャッシュ


def load_pixels(image: Image.Image, max_pixels: int = 100_000) -> np.ndarray:
    """PIL 画像を (N, 3) の RGB ピクセル配列に変換する.

    速度のため max_pixels を超える場合は縮小（サンプリング）する。
    """
    rgb = image.convert("RGB")

    # アスペクト比を保ったまま、総ピクセル数が max_pixels 以下になるよう縮小
    w, h = rgb.size
    n = w * h
    if max_pixels and n > max_pixels:
        scale = (max_pixels / n) ** 0.5
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
        rgb = rgb.resize(new_size, Image.BILINEAR)

    arr = np.asarray(rgb, dtype=np.float64).reshape(-1, 3)
    return arr


def make_palette(
    pixels: np.ndarray,
    k: int,
    accent_low: float = DEFAULT_ACCENT_LOW,
    accent_high: float = DEFAULT_ACCENT_HIGH,
    seed: int = DEFAULT_SEED,
) -> Palette:
    """ピクセル配列を k クラスタにクラスタリングしてパレットを作る.

    各クラスタの中心色を代表色、要素数の割合をその色の割合とする。
    割合が [accent_low, accent_high] に入る色をアクセントカラーとする。
    結果は割合の降順に並べる。
    """
    k = max(1, min(k, len(pixels)))

    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    with warnings.catch_warnings():
        # べた塗り画像など、実際の色数 < k のときの警告は想定内なので抑制
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        labels = km.fit_predict(pixels)
    centers = km.cluster_centers_

    counts = np.bincount(labels, minlength=k)
    total = counts.sum()
    proportions = counts / total if total else np.zeros(k)

    entries: list[ColorEntry] = []
    for i in range(k):
        r, g, b = (int(round(v)) for v in centers[i])
        r, g, b = (max(0, min(255, c)) for c in (r, g, b))
        prop = float(proportions[i])
        is_accent = accent_low <= prop <= accent_high
        entries.append(ColorEntry(rgb=(r, g, b), proportion=prop, is_accent=is_accent))

    # 割合の降順に並べ替え（多い順）
    entries.sort(key=lambda c: c.proportion, reverse=True)
    return Palette(k=k, colors=entries)


def find_min_accent_k(
    pixels: np.ndarray,
    k_min: int = DEFAULT_K_MIN,
    k_max: int = DEFAULT_K_MAX,
    accent_low: float = DEFAULT_ACCENT_LOW,
    accent_high: float = DEFAULT_ACCENT_HIGH,
    seed: int = DEFAULT_SEED,
) -> SearchResult:
    """アクセントカラーが抽出できる最小クラスタ数を二分探索で探す.

    手順（研究プロトタイプの再現）:
      - まず k_max（既定 16）でパレットを作り、アクセントカラーを定義・確認する。
      - 述語 P(k) = 「k クラスタでアクセントカラーが 1 つ以上出る」とし、
        P(k) を満たす最小の k を [k_min, k_max] で二分探索する。
        例: 16(定義)→8→4(✗)→6(○)→5(○) ⇒ 最小は 5

    注意: P(k) は厳密には単調とは限らないため、二分探索は近似的な探索である
    （ユーザー指定に基づき二分探索を採用）。各 k の結果は palettes にキャッシュする。
    """
    palettes: dict[int, Palette] = {}
    trace: list[tuple[int, bool]] = []

    def predicate(k: int) -> bool:
        if k not in palettes:
            palettes[k] = make_palette(pixels, k, accent_low, accent_high, seed)
        has = palettes[k].has_accent
        trace.append((k, has))
        return has

    # 初期クラスタ数 k_max でアクセントカラーを定義・確認
    base_palette = make_palette(pixels, k_max, accent_low, accent_high, seed)
    palettes[k_max] = base_palette

    # k_max でもアクセントが無ければ探索不能
    if not base_palette.has_accent:
        trace.append((k_max, False))
        return SearchResult(
            min_k=None,
            final_palette=None,
            base_palette=base_palette,
            trace=trace,
            palettes=palettes,
        )

    # P(k) を満たす最小の k を二分探索（P(k_max) は真であることが保証済み）
    lo, hi = k_min, k_max
    while lo < hi:
        mid = (lo + hi) // 2
        if predicate(mid):
            hi = mid
        else:
            lo = mid + 1

    min_k = lo
    return SearchResult(
        min_k=min_k,
        final_palette=palettes[min_k],
        base_palette=base_palette,
        trace=trace,
        palettes=palettes,
    )
