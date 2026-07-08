"""カラーパレット抽出のコアロジック.

画像のピクセル色を k-medoids（教師なし学習：クラスタリング）で減色し、
各色の割合を求めて「アクセントカラー」を定義する。
さらに、そのアクセントカラーが抽出できる最小クラスタ数を二分探索で探す。

大学時代の研究プロトタイプの再現:
  1. 画像を入力
  2. クラスタリングで色を減らす（初期クラスタ数=16）
  3. クラスタの要素数から割合を算出し、降順に並べ替え
  4. 0〜10% のクラスタを「アクセントカラー」と定義
  5. アクセントカラーが抽出できる最小クラスタ数を、
     様々なクラスタ数でクラスタリングして探す（二分探索）
  6. 最小クラスタ数でパレットを作成し、アクセントカラー・16進数・割合を示す
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from clustering import KMedoidsLite
from color_naming import (
    ACHROMATIC_NAMES_JA,
    BASIC_NAMES_JA,
    classify_basic_index,
    nearest_basic_index,
    nearest_real_pixel,
    srgb_to_lab,
)

# 色名の判定方式
NAMING_KNN = "knn"  # 人間の色名データ（XKCD）の k 近傍多数決
NAMING_ANCHOR = "anchor"  # 基本色アンカーへの最近傍（旧方式）
DEFAULT_NAMING = NAMING_KNN


def _name_indices(centers: np.ndarray, naming: str) -> np.ndarray:
    """代表色を基本色名インデックスに変換（方式で切替）."""
    if naming == NAMING_ANCHOR:
        return nearest_basic_index(centers)
    return classify_basic_index(centers)


# クラスタリングの再現性を保つための既定シード
DEFAULT_SEED = 42
# 「アクセントカラー」とみなす割合の既定範囲（0〜10%）
DEFAULT_ACCENT_LOW = 0.00
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
    name: str = ""  # 基本色名（集約時のみ。例: "赤"）

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


@dataclass
class ClusteredImage:
    """ある k でクラスタリングした「量子化画像」と「生の k 色パレット」."""

    image: Image.Image  # 各画素を所属クラスタの中心色に置換した減色画像
    palette: Palette  # 集約せず各クラスタ中心をそのまま 1 色としたパレット


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


def _to_rgb_tuple(arr) -> tuple[int, int, int]:
    """float の色配列を 0-255 の int タプルに丸めてクランプする."""
    r, g, b = (int(round(float(v))) for v in arr)
    return (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))


def make_palette(
    pixels: np.ndarray,
    k: int,
    accent_low: float = DEFAULT_ACCENT_LOW,
    accent_high: float = DEFAULT_ACCENT_HIGH,
    seed: int = DEFAULT_SEED,
    aggregate: bool = True,
    exclude_achromatic: bool = True,
    lab_space: bool = True,
    contrast_min: float = 0.0,
    naming: str = DEFAULT_NAMING,
) -> Palette:
    """ピクセル配列を k クラスタにクラスタリングしてパレットを作る.

    aggregate=True（既定）の場合:
      各クラスタ中心を CIELAB ΔE で最も近い基本色名に対応づけ、同じ色名の
      クラスタの割合を合算する。各色名の代表色（スウィッチ）は、そのグループの
      割合加重平均色に最も近い「実在画素色」を画像から選ぶ。
      これにより、分裂した同系色がまとまりアクセントが安定して出やすくなる。

    aggregate=False の場合:
      各クラスタ中心をそのまま 1 色として扱う（従来動作）。

    いずれも割合が [accent_low, accent_high] に入る色をアクセントとし、
    結果は割合の降順に並べる。
    """
    k = max(1, min(k, len(pixels)))

    km = KMedoidsLite(n_clusters=k, random_state=seed, lab_space=lab_space)
    labels = km.fit_predict(pixels)
    centers = km.cluster_centers_

    counts = np.bincount(labels, minlength=k)
    total = counts.sum()
    proportions = counts / total if total else np.zeros(k)

    def _entry(prop: float, rgb: tuple[int, int, int], name: str = "") -> ColorEntry:
        in_range = accent_low <= prop <= accent_high
        # 無彩色（白・灰・黒）はアクセントから除外（exclude_achromatic=True のとき）
        is_achromatic = exclude_achromatic and name in ACHROMATIC_NAMES_JA
        return ColorEntry(
            rgb=rgb,
            proportion=prop,
            is_accent=in_range and not is_achromatic,
            name=name,
        )

    entries: list[ColorEntry] = []

    if not aggregate:
        for i in range(k):
            entries.append(_entry(float(proportions[i]), _to_rgb_tuple(centers[i])))
    else:
        # 各クラスタ中心を基本色名に対応づけ（方式で切替）
        cluster_basic = _name_indices(centers, naming)  # shape (k,)
        for b_idx in np.unique(cluster_basic):
            member_clusters = np.where(cluster_basic == b_idx)[0]
            group_count = counts[member_clusters].sum()
            if group_count == 0:
                continue
            prop = float(group_count / total) if total else 0.0

            # グループの割合加重平均色（合成色になりうる）
            weights = counts[member_clusters].astype(np.float64)
            avg = np.average(centers[member_clusters], axis=0, weights=weights)

            # 平均色に最も近い「実在画素」をグループ内から選んでスウィッチにする
            member_mask = np.isin(labels, member_clusters)
            member_pixels = pixels[member_mask]
            swatch = nearest_real_pixel(avg, member_pixels)

            entries.append(_entry(prop, _to_rgb_tuple(swatch), BASIC_NAMES_JA[b_idx]))

    # コントラスト・ゲート: アクセントは「主要色から ΔE で際立つ色」だけに絞る。
    # 主要色 = アクセント帯より割合が大きい色（無ければ最大の色）。
    if contrast_min > 0 and entries:
        dominant = [e for e in entries if e.proportion > accent_high]
        if not dominant:
            dominant = [max(entries, key=lambda e: e.proportion)]
        dom_lab = srgb_to_lab(np.array([e.rgb for e in dominant], dtype=np.float64))
        for e in entries:
            if not e.is_accent:
                continue
            e_lab = srgb_to_lab(np.array([e.rgb], dtype=np.float64))[0]
            if float(np.linalg.norm(dom_lab - e_lab, axis=1).min()) < contrast_min:
                e.is_accent = False

    # 割合の降順に並べ替え（多い順）
    entries.sort(key=lambda c: c.proportion, reverse=True)
    return Palette(k=k, colors=entries)


def cluster_and_quantize(
    image: Image.Image,
    k: int,
    accent_names: set[str] | None = None,
    seed: int = DEFAULT_SEED,
    max_fit_pixels: int = 100_000,
    max_display_pixels: int = 480_000,
    lab_space: bool = True,
    chroma_gamma: float = 0.0,
    naming: str = DEFAULT_NAMING,
) -> ClusteredImage:
    """k クラスタでクラスタリングし、「量子化画像」と「生の k 色パレット」を返す.

    - 量子化画像: 各画素を所属クラスタの中心色に置き換えた減色画像。
    - パレット: 集約（基本色名での合算）はせず、各クラスタ中心をそのまま 1 色とする。
      accent_names に含まれる基本色名に対応するクラスタを「アクセント」として印付けする
      （アクセント判定自体は集約ベースの探索結果を渡して使う）。
    """
    accent_names = accent_names or set()

    fit_pixels = load_pixels(image, max_pixels=max_fit_pixels)
    k = max(1, min(k, len(fit_pixels)))

    km = KMedoidsLite(
        n_clusters=k, random_state=seed, lab_space=lab_space, chroma_gamma=chroma_gamma
    )
    km.fit(fit_pixels)
    centers = km.cluster_centers_

    counts = np.bincount(km.labels_, minlength=k)
    total = counts.sum()
    proportions = counts / total if total else np.zeros(k)

    # --- 量子化画像（表示用に縮小してから各画素を最近傍中心の色へ置換）---
    disp = image.convert("RGB")
    w, h = disp.size
    n = w * h
    if max_display_pixels and n > max_display_pixels:
        scale = (max_display_pixels / n) ** 0.5
        disp = disp.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
    disp_arr = np.asarray(disp, dtype=np.float64).reshape(-1, 3)
    disp_labels = km.predict(disp_arr)
    quant = centers[disp_labels].reshape(disp.height, disp.width, 3)
    quant_img = Image.fromarray(np.clip(np.round(quant), 0, 255).astype(np.uint8))

    # --- 生の k 色パレット（集約しない）---
    cluster_basic = _name_indices(centers, naming)
    entries: list[ColorEntry] = []
    for i in range(k):
        if counts[i] == 0:
            continue
        name = BASIC_NAMES_JA[cluster_basic[i]]
        entries.append(
            ColorEntry(
                rgb=_to_rgb_tuple(centers[i]),
                proportion=float(proportions[i]),
                is_accent=name in accent_names,
                name=name,
            )
        )
    entries.sort(key=lambda c: c.proportion, reverse=True)
    return ClusteredImage(image=quant_img, palette=Palette(k=k, colors=entries))


def find_min_accent_k(
    pixels: np.ndarray,
    k_min: int = DEFAULT_K_MIN,
    k_max: int = DEFAULT_K_MAX,
    accent_low: float = DEFAULT_ACCENT_LOW,
    accent_high: float = DEFAULT_ACCENT_HIGH,
    seed: int = DEFAULT_SEED,
    aggregate: bool = True,
    exclude_achromatic: bool = True,
    lab_space: bool = True,
    contrast_min: float = 0.0,
    naming: str = DEFAULT_NAMING,
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
            palettes[k] = make_palette(
                pixels, k, accent_low, accent_high, seed, aggregate,
                exclude_achromatic, lab_space, contrast_min, naming,
            )
        has = palettes[k].has_accent
        trace.append((k, has))
        return has

    # 初期クラスタ数 k_max でアクセントカラーを定義・確認
    base_palette = make_palette(
        pixels, k_max, accent_low, accent_high, seed, aggregate,
        exclude_achromatic, lab_space, contrast_min, naming,
    )
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
