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
    BASIC_NAMES_JA,
    NEUTRAL_CHROMA_FLOOR,
    NON_ACCENT_NAMES_JA,
    classify_basic_index_lab,
    nearest_basic_index_lab,
    nearest_real_pixel,
    srgb_to_lab,
)
from color_naming import _lab_chroma  # noqa: E402


def _stretch_ab(lab: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """normalize モード: 有彩色（彩度フロア以上）の a*,b* だけ引き伸ばす.

    無彩色（低彩度）は微小ノイズを増幅して誤命名しないよう、そのまま残す。
    """
    lab = lab.copy()
    chroma0 = _lab_chroma(lab)
    w = weights if (weights is not None and np.sum(weights) > 0) else np.ones(len(lab))
    typ = float(np.average(chroma0, weights=w))
    s = min(_NORM_MAX, max(1.0, _NORM_TARGET / (typ + 1e-6)))
    mask = chroma0 >= NEUTRAL_CHROMA_FLOOR
    lab[mask, 1:] *= s
    return lab

# 色名の判定方式
NAMING_KNN = "knn"  # 人間の色名データ（XKCD）の k 近傍多数決
NAMING_ANCHOR = "anchor"  # 基本色アンカーへの最近傍（旧方式）
DEFAULT_NAMING = NAMING_KNN

# アクセント判定モード
ACCENT_ABSOLUTE = "absolute"  # 現行: 割合帯＋有彩色＋絶対コントラスト
ACCENT_RELATIVE = "relative"  # 画像全体の彩度分布に対する相対外れ値
ACCENT_NORMALIZE = "normalize"  # 画像ごとに彩度を正規化してから絶対判定
DEFAULT_ACCENT_MODE = ACCENT_ABSOLUTE

# 相対アクセントのパラメータ（定数）
_REL_MULT = 2.0  # 画像の典型彩度の何倍で外れ値とみなすか
_REL_FLOOR = 6.0  # 相対でも最低これだけ彩度が要る（純グレー画像の誤検出防止）
# 正規化アクセントのパラメータ（定数）
_NORM_TARGET = 30.0  # 典型彩度をこの水準へ引き伸ばす
_NORM_MAX = 4.0  # 引き伸ばし倍率の上限（ノイズ増幅抑制）


def _name_indices_lab(lab: np.ndarray, naming: str) -> np.ndarray:
    """Lab を基本色名インデックスに変換（方式で切替）."""
    if naming == NAMING_ANCHOR:
        return nearest_basic_index_lab(lab)
    return classify_basic_index_lab(lab)


def _name_indices(rgb: np.ndarray, naming: str, accent_mode: str = DEFAULT_ACCENT_MODE,
                  counts: np.ndarray | None = None) -> np.ndarray:
    """RGB を基本色名インデックスに変換（normalize モードでは彩度を引き伸ばす）."""
    lab = srgb_to_lab(np.asarray(rgb, dtype=np.float64).reshape(-1, 3))
    if accent_mode == ACCENT_NORMALIZE:
        lab = _stretch_ab(lab, counts)
    return _name_indices_lab(lab, naming)


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


def _merge_small(centers: np.ndarray, counts: np.ndarray, min_prop: float):
    """割合が min_prop 未満の小クラスタを、凝集的に最も近い色へ併合する.

    最小の（フロア未満の）クラスタを最近傍（Lab）へ併合、を繰り返す。近い色どうしが
    先にまとまるので、複数サブクラスタに割れた小さな色（例: 花の赤）は互いに併合されて
    生き残り、バラバラなノイズだけが大きな色へ吸収される。

    戻り値 (keep_idx, remap):
      keep_idx: 残すクラスタの元インデックス
      remap:    元クラスタ index -> keep_idx 内の位置（新インデックス）
    """
    counts = np.asarray(counts, dtype=np.float64)
    n = len(centers)
    total = counts.sum()
    if min_prop <= 0 or total == 0 or n <= 1:
        return np.arange(n), np.arange(n)

    lab = srgb_to_lab(np.asarray(centers, dtype=np.float64))
    floor = min_prop * total
    grp = np.arange(n)  # 各元クラスタが属する現在のグループ根
    gcount = counts.copy()
    active = list(range(n))

    while len(active) > 1:
        below = [g for g in active if gcount[g] < floor]
        if not below:
            break
        g = min(below, key=lambda x: gcount[x])  # 最小のフロア未満グループ
        others = [o for o in active if o != g]
        tgt = others[int(np.argmin([np.linalg.norm(lab[g] - lab[o]) for o in others]))]
        grp[grp == g] = tgt
        gcount[tgt] += gcount[g]
        gcount[g] = 0.0
        active.remove(g)

    keep_idx = np.array(sorted(active))
    pos = {g: i for i, g in enumerate(keep_idx)}
    remap = np.array([pos[grp[c]] for c in range(n)], dtype=int)
    return keep_idx, remap


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
    accent_mode: str = DEFAULT_ACCENT_MODE,
    min_prop: float = 0.0,
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

    # 極小クラスタを近い色へ併合（min_prop 未満を吸収）
    if min_prop > 0:
        keep_idx, remap = _merge_small(centers, counts, min_prop)
        labels = remap[labels]
        centers = centers[keep_idx]
        k = len(keep_idx)
        counts = np.bincount(labels, minlength=k)

    total = counts.sum()
    proportions = counts / total if total else np.zeros(k)

    def _entry(prop: float, rgb: tuple[int, int, int], name: str = "") -> ColorEntry:
        in_range = accent_low <= prop <= accent_high
        # 無彩色（白・灰・黒）と地味色（茶）はアクセントから除外
        is_excluded = exclude_achromatic and name in NON_ACCENT_NAMES_JA
        return ColorEntry(
            rgb=rgb,
            proportion=prop,
            is_accent=in_range and not is_excluded,
            name=name,
        )

    entries: list[ColorEntry] = []

    if not aggregate:
        for i in range(k):
            entries.append(_entry(float(proportions[i]), _to_rgb_tuple(centers[i])))
    else:
        # 命名用の Lab を用意（normalize モードでは有彩色の a*,b* を引き伸ばす）
        centers_lab = srgb_to_lab(centers)
        if accent_mode == ACCENT_NORMALIZE and total:
            lab_named = _stretch_ab(centers_lab, counts)
        else:
            lab_named = centers_lab

        # 各クラスタ中心を基本色名に対応づけ（方式で切替）
        cluster_basic = _name_indices_lab(lab_named, naming)  # shape (k,)
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

    # --- アクセント判定モードごとの後処理 ---
    if accent_mode == ACCENT_RELATIVE and entries:
        # 相対: 画像全体の典型彩度に対して十分外れた（彩度が高い）色だけをアクセントに。
        # 白っぽい画像なら淡い色でも“外れ値”として拾える。
        chroma = _lab_chroma(srgb_to_lab(np.array([e.rgb for e in entries], dtype=np.float64)))
        props = np.array([e.proportion for e in entries])
        typ = float(np.average(chroma, weights=props)) if props.sum() else 0.0
        thr = max(_REL_FLOOR, _REL_MULT * typ)
        for e, c in zip(entries, chroma):
            if e.is_accent and c < thr:
                e.is_accent = False
    elif accent_mode == ACCENT_ABSOLUTE and contrast_min > 0 and entries:
        # 絶対: アクセントは「主要色から ΔE で際立つ色」だけに絞る。
        # 主要色 = アクセント帯より割合が大きい色（無ければ最大の色）。
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
    # ACCENT_NORMALIZE: 命名を引き伸ばし済み色で行っているため、割合帯＋有彩色の
    # 基本判定（_entry）をそのまま使う（追加のゲートなし）。

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
    accent_mode: str = DEFAULT_ACCENT_MODE,
    min_prop: float = 0.0,
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

    # 極小クラスタを近い色へ併合（ノイズ色の除去）
    remap = np.arange(k)
    if min_prop > 0:
        keep_idx, remap = _merge_small(centers, counts, min_prop)
        centers = centers[keep_idx]
        k = len(keep_idx)
        counts = np.bincount(remap[km.labels_], minlength=k)

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
    disp_labels = remap[km.predict(disp_arr)]
    quant = centers[disp_labels].reshape(disp.height, disp.width, 3)
    quant_img = Image.fromarray(np.clip(np.round(quant), 0, 255).astype(np.uint8))

    # --- 生の k 色パレット（集約しない）---
    cluster_basic = _name_indices(centers, naming, accent_mode, counts)
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
    accent_mode: str = DEFAULT_ACCENT_MODE,
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
                exclude_achromatic, lab_space, contrast_min, naming, accent_mode,
            )
        has = palettes[k].has_accent
        trace.append((k, has))
        return has

    # 初期クラスタ数 k_max でアクセントカラーを定義・確認
    base_palette = make_palette(
        pixels, k_max, accent_low, accent_high, seed, aggregate,
        exclude_achromatic, lab_space, contrast_min, naming, accent_mode,
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
