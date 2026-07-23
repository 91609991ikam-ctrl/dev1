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
    flatten_on_white,
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
# 重複除去（似た大きい色の近くのアクセントを外す）の定数
_DEDUP_DE = 15.0  # この ΔE 未満なら「似た色」
_DEDUP_RATIO = 2.0  # 相手がこの倍以上大きいとき外す
# 小さな鮮やかな色をノイズ併合から守る（＝小さくてもアクセント候補として残す）閾値
_VIVID_CHROMA = 45.0  # この彩度 C* 以上なら「鮮やか」
_VIVID_ISO_DE = 35.0  # 他の全色から Lab でこの ΔE 以上離れていれば「孤立」
# 影の畳み込み（同色相で暗いだけの色を基の色へ併合）の定数
_SHADOW_L_GAP = 8.0    # 基よりこれ以上暗い（L* 差）ときに影候補
_SHADOW_L_MAX = 45.0   # L* 差がこれを超えると別色とみなし畳み込まない（黒縁・濃色を保護）
_SHADOW_COS = 0.87     # a*b* 方向（色相）の一致度 cos。約 30°までのズレを許容
_SHADOW_NEUTRAL_MAG = 4.0  # a*b* の大きさがこれ未満なら「ほぼ無彩色」
_SHADOW_C_RATIO = 1.15  # 影の彩度は基の C* × これ + 加算値 まで（影は彩度が増えない）
_SHADOW_C_ADD = 3.0


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
# 「アクセントカラー」とみなす割合の既定範囲（0〜5%）
DEFAULT_ACCENT_LOW = 0.00
DEFAULT_ACCENT_HIGH = 0.05
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


_BG_TOL = 16.0  # 背景色とみなす ΔE（この範囲の色を背景として除外）


def _has_alpha(image: Image.Image) -> bool:
    return image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info)


def _detect_corner_bg(arr: np.ndarray) -> np.ndarray | None:
    """四隅が十分一致していれば、その色を「背景色」として返す（なければ None）."""
    h, w = arr.shape[:2]
    ps = max(4, min(h, w) // 12)
    corners = [arr[:ps, :ps], arr[:ps, -ps:], arr[-ps:, :ps], arr[-ps:, -ps:]]
    meds = np.array([np.median(c.reshape(-1, 3), axis=0) for c in corners])
    labs = srgb_to_lab(meds)
    mx = max(
        float(np.linalg.norm(labs[i] - labs[j]))
        for i in range(4) for j in range(i + 1, 4)
    )
    return np.median(meds, axis=0) if mx < 15.0 else None


def _display_background_mask(image: Image.Image, disp: Image.Image) -> np.ndarray | None:
    """表示画像 disp（RGB）上で「背景」とみなせる画素の 1D bool マスクを返す.

    透過画像は不透明部分以外、非透過画像は四隅一致の背景色に近い画素を背景とする。
    背景が見つからなければ None。load_pixels の除外条件と対応させ、減色画像で背景を
    元の色のまま残す（前景色で塗りつぶさない）ために使う。
    """
    disp_arr = np.asarray(disp, dtype=np.float64)
    flat = disp_arr.reshape(-1, 3)
    if _has_alpha(image):
        a = image.convert("RGBA").split()[-1].resize(disp.size, Image.BILINEAR)
        return np.asarray(a).reshape(-1) < 128
    bg = _detect_corner_bg(disp_arr)
    if bg is None:
        return None
    d = np.linalg.norm(srgb_to_lab(flat) - srgb_to_lab(bg[None, :]), axis=1)
    bg_mask = d < _BG_TOL
    # 前景が十分残る場合のみ背景として扱う（画像全体が一様な場合は塗らない）
    return bg_mask if (~bg_mask).sum() >= 50 else None


def load_pixels(
    image: Image.Image, max_pixels: int = 100_000, drop_background: bool = True
) -> np.ndarray:
    """PIL 画像を (N, 3) の RGB ピクセル配列に変換する（背景は除外）.

    - 透過画像は不透明部分のみを使う（透明部分＝背景は除外）。
    - 非透過画像は四隅が一致すればその色を背景とみなして除外する。
    速度のため max_pixels を超える場合は縮小する。
    """
    has_alpha = _has_alpha(image)
    img = image.convert("RGBA") if has_alpha else flatten_on_white(image)

    w, h = img.size
    n = w * h
    if max_pixels and n > max_pixels:
        scale = (max_pixels / n) ** 0.5
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)

    arr = np.asarray(img, dtype=np.float64)
    if has_alpha:
        rgb = arr[..., :3].reshape(-1, 3)
        keep = arr[..., 3].reshape(-1) >= 128
    else:
        rgb = arr.reshape(-1, 3)
        keep = np.ones(len(rgb), dtype=bool)
        if drop_background:
            bg = _detect_corner_bg(arr)
            if bg is not None:
                d = np.linalg.norm(srgb_to_lab(rgb) - srgb_to_lab(bg[None, :]), axis=1)
                cand = d >= _BG_TOL
                if cand.sum() >= 50:  # 主体が残る場合のみ背景を除外
                    keep = cand

    out = rgb[keep]
    return out if len(out) else rgb


def _to_rgb_tuple(arr) -> tuple[int, int, int]:
    """float の色配列を 0-255 の int タプルに丸めてクランプする."""
    r, g, b = (int(round(float(v))) for v in arr)
    return (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))


def _merge_small(
    centers: np.ndarray, counts: np.ndarray, min_prop: float, protect_vivid: bool = False
):
    """割合が min_prop 未満の小クラスタを、凝集的に最も近い色へ併合する.

    最小の（フロア未満の）クラスタを最近傍（Lab）へ併合、を繰り返す。近い色どうしが
    先にまとまるので、複数サブクラスタに割れた小さな色（例: 花の赤）は互いに併合されて
    生き残り、バラバラなノイズだけが大きな色へ吸収される。

    protect_vivid=True のとき、フロア未満でも「鮮やか（高彩度）かつ孤立（他の全色から
    Lab で遠い）」な色は併合しない。小さくても目立つアクセント（額の宝石など）が
    ノイズとして吸収されて消えるのを防ぐ。くすんだ色や近い色のノイズは従来通り吸収される。

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
    chroma = _lab_chroma(lab)
    floor = min_prop * total
    grp = np.arange(n)  # 各元クラスタが属する現在のグループ根
    gcount = counts.copy()
    active = list(range(n))

    def _protected(g: int, others: list[int]) -> bool:
        # 鮮やかで、他の全グループから十分離れている（孤立）小色は守る。
        # 近い色（同系の陰影など）が残っていれば孤立ではないので守らない
        # （その場合はまず同士で併合されてから判定される）。
        if not protect_vivid or chroma[g] < _VIVID_CHROMA or not others:
            return False
        dmin = min(float(np.linalg.norm(lab[g] - lab[o])) for o in others)
        return dmin >= _VIVID_ISO_DE

    while len(active) > 1:
        below = [
            g for g in active
            if gcount[g] < floor and not _protected(g, [o for o in active if o != g])
        ]
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


def _fold_shadows(centers: np.ndarray, counts: np.ndarray):
    """影（同色相で暗いだけの色）を、明るい方の大きな色へ畳み込む.

    イラストの陰影は多くの場合「同じ色相のまま L* が下がった（彩度も落ちた）」色で、
    別クラスタ・別色名として不要に増える。次を満たす小クラスタ c を基クラスタ t へ
    畳み込む（c の画素は t の割合へ合算するので割合は正直）:
      - t の方が画素数が多い（＝基の色）
      - c が t より _SHADOW_L_GAP〜_SHADOW_L_MAX だけ暗い（暗すぎる別色は畳まない）
      - c の彩度が t を超えて増えていない（鮮やかなアクセントを基へ吸わせない）
      - a*b* 方向（色相）が一致（cos ≥ _SHADOW_COS）、または両方ほぼ無彩色

    戻り値は _merge_small と同じ (keep_idx, remap)。
    """
    counts = np.asarray(counts, dtype=np.float64)
    n = len(centers)
    if n <= 1:
        return np.arange(n), np.arange(n)

    lab = srgb_to_lab(np.asarray(centers, dtype=np.float64))
    L = lab[:, 0]
    ab = lab[:, 1:]
    chroma = _lab_chroma(lab)
    mag = np.linalg.norm(ab, axis=1)

    root = np.arange(n)

    def find(x: int) -> int:
        while root[x] != x:
            root[x] = root[root[x]]
            x = root[x]
        return x

    def _same_hue(c: int, t: int) -> bool:
        nc, nt = mag[c], mag[t]
        if nc >= _SHADOW_NEUTRAL_MAG and nt >= _SHADOW_NEUTRAL_MAG:
            return float(ab[c] @ ab[t]) / (nc * nt) >= _SHADOW_COS
        return nc < _SHADOW_NEUTRAL_MAG and nt < _SHADOW_NEUTRAL_MAG

    # 暗い色から順に、基となる（明るく大きい同色相の）色へ畳み込む
    for c in np.argsort(L):
        c = int(c)
        best, best_d = -1, np.inf
        for t in range(n):
            if t == c or counts[t] <= counts[c]:
                continue
            dL = L[t] - L[c]
            if dL < _SHADOW_L_GAP or dL > _SHADOW_L_MAX:
                continue
            if chroma[c] > chroma[t] * _SHADOW_C_RATIO + _SHADOW_C_ADD:
                continue
            if not _same_hue(c, t):
                continue
            d = float(np.linalg.norm(ab[c] - ab[t]))
            if d < best_d:
                best_d, best = d, t
        if best >= 0:
            root[find(c)] = find(best)

    final = np.array([find(i) for i in range(n)], dtype=int)
    keep_idx = np.array(sorted(set(final.tolist())))
    pos = {g: i for i, g in enumerate(keep_idx)}
    remap = np.array([pos[final[c]] for c in range(n)], dtype=int)
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
    exclude_background: bool = False,
    area_gamma: float = 1.0,
    fold_shadows: bool = True,
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

    km = KMedoidsLite(
        n_clusters=k, random_state=seed, lab_space=lab_space, area_gamma=area_gamma
    )
    labels = km.fit_predict(pixels)
    centers = km.cluster_centers_

    counts = np.bincount(labels, minlength=k)

    # 影（同色相で暗いだけの色）を基の色へ畳み込み、不要な影色を減らす
    if fold_shadows and k > 1:
        keep_idx, remap = _fold_shadows(centers, counts)
        labels = remap[labels]
        centers = centers[keep_idx]
        k = len(keep_idx)
        counts = np.bincount(labels, minlength=k)

    # 極小クラスタを近い色へ併合（min_prop 未満を吸収）。救済ON なら鮮やか孤立色は守る
    if min_prop > 0:
        keep_idx, remap = _merge_small(
            centers, counts, min_prop, protect_vivid=area_gamma < 1.0
        )
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

            # グループの代表色は所属画素の「中央値」（縁の混色など外れ値に頑健）
            member_mask = np.isin(labels, member_clusters)
            member_pixels = pixels[member_mask]
            swatch = np.median(member_pixels, axis=0)

            entries.append(_entry(prop, _to_rgb_tuple(swatch), BASIC_NAMES_JA[b_idx]))

    # 背景（最大色）の割合を除いて再正規化した割合でアクセント帯を判定し直す。
    # 例: 黒背景にキャラがドンの場合、キャラの色が背景込みだと小さく見えて誤検出
    # されるのを防ぐ（背景を除いた中での割合で判定）。表示用 proportion は元のまま。
    if exclude_background and len(entries) > 1:
        bg = max(entries, key=lambda e: e.proportion)
        denom = max(1e-9, 1.0 - bg.proportion)
        for e in entries:
            if e is bg:
                e.is_accent = False
                continue
            p = e.proportion / denom
            excluded = exclude_achromatic and e.name in NON_ACCENT_NAMES_JA
            e.is_accent = (accent_low <= p <= accent_high) and not excluded

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

    # 重複除去: アクセントが「はるかに大きい似た色」の近く（ΔE小）にあるなら外す。
    # 例: 橙 0.5% が 茶 8.4% と似ている場合、同じ色の一部なのでアクセントにしない。
    if entries:
        e_lab = srgb_to_lab(np.array([e.rgb for e in entries], dtype=np.float64))
        for i, e in enumerate(entries):
            if not e.is_accent:
                continue
            for j, o in enumerate(entries):
                if o.proportion >= _DEDUP_RATIO * e.proportion and \
                        float(np.linalg.norm(e_lab[i] - e_lab[j])) < _DEDUP_DE:
                    e.is_accent = False
                    break

    # 小さく鮮やかで孤立した有彩色の救済: コントラスト/重複除去などで誤って外れても、
    # 「アクセント帯・有彩色・高彩度・自分より十分大きい色から Lab で孤立」なら拾い直す。
    # 額の宝石のような明確なアクセントが、各ゲートのすり抜けで消えるのを防ぐ（救済ON時）。
    if area_gamma < 1.0 and entries:
        v_lab = srgb_to_lab(np.array([e.rgb for e in entries], dtype=np.float64))
        v_chroma = _lab_chroma(v_lab)
        v_props = np.array([e.proportion for e in entries])
        for i, e in enumerate(entries):
            if e.is_accent or not (accent_low <= e.proportion <= accent_high):
                continue
            if exclude_achromatic and e.name in NON_ACCENT_NAMES_JA:
                continue
            if v_chroma[i] < _VIVID_CHROMA:
                continue
            larger = v_props >= _DEDUP_RATIO * e.proportion
            if larger.any():
                dmin = float(np.linalg.norm(v_lab[larger] - v_lab[i], axis=1).min())
                if dmin < _VIVID_ISO_DE:
                    continue
            e.is_accent = True

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
    area_gamma: float = 1.0,
    fold_shadows: bool = True,
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
        n_clusters=k, random_state=seed, lab_space=lab_space,
        chroma_gamma=chroma_gamma, area_gamma=area_gamma,
    )
    km.fit(fit_pixels)
    centers = km.cluster_centers_
    counts = np.bincount(km.labels_, minlength=k)

    # remap: 元クラスタ index -> 現在の index（各後処理で合成していく）
    remap = np.arange(k)
    # 影（同色相で暗いだけの色）を基の色へ畳み込む
    if fold_shadows and k > 1:
        keep_idx, rm = _fold_shadows(centers, counts)
        centers = centers[keep_idx]
        k = len(keep_idx)
        remap = rm[remap]
        counts = np.bincount(remap[km.labels_], minlength=k)
    # 極小クラスタを近い色へ併合（ノイズ色の除去）。救済ON なら鮮やか孤立色は守る
    if min_prop > 0:
        keep_idx, rm = _merge_small(
            centers, counts, min_prop, protect_vivid=area_gamma < 1.0
        )
        centers = centers[keep_idx]
        k = len(keep_idx)
        remap = rm[remap]
        counts = np.bincount(remap[km.labels_], minlength=k)

    total = counts.sum()
    proportions = counts / total if total else np.zeros(k)

    # --- 量子化画像（表示用に縮小してから各画素を最近傍中心の色へ置換）---
    disp = flatten_on_white(image)
    w, h = disp.size
    n = w * h
    if max_display_pixels and n > max_display_pixels:
        scale = (max_display_pixels / n) ** 0.5
        disp = disp.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
    disp_arr = np.asarray(disp, dtype=np.float64).reshape(-1, 3)
    disp_labels = remap[km.predict(disp_arr)]
    quant = centers[disp_labels]
    # 背景は学習から除外している。表示でも背景画素を最寄り前景色で塗らず元のまま残す
    # （白背景がクリーム等に化けて見える不具合を防ぐ）。
    bg_mask = _display_background_mask(image, disp)
    if bg_mask is not None:
        quant[bg_mask] = disp_arr[bg_mask]
    quant = quant.reshape(disp.height, disp.width, 3)
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
