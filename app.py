"""カラーパレット抽出 Streamlit アプリ.

イラスト・絵画・写真などの画像をアップロードすると、使われている色を
K-means クラスタリングで抽出し、CIELAB ΔE で基本色名（赤・橙・黄…）に
近似・集約してカラーパレットを作成する。
アクセントカラー（5〜10% 付近の色）が抽出できる最小クラスタ数を
二分探索で探し、そのパレットを表示する。

表示レイアウト:
    入力画像 → 横向きの積み上げ棒グラフ → 丸＋色名＋#RRGGBB＋割合 の凡例

起動:
    streamlit run app.py
"""

from __future__ import annotations

import io

import streamlit as st
from PIL import Image

from color_palette import (
    DEFAULT_ACCENT_HIGH,
    DEFAULT_ACCENT_LOW,
    DEFAULT_K_MAX,
    DEFAULT_SEED,
    ACCENT_ABSOLUTE,
    ACCENT_NORMALIZE,
    ACCENT_RELATIVE,
    NAMING_ANCHOR,
    NAMING_KNN,
    Palette,
    cluster_and_quantize,
    load_pixels,
    make_palette,
)
from segmentation import (
    METHOD_FELZENSZWALB,
    METHOD_QUICKSHIFT,
    METHOD_SLIC,
    segment,
)


st.set_page_config(page_title="カラーパレット抽出", page_icon="🎨", layout="wide")


def _text_color_for(rgb: tuple[int, int, int]) -> str:
    """背景色に対して読みやすい文字色（黒/白）を返す."""
    r, g, b = rgb
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    return "#000000" if luminance > 0.55 else "#FFFFFF"


def palette_html(palette: Palette) -> str:
    """パレットの「横向き積み上げ棒グラフ＋丸の凡例」HTML を組み立てて返す.

    1) 各色を幅＝割合のセグメントとして並べた横棒
    2) その下に、丸（その色）＋色名＋#RRGGBB＋割合 の凡例リスト
    """
    if not palette.colors:
        return ""

    # 描画は割合の小さい順（パレット内部は大きい順で保持）
    colors = list(reversed(palette.colors))

    # --- 1) 横向き積み上げ棒グラフ ---
    segments = ""
    for c in colors:
        width = c.proportion * 100.0
        fg = _text_color_for(c.rgb)
        # 幅が十分あるセグメントにだけ割合ラベルを載せる
        label = f"{c.percent:.0f}%" if width >= 7 else ""
        segments += (
            f'<div title="{c.name} {c.hex} {c.percent:.1f}%" '
            f'style="width:{width}%;background:{c.hex};color:{fg};'
            f"display:flex;align-items:center;justify-content:center;"
            f'font-size:0.8rem;font-family:monospace;">{label}</div>'
        )
    bar = (
        '<div style="display:flex;width:100%;height:56px;border-radius:8px;'
        f'overflow:hidden;border:1px solid #ccc;">{segments}</div>'
    )

    # --- 2) 丸の凡例リスト ---
    rows = ""
    for c in colors:
        name = f"{c.name}　" if c.name else ""
        star = (
            '<span style="color:#FF3B30;font-weight:bold;">　⭐ アクセント</span>'
            if c.is_accent
            else ""
        )
        rows += (
            '<div style="display:flex;align-items:center;gap:12px;margin:8px 0;">'
            f'<span style="display:inline-block;width:24px;height:24px;'
            f"border-radius:50%;background:{c.hex};border:1px solid #bbb;"
            'flex:none;"></span>'
            f'<span style="font-family:monospace;font-size:1rem;">'
            f"{name}{c.hex}　—　{c.percent:.1f}%{star}</span>"
            "</div>"
        )

    return bar + '<div style="height:14px;"></div>' + rows


def render_palette(palette: Palette) -> None:
    """パレットを描画する（横棒グラフ＋丸の凡例）."""
    html = palette_html(palette)
    if html:
        st.markdown(html, unsafe_allow_html=True)


def palette_table(palette: Palette) -> list[dict]:
    """パレットを表形式のデータに変換."""
    rows = []
    for i, c in enumerate(palette.colors, start=1):
        rows.append(
            {
                "順位": i,
                "色名": c.name,
                "16進数": c.hex,
                "RGB": f"{c.rgb[0]}, {c.rgb[1]}, {c.rgb[2]}",
                "割合(%)": round(c.percent, 2),
                "アクセント": "⭐" if c.is_accent else "",
            }
        )
    return rows


def _png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


@st.cache_data(show_spinner=False)
def segment_image(
    file_bytes: bytes, method: str, n_segments: int, compactness: float,
    scale: float, max_dist: float,
):
    """領域分割し、平均色マップ・境界オーバーレイ・領域数を返す（キャッシュ対象）."""
    seg = segment(
        Image.open(io.BytesIO(file_bytes)),
        method=method,
        n_segments=n_segments,
        compactness=compactness,
        scale=scale,
        max_dist=max_dist,
    )
    return seg.mean_color_image, seg.boundary_overlay, seg.n_regions


@st.cache_data(show_spinner=False)
def analyze_image(
    file_bytes: bytes,
    max_pixels: int,
    k: int,
    accent_low: float,
    accent_high: float,
    seed: int,
    aggregate: bool,
    exclude_achromatic: bool,
    lab_space: bool,
    contrast_min: float,
    chroma_gamma: float,
    naming: str,
    accent_mode: str,
    min_prop: float,
    exclude_background: bool,
) -> dict:
    """1 枚の画像を、固定クラスタ数 k で解析して結果を返す（キャッシュ対象）."""
    image = Image.open(io.BytesIO(file_bytes))
    pixels = load_pixels(image, max_pixels=max_pixels)

    # 固定 k で集約パレットを作りアクセントを判定（最小 k 探索はしない）
    agg_palette = make_palette(
        pixels, k, accent_low, accent_high, seed, aggregate,
        exclude_achromatic, lab_space, contrast_min, naming, accent_mode,
        min_prop, exclude_background,
    )
    accent_names = {c.name for c in agg_palette.accent_colors}
    clustered = cluster_and_quantize(
        image,
        k,
        accent_names=accent_names,
        seed=seed,
        max_fit_pixels=max_pixels,
        lab_space=lab_space,
        chroma_gamma=chroma_gamma,
        naming=naming,
        accent_mode=accent_mode,
        min_prop=min_prop,
    )
    # 集約カラー（全色: 名前・hex・割合・アクセント可否）
    agg_colors = [
        {"name": c.name, "hex": c.hex, "percent": round(c.percent, 2), "accent": c.is_accent}
        for c in agg_palette.colors
    ]
    return {
        "k": k,
        "no_accent": not agg_palette.has_accent,
        "palette": clustered.palette,
        "quant": clustered.image,
        "accents": agg_palette.accent_colors,
        "agg_colors": agg_colors,
    }


def render_result(
    image: Image.Image, res: dict, accent_low: float, accent_high: float, seg=None
) -> None:
    """1 枚分の結果（入力画像・領域分割・減色画像・パレット）を描画する."""
    img_col, _ = st.columns([2, 1])
    img_col.subheader("入力画像")
    img_col.image(image, use_container_width=True)

    # 領域分割の確認ビュー（境界オーバーレイ ＋ 領域平均色マップ）
    if seg is not None:
        mean_img, overlay_img, n_regions = seg
        img_col.subheader(f"領域分割（境界／{n_regions} 領域）")
        img_col.image(overlay_img, use_container_width=True)
        img_col.subheader("領域の平均色マップ（＝色分析の入力）")
        img_col.image(mean_img, use_container_width=True)

    img_col.subheader(f"クラスタリング後の画像（k = {res['k']}）")
    img_col.image(res["quant"], use_container_width=True)

    if res["no_accent"]:
        st.warning(
            f"指定した範囲（{accent_low*100:.1f}〜{accent_high*100:.1f}%）で、"
            f"k={res['k']} ではアクセントカラーが見つかりませんでした。"
            "アクセント範囲を広げるか、クラスタ数 k を変えて試してください。"
        )
    else:
        st.success(f"✅ k = {res['k']} でアクセントカラーが見つかりました。")

    final = res["palette"]
    st.subheader(f"🎨 カラーパレット（k = {res['k']}・{len(final.colors)}色）")
    render_palette(final)

    accents = res["accents"]
    if accents:
        st.markdown(
            "**アクセントカラー（集約ベース）:** "
            + " / ".join(
                f"{c.name + ' ' if c.name else ''}`{c.hex}` ({c.percent:.1f}%)"
                for c in accents
            )
        )

    # 集約カラー（全色）: アクセント以外も含めた基本色名ごとの割合
    st.markdown("**集約カラー（全色・割合の多い順）:**")
    st.table(
        [
            {
                "色名": c["name"],
                "16進数": c["hex"],
                "割合(%)": c["percent"],
                "アクセント": "⭐" if c["accent"] else "",
            }
            for c in res["agg_colors"]
        ]
    )

    with st.expander("📋 パレットの詳細（生の k 色）"):
        st.table(palette_table(final))


def main() -> None:
    st.title("🎨 カラーパレット抽出アプリ")
    st.caption(
        "画像を領域分割 → k-medoids で固定クラスタ数 k に減色 → 基本色名に集約して"
        "パレットを作り、アクセントカラーを判定します。"
    )

    # ---- サイドバー：パラメータ ----
    with st.sidebar:
        st.header("⚙️ パラメータ")

        st.subheader("領域分割（前段）")
        seg_on = st.toggle(
            "領域分割を前段に使う",
            value=True,
            help="ON: 先にスーパーピクセルで小領域にまとめ、領域平均色を色分析に使う。"
            "小さな鮮やか領域を面として拾い、にじみノイズを均します。",
        )
        seg_method_label = st.radio(
            "分割手法",
            options=[
                "Felzenszwalb（グラフ・推奨）",
                "Quickshift（内容密着・やや遅い）",
                "SLIC（格子・高速）",
            ],
            index=0,
            disabled=not seg_on,
            help="Felzenszwalb は速くて内容に沿う（既定）。Quickshift はより密着だが遅い。"
            "SLIC は高速だが compactness を下げないと格子っぽい。",
        )
        if seg_method_label.startswith("SLIC"):
            seg_method = METHOD_SLIC
        elif seg_method_label.startswith("Quickshift"):
            seg_method = METHOD_QUICKSHIFT
        else:
            seg_method = METHOD_FELZENSZWALB

        seg_max_dist = st.slider(
            "Quickshift: 粒度 (max_dist)",
            min_value=4.0, max_value=300.0, value=250.0, step=2.0,
            disabled=(not seg_on) or seg_method != METHOD_QUICKSHIFT,
            help="小さいほど細かく（小さな色を保持）、大きいほど大まかに。"
            "大きくしても領域数は頭打ちします。",
        )
        seg_n_segments = st.slider(
            "SLIC: 領域数の目安",
            min_value=50, max_value=2000, value=500, step=50,
            disabled=(not seg_on) or seg_method != METHOD_SLIC,
            help="大きいほど細かく分割（小さな色も残る）。",
        )
        seg_compactness = st.slider(
            "SLIC: compactness（小さいほど色に沿う）",
            min_value=1.0, max_value=20.0, value=5.0, step=1.0,
            disabled=(not seg_on) or seg_method != METHOD_SLIC,
            help="小さいほど色の境界に沿い、大きいほど格子状に整います。",
        )
        seg_scale = st.slider(
            "Felzenszwalb: scale",
            min_value=50.0, max_value=600.0, value=200.0, step=25.0,
            disabled=(not seg_on) or seg_method != METHOD_FELZENSZWALB,
            help="大きいほど大まかな領域になります。",
        )

        st.subheader("クラスタリング")
        st.caption("手法: k-medoids（代表色に実在画素を使い色のくすみを避ける）")
        lab_space = st.toggle(
            "知覚的距離 CIELAB で分割",
            value=True,
            help="ON: 距離を CIELAB(ΔE) で計算し知覚的に分割（赤と背景が分かれやすい）。"
            "OFF: RGB 距離。",
        )
        tip_weight = st.toggle(
            "先端寄せ（彩度重み γ=2）",
            value=True,
            help="ON: 表示する代表色を彩度 C² 重みで高彩度側に寄せ、くすみを抑える"
            "（表示のみ。割合/アクセント判定は medoid のまま）。",
        )
        chroma_gamma = 2.0 if tip_weight else 0.0

        st.subheader("色名の判定")
        naming_knn = st.toggle(
            "人間の色名データで判定（XKCD・推奨）",
            value=True,
            help="ON: 人が付けた色名の分布(k近傍)で基本色名を判定。暗い/くすんだ有彩色が"
            "無彩色に誤判定されにくい。OFF: 基本色アンカーへの最近傍（旧方式）。",
        )
        naming = NAMING_KNN if naming_knn else NAMING_ANCHOR

        st.subheader("アクセントカラーの範囲")
        accent_range = st.slider(
            "割合の下限〜上限 (%)",
            min_value=0.0,
            max_value=50.0,
            value=(DEFAULT_ACCENT_LOW * 100, DEFAULT_ACCENT_HIGH * 100),
            step=0.5,
            help="この割合の範囲に入る色を『アクセントカラー』とみなします（既定 5〜10%）。",
        )
        accent_low = accent_range[0] / 100.0
        accent_high = accent_range[1] / 100.0

        aggregate = st.toggle(
            "基本色名に近似・集約する",
            value=True,
            help="ON: 似た色を基本色名（赤・橙…）にまとめてからアクセント判定。"
            "OFF: 各クラスタをそのまま色として扱う（従来動作）。",
        )

        exclude_achromatic = st.toggle(
            "無彩色・地味色（白・灰・黒・茶）はアクセントにしない",
            value=True,
            help="ON: 白・灰・黒・茶（くすんだ地味な色）はアクセント対象外にします。"
            "OFF: これらもアクセントになりえます。",
        )
        exclude_background = st.toggle(
            "背景（最大色）を除いて割合を判定",
            value=False,
            help="ON: 最も割合の大きい色を背景とみなし、それを除いた中での割合で"
            "アクセントを判定します。黒背景にキャラがドンの絵などで誤検出を防げます。",
        )

        accent_mode_label = st.radio(
            "アクセント判定モード",
            options=[
                "絶対（現行・コントラスト）",
                "① 相対（画像内で際立つ色）",
                "② 正規化（彩度を引き伸ばして判定）",
            ],
            index=0,
            help="淡い画像で薄い色を拾いたいときは ① か ②。"
            "① は画像全体の彩度に対する外れ値で判定、② は彩度を画像ごとに引き伸ばして判定。",
        )
        if accent_mode_label.startswith("①"):
            accent_mode = ACCENT_RELATIVE
        elif accent_mode_label.startswith("②"):
            accent_mode = ACCENT_NORMALIZE
        else:
            accent_mode = ACCENT_ABSOLUTE

        contrast_on = st.toggle(
            "主要色から際立つ色のみアクセント（コントラスト）",
            value=True,
            disabled=accent_mode != ACCENT_ABSOLUTE,
            help="ON: 割合が小さいだけでなく、主要色から色差 ΔE で十分離れた色だけを"
            "アクセントとします（隣接色相を弾く）。絶対モードのみ有効。",
        )
        contrast_min = st.slider(
            "コントラスト閾値 ΔE",
            min_value=0.0, max_value=80.0, value=25.0, step=5.0,
            disabled=(not contrast_on) or accent_mode != ACCENT_ABSOLUTE,
            help="大きいほど『主要色と大きく違う色』だけをアクセントに。"
            "参考: 赤-橙≈41, 赤-青≈127。",
        )
        contrast_min = contrast_min if contrast_on else 0.0

        st.subheader("クラスタ数")
        k = st.number_input(
            "クラスタ数 (k)",
            min_value=2,
            max_value=64,
            value=32,
            help="このクラスタ数（固定）でパレットとアクセントを判定します（既定 32）。",
        )
        min_prop = st.slider(
            "最小割合フロア (%)",
            min_value=0.0, max_value=3.0, value=0.5, step=0.1,
            help="この割合未満の極小クラスタ（ノイズ色）を、最も近い色へ併合します。"
            "k を増やすと出る 0.1〜0.5% のノイズ色を掃除できます。0 で無効。",
        ) / 100.0

        st.subheader("詳細設定")
        max_pixels = st.select_slider(
            "サンプリング上限ピクセル数",
            options=[10_000, 50_000, 100_000, 200_000, 500_000],
            value=50_000,
            help="速度のため画像を縮小してから処理します。大きいほど精密・低速。",
        )
        seed = st.number_input(
            "乱数シード", min_value=0, max_value=9999, value=DEFAULT_SEED,
            help="クラスタリングの再現性のため固定しています。",
        )

    # ---- 画像入力（複数可）----
    uploaded = st.file_uploader(
        "画像をアップロード（複数選択可。フォルダ内を全選択すれば一括処理できます）",
        type=["png", "jpg", "jpeg", "bmp", "webp"],
        accept_multiple_files=True,
    )

    if not uploaded:
        st.info("👆 画像を 1 枚以上アップロードしてください（複数選択可）。")
        return

    st.caption(f"{len(uploaded)} 件のファイルを処理します。")

    # ---- 各ファイルを順に解析・表示 ----
    for i, file in enumerate(uploaded):
        data = file.getvalue()
        st.header(f"📄 {file.name}")
        try:
            with st.spinner(f"{file.name} を解析中..."):
                image = Image.open(io.BytesIO(data))
                # 前段: 領域分割（ON なら領域平均色マップを色分析の入力にする）
                seg = None
                analysis_bytes = data
                if seg_on:
                    mean_img, overlay_img, n_regions = segment_image(
                        data, seg_method, int(seg_n_segments),
                        float(seg_compactness), float(seg_scale), float(seg_max_dist),
                    )
                    seg = (mean_img, overlay_img, n_regions)
                    analysis_bytes = _png_bytes(mean_img)

                res = analyze_image(
                    analysis_bytes,
                    int(max_pixels),
                    int(k),
                    accent_low,
                    accent_high,
                    int(seed),
                    aggregate,
                    exclude_achromatic,
                    lab_space,
                    float(contrast_min),
                    float(chroma_gamma),
                    naming,
                    accent_mode,
                    float(min_prop),
                    exclude_background,
                )
        except Exception as e:  # noqa: BLE001 - 1 件失敗しても残りは続行
            st.error(f"{file.name} の処理に失敗しました: {e}")
            continue

        render_result(image, res, accent_low, accent_high, seg=seg)

        if i < len(uploaded) - 1:
            st.divider()


if __name__ == "__main__":
    main()
