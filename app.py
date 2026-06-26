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
    DEFAULT_K_MIN,
    DEFAULT_SEED,
    Palette,
    cluster_and_quantize,
    find_min_accent_k,
    load_pixels,
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


@st.cache_data(show_spinner=False)
def analyze_image(
    file_bytes: bytes,
    max_pixels: int,
    k_min: int,
    k_max: int,
    accent_low: float,
    accent_high: float,
    seed: int,
    aggregate: bool,
    exclude_achromatic: bool,
    lab_space: bool,
) -> dict:
    """1 枚の画像を解析し、表示に必要な結果をまとめて返す（キャッシュ対象）."""
    image = Image.open(io.BytesIO(file_bytes))
    pixels = load_pixels(image, max_pixels=max_pixels)
    result = find_min_accent_k(
        pixels,
        k_min=k_min,
        k_max=k_max,
        accent_low=accent_low,
        accent_high=accent_high,
        seed=seed,
        aggregate=aggregate,
        exclude_achromatic=exclude_achromatic,
        lab_space=lab_space,
    )
    no_accent = result.min_k is None
    disp_k = result.base_palette.k if no_accent else result.min_k
    agg_palette = result.base_palette if no_accent else result.final_palette
    accent_names = {c.name for c in agg_palette.accent_colors}
    clustered = cluster_and_quantize(
        image,
        disp_k,
        accent_names=accent_names,
        seed=seed,
        max_fit_pixels=max_pixels,
        lab_space=lab_space,
    )
    return {
        "min_k": result.min_k,
        "base_k": result.base_palette.k,
        "disp_k": disp_k,
        "no_accent": no_accent,
        "trace": result.trace,
        "palette": clustered.palette,
        "quant": clustered.image,
        "accents": agg_palette.accent_colors,
    }


def render_result(image: Image.Image, res: dict, accent_low: float, accent_high: float) -> None:
    """1 枚分の結果（入力画像・減色画像・パレット）を描画する."""
    img_col, _ = st.columns([2, 1])
    img_col.subheader("入力画像")
    img_col.image(image, use_container_width=True)
    img_col.subheader(f"クラスタリング後の画像（k = {res['disp_k']}）")
    img_col.image(res["quant"], use_container_width=True)

    if res["no_accent"]:
        st.warning(
            f"指定した範囲（{accent_low*100:.1f}〜{accent_high*100:.1f}%）の"
            f"アクセントカラーは k={res['base_k']} でも見つかりませんでした。"
            "アクセント範囲を広げて再試行してみてください。"
        )
    else:
        st.success(
            f"✅ アクセントカラーが抽出できる最小クラスタ数は **k = {res['min_k']}** です。"
        )

    final = res["palette"]
    st.subheader(f"🎨 カラーパレット（k = {res['disp_k']}・{len(final.colors)}色）")
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

    with st.expander("📋 パレットの詳細（表）"):
        st.table(palette_table(final))

    with st.expander("🔍 二分探索の経過を見る"):
        trace_rows = [
            {"試したクラスタ数 k": k, "アクセント抽出": "○ できた" if ok else "✗ できない"}
            for k, ok in res["trace"]
        ]
        st.table(trace_rows)
        st.caption(
            "k_max でアクセントを定義し、P(k)=「アクセントが出るか」を満たす"
            "最小の k を二分探索しています。"
        )


def main() -> None:
    st.title("🎨 カラーパレット抽出アプリ")
    st.caption(
        "画像の色を K-means で減色し、基本色名（赤・橙・黄…）に近似・集約して"
        "パレットを作ります。アクセントカラーが抽出できる最小クラスタ数を"
        "二分探索で探します。"
    )

    # ---- サイドバー：パラメータ ----
    with st.sidebar:
        st.header("⚙️ パラメータ")

        st.subheader("クラスタリング")
        st.caption("手法: k-medoids（代表色に実在画素を使い色のくすみを避ける）")
        lab_space = st.toggle(
            "知覚的距離 CIELAB で分割",
            value=True,
            help="ON: 距離を CIELAB(ΔE) で計算し知覚的に分割（赤と背景が分かれやすい）。"
            "OFF: RGB 距離。",
        )

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
            "無彩色（白・灰・黒）はアクセントにしない",
            value=True,
            help="ON: 白・灰・黒はアクセントカラーの対象外にします。"
            "OFF: 無彩色もアクセントになりえます。",
        )

        st.subheader("クラスタ数の探索範囲")
        k_max = st.number_input(
            "初期クラスタ数 (k_max)",
            min_value=2,
            max_value=64,
            value=DEFAULT_K_MAX,
            help="最初にアクセントを定義するクラスタ数（既定 16）。",
        )
        k_min = st.number_input(
            "最小クラスタ数 (k_min)",
            min_value=1,
            max_value=int(k_max),
            value=min(DEFAULT_K_MIN, int(k_max)),
            help="探索する最小のクラスタ数。",
        )

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
                res = analyze_image(
                    data,
                    int(max_pixels),
                    int(k_min),
                    int(k_max),
                    accent_low,
                    accent_high,
                    int(seed),
                    aggregate,
                    exclude_achromatic,
                    lab_space,
                )
                image = Image.open(io.BytesIO(data))
        except Exception as e:  # noqa: BLE001 - 1 件失敗しても残りは続行
            st.error(f"{file.name} の処理に失敗しました: {e}")
            continue

        render_result(image, res, accent_low, accent_high)

        if i < len(uploaded) - 1:
            st.divider()


if __name__ == "__main__":
    main()
