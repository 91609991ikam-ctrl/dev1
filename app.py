"""カラーパレット抽出 Streamlit アプリ.

イラスト・絵画・写真などの画像をアップロードすると、使われている色を
K-means クラスタリングで抽出し、数色のカラーパレットを作成する。
アクセントカラー（5〜10% 付近の色）が抽出できる最小クラスタ数を
二分探索で探し、そのパレットを表示する。

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
    find_min_accent_k,
    load_pixels,
)


st.set_page_config(page_title="カラーパレット抽出", page_icon="🎨", layout="wide")


def _text_color_for(rgb: tuple[int, int, int]) -> str:
    """背景色に対して読みやすい文字色（黒/白）を返す."""
    r, g, b = rgb
    # 相対輝度（簡易）
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    return "#000000" if luminance > 0.55 else "#FFFFFF"


def render_palette(palette: Palette) -> None:
    """パレットを横並びの色バーで描画する."""
    cols = st.columns(len(palette.colors))
    for col, color in zip(cols, palette.colors):
        fg = _text_color_for(color.rgb)
        border = "3px solid #FF3B30" if color.is_accent else "1px solid #ddd"
        badge = "⭐ アクセント" if color.is_accent else "&nbsp;"
        col.markdown(
            f"""
            <div style="background-color:{color.hex};
                        color:{fg};
                        border:{border};
                        border-radius:8px;
                        padding:14px 8px;
                        text-align:center;
                        font-family:monospace;">
                <div style="font-size:0.7rem;height:1rem;">{badge}</div>
                <div style="font-weight:bold;margin-top:6px;">{color.hex}</div>
                <div style="font-size:1.1rem;margin-top:4px;">{color.percent:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def palette_table(palette: Palette) -> list[dict]:
    """パレットを表形式のデータに変換."""
    rows = []
    for i, c in enumerate(palette.colors, start=1):
        rows.append(
            {
                "順位": i,
                "16進数": c.hex,
                "RGB": f"{c.rgb[0]}, {c.rgb[1]}, {c.rgb[2]}",
                "割合(%)": round(c.percent, 2),
                "アクセント": "⭐" if c.is_accent else "",
            }
        )
    return rows


def main() -> None:
    st.title("🎨 カラーパレット抽出アプリ")
    st.caption(
        "画像の色を K-means クラスタリングで減色し、アクセントカラーが抽出できる"
        "最小クラスタ数を二分探索で探してパレットを作ります。"
    )

    # ---- サイドバー：パラメータ ----
    with st.sidebar:
        st.header("⚙️ パラメータ")

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
            value=100_000,
            help="速度のため画像を縮小してから処理します。大きいほど精密・低速。",
        )
        seed = st.number_input(
            "乱数シード", min_value=0, max_value=9999, value=DEFAULT_SEED,
            help="クラスタリングの再現性のため固定しています。",
        )

    # ---- 画像入力 ----
    uploaded = st.file_uploader(
        "画像をアップロード（PNG / JPG など）",
        type=["png", "jpg", "jpeg", "bmp", "webp"],
    )

    if uploaded is None:
        st.info("👈 まず画像をアップロードしてください。")
        return

    image = Image.open(io.BytesIO(uploaded.read()))

    left, right = st.columns([1, 1])
    with left:
        st.subheader("入力画像")
        st.image(image, use_container_width=True)

    # ---- 解析 ----
    with st.spinner("クラスタリング中..."):
        pixels = load_pixels(image, max_pixels=int(max_pixels))
        result = find_min_accent_k(
            pixels,
            k_min=int(k_min),
            k_max=int(k_max),
            accent_low=accent_low,
            accent_high=accent_high,
            seed=int(seed),
        )

    with right:
        st.subheader(f"初期クラスタ数 k={result.base_palette.k} のパレット")
        render_palette(result.base_palette)

    st.divider()

    # ---- 探索結果 ----
    if result.min_k is None:
        st.warning(
            f"指定した範囲（{accent_low*100:.1f}〜{accent_high*100:.1f}%）の"
            f"アクセントカラーは k={result.base_palette.k} でも見つかりませんでした。"
            "アクセント範囲を広げて再試行してみてください。"
        )
        return

    st.success(
        f"✅ アクセントカラーが抽出できる最小クラスタ数は **k = {result.min_k}** です。"
    )

    # 探索の経過を表示
    with st.expander("🔍 二分探索の経過を見る"):
        trace_rows = [
            {"試したクラスタ数 k": k, "アクセント抽出": "○ できた" if ok else "✗ できない"}
            for k, ok in result.trace
        ]
        st.table(trace_rows)
        st.caption(
            "k_max でアクセントを定義し、P(k)=「アクセントが出るか」を満たす"
            "最小の k を二分探索しています。"
        )

    # ---- 最終パレット ----
    final = result.final_palette
    st.subheader(f"🎨 最終カラーパレット（k = {final.k}）")
    render_palette(final)

    st.markdown("#### 詳細")
    st.table(palette_table(final))

    accents = final.accent_colors
    if accents:
        st.markdown(
            "**アクセントカラー:** "
            + " / ".join(f"`{c.hex}` ({c.percent:.1f}%)" for c in accents)
        )


if __name__ == "__main__":
    main()
