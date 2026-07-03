"""色の 3D 空間プロット（診断用の別アプリ）.

メインの `app.py` とは独立した、色分布の可視化ツール。画像をアップロードすると
RGB / CIELAB 空間の 3D 散布図を表示し、ブラウザ上で回転・ズームして
「なぜ色がくすむのか」を直接観察できる。

起動:
    streamlit run color_space_app.py
"""

from __future__ import annotations

import io

import streamlit as st
from PIL import Image

from color_space import make_color_figure


st.set_page_config(page_title="色の3D空間プロット", page_icon="🧊", layout="wide")


@st.cache_data(show_spinner=False)
def _figure(file_bytes: bytes, k: int, max_points: int, seed: int, lab_space: bool):
    return make_color_figure(
        Image.open(io.BytesIO(file_bytes)),
        k=k,
        max_points=max_points,
        seed=seed,
        lab_space=lab_space,
    )


def main() -> None:
    st.title("🧊 色の 3D 空間プロット（診断ツール）")
    st.caption(
        "画像の色を RGB / CIELAB 空間の 3D 散布図で表示します。各画素をその色で点描し、"
        "k-medoids の代表色（◆）を重ねます。ボタンで点の色を「実際の色 / 割り当て代表色」に"
        "切り替えると、量子化でどの色がどこに潰れるか（くすみの原因）が見えます。"
    )

    with st.sidebar:
        st.header("⚙️ パラメータ")
        k = st.number_input("medoid のクラスタ数 (k)", min_value=1, max_value=64, value=16)
        max_points = st.select_slider(
            "散布図に描く最大点数",
            options=[1000, 2000, 5000, 10000, 20000],
            value=5000,
            help="多いほど詳細ですが、回転が重くなります。",
        )
        lab_space = st.toggle(
            "medoid の分割は CIELAB 距離", value=True,
            help="メインアプリと同じ設定。OFF で RGB 距離。",
        )
        seed = st.number_input("乱数シード", min_value=0, max_value=9999, value=42)

    uploaded = st.file_uploader(
        "画像をアップロード（複数選択可）",
        type=["png", "jpg", "jpeg", "bmp", "webp"],
        accept_multiple_files=True,
    )

    if not uploaded:
        st.info("👆 画像をアップロードすると 3D プロットを表示します。")
        return

    for i, file in enumerate(uploaded):
        st.header(f"📄 {file.name}")
        data = file.getvalue()
        col_img, _ = st.columns([1, 2])
        col_img.image(Image.open(io.BytesIO(data)), caption="入力画像", use_container_width=True)
        try:
            with st.spinner(f"{file.name} の色分布を計算中..."):
                fig = _figure(data, int(k), int(max_points), int(seed), lab_space)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:  # noqa: BLE001 - 1 件失敗しても続行
            st.error(f"{file.name} の処理に失敗しました: {e}")

        if i < len(uploaded) - 1:
            st.divider()


if __name__ == "__main__":
    main()
