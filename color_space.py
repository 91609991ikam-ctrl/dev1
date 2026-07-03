"""画像の色を 3 次元空間にプロットする診断ツール（メインアプリから独立）.

「なぜ色がくすむのか」を直接見るために、画像のピクセル色を RGB / CIELAB の
3D 散布図で可視化する。各画素をその色で点描し、k-medoids のクラスタ中心
（medoid）を大きなマーカーで重ねる。ボタンで点の色を「実際の色 / 割り当て
られた代表色」に切り替えられるので、量子化でどの色がどこに潰れるかが分かる。

使い方（CLI）:
    python color_space.py <画像パス> [--k 16] [--max-points 5000] [--out plot.html]

使い方（コード）:
    fig = make_color_figure(Image.open("x.png"), k=16)
    fig.write_html("plot.html")
"""

from __future__ import annotations

import argparse

import numpy as np
import plotly.graph_objects as go
from PIL import Image
from plotly.subplots import make_subplots

from clustering import KMedoidsLite
from color_naming import srgb_to_lab


def _rgb_strings(rgb: np.ndarray) -> list[str]:
    """(N,3) の RGB 配列を 'rgb(r,g,b)' 文字列のリストにする."""
    rgb = np.clip(np.round(np.asarray(rgb)), 0, 255).astype(int)
    return [f"rgb({r},{g},{b})" for r, g, b in rgb]


def sample_pixels(image: Image.Image, max_points: int, seed: int = 0) -> np.ndarray:
    """画像から最大 max_points 個の画素をランダムサンプリングして (M,3) で返す."""
    arr = np.asarray(image.convert("RGB"), dtype=np.float64).reshape(-1, 3)
    if len(arr) > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(arr), size=max_points, replace=False)
        arr = arr[idx]
    return arr


def make_color_figure(
    image: Image.Image,
    k: int = 16,
    max_points: int = 5000,
    seed: int = 42,
    lab_space: bool = True,
    ab_scale: float = 1.0,
    chroma_gamma: float = 0.0,
    marker_size: int = 2,
) -> go.Figure:
    """RGB と CIELAB の 3D 散布図（medoid 重ね・色切替つき）を作って返す.

    ab_scale（α: 彩度方向の距離強調＝分離）と chroma_gamma（γ: 彩度重み＝先端寄せ）を
    渡すと、代表色（◆）が高彩度側にどう動くかを観察できる。
    """
    pixels = sample_pixels(image, max_points, seed)
    lab = srgb_to_lab(pixels)

    # k-medoids でクラスタ中心（実在画素）を求め、各画素の割り当て代表色を得る
    km = KMedoidsLite(
        n_clusters=k,
        random_state=seed,
        lab_space=lab_space,
        ab_scale=ab_scale,
        chroma_gamma=chroma_gamma,
    )
    km.fit(pixels)
    centers = km.cluster_centers_
    centers_lab = srgb_to_lab(centers)
    assigned = centers[km.predict(pixels)]  # 各画素に割り当てられた代表色

    true_colors = _rgb_strings(pixels)
    assigned_colors = _rgb_strings(assigned)
    center_colors = _rgb_strings(centers)

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=("RGB 空間", "CIELAB 空間"),
        horizontal_spacing=0.02,
    )

    # trace 0: 画素(RGB), 1: medoid(RGB), 2: 画素(Lab), 3: medoid(Lab)
    fig.add_trace(
        go.Scatter3d(
            x=pixels[:, 0], y=pixels[:, 1], z=pixels[:, 2],
            mode="markers",
            marker=dict(size=marker_size, color=true_colors, opacity=0.6),
            name="画素",
            hoverinfo="skip",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter3d(
            x=centers[:, 0], y=centers[:, 1], z=centers[:, 2],
            mode="markers",
            marker=dict(size=8, color=center_colors, symbol="diamond",
                        line=dict(color="black", width=1)),
            name="代表色 (medoid)",
            hoverinfo="text",
            text=center_colors,
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter3d(
            x=lab[:, 1], y=lab[:, 2], z=lab[:, 0],  # a*, b*, L*
            mode="markers",
            marker=dict(size=marker_size, color=true_colors, opacity=0.6),
            name="画素",
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1, col=2,
    )
    fig.add_trace(
        go.Scatter3d(
            x=centers_lab[:, 1], y=centers_lab[:, 2], z=centers_lab[:, 0],
            mode="markers",
            marker=dict(size=8, color=center_colors, symbol="diamond",
                        line=dict(color="black", width=1)),
            name="代表色 (medoid)",
            hoverinfo="text",
            text=center_colors,
            showlegend=False,
        ),
        row=1, col=2,
    )

    # 点の色を「実際の色 / 割り当て代表色」で切替（画素トレース 0,2 を restyle）
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=0.5, xanchor="center", y=1.12, yanchor="top",
                buttons=[
                    dict(label="実際の色", method="restyle",
                         args=[{"marker.color": [true_colors, true_colors]}, [0, 2]]),
                    dict(label="割り当て代表色", method="restyle",
                         args=[{"marker.color": [assigned_colors, assigned_colors]}, [0, 2]]),
                ],
            )
        ],
        scene=dict(
            xaxis_title="R", yaxis_title="G", zaxis_title="B",
            xaxis=dict(range=[0, 255]), yaxis=dict(range=[0, 255]), zaxis=dict(range=[0, 255]),
        ),
        scene2=dict(
            xaxis_title="a* (緑→赤)", yaxis_title="b* (青→黄)", zaxis_title="L* (明度)",
        ),
        margin=dict(l=0, r=0, t=60, b=0),
        legend=dict(x=0.0, y=1.0),
        title=(
            f"色の 3D 分布（サンプル {len(pixels)} 点 / medoid k={k} / "
            f"α={ab_scale:g} γ={chroma_gamma:g}）"
        ),
    )
    return fig


def _main() -> None:
    parser = argparse.ArgumentParser(description="画像の色を 3D プロットする")
    parser.add_argument("image", help="画像ファイルのパス")
    parser.add_argument("--k", type=int, default=16, help="medoid のクラスタ数")
    parser.add_argument("--max-points", type=int, default=5000, help="散布図に描く最大点数")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ab-scale", type=float, default=1.0, help="α: a*,b* 軸の強調（分離）")
    parser.add_argument("--chroma-gamma", type=float, default=0.0, help="γ: 彩度重み（先端寄せ）")
    parser.add_argument("--out", default="color_space.html", help="出力 HTML パス")
    args = parser.parse_args()

    fig = make_color_figure(
        Image.open(args.image),
        k=args.k,
        max_points=args.max_points,
        seed=args.seed,
        ab_scale=args.ab_scale,
        chroma_gamma=args.chroma_gamma,
    )
    fig.write_html(args.out, include_plotlyjs=True)
    print(f"書き出しました: {args.out}")


if __name__ == "__main__":
    _main()
