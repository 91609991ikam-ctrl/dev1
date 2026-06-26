"""軽量な k-medoids 実装（CIELAB 距離・彩度考慮の代表色）.

k-means は各クラスタを「平均色（重心）」で代表するため、鮮やかな色が周囲と
平均されてくすみやすい。k-medoids は代表色を「実在する画素」にするため、平均に
よる色の濁りを避けられる。

本実装の特徴:
  - 距離を **CIELAB（ΔE76）** で計算（知覚的に分割。赤と背景が分かれやすい）。
    lab_space=False で RGB 距離にも切替可能（手法比較用）。
  - 代表色を **彩度考慮** で選ぶ（saturation_aware）。クラスタ中心付近の実画素の
    うち、Lab 彩度 C=√(a²+b²) が最大の画素を代表色にする。中心付近に限ることで
    外れ値（極端に派手な1画素）を拾わず、くすみだけを抑える。

medoid 法は距離行列が O(n^2) になり全画素には適用できないので、学習は部分標本
（sample_size 個）で行い、全画素は最近傍 medoid へ割り当てる（CLARA 的手法）。

scikit-learn と同様の API（fit / fit_predict / predict / cluster_centers_ / labels_）
を持たせ、既存コードで KMeans と差し替え可能にする。cluster_centers_ は RGB。
"""

from __future__ import annotations

import numpy as np

from color_naming import srgb_to_lab


def _lab_chroma(lab: np.ndarray) -> np.ndarray:
    """Lab 配列 (...,3) の彩度 C=√(a²+b²) を返す."""
    return np.sqrt(lab[..., 1] ** 2 + lab[..., 2] ** 2)


class KMedoidsLite:
    """部分標本＋交互最適化（PAM 風）による k-medoids."""

    def __init__(
        self,
        n_clusters: int,
        random_state: int = 0,
        sample_size: int = 2500,
        max_iter: int = 50,
        lab_space: bool = True,
        saturation_aware: bool = True,
        sat_weight: float = 0.5,
    ):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.sample_size = sample_size
        self.max_iter = max_iter
        self.lab_space = lab_space
        self.saturation_aware = saturation_aware
        self.sat_weight = sat_weight
        self.cluster_centers_: np.ndarray | None = None
        self.labels_: np.ndarray | None = None
        self._medoid_feat: np.ndarray | None = None  # predict 用の medoid 特徴量

    def _features(self, rgb: np.ndarray) -> np.ndarray:
        """距離計算に使う特徴量（Lab または RGB）に変換する."""
        rgb = np.asarray(rgb, dtype=np.float64)
        return srgb_to_lab(rgb) if self.lab_space else rgb

    @staticmethod
    def _pairwise(a: np.ndarray) -> np.ndarray:
        """点群 a (m,d) の総当たりユークリッド距離行列 (m,m)."""
        sq = np.sum(a * a, axis=1)
        d2 = sq[:, None] + sq[None, :] - 2.0 * (a @ a.T)
        return np.sqrt(np.maximum(d2, 0.0))

    def _kpp_init(self, dist: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
        """k-means++ 風に初期 medoid を選ぶ（距離行列ベース）."""
        n = dist.shape[0]
        chosen = [int(rng.integers(n))]
        closest = dist[chosen[0]].copy()
        for _ in range(1, k):
            probs = closest ** 2
            total = probs.sum()
            nxt = int(rng.integers(n)) if total <= 0 else int(rng.choice(n, p=probs / total))
            chosen.append(nxt)
            closest = np.minimum(closest, dist[nxt])
        return np.array(chosen, dtype=int)

    def _representative(
        self,
        members: np.ndarray,
        sample_feat: np.ndarray,
        sample_rgb: np.ndarray,
        medoid: int,
    ) -> int:
        """クラスタの代表となる実画素のインデックスを返す（彩度考慮）.

        「彩度が高く、かつクラスタ中心(medoid)から離れすぎない」実画素を選ぶ。
        各メンバの 彩度 と medoid からの距離を 0..1 に正規化し、
        score = 彩度 - sat_weight × 距離 を最大化する画素を代表色にする。
        これにより、くすみを抑えつつ、極端に派手な外れ値の 1 画素も拾わない。
        """
        if not self.saturation_aware or len(members) == 1:
            return int(medoid)

        chroma = _lab_chroma(srgb_to_lab(sample_rgb[members]))
        dist = np.linalg.norm(sample_feat[members] - sample_feat[medoid], axis=1)

        def _norm(x: np.ndarray) -> np.ndarray:
            span = x.max() - x.min()
            return (x - x.min()) / span if span > 0 else np.zeros_like(x)

        score = _norm(chroma) - self.sat_weight * _norm(dist)
        return int(members[int(np.argmax(score))])

    def fit(self, X: np.ndarray) -> "KMedoidsLite":
        X = np.asarray(X, dtype=np.float64)
        n = len(X)
        k = max(1, min(self.n_clusters, n))
        rng = np.random.default_rng(self.random_state)

        # 部分標本を取り出して距離行列を作る
        m = min(self.sample_size, n)
        idx = np.arange(n) if m >= n else rng.choice(n, size=m, replace=False)
        sample_rgb = X[idx]
        sample_feat = self._features(sample_rgb)
        dist = self._pairwise(sample_feat)

        # 初期 medoid → 交互最適化（割当 → 各クラスタの medoid 更新）
        medoids = self._kpp_init(dist, k, rng)
        labels = np.argmin(dist[:, medoids], axis=1)
        for _ in range(self.max_iter):
            new_medoids = medoids.copy()
            for c in range(k):
                members = np.where(labels == c)[0]
                if len(members) == 0:
                    continue
                intra = dist[np.ix_(members, members)].sum(axis=1)
                new_medoids[c] = members[int(np.argmin(intra))]
            if np.array_equal(new_medoids, medoids):
                break
            medoids = new_medoids
            labels = np.argmin(dist[:, medoids], axis=1)

        # 割当（predict）は medoid（中心）を基準にする
        self._medoid_feat = sample_feat[medoids].copy()

        # 表示用の代表色は彩度考慮で選んだ実画素（RGB）
        reps = np.array(
            [
                self._representative(np.where(labels == c)[0], sample_feat, sample_rgb, medoids[c])
                for c in range(k)
            ]
        )
        self.cluster_centers_ = sample_rgb[reps].copy()
        self.labels_ = self.predict(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        feat = self._features(X)
        c = self._medoid_feat
        d2 = (
            np.sum(feat * feat, axis=1)[:, None]
            + np.sum(c * c, axis=1)[None, :]
            - 2.0 * (feat @ c.T)
        )
        return np.argmin(np.maximum(d2, 0.0), axis=1)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).labels_
