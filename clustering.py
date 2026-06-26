"""軽量な k-medoids 実装（CIELAB 距離）.

k-means は各クラスタを「平均色（重心）」で代表するため、鮮やかな色が周囲と
平均されてくすみやすい。k-medoids は代表色を「実在する画素（medoid）」にする
ため、平均による色の濁りを避けられる。

距離は **CIELAB（ΔE76）** で計算する（知覚的に分割。赤と背景が分かれやすい）。
lab_space=False で RGB 距離にも切替可能（手法比較用）。

medoid 法は距離行列が O(n^2) になり全画素には適用できないので、学習は部分標本
（sample_size 個）で行い、全画素は最近傍 medoid へ割り当てる（CLARA 的手法）。

scikit-learn と同様の API（fit / fit_predict / predict / cluster_centers_ / labels_）
を持たせる。cluster_centers_ は RGB。
"""

from __future__ import annotations

import numpy as np

from color_naming import srgb_to_lab


class KMedoidsLite:
    """部分標本＋交互最適化（PAM 風）による k-medoids."""

    def __init__(
        self,
        n_clusters: int,
        random_state: int = 0,
        sample_size: int = 2500,
        max_iter: int = 50,
        lab_space: bool = True,
    ):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.sample_size = sample_size
        self.max_iter = max_iter
        self.lab_space = lab_space
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

        # 代表色は medoid（実在画素）の RGB
        self._medoid_feat = sample_feat[medoids].copy()
        self.cluster_centers_ = sample_rgb[medoids].copy()
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
