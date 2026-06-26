"""軽量な k-medoids 実装.

k-means は各クラスタを「平均色（重心）」で代表するため、鮮やかな色が周囲と
平均されてくすみやすい。k-medoids は代表色を「実在する画素（medoid）」にする
ため、平均による色の濁りを避けられる。

medoid 法は距離行列が O(n^2) になり全画素には適用できないので、学習は部分標本
（sample_size 個）で行い、全画素は最近傍 medoid へ割り当てる（CLARA 的手法）。
距離は k-means と同じ RGB ユークリッドで計算する（平均→medoid の効果を切り分ける）。

scikit-learn と同様の API（fit / fit_predict / predict / cluster_centers_ / labels_）
を持たせ、既存コードで KMeans と差し替え可能にする。
"""

from __future__ import annotations

import numpy as np


class KMedoidsLite:
    """部分標本＋交互最適化（PAM 風）による k-medoids（RGB 距離）."""

    def __init__(
        self,
        n_clusters: int,
        random_state: int = 0,
        sample_size: int = 2500,
        max_iter: int = 50,
    ):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.sample_size = sample_size
        self.max_iter = max_iter
        self.cluster_centers_: np.ndarray | None = None
        self.labels_: np.ndarray | None = None

    @staticmethod
    def _pairwise(a: np.ndarray) -> np.ndarray:
        """点群 a (m,3) の総当たりユークリッド距離行列 (m,m)."""
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
        sample = X[idx]
        dist = self._pairwise(sample)

        # 初期 medoid → 交互最適化（割当 → 各クラスタの medoid 更新）
        medoids = self._kpp_init(dist, k, rng)
        for _ in range(self.max_iter):
            labels = np.argmin(dist[:, medoids], axis=1)
            new_medoids = medoids.copy()
            for c in range(k):
                members = np.where(labels == c)[0]
                if len(members) == 0:
                    continue
                # クラスタ内距離和が最小の点を新 medoid にする
                intra = dist[np.ix_(members, members)].sum(axis=1)
                new_medoids[c] = members[int(np.argmin(intra))]
            if np.array_equal(new_medoids, medoids):
                break
            medoids = new_medoids

        # 代表色は「実在画素」（medoid の色）
        self.cluster_centers_ = sample[medoids].copy()
        self.labels_ = self.predict(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        c = self.cluster_centers_
        d2 = (
            np.sum(X * X, axis=1)[:, None]
            + np.sum(c * c, axis=1)[None, :]
            - 2.0 * (X @ c.T)
        )
        return np.argmin(np.maximum(d2, 0.0), axis=1)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).labels_
