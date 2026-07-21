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


def _lab_chroma(lab: np.ndarray) -> np.ndarray:
    """Lab 配列 (...,3) の彩度 C=√(a²+b²) を返す."""
    return np.sqrt(lab[..., 1] ** 2 + lab[..., 2] ** 2)


class KMedoidsLite:
    """部分標本＋交互最適化（PAM 風）による k-medoids.

    精度改善の研究レバー（既定は無効でメインアプリと同一挙動）:
      - ab_scale (α): Lab の a*,b* 軸を α 倍して彩度方向の距離を強調する。
        鮮やかな色（軸から遠い先端）が自前のクラスタに分離されやすくなる（割合は正直）。
      - chroma_gamma (γ): 各画素を彩度 C^γ で重み付けし、medoid 更新を重み付きにする。
        代表色が分布の先端（高彩度）へ寄る＝見た目が鮮やかになる（割合は水増しに注意）。
      - area_gamma: 学習時の「面積重み」を count^area_gamma に圧縮する（1.0=従来の
        面積重み、0.0=相異なる色を等価に扱う）。小さな領域の目立つ色が、画素数の多い
        色に呑まれて medoid を得られず消えるのを防ぐ。1.0 未満のとき、相異なる色
        （量子化した色）を count^area_gamma で重み付け標本化し、PAM コストも同じ重みに
        する。割合は predict で全画素から数え直すため水増しにはならない。
    """

    def __init__(
        self,
        n_clusters: int,
        random_state: int = 0,
        sample_size: int = 2500,
        max_iter: int = 50,
        lab_space: bool = True,
        ab_scale: float = 1.0,
        chroma_gamma: float = 0.0,
        area_gamma: float = 1.0,
    ):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.sample_size = sample_size
        self.max_iter = max_iter
        self.lab_space = lab_space
        self.ab_scale = ab_scale
        self.chroma_gamma = chroma_gamma
        self.area_gamma = area_gamma
        self.cluster_centers_: np.ndarray | None = None
        self.labels_: np.ndarray | None = None
        self._medoid_feat: np.ndarray | None = None  # predict 用の medoid 特徴量

    def _features(self, rgb: np.ndarray) -> np.ndarray:
        """距離計算に使う特徴量（Lab または RGB）に変換する.

        lab_space かつ ab_scale≠1 のとき、a*,b* 軸を ab_scale 倍して彩度方向を強調する。
        """
        rgb = np.asarray(rgb, dtype=np.float64)
        if not self.lab_space:
            return rgb
        lab = srgb_to_lab(rgb)
        if self.ab_scale != 1.0:
            lab = lab.copy()
            lab[..., 1] *= self.ab_scale
            lab[..., 2] *= self.ab_scale
        return lab

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

    def _distinct_sample(
        self, X: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """相異なる色を count^area_gamma で重み付け標本化する（小さな色の救済）.

        近い色を 8 段階に量子化してユニーク色にまとめ、その出現数 count を
        area_gamma 乗した重みで標本を選ぶ。面積の大きい色ほど重みを圧縮するので、
        画素数の少ない鮮やかな色も標本に残り、自前の medoid を得やすくなる。
        戻り値: (代表色 RGB, PAM コスト用の重み)。
        """
        bin_ = 8
        q = np.floor(np.clip(X, 0, 255) / bin_).astype(np.int64)
        _, inv, cnt = np.unique(q, axis=0, return_inverse=True, return_counts=True)
        nuniq = len(cnt)
        # 各ユニーク色の代表＝そのビン内の平均色（幅 8 なので実色にごく近い）
        reps = np.zeros((nuniq, 3), dtype=np.float64)
        np.add.at(reps, inv, X)
        reps /= cnt[:, None]
        w = np.power(cnt.astype(np.float64), self.area_gamma)
        m = min(self.sample_size, nuniq)
        if m >= nuniq:
            sel = np.arange(nuniq)
        else:
            sel = rng.choice(nuniq, size=m, replace=False, p=w / w.sum())
        return reps[sel], w[sel]

    def fit(self, X: np.ndarray) -> "KMedoidsLite":
        X = np.asarray(X, dtype=np.float64)
        n = len(X)
        k = max(1, min(self.n_clusters, n))
        rng = np.random.default_rng(self.random_state)

        # 部分標本を取り出して距離行列を作る
        sample_w = None
        if self.area_gamma >= 1.0:
            # 従来: 一様ランダム部分標本（面積重み＝画素数そのまま）
            m = min(self.sample_size, n)
            idx = np.arange(n) if m >= n else rng.choice(n, size=m, replace=False)
            sample_rgb = X[idx]
        else:
            # 小さな色の救済: 相異なる色を count^area_gamma で重み付け標本化
            sample_rgb, sample_w = self._distinct_sample(X, rng)
        sample_feat = self._features(sample_rgb)
        dist = self._pairwise(sample_feat)

        # 彩度重み（chroma_gamma>0 のとき）。重みは元の（スケール前）Lab 彩度から作る
        weights = None
        if self.chroma_gamma > 0:
            chroma = _lab_chroma(srgb_to_lab(sample_rgb))
            weights = np.power(chroma, self.chroma_gamma)
        # 面積重み（圧縮済み）を掛け合わせて PAM コストに反映（小さな色が medoid を得やすく）
        if sample_w is not None:
            weights = sample_w if weights is None else weights * sample_w

        # 初期 medoid → 交互最適化（割当 → 各クラスタの medoid 更新）
        medoids = self._kpp_init(dist, k, rng)
        labels = np.argmin(dist[:, medoids], axis=1)
        for _ in range(self.max_iter):
            new_medoids = medoids.copy()
            for c in range(k):
                members = np.where(labels == c)[0]
                if len(members) == 0:
                    continue
                sub = dist[np.ix_(members, members)]
                # 重み付きなら Σ w_j·d(candidate, j) を最小化（代表色が高彩度側へ寄る）
                if weights is None:
                    cost = sub.sum(axis=1)
                else:
                    cost = (sub * weights[members][None, :]).sum(axis=1)
                new_medoids[c] = members[int(np.argmin(cost))]
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
