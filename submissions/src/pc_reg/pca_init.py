from __future__ import annotations
from os import RTLD_DEEPBIND
from re import M
from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass
class InitResult:
    T: np.ndarray
    score: float


def _pca_axes(xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    点群の主成分方向を求める。

    引数:
    xyz: (N,3)

    Returns:
    mu: (3,) 重心        
    axes: (3,3) ３本の直交軸を列に持つ行列
    """
    mu = xyz.mean(axis=0)
    X = xyz - mu
    # Cは共分散行列
    C = (X.T @ X) / max(X.shape[0] - 1, 1)
    # wは固有値、vは固有ベクトル
    w, V = np.linalg.eigh(C)
    # 小さいほうから並び替える。
    idx = np.argsort(w)[::-1]
    V = V[:, idx]
    if np.linalg.det(V) < 0:
        V[:, 2] *= -1
    return mu, V


def _compose_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def initial_align_pca(
    src: np.ndarray,
    tgt: np.ndarray,
    max_dist: float = 1.0,
) -> InitResult:
    """
    ダウンサンプルされたsrc,tgt点群に対してPCAで得た3本の軸を合わせる回転を作り、
    軸の符号を8通り試し、最近傍距離の平均が一番小さいものを返す。

    引数:
        src: (N,3) source ダウンサンプルされた点群
        tgt: (M,3) target ダウンサンプルされた点群
        max_dist: 近いとみなす距離の上限
    Returns:
        同次変換行列
    """
    mu_s, A_s = _pca_axes(src)
    mu_t, A_t = _pca_axes(tgt)

    best_T = _compose_T(np.eye(3), mu_t - mu_s)
    best_score = float("inf")

    # 軸の向きの組み合わせを全通り計算
    signs = np.array(
        [
            [1, 1, 1],
            [1, 1, -1],
            [1, -1, 1],
            [1, -1, -1],
            [-1, 1, 1],
            [-1, 1, -1],
            [-1, -1, 1],
            [-1, -1, -1],
        ],
        dtype=np.float64,
    )

    for s in signs:
        S = np.diag(s)
        R = (A_t @ S) @ A_s.T

        t = mu_t - R @ mu_s
        T = _compose_T(R, t)

        src0 = (R @ src.T).T + t
        score = _score_nn_mean(src0, tgt, max_dist=max_dist)

        if score < best_score:
            best_score = score
            best_T = T

    return InitResult(T=best_T, score=best_score)


def _score_nn_mean(src: np.ndarray, tgt: np.ndarray, max_dist: float) -> float:
    """
    source点群をtarget点群に合わせたとき、どれほど平均的に近いかを測る関数
    """
    dists = []
    for i in range(src.shape[0]):
        diff = tgt - src[i]
        d2 = np.sum(diff * diff, axis=1)
        d = float(np.sqrt(d2.min()))
        if d < max_dist:
            dists.append(d)
    if len(dists) < 50:
        return float("inf")
    return float(np.mean(dists))
