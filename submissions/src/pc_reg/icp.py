import numpy as np
from typing import Tuple, List

from pc_reg.metrics import compute_rmse_full
from pc_reg.transform import apply_transform


def find_nearest_neighbors_bruteforce(
    src: np.ndarray,
    tgt: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    全探索で最近傍を見つける

    引数:
        src: (N, 3) source点群
        tgt: (M, 3) target点群

    Returns:
        matched_tgt: (N, 3) 各src点群に関する最近傍tgt点群
        distances: (N,) 距離
    """
    matched = np.zeros_like(src)
    distances = np.zeros(src.shape[0])

    for i in range(src.shape[0]):
        diff = tgt - src[i]
        dist2 = np.sum(diff**2, axis=1)
        idx = np.argmin(dist2)
        matched[i] = tgt[idx]
        distances[i] = np.sqrt(dist2[idx])

    return matched, distances


def estimate_rigid_transform_svd(
    src: np.ndarray,
    tgt: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    対応付けられたsrc,tgtに対して、平均二乗誤差を最小にする
    回転Rと並進tを求める

    引数:
        src: (N, 3) source 点群
        tgt: (N, 3) tgt 点群

    Returns:
        R: (3, 3) 回転行列
        t: (3,) 並進移動ベクトル
    """
    src_mean = src.mean(axis=0)
    tgt_mean = tgt.mean(axis=0)

    src_centered = src - src_mean
    tgt_centered = tgt - tgt_mean

    # Hは相互共分散行列
    H = src_centered.T @ tgt_centered
    U, _, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = tgt_mean - R @ src_mean

    return R, t


def icp_point_to_point(
    src: np.ndarray,
    tgt: np.ndarray,
    init_T: np.ndarray | None = None,
    max_iter: int = 30,
    tol: float = 1e-5,
    max_dist: float = 0.3,
) -> Tuple[np.ndarray, List[float]]:

    # 初期化
    if init_T is None:
        src_trans = src.copy()
        src_trans += tgt.mean(axis=0) - src.mean(axis=0)
        T = np.eye(4)
        T[:3, 3] = tgt.mean(axis=0) - src.mean(axis=0)
    else:
        src_trans = apply_transform(src, init_T)
        T = init_T.copy()

    rmse_list = []

    prev_rmse = None

    for it in range(max_iter):
        matched_tgt, distances = find_nearest_neighbors_bruteforce(
            src_trans, tgt)
        # 距離カット
        mask = distances < max_dist

        src_valid = src_trans[mask]
        tgt_valid = matched_tgt[mask]
        dist_valid = distances[mask]

        if src_valid.shape[0] < 50:
            print("Too few correspondences after distance filter")
            break

        # トリム
        trim_ratio = 0.7
        k = int(len(dist_valid) * trim_ratio)
        idx = np.argsort(dist_valid)[:k]

        src_valid = src_valid[idx]
        tgt_valid = tgt_valid[idx]
        dist_valid = dist_valid[idx]

        rmse = np.sqrt(np.mean(dist_valid**2))
        rmse_list.append(rmse)

        if it > 0 and abs(prev_rmse - rmse) < tol:
            break

        R, t = estimate_rigid_transform_svd(src_valid, tgt_valid)

        src_trans = (R @ src_trans.T).T + t

        T_inter = np.eye(4)
        T_inter[:3, :3] = R
        T_inter[:3, 3] = t
        T = T_inter @ T

        prev_rmse = rmse

    return T, rmse_list,
