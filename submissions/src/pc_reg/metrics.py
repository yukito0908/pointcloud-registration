import numpy as np


def compute_rmse_full(
    src: np.ndarray,
    tgt: np.ndarray,
    max_points: int = 2000,
) -> float:
    """
    同次変換行列を適用したのち、source点群とtarget点群の近さを測る。

    引数:
        src: (N, 3) 変換された点群
        tgt: (M, 3) 点群
        max_points: source点群のサンプルの数

    Returns:
        rmse
    """
    if src.shape[0] > max_points:
        idx = np.random.choice(src.shape[0], max_points, replace=False)
        src = src[idx]

    dists = []

    for i in range(src.shape[0]):
        diff = tgt - src[i]
        dist2 = np.sum(diff**2, axis=1)
        d = np.sqrt(dist2.min())
        dists.append(d)

    rmse = np.sqrt(np.mean(np.array(dists) ** 2))
    return rmse
