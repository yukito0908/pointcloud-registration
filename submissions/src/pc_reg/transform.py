import numpy as np


def apply_transform(
    xyz: np.ndarray,
    T: np.ndarray,
) -> np.ndarray:
    """
    同次変換行列を点群に適用する。

    Args:
        xyz: (N, 3) 点群
        T: (4, 4) 同次変換行列

    Returns:
        xyz_trans: (N, 3) 変換された行列
    """
    xyz_h = np.hstack([xyz, np.ones((xyz.shape[0], 1))])
    xyz_trans_h = (T @ xyz_h.T).T
    return xyz_trans_h[:, :3]
