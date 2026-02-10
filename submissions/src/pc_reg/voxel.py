from typing import Tuple
import numpy as np


def voxel_downsample(
    xyz: np.ndarray,
    voxel_size: float,
) -> np.ndarray:
    """
    voxel gridを用いて点群を間引く。

    引数はxyz: (N, 3) の点群
    voxel_sizeの単位は(m)

    Returns:
    xyz_dsは(M, 3)のダウンサンプルされた点群
    """
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positve")

    # 各点が属する voxel index を計算
    voxel_idx = np.floor(xyz / voxel_size).astype(np.int64)

    # 同じ voxel に入った点をまとめる
    # unique voxel index を取得
    unique_voxels, inverse = np.unique(voxel_idx, axis=0, return_inverse=True)

    # 各voxelの代表点　平均との差分
    xyz_ds = np.zeros((unique_voxels.shape[0], 3), dtype=np.float64)
    counts = np.zeros(unique_voxels.shape[0], dtype=np.int64)

    for i, idx in enumerate(inverse):
        xyz_ds[idx] += xyz[i]
        counts[idx] += 1

    xyz_ds /= counts[:, None]
    return xyz_ds
