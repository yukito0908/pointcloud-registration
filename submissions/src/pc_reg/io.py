from pathlib import Path
from typing import Tuple

import numpy as np


def load_point_cloud(
    path: str | Path,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    テキストファイルから点群を読み込む。

    点群のデータ構造：
    x y z r g b

    引数:pathはテキストファイルへのパス

    Return:
    xyz: (N, 3) 
    rgb: (N, 3) 
    """
    path = Path(path)
    data = np.loadtxt(path)

    if data.shape[1] != 6:
        raise ValueError(
            f"Expected 6 columns in the txt file, but got {data.shape[1]}")

    xyz = data[:, 0:3].astype(np.float64)
    rgb = data[:, 3:6].astype(np.uint8)
    return xyz, rgb
