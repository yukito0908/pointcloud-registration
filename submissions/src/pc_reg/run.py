import argparse
from json import load
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from pc_reg.io import load_point_cloud
from pc_reg.viz import plot_point_clouds
from pc_reg.icp import icp_point_to_point
from pc_reg.voxel import voxel_downsample
from pc_reg.transform import apply_transform
from pc_reg.metrics import compute_rmse_full
from pc_reg.viz import plot_2d_projections
from pc_reg.pca_init import initial_align_pca


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--outdir", default="results")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    # 読み込み
    xyz_s, rgb_s = load_point_cloud(args.source)
    xyz_t, rgb_t = load_point_cloud(args.target)

    # ダウンサンプル
    xyz_s_ds = voxel_downsample(xyz_s, voxel_size=0.2)
    xyz_t_ds = voxel_downsample(xyz_t, voxel_size=0.2)

    # PCAによる粗い回転初期化
    init = initial_align_pca(xyz_s_ds, xyz_t_ds, max_dist=1.0)
    print("PCA init score:", init.score)

    # ICP
    T, rmse_list = icp_point_to_point(
        xyz_s_ds,
        xyz_t_ds,
        init_T=init.T)
    print("Final RMSE (downsampled):", rmse_list[-1])

    xyz_s_full_trans = apply_transform(xyz_s, T)
    R = T[:3, :3]
    t = T[:3, 3]
    angle = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    print("Final |t| =", np.linalg.norm(t), "angle(deg) =", angle)

    # after.png
    xyz_s_aligned = apply_transform(xyz_s, T)

    plot_point_clouds(
        xyz_s_aligned,
        xyz_t,
        outdir / "after.png",
        "After registrarion (raw point clouds)",
    )
    plot_2d_projections(xyz_s_aligned, xyz_t, outdir /
                        "after", max_points=6000)
    # before.png
    plot_point_clouds(
        xyz_s,
        xyz_t,
        outdir / "before.png",
        "Before registrarion (raw point clouds)",
    )
    plot_2d_projections(xyz_s, xyz_t, outdir / "before", max_points=6000)

    # ログ
    print("centroid diff before:", np.linalg.norm(
        xyz_s.mean(0) - xyz_t.mean(0)))
    print(
        "centroid diff after :", np.linalg.norm(
            xyz_s_aligned.mean(0) - xyz_t.mean(0))
    )

    rmse_full = compute_rmse_full(
        xyz_s_full_trans,
        xyz_t,
        max_points=2000,
    )
    print(f"rmse={rmse_full:}")

    plt.figure()
    plt.plot(range(len(rmse_list)), rmse_list, marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("RMSE")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(outdir / "rmse_curve.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
