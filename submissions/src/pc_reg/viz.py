from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def plot_point_clouds(
    xyz1: np.ndarray,
    xyz2: np.ndarray,
    out_path: str | Path,
    title: str,
    max_points: int = 6000,
) -> None:

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def sample(xyz: np.ndarray) -> np.ndarray:
        if xyz.shape[0] <= max_points:
            return xyz
        step = xyz.shape[0] // max_points
        return xyz[::step]

    p1 = sample(xyz1)
    p2 = sample(xyz2)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(p1[:, 0], p1[:, 1], p1[:, 2], s=1,
               c="red", alpha=0.2, label="source")
    ax.scatter(p2[:, 0], p2[:, 1], p2[:, 2], s=1,
               c="blue", alpha=0.2, label="target")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    ax.set_title(title)
    ax.view_init(elev=20, azim=30)

    xyz_all = np.vstack([p1, p2])
    mins = xyz_all.min(axis=0)
    maxs = xyz_all.max(axis=0)
    centers = (mins + maxs) / 2
    max_range = (maxs - mins).max() / 2

    ax.set_xlim(centers[0] - max_range, centers[0] + max_range)
    ax.set_ylim(centers[1] - max_range, centers[1] + max_range)
    ax.set_zlim(centers[2] - max_range, centers[2] + max_range)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_2d_projections(
    xyz1: np.ndarray,
    xyz2: np.ndarray,
    out_prefix: str | Path,
    max_points: int = 6000,
) -> None:
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    def sample(xyz: np.ndarray) -> np.ndarray:
        if xyz.shape[0] <= max_points:
            return xyz
        idx = np.random.choice(xyz.shape[0], max_points, replace=False)
        return xyz[idx]

    p1 = sample(xyz1)
    p2 = sample(xyz2)

    xmin = min(p1[:, 0].min(), p2[:, 0].min())
    xmax = max(p1[:, 0].max(), p2[:, 0].max())
    ymin = min(p1[:, 1].min(), p2[:, 1].min())
    ymax = max(p1[:, 1].max(), p2[:, 1].max())
    zmin = min(p1[:, 2].min(), p2[:, 2].min())
    zmax = max(p1[:, 2].max(), p2[:, 2].max())

    def save_plot(xa, ya, xb, yb, xlim, ylim, title, out_path):
        plt.figure(figsize=(6, 6))
        plt.scatter(xa, ya, s=1, alpha=0.25, label="source")
        plt.scatter(xb, yb, s=1, alpha=0.25, label="target")
        plt.xlim(*xlim)
        plt.ylim(*ylim)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.grid(True, linewidth=0.5, alpha=0.4)
        plt.title(title)
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()

    save_plot(
        p1[:, 0],
        p1[:, 1],
        p2[:, 0],
        p2[:, 1],
        (xmin, xmax),
        (ymin, ymax),
        "XY projection",
        out_prefix.with_name(out_prefix.name + "_xy.png"),
    )
    save_plot(
        p1[:, 0],
        p1[:, 2],
        p2[:, 0],
        p2[:, 2],
        (xmin, xmax),
        (zmin, zmax),
        "XZ projection",
        out_prefix.with_name(out_prefix.name + "_xz.png"),
    )
    save_plot(
        p1[:, 1],
        p1[:, 2],
        p2[:, 1],
        p2[:, 2],
        (ymin, ymax),
        (zmin, zmax),
        "YZ projection",
        out_prefix.with_name(out_prefix.name + "_yz.png"),
    )
