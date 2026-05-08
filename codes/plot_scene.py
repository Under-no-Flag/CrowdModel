from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

from crowd_bellman.loaders.config_loader import load_scene_spec
from crowd_bellman.compilers.config_compiler import compile_scene
from crowd_bellman.scenes import SimulationConfig


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot a scene mask from scene.toml.")
    parser.add_argument("--scene", required=True, help="Path to scene.toml")
    parser.add_argument("--output", default="scene.png", help="Output image path")
    parser.add_argument("--nx", type=int, default=128, help="Grid width")
    parser.add_argument("--ny", type=int, default=96, help="Grid height")
    parser.add_argument("--background", default="#f5f2e9", help="Walkable background color")
    parser.add_argument("--obstacle", default="#000000", help="Obstacle dot color")
    parser.add_argument("--obstacle-bg", default="#ffffff", help="Obstacle background color")
    parser.add_argument("--dpi", type=int, default=150, help="Output DPI")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    scene_path = Path(args.scene)
    scene_spec = load_scene_spec(scene_path)

    cfg = SimulationConfig(nx=args.nx, ny=args.ny)
    bundle = compile_scene(scene_spec, cfg)
    walkable = bundle.scene.walkable

    base = np.full(walkable.shape, np.nan, dtype=float)
    base[walkable] = 1.0
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("scene_bg", [args.background, args.background])

    fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=args.dpi)
    ax.set_facecolor(args.obstacle_bg)
    ax.imshow(base, origin="lower", interpolation="nearest", cmap=cmap, vmin=0.0, vmax=1.0)
    oy, ox = np.where(~walkable)
    if ox.size > 0:
        ax.scatter(ox, oy, s=2, c=args.obstacle, marker="s", linewidths=0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    fig.tight_layout(pad=0)
    fig.savefig(args.output, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


if __name__ == "__main__":
    main()
