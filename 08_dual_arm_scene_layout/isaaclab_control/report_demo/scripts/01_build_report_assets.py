#!/usr/bin/env python3
"""Build static, auditable figures for the successful report demonstration.

Inputs are existing capture/network artifacts.  This script performs no network
inference and issues no robot command.  Its outputs are presentation PNG files.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# Keep Matplotlib cache local to this report tool instead of writing into the
# user's home configuration directory.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/dgn2_wuji2_report_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEMO_ROOT = PROJECT_ROOT / "08_dual_arm_scene_layout/isaaclab_control/report_demo"
CAPTURE_ROOT = PROJECT_ROOT / "08_dual_arm_scene_layout/captures/live_dynamic_scene0000"
TARGET_ROOT = CAPTURE_ROOT / "dgn2/dog"
OUTPUT_ROOT = DEMO_ROOT / "assets/generated"
SELECTED_CANDIDATE = 3800


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def set_equal_3d(ax, xyz: np.ndarray) -> None:
    low = xyz.min(axis=0)
    high = xyz.max(axis=0)
    center = (low + high) * 0.5
    radius = max(float(np.max(high - low)) * 0.52, 1e-3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def build_point_cloud_figure(network: np.lib.npyio.NpzFile) -> Path:
    """Render the exact 40K network points from the capturing camera viewpoint.

    ``pixel_indices`` was sampled with the same permutation as ``pc`` and
    ``target_membership``.  The one-to-one contract is therefore preserved:
    ``pc[i] <-> pixel_indices[i] <-> target_membership[i]``.
    """
    points = np.asarray(network["pc"][0], dtype=np.float32)
    membership = np.asarray(network["target_membership"][0], dtype=bool)
    pixel_indices = np.asarray(network["pixel_indices"][0], dtype=np.int64)
    width, height = Image.open(CAPTURE_ROOT / "rgb.png").size
    expected_pixels = width * height
    if np.any(pixel_indices < 0) or np.any(pixel_indices >= expected_pixels):
        raise RuntimeError("network pixel_indices exceed the captured RGB resolution")

    pixel_u = pixel_indices % width
    pixel_v = pixel_indices // width
    scene_mask = ~membership

    fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=120, facecolor="#111111")
    ax.set_facecolor("#111111")
    ax.scatter(
        pixel_u[scene_mask], pixel_v[scene_mask], s=1.6, c="#a9a9a9",
        alpha=0.70, linewidths=0, label="scene point", rasterized=True,
    )
    ax.scatter(
        pixel_u[membership], pixel_v[membership], s=4.2, c="#00e5ff",
        alpha=0.98, linewidths=0, label="selected target point", rasterized=True,
    )
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(
        "Camera-view projection of the exact 40,000 network points",
        color="white", fontsize=15,
    )
    ax.set_xlabel("image x / pixel", color="white")
    ax.set_ylabel("image y / pixel", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#777777")
    legend = ax.legend(loc="upper right", facecolor="#222222", edgecolor="#888888")
    for text_item in legend.get_texts():
        text_item.set_color("white")
    ax.text(
        0.01, 0.018,
        f"scene+target: {len(points):,} points   selected target: {int(membership.sum()):,} points",
        color="white", transform=ax.transAxes,
        bbox={"facecolor": "#111111", "alpha": 0.75, "edgecolor": "none"},
    )
    output = OUTPUT_ROOT / "04_network_point_cloud.png"
    fig.tight_layout()
    fig.savefig(output, facecolor=fig.get_facecolor())
    plt.close(fig)
    return output


def build_score_figure(prediction: np.lib.npyio.NpzFile) -> Path:
    target_indices = np.asarray(prediction["target_candidate_index"], dtype=np.int64)
    target_order = np.asarray(
        prediction["target_score_descending_candidate_index"], dtype=np.int64
    )
    score = np.asarray(prediction["score"], dtype=np.float32)
    graspness = np.asarray(prediction["graspness"], dtype=np.float32)
    log_prob = np.asarray(prediction["log_prob"], dtype=np.float32)
    seed = np.asarray(prediction["seed_point_world"], dtype=np.float32)
    selected_rank = int(np.where(target_order == SELECTED_CANDIDATE)[0][0]) + 1

    # Exact scores are preserved.  For readability the spatial panel displays
    # the top 1000 target candidates; no score is recomputed or renormalized.
    visible = target_order[: min(1000, len(target_order))]
    fig = plt.figure(figsize=(15, 6), dpi=150)
    ax0 = fig.add_subplot(121, projection="3d")
    scatter = ax0.scatter(
        seed[visible, 0],
        seed[visible, 1],
        seed[visible, 2],
        c=score[visible],
        cmap="viridis",
        s=8,
        alpha=0.82,
    )
    ax0.scatter(
        seed[SELECTED_CANDIDATE, 0],
        seed[SELECTED_CANDIDATE, 1],
        seed[SELECTED_CANDIDATE, 2],
        c="red",
        marker="*",
        s=260,
        edgecolors="black",
        linewidths=0.7,
        label=f"selected candidate {SELECTED_CANDIDATE}",
    )
    set_equal_3d(ax0, seed[target_indices])
    ax0.view_init(elev=26, azim=-52)
    ax0.set_title("Top 1,000 target candidates in world coordinates")
    ax0.set_xlabel("world x / m")
    ax0.set_ylabel("world y / m")
    ax0.set_zlabel("world z / m")
    ax0.legend(loc="upper left")
    fig.colorbar(scatter, ax=ax0, shrink=0.70, label="official score")

    ax1 = fig.add_subplot(122)
    ranks = np.arange(1, 31)
    top = target_order[:30]
    ax1.plot(ranks, score[top], "o-", color="#345995", linewidth=1.8, markersize=4)
    ax1.scatter(
        [selected_rank], [score[SELECTED_CANDIDATE]], c="red", marker="*", s=220, zorder=5
    )
    ax1.annotate(
        f"candidate {SELECTED_CANDIDATE}\nrank {selected_rank}\nscore {score[SELECTED_CANDIDATE]:.3f}",
        (selected_rank, score[SELECTED_CANDIDATE]),
        xytext=(selected_rank + 2, score[SELECTED_CANDIDATE] - 0.55),
        arrowprops={"arrowstyle": "->", "color": "red"},
    )
    ax1.set_title("Exact official target ranking")
    ax1.set_xlabel("target rank (1 = highest score)")
    ax1.set_ylabel("score = log_prob + 5 * graspness")
    ax1.grid(alpha=0.25)
    ax1.text(
        0.03,
        0.05,
        f"selected log_prob={log_prob[SELECTED_CANDIDATE]:.3f}\n"
        f"selected graspness={graspness[SELECTED_CANDIDATE]:.3f}",
        transform=ax1.transAxes,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#aaaaaa"},
    )
    output = OUTPUT_ROOT / "05_official_grasp_scores.png"
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def build_gate_figure() -> Path:
    labels = [
        "diffusion\nproposals",
        "target seed\nmembership",
        "scene/table\ncollision",
        "coarse arm\nreachability",
        "exact IK +\npath collision",
        "physical\nlift pass",
    ]
    values = np.array([8192, 7688, 6614, 30, 1, 1], dtype=float)
    colors = ["#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#08519c", "#2ca25f"]
    fig, ax = plt.subplots(figsize=(13, 5.5), dpi=150)
    bars = ax.bar(np.arange(len(values)), values, color=colors)
    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(labels)), labels)
    ax.set_ylabel("candidate count (log scale)")
    ax.set_title("Auditable selection gates: score alone is not executable grasp")
    ax.grid(axis="y", alpha=0.25, which="both")
    for bar, value in zip(bars, values.astype(int)):
        ax.text(bar.get_x() + bar.get_width() / 2, value * 1.15, f"{value:,}", ha="center")
    ax.text(
        0.99,
        0.95,
        "candidate 3800: official target rank 7\n17.78 s action | 123.20 mm lift | PASS",
        ha="right",
        va="top",
        transform=ax.transAxes,
        bbox={"facecolor": "#e5f5e0", "edgecolor": "#31a354", "alpha": 0.95},
    )
    output = OUTPUT_ROOT / "06_candidate_gate_funnel.png"
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def build_dashboard(generated: list[Path]) -> Path:
    sources = [
        CAPTURE_ROOT / "rgb.png",
        CAPTURE_ROOT / "depth_preview.png",
        CAPTURE_ROOT / "grounded_sam/dog/overlay.png",
        *generated,
    ]
    titles = [
        "1. Top camera RGB",
        "2. Depth",
        "3. GroundingDINO + SAM",
        "4. Official 40K point input",
        "5. Official grasp scores",
        "6. Execution gates",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=130)
    for ax, source, title in zip(axes.flat, sources, titles):
        ax.imshow(Image.open(source).convert("RGB"))
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(
        "DexGraspNet 2.0 -> Wuji2 -> dual-arm report evidence\n"
        "official test scene_0000, dynamically settled live capture",
        fontsize=16,
    )
    output = OUTPUT_ROOT / "00_full_pipeline_dashboard.png"
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    network = np.load(TARGET_ROOT / "network_input.npz", allow_pickle=False)
    prediction = np.load(
        TARGET_ROOT / "official_leap_1024_target_ranked.npz", allow_pickle=False
    )
    point_cloud = build_point_cloud_figure(network)
    scores = build_score_figure(prediction)
    gates = build_gate_figure()
    dashboard = build_dashboard([point_cloud, scores, gates])

    manifest = {
        "schema_version": 1,
        "status": "PASS",
        "mode": "cached_evidence_render_only",
        "source_capture": str(CAPTURE_ROOT),
        "target": "dog",
        "selected_candidate": SELECTED_CANDIDATE,
        "outputs": [str(path) for path in [dashboard, point_cloud, scores, gates]],
    }
    (OUTPUT_ROOT / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[REPORT ASSETS PASS] {OUTPUT_ROOT}")
    for path in [dashboard, point_cloud, scores, gates]:
        print(path)


if __name__ == "__main__":
    main()
