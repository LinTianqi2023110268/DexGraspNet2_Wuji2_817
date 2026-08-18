#!/usr/bin/env python3
"""
Right-Arm Workspace / Table Layout Calibration

Goal
----
Search the DualArmMount translation (x, y, z) while keeping the current mount
rotation fixed, and rank layouts by STRICT cuRobo 6-D IK reachability over the
table SourceZone (primary) and PlacementZone (secondary).

This is a read-only calibration tool:
- it does NOT modify manual_layout_calibrated.json;
- it does NOT start Isaac Sim;
- it does NOT command the robot;
- it uses the project's real right-arm URDF and CuroboGpuIK;
- success uses the project's strict 5 mm / 5 deg / 3 deg inner-joint-margin
  acceptance contract.

Instead of inventing arbitrary wrist orientations, the scanner extracts
realistic COVER task templates from already-finalized Wuji2 candidate cases.
Each template stores the target-object -> right-flange transform.  During the
workspace scan, that same real grasp relationship is translated across the
SourceZone / PlacementZone, so we ask a physically meaningful question:

    "If an object requiring a grasp like this were located at this table
     position, would the right arm have an exact 6-D IK solution?"

The tool uses a two-stage search:
1) coarse XYZ grid (default 315 mount locations);
2) local refinement around the coarse best (default 125 locations).

Typical total: 440 tested DualArmMount centers.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def rotation_angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    rel = np.asarray(Ra, dtype=np.float64).T @ np.asarray(Rb, dtype=np.float64)
    c = float(np.clip((np.trace(rel) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(c))


def parse_triplet(text: str) -> tuple[float, float, float]:
    vals = [float(x.strip()) for x in text.split(",")]
    if len(vals) != 3:
        raise argparse.ArgumentTypeError("expected min,max,step")
    lo, hi, step = vals
    if step <= 0 or hi < lo:
        raise argparse.ArgumentTypeError("require step>0 and max>=min")
    return lo, hi, step


def axis_values(spec: tuple[float, float, float]) -> np.ndarray:
    lo, hi, step = spec
    n = int(math.floor((hi - lo) / step + 1.0e-9)) + 1
    vals = lo + step * np.arange(n, dtype=np.float64)
    if vals[-1] < hi - 1.0e-9:
        vals = np.append(vals, hi)
    return vals


@dataclass
class TaskTemplate:
    group: str
    case_root: str
    candidate_index: int
    target_rank: int | None
    T_world_object_template: np.ndarray
    T_object_flange: np.ndarray

    @property
    def object_z_world(self) -> float:
        return float(self.T_world_object_template[2, 3])

    def flange_at_xy(self, x: float, y: float) -> np.ndarray:
        T = self.T_world_object_template.copy()
        T[0, 3] = float(x)
        T[1, 3] = float(y)
        return T @ self.T_object_flange


@dataclass
class ZoneTaskSet:
    name: str
    points_xy: np.ndarray             # [P,2]
    poses_world: np.ndarray           # [P*T,4,4]
    template_count: int


@dataclass
class ZoneMetrics:
    any_coverage: float
    robust_coverage: float
    mean_template_reach_fraction: float
    median_template_reach_fraction: float
    mean_best_margin_deg: float
    pose_success_fraction: float
    point_reach_fraction: np.ndarray

    def score_source(self) -> float:
        return (
            0.50 * self.robust_coverage
            + 0.30 * self.any_coverage
            + 0.20 * self.mean_template_reach_fraction
        )

    def score_placement(self) -> float:
        return (
            0.70 * self.any_coverage
            + 0.30 * self.mean_template_reach_fraction
        )


def parse_template_group(text: str) -> tuple[str, Path, Path | None]:
    """
    Syntax:
        NAME|ROOT
        NAME|ROOT|PASS_JSON

    PASS_JSON is optional.  If supplied, only rows with pass=true are used.
    """
    parts = text.split("|")
    if len(parts) not in (2, 3):
        raise argparse.ArgumentTypeError(
            "template-group format must be NAME|ROOT or NAME|ROOT|PASS_JSON"
        )
    name = parts[0].strip()
    root = Path(parts[1]).expanduser()
    pass_json = Path(parts[2]).expanduser() if len(parts) == 3 and parts[2].strip() else None
    if not name:
        raise argparse.ArgumentTypeError("template group name must be nonempty")
    return name, root, pass_json


def target_rank_from_path(case_root: Path) -> int | None:
    for part in case_root.parts[::-1]:
        m = re.fullmatch(r"rank_(\d+)", part)
        if m:
            return int(m.group(1))
        m = re.search(r"_r(\d+)_cand", part)
        if m:
            return int(m.group(1))
    return None


def load_pass_candidates(path: Path | None) -> set[int] | None:
    if path is None:
        return None
    payload = load_json(path.resolve())
    rows = payload.get("rows", [])
    return {
        int(row["candidate_index"])
        for row in rows
        if bool(row.get("pass", False))
    }


def load_case_template(case_root: Path, group_name: str) -> TaskTemplate:
    case_root = case_root.resolve()
    case_meta = load_json(case_root / "case.json")
    target_id = int(case_meta["target_segmentation_id"])
    candidate_index = int(case_meta["source_candidate_index"])

    arm_path = case_root / "07_arm_execution/arm_flange_targets.npz"
    manifests = sorted((case_root / "01_input").glob("scene_*_manifest.json"))
    if len(manifests) != 1:
        raise RuntimeError(f"{case_root}: expected one scene manifest, got {len(manifests)}")
    scene = load_json(manifests[0])
    target = next(
        row for row in scene["objects"]
        if int(row["segmentation_id"]) == target_id
    )

    with np.load(arm_path, allow_pickle=False) as z:
        names = [str(x) for x in z["waypoint_names"].tolist()]
        flange_world = np.asarray(z["world_from_right_flange"], dtype=np.float64)
        world_from_source = np.asarray(z["world_from_source_zone"], dtype=np.float64)

    cover_i = names.index("cover")
    T_world_flange = flange_world[cover_i]

    # Keep the exact current project convention used by flexible_route_search:
    # pose_world_object in the scene manifest is composed with world_from_source.
    T_source_object = np.asarray(target["pose_world_object"], dtype=np.float64)
    T_world_object = world_from_source @ T_source_object
    T_object_flange = np.linalg.inv(T_world_object) @ T_world_flange

    return TaskTemplate(
        group=group_name,
        case_root=str(case_root),
        candidate_index=candidate_index,
        target_rank=target_rank_from_path(case_root),
        T_world_object_template=T_world_object,
        T_object_flange=T_object_flange,
    )


def discover_group_templates(
    group_name: str,
    root: Path,
    pass_json: Path | None,
) -> list[TaskTemplate]:
    root = root.resolve()
    if not root.exists():
        raise FileNotFoundError(root)
    allowed = load_pass_candidates(pass_json)
    arm_files = sorted(root.rglob("07_arm_execution/arm_flange_targets.npz"))
    if not arm_files:
        # Also accept a root that directly contains case directories with a slightly
        # different recursive shape.
        arm_files = sorted(root.rglob("arm_flange_targets.npz"))
    templates: list[TaskTemplate] = []
    errors = 0
    for arm in arm_files:
        if arm.name != "arm_flange_targets.npz":
            continue
        case_root = arm.parent.parent
        case_json = case_root / "case.json"
        if not case_json.is_file():
            continue
        try:
            candidate_index = int(load_json(case_json)["source_candidate_index"])
            if allowed is not None and candidate_index not in allowed:
                continue
            templates.append(load_case_template(case_root, group_name))
        except Exception as exc:
            errors += 1
            if errors <= 5:
                print(f"[template warning] {case_root}: {type(exc).__name__}: {exc}")
    print(
        f"[templates] group={group_name} discovered={len(templates)} "
        f"errors={errors} filter={'PASS only' if allowed is not None else 'all finalized'}"
    )
    return templates


def pose_template_distance(a: TaskTemplate, b: TaskTemplate) -> float:
    # Translation of flange relative to object, normalized by 5 cm,
    # plus orientation difference normalized by 20 deg.
    ta = a.T_object_flange[:3, 3]
    tb = b.T_object_flange[:3, 3]
    dt = float(np.linalg.norm(ta - tb)) / 0.05
    dr = rotation_angle_deg(
        a.T_object_flange[:3, :3],
        b.T_object_flange[:3, :3],
    ) / 20.0
    return math.sqrt(dt * dt + dr * dr)


def diverse_subset(rows: list[TaskTemplate], count: int) -> list[TaskTemplate]:
    if len(rows) <= count:
        return list(rows)
    # Deterministic farthest-first sampling.
    chosen = [rows[0]]
    remaining = list(rows[1:])
    while remaining and len(chosen) < count:
        distances = [
            min(pose_template_distance(row, prev) for prev in chosen)
            for row in remaining
        ]
        idx = int(np.argmax(distances))
        chosen.append(remaining.pop(idx))
    return chosen


def balanced_template_library(
    groups: list[tuple[str, Path, Path | None]],
    max_templates: int,
) -> list[TaskTemplate]:
    if not groups:
        raise RuntimeError("At least one --template-group is required")
    per_group = max(1, int(math.ceil(max_templates / len(groups))))
    selected: list[TaskTemplate] = []
    for name, root, pass_json in groups:
        rows = discover_group_templates(name, root, pass_json)
        if not rows:
            raise RuntimeError(f"No usable task templates in group {name}: {root}")
        selected.extend(diverse_subset(rows, per_group))

    # If group balancing produced slightly more than the global cap, run a second
    # diversity pass over the combined library.
    selected = diverse_subset(selected, max_templates)
    print(f"[templates] final balanced diverse library={len(selected)}")
    for i, row in enumerate(selected):
        rel = row.T_object_flange[:3, 3]
        print(
            f"    T{i:02d} group={row.group:<18} cand={row.candidate_index:<5d} "
            f"rank={row.target_rank} object->flange t={np.round(rel, 4).tolist()}"
        )
    return selected


def zone_bounds(layout: dict, key: str, edge_margin_m: float) -> tuple[float, float, float, float]:
    rec = layout["transforms"][key]
    geom_key = "source_zone_size_m" if key == "source_zone" else "placement_zone_size_m"
    size = np.asarray(layout["geometry"][geom_key], dtype=np.float64)
    center = np.asarray(rec["position_world_m"], dtype=np.float64)
    x0 = float(center[0] - 0.5 * size[0] + edge_margin_m)
    x1 = float(center[0] + 0.5 * size[0] - edge_margin_m)
    y0 = float(center[1] - 0.5 * size[1] + edge_margin_m)
    y1 = float(center[1] + 0.5 * size[1] - edge_margin_m)
    if x0 >= x1 or y0 >= y1:
        raise RuntimeError(f"{key}: edge margin too large")
    return x0, x1, y0, y1


def make_zone_tasks(
    name: str,
    layout: dict,
    zone_key: str,
    grid_nx: int,
    grid_ny: int,
    edge_margin_m: float,
    templates: list[TaskTemplate],
) -> ZoneTaskSet:
    x0, x1, y0, y1 = zone_bounds(layout, zone_key, edge_margin_m)
    xs = np.linspace(x0, x1, int(grid_nx), dtype=np.float64)
    ys = np.linspace(y0, y1, int(grid_ny), dtype=np.float64)
    points = np.asarray([(x, y) for y in ys for x in xs], dtype=np.float64)
    poses = np.stack([
        tpl.flange_at_xy(float(x), float(y))
        for x, y in points
        for tpl in templates
    ])
    return ZoneTaskSet(
        name=name,
        points_xy=points,
        poses_world=poses,
        template_count=len(templates),
    )


def candidate_world_from_base(baseline: np.ndarray, xyz: tuple[float, float, float]) -> np.ndarray:
    T = np.asarray(baseline, dtype=np.float64).copy()
    T[:3, 3] = np.asarray(xyz, dtype=np.float64)
    return T


def solve_zone(
    ik,
    zone: ZoneTaskSet,
    T_world_base: np.ndarray,
    robust_min_fraction: float,
) -> ZoneMetrics:
    T_base_world = np.linalg.inv(T_world_base)
    targets_base = T_base_world[None] @ zone.poses_world
    result = ik.solve(targets_base)

    success = np.any(result.accepted, axis=1)
    P = len(zone.points_xy)
    T = int(zone.template_count)
    success_pt = success.reshape(P, T)
    fractions = success_pt.mean(axis=1)

    margin = np.asarray(result.inner_limit_margin_rad, dtype=np.float64)
    accepted = np.asarray(result.accepted, dtype=bool)
    masked = np.where(accepted, margin, -np.inf)
    best_margin = np.max(masked, axis=1)
    finite_margin = np.isfinite(best_margin)

    return ZoneMetrics(
        any_coverage=float(np.mean(np.any(success_pt, axis=1))),
        robust_coverage=float(np.mean(fractions >= float(robust_min_fraction))),
        mean_template_reach_fraction=float(np.mean(fractions)),
        median_template_reach_fraction=float(np.median(fractions)),
        mean_best_margin_deg=(
            float(np.degrees(np.mean(best_margin[finite_margin])))
            if np.any(finite_margin) else float("-inf")
        ),
        pose_success_fraction=float(np.mean(success)),
        point_reach_fraction=fractions,
    )


def table_forward_edge_y(layout: dict) -> float:
    center = np.asarray(layout["transforms"]["table"]["position_world_m"], dtype=np.float64)
    size = np.asarray(layout["geometry"]["table_size_m"], dtype=np.float64)
    return float(center[1] + 0.5 * size[1])


def score_layout(
    source: ZoneMetrics,
    placement: ZoneMetrics,
    source_weight: float,
    placement_weight: float,
) -> float:
    sw = float(source_weight)
    pw = float(placement_weight)
    denom = sw + pw
    return (sw * source.score_source() + pw * placement.score_placement()) / denom


def evaluate_layouts(
    *,
    ik,
    layouts_xyz: list[tuple[float, float, float]],
    baseline_world_from_base: np.ndarray,
    source_tasks: ZoneTaskSet,
    placement_tasks: ZoneTaskSet,
    robust_min_fraction: float,
    source_weight: float,
    placement_weight: float,
    table_edge_y: float,
    stage: str,
) -> list[dict]:
    rows = []
    total = len(layouts_xyz)
    started = time.perf_counter()
    for i, xyz in enumerate(layouts_xyz, start=1):
        T_world_base = candidate_world_from_base(baseline_world_from_base, xyz)
        s = solve_zone(ik, source_tasks, T_world_base, robust_min_fraction)
        p = solve_zone(ik, placement_tasks, T_world_base, robust_min_fraction)
        score = score_layout(s, p, source_weight, placement_weight)
        row = {
            "stage": stage,
            "x_m": float(xyz[0]),
            "y_m": float(xyz[1]),
            "z_m": float(xyz[2]),
            "score": float(score),
            "source_score": float(s.score_source()),
            "source_any_coverage": s.any_coverage,
            "source_robust_coverage": s.robust_coverage,
            "source_mean_template_fraction": s.mean_template_reach_fraction,
            "source_median_template_fraction": s.median_template_reach_fraction,
            "source_pose_success_fraction": s.pose_success_fraction,
            "source_mean_best_margin_deg": s.mean_best_margin_deg,
            "placement_score": float(p.score_placement()),
            "placement_any_coverage": p.any_coverage,
            "placement_robust_coverage": p.robust_coverage,
            "placement_mean_template_fraction": p.mean_template_reach_fraction,
            "placement_pose_success_fraction": p.pose_success_fraction,
            "placement_mean_best_margin_deg": p.mean_best_margin_deg,
            "base_center_gap_to_table_forward_edge_y_m": float(xyz[1] - table_edge_y),
        }
        rows.append(row)
        if i == 1 or i % 10 == 0 or i == total:
            elapsed = time.perf_counter() - started
            rate = elapsed / i
            eta = rate * (total - i)
            print(
                f"[{stage}] {i:4d}/{total:4d} "
                f"xyz=({xyz[0]:+.3f},{xyz[1]:+.3f},{xyz[2]:+.3f}) "
                f"score={score:.4f} "
                f"Srob={s.robust_coverage:.3f} Sany={s.any_coverage:.3f} "
                f"Pany={p.any_coverage:.3f} ETA={eta:.1f}s",
                flush=True,
            )
    return rows


def unique_xyz(rows: list[dict]) -> list[dict]:
    seen = {}
    for row in rows:
        key = (
            round(float(row["x_m"]), 6),
            round(float(row["y_m"]), 6),
            round(float(row["z_m"]), 6),
        )
        old = seen.get(key)
        if old is None or float(row["score"]) > float(old["score"]):
            seen[key] = row
    return list(seen.values())


def cartesian_xyz(
    xs: np.ndarray, ys: np.ndarray, zs: np.ndarray
) -> list[tuple[float, float, float]]:
    return [
        (float(x), float(y), float(z))
        for x in xs for y in ys for z in zs
    ]


def refinement_xyz(
    center: tuple[float, float, float],
    xy_radius: float,
    z_radius: float,
    xy_step: float,
    z_step: float,
) -> list[tuple[float, float, float]]:
    x0, y0, z0 = center
    xs = axis_values((x0 - xy_radius, x0 + xy_radius, xy_step))
    ys = axis_values((y0 - xy_radius, y0 + xy_radius, xy_step))
    zs = axis_values((z0 - z_radius, z0 + z_radius, z_step))
    return cartesian_xyz(xs, ys, zs)


def select_recommendations(
    rows: list[dict],
    current_xyz: np.ndarray,
    min_clearance_indicator_m: float,
    plateau_abs_tol: float,
) -> dict:
    ordered = sorted(rows, key=lambda r: float(r["score"]), reverse=True)
    best = ordered[0]
    best_score = float(best["score"])
    plateau = [
        row for row in ordered
        if float(row["score"]) >= best_score - float(plateau_abs_tol)
    ]
    clearance_plateau = [
        row for row in plateau
        if float(row["base_center_gap_to_table_forward_edge_y_m"])
        >= float(min_clearance_indicator_m)
    ]
    pool = clearance_plateau or plateau

    def movement(row):
        xyz = np.asarray([row["x_m"], row["y_m"], row["z_m"]], dtype=np.float64)
        return float(np.linalg.norm(xyz - current_xyz))

    nearest = min(pool, key=movement)
    return {
        "best_score_layout": best,
        "plateau_abs_tolerance": float(plateau_abs_tol),
        "plateau_count": len(plateau),
        "plateau_clearance_count": len(clearance_plateau),
        "plateau_nearest_current_layout": {
            **nearest,
            "movement_from_current_m": movement(nearest),
        },
    }


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def plot_heatmap(
    path: Path,
    zone: ZoneTaskSet,
    metrics: ZoneMetrics,
    title: str,
    nx: int,
    ny: int,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[plot warning] matplotlib unavailable: {exc}")
        return
    values = metrics.point_reach_fraction.reshape(int(ny), int(nx))
    xs = zone.points_xy[:, 0].reshape(int(ny), int(nx))
    ys = zone.points_xy[:, 1].reshape(int(ny), int(nx))
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    im = ax.imshow(
        values,
        origin="lower",
        extent=[float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max())],
        aspect="auto",
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_xlabel("world x [m]")
    ax.set_ylabel("world y [m]")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="reachable task-template fraction")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--project-root",
        type=Path,
        default=Path("/home/lin/Projects/DexGraspNet2_Wuji2"),
    )
    p.add_argument(
        "--layout",
        type=Path,
        default=Path("08_dual_arm_scene_layout/config/manual_layout_calibrated.json"),
    )
    p.add_argument(
        "--template-group",
        action="append",
        type=parse_template_group,
        required=True,
        help="repeatable: NAME|ROOT or NAME|ROOT|PASS_JSON",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "08_dual_arm_scene_layout/isaaclab_control/outputs/"
            "right_arm_workspace_layout_calibration"
        ),
    )

    # 315 coarse candidates around the geometry-based first recommendation.
    p.add_argument("--x-range", type=parse_triplet, default=(-0.16, 0.00, 0.02))
    p.add_argument("--y-range", type=parse_triplet, default=(0.10, 0.22, 0.02))
    p.add_argument("--z-range", type=parse_triplet, default=(0.70, 0.80, 0.025))

    # Local 5x5x5 = 125 candidate refinement around coarse best.
    p.add_argument("--refine-xy-radius-m", type=float, default=0.02)
    p.add_argument("--refine-z-radius-m", type=float, default=0.025)
    p.add_argument("--refine-xy-step-m", type=float, default=0.01)
    p.add_argument("--refine-z-step-m", type=float, default=0.0125)

    p.add_argument("--max-templates", type=int, default=12)
    p.add_argument("--coarse-source-grid", default="12,8")
    p.add_argument("--coarse-placement-grid", default="8,4")
    p.add_argument("--refine-source-grid", default="20,12")
    p.add_argument("--refine-placement-grid", default="12,5")
    p.add_argument("--zone-edge-margin-m", type=float, default=0.02)
    p.add_argument("--robust-min-template-fraction", type=float, default=0.25)

    p.add_argument("--source-weight", type=float, default=0.85)
    p.add_argument("--placement-weight", type=float, default=0.15)

    # The real project strict exact-IK contract.
    p.add_argument("--seeds", type=int, default=48)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--device", default="cuda:0")

    # This is only an indicator, NOT a collision proof.  Static Isaac validation
    # is still required for the top layouts.
    p.add_argument("--min-clearance-indicator-m", type=float, default=0.07)
    p.add_argument("--plateau-abs-tol", type=float, default=0.005)
    return p


def parse_grid(text: str) -> tuple[int, int]:
    parts = [int(x.strip()) for x in text.split(",")]
    if len(parts) != 2 or any(x < 2 for x in parts):
        raise ValueError("grid must be NX,NY with both >=2")
    return parts[0], parts[1]


def main() -> int:
    args = build_parser().parse_args()
    project_root = args.project_root.expanduser().resolve()
    layout_path = args.layout.expanduser()
    if not layout_path.is_absolute():
        layout_path = (project_root / layout_path).resolve()
    output_dir = args.output_dir.expanduser()
    if not output_dir.is_absolute():
        output_dir = (project_root / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    layout = load_json(layout_path)
    baseline_row = np.asarray(
        layout["transforms"]["dual_arm_mount"]["Gf_local_to_world_row_major"],
        dtype=np.float64,
    )
    baseline_world_from_base = baseline_row.T
    current_xyz = baseline_world_from_base[:3, 3].copy()

    print("=" * 100)
    print("RIGHT-ARM WORKSPACE / TABLE LAYOUT CALIBRATION")
    print("read-only | no Isaac | strict cuRobo 6-D IK")
    print("=" * 100)
    print("current DualArmMount xyz =", np.round(current_xyz, 6).tolist())
    print("mount rotation is FROZEN to the current calibrated rotation")

    # Resolve group paths relative to project root.
    groups = []
    for name, root, pass_json in args.template_group:
        r = root if root.is_absolute() else project_root / root
        pj = None if pass_json is None else (
            pass_json if pass_json.is_absolute() else project_root / pass_json
        )
        groups.append((name, r.resolve(), None if pj is None else pj.resolve()))

    templates = balanced_template_library(groups, int(args.max_templates))
    write_json(
        output_dir / "task_template_library.json",
        {
            "template_count": len(templates),
            "templates": [
                {
                    "group": t.group,
                    "case_root": t.case_root,
                    "candidate_index": t.candidate_index,
                    "target_rank": t.target_rank,
                    "object_z_world_m": t.object_z_world,
                    "T_object_flange": t.T_object_flange.tolist(),
                }
                for t in templates
            ],
        },
    )

    control_root = project_root / "08_dual_arm_scene_layout/isaaclab_control"
    sys.path.insert(0, str(control_root))
    from core.config import IKConfig
    from core.ik import CuroboGpuIK

    robot_urdf = (
        project_root
        / "01_environment/vendor/wuji-description/"
        "dual_arm_right_wuji2/urdf/dual_arm_right_wuji2.urdf"
    )
    if not robot_urdf.is_file():
        raise FileNotFoundError(robot_urdf)

    ik_cfg = IKConfig(
        device=str(args.device),
        num_seeds=int(args.seeds),
        batch_size=int(args.batch_size),
        return_seeds=int(args.seeds),
    )
    ik = CuroboGpuIK(robot_urdf, ik_cfg)
    print(
        f"cuRobo={ik.version} | seeds={args.seeds} | batch={args.batch_size} "
        f"| tolerance=5mm/5deg | margin=3deg"
    )

    table_edge_y = table_forward_edge_y(layout)

    # ---------------- coarse search ----------------
    csx, csy = parse_grid(args.coarse_source_grid)
    cpx, cpy = parse_grid(args.coarse_placement_grid)
    coarse_source = make_zone_tasks(
        "source", layout, "source_zone",
        csx, csy, args.zone_edge_margin_m, templates,
    )
    coarse_place = make_zone_tasks(
        "placement", layout, "placement_zone",
        cpx, cpy, args.zone_edge_margin_m, templates,
    )
    coarse_xyz = cartesian_xyz(
        axis_values(args.x_range),
        axis_values(args.y_range),
        axis_values(args.z_range),
    )
    print(
        f"[coarse] layouts={len(coarse_xyz)} "
        f"source points={len(coarse_source.points_xy)} "
        f"placement points={len(coarse_place.points_xy)} "
        f"templates={len(templates)}"
    )
    coarse_rows = evaluate_layouts(
        ik=ik,
        layouts_xyz=coarse_xyz,
        baseline_world_from_base=baseline_world_from_base,
        source_tasks=coarse_source,
        placement_tasks=coarse_place,
        robust_min_fraction=args.robust_min_template_fraction,
        source_weight=args.source_weight,
        placement_weight=args.placement_weight,
        table_edge_y=table_edge_y,
        stage="coarse",
    )
    coarse_best = max(coarse_rows, key=lambda r: float(r["score"]))
    coarse_best_xyz = (
        float(coarse_best["x_m"]),
        float(coarse_best["y_m"]),
        float(coarse_best["z_m"]),
    )
    print("\n[coarse best]", coarse_best_xyz, "score=", coarse_best["score"])

    # ---------------- local refinement ----------------
    rsx, rsy = parse_grid(args.refine_source_grid)
    rpx, rpy = parse_grid(args.refine_placement_grid)
    refine_source = make_zone_tasks(
        "source", layout, "source_zone",
        rsx, rsy, args.zone_edge_margin_m, templates,
    )
    refine_place = make_zone_tasks(
        "placement", layout, "placement_zone",
        rpx, rpy, args.zone_edge_margin_m, templates,
    )
    refine_xyz = refinement_xyz(
        coarse_best_xyz,
        args.refine_xy_radius_m,
        args.refine_z_radius_m,
        args.refine_xy_step_m,
        args.refine_z_step_m,
    )
    print(
        f"\n[refine] layouts={len(refine_xyz)} "
        f"source points={len(refine_source.points_xy)} "
        f"placement points={len(refine_place.points_xy)}"
    )
    refine_rows = evaluate_layouts(
        ik=ik,
        layouts_xyz=refine_xyz,
        baseline_world_from_base=baseline_world_from_base,
        source_tasks=refine_source,
        placement_tasks=refine_place,
        robust_min_fraction=args.robust_min_template_fraction,
        source_weight=args.source_weight,
        placement_weight=args.placement_weight,
        table_edge_y=table_edge_y,
        stage="refine",
    )

    all_rows = unique_xyz(coarse_rows + refine_rows)
    all_rows_sorted = sorted(all_rows, key=lambda r: float(r["score"]), reverse=True)
    recommendations = select_recommendations(
        all_rows_sorted,
        current_xyz=current_xyz,
        min_clearance_indicator_m=args.min_clearance_indicator_m,
        plateau_abs_tol=args.plateau_abs_tol,
    )
    best = recommendations["best_score_layout"]
    nearest = recommendations["plateau_nearest_current_layout"]

    # For final heatmaps, evaluate the best-score layout once on dense refinement grids.
    best_xyz = (float(best["x_m"]), float(best["y_m"]), float(best["z_m"]))
    T_best = candidate_world_from_base(baseline_world_from_base, best_xyz)
    best_source_metrics = solve_zone(
        ik, refine_source, T_best, args.robust_min_template_fraction
    )
    best_place_metrics = solve_zone(
        ik, refine_place, T_best, args.robust_min_template_fraction
    )

    plot_heatmap(
        output_dir / "best_source_reachability_heatmap.png",
        refine_source,
        best_source_metrics,
        f"SourceZone exact-IK reachability @ DualArmMount {np.round(best_xyz, 3).tolist()}",
        rsx,
        rsy,
    )
    plot_heatmap(
        output_dir / "best_placement_reachability_heatmap.png",
        refine_place,
        best_place_metrics,
        f"PlacementZone exact-IK reachability @ DualArmMount {np.round(best_xyz, 3).tolist()}",
        rpx,
        rpy,
    )

    save_csv(output_dir / "layout_scan_all.csv", all_rows_sorted)
    write_json(output_dir / "top20_layouts.json", {"rows": all_rows_sorted[:20]})

    delta_best = np.asarray(best_xyz) - current_xyz
    nearest_xyz = np.asarray(
        [nearest["x_m"], nearest["y_m"], nearest["z_m"]], dtype=np.float64
    )
    delta_nearest = nearest_xyz - current_xyz

    report = {
        "schema_version": 1,
        "status": "PASS",
        "purpose": "optimize DualArmMount xyz for right-arm table workspace",
        "does_not_modify_layout": True,
        "does_not_start_isaac": True,
        "strict_ik_contract": {
            "position_tolerance_m": 0.005,
            "orientation_tolerance_deg": 5.0,
            "minimum_inner_joint_margin_deg": 3.0,
            "seeds": int(args.seeds),
            "tool_frame": "arm_r_link_tf",
        },
        "layout_source": str(layout_path),
        "current_dual_arm_mount_xyz_m": current_xyz.tolist(),
        "mount_rotation_frozen": baseline_world_from_base[:3, :3].tolist(),
        "search": {
            "coarse_layout_count": len(coarse_xyz),
            "refine_layout_count": len(refine_xyz),
            "total_unique_layouts": len(all_rows_sorted),
            "source_weight": float(args.source_weight),
            "placement_weight": float(args.placement_weight),
            "robust_min_template_fraction": float(args.robust_min_template_fraction),
            "task_template_count": len(templates),
            "zone_edge_margin_m": float(args.zone_edge_margin_m),
        },
        "recommendations": {
            **recommendations,
            "best_score_delta_from_current_m": delta_best.tolist(),
            "plateau_nearest_current_delta_m": delta_nearest.tolist(),
        },
        "best_layout_dense_zone_metrics": {
            "source": {
                "any_coverage": best_source_metrics.any_coverage,
                "robust_coverage": best_source_metrics.robust_coverage,
                "mean_template_reach_fraction": best_source_metrics.mean_template_reach_fraction,
                "pose_success_fraction": best_source_metrics.pose_success_fraction,
                "mean_best_margin_deg": best_source_metrics.mean_best_margin_deg,
            },
            "placement": {
                "any_coverage": best_place_metrics.any_coverage,
                "robust_coverage": best_place_metrics.robust_coverage,
                "mean_template_reach_fraction": best_place_metrics.mean_template_reach_fraction,
                "pose_success_fraction": best_place_metrics.pose_success_fraction,
                "mean_best_margin_deg": best_place_metrics.mean_best_margin_deg,
            },
        },
        "static_collision_note": (
            "base_center_gap_to_table_forward_edge_y_m is only a geometric indicator. "
            "The final top layouts still require an Isaac/PhysX static penetration check."
        ),
        "outputs": {
            "csv": str(output_dir / "layout_scan_all.csv"),
            "top20": str(output_dir / "top20_layouts.json"),
            "template_library": str(output_dir / "task_template_library.json"),
            "source_heatmap": str(output_dir / "best_source_reachability_heatmap.png"),
            "placement_heatmap": str(output_dir / "best_placement_reachability_heatmap.png"),
        },
    }
    report_path = output_dir / "layout_calibration_report.json"
    write_json(report_path, report)

    print("\n" + "=" * 100)
    print("LAYOUT CALIBRATION COMPLETE")
    print("=" * 100)
    print("Current DualArmMount :", np.round(current_xyz, 4).tolist())
    print(
        "Best score layout    :",
        np.round(np.asarray(best_xyz), 4).tolist(),
        "delta=",
        np.round(delta_best, 4).tolist(),
    )
    print(
        f"  Source any/robust   : "
        f"{best_source_metrics.any_coverage:.3f} / {best_source_metrics.robust_coverage:.3f}"
    )
    print(
        f"  Placement any       : {best_place_metrics.any_coverage:.3f}"
    )
    print(
        "Plateau nearest curr :",
        np.round(nearest_xyz, 4).tolist(),
        "delta=",
        np.round(delta_nearest, 4).tolist(),
    )
    print("report:", report_path)
    print("IMPORTANT: do not edit the production layout until top layouts pass static Isaac validation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
