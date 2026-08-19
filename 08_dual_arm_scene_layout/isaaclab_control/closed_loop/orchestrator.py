#!/usr/bin/env python3
"""One-command persistent semantic dexterous grasp loop.

V2 architecture
---------------
* Isaac Lab/Sim starts once and keeps the same physical world for every capture
  and every grasp cycle.
* cuRobo starts once per candidate batch and is released before Isaac execution.
* legacy approximate GRASP/PREGRASP coarse IK gates are configurable and OFF by
  default.
* after LEAP->Wuji2, exact COVER is the hard grasp-root IK gate.
* PREGRASP/LIFT/TRANSFER/PLACE/RETREAT use large configurable 6D task sets;
  strict IK accuracy is unchanged.
* the q7 route produced by planning is executed directly in the same Isaac
  world: no second runtime IK and no pre-execution FK gate.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import re
import shlex
import struct
import subprocess
import sys
import time
import zlib
from datetime import datetime

import numpy as np


HERE = Path(__file__).resolve().parent
CONTROL_ROOT = HERE.parent
SCRIPTS = HERE / "scripts"
DEFAULT_CONFIG = HERE / "config/closed_loop.json"
sys.path.insert(0, str(CONTROL_ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(SCRIPTS))

from core.bridge import CuroboWorkerClient  # noqa: E402
from core.config import WorkerConfig  # noqa: E402
from persistent_isaac import PersistentIsaacClient  # noqa: E402
from planning.flexible_route_search import (  # noqa: E402
    screen_exact_cover_batch,
    summarize_exact_cover_subfunnel,
)
from planning.simplified_route_search import plan_flexible_route  # noqa: E402
from planning.candidate_rfs_v2_runtime import run_candidate_rfs_v2  # noqa: E402
from all_candidate_gpu_prefilter import load_targets  # noqa: E402


VERBOSE = False
DEBUG_LOG: Path | None = None
WORKSPACE_ROI_XYXY = (170, 0, 970, 700)


def load_json(path: Path) -> dict:
    return json.loads(Path(path).resolve().read_text(encoding="utf-8"))


def resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def debug_write(text: str) -> None:
    if DEBUG_LOG is None:
        return
    DEBUG_LOG.parent.mkdir(parents=True, exist_ok=True)
    with DEBUG_LOG.open("a", encoding="utf-8") as stream:
        stream.write(str(text))
        if text and not str(text).endswith("\n"):
            stream.write("\n")


def gpu_memory_snapshot() -> str:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=5.0,
        )
    except Exception as exc:
        return f"unavailable ({type(exc).__name__}: {exc})"
    if completed.returncode != 0:
        reason = (completed.stderr or completed.stdout or "").strip()
        return f"unavailable ({reason})"
    rows = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    return "; ".join(f"gpu{index}: used/free MiB={row}" for index, row in enumerate(rows)) or "unavailable"


def prepare_roi_depth_for_esdf(capture_root: Path) -> tuple[Path, Path]:
    """Keep full K/T/depth shape, but invalidate depth outside the workspace ROI."""
    depth_path = Path(capture_root) / "depth_m.npy"
    depth = np.load(depth_path).astype(np.float32, copy=True)
    height, width = depth.shape
    x1, y1, x2, y2 = WORKSPACE_ROI_XYXY
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(f"Invalid workspace ROI {WORKSPACE_ROI_XYXY} for depth shape {(height, width)}")
    roi_depth = np.zeros_like(depth, dtype=np.float32)
    roi_depth[y1:y2, x1:x2] = depth[y1:y2, x1:x2]
    out = Path(capture_root) / "depth_m_workspace_roi.npy"
    np.save(out, roi_depth)
    metadata = {
        "schema_version": 1,
        "purpose": "planner ESDF workspace ROI; DGN2 40k input still uses full depth_m.npy",
        "roi_xyxy_pixels": [x1, y1, x2, y2],
        "full_depth_shape_hw": [height, width],
        "invalidated_outside_roi": True,
        "intrinsics_unchanged": True,
        "T_world_camera_unchanged": True,
    }
    meta_path = Path(capture_root) / "depth_workspace_roi_metadata.json"
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out, meta_path


def safe_slug(text: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z._-]+", "_", text.strip()).strip("._")
    return slug[:64] or "target"


def run(label: str, cmd: list, *, cwd: Path, env=None, capture_json: bool = False):
    command_line = " ".join(shlex.quote(str(value)) for value in cmd)
    debug_write(f"\n{'='*18} {label} {'='*18}\n$ {command_line}\n")
    if VERBOSE:
        print(f"\n{'='*18} {label} {'='*18}")
        print("$", command_line, flush=True)
    completed = subprocess.run(
        [str(value) for value in cmd],
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    debug_write(completed.stdout or "")
    if VERBOSE and completed.stdout:
        print(completed.stdout, end="")
    if completed.returncode:
        tail = "\n".join((completed.stdout or "").splitlines()[-30:])
        raise RuntimeError(f"{label} failed: {completed.returncode}\n{tail}")
    if not capture_json:
        return None
    lines = [line for line in (completed.stdout or "").splitlines() if line.strip()]
    for line in reversed(lines):
        try:
            return json.loads(line)
        except Exception:
            pass
    raise RuntimeError(f"{label} did not emit a final JSON object")


def show_async(template, **kwargs) -> None:
    if not template:
        return
    cmd = [str(value).format(**kwargs) for value in template]
    try:
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    except Exception as exc:
        debug_write(f"viewer failed: {exc}")


def show_image_path(path: Path, cfg: dict) -> None:
    path = Path(path)
    if not path.is_file():
        return
    template = cfg.get("show_overlay_command") or ["xdg-open", "{overlay}"]
    try:
        show_async(template, overlay=str(path), rgb=str(path))
    except Exception as exc:
        print(f"    ⚠ 图片打开失败：{path} ({exc})")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_planning_funnel(cycle_root: Path, funnel: dict) -> Path:
    path = cycle_root / "planning_funnel.json"
    write_json(path, funnel)
    return path


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def _read_png_rgb(path: Path) -> np.ndarray:
    data = Path(path).read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"not a PNG: {path}")
    offset = 8
    width = height = color_type = bit_depth = None
    idat = bytearray()
    while offset < len(data):
        length = struct.unpack(">I", data[offset:offset + 4])[0]
        chunk_type = data[offset + 4:offset + 8]
        chunk = data[offset + 8:offset + 8 + length]
        offset += 12 + length
        if chunk_type == b"IHDR":
            width, height, bit_depth, color_type, _comp, _filter, _interlace = struct.unpack(">IIBBBBB", chunk)
        elif chunk_type == b"IDAT":
            idat.extend(chunk)
        elif chunk_type == b"IEND":
            break
    if bit_depth != 8 or color_type not in (2, 6):
        raise ValueError(f"unsupported PNG format bit_depth={bit_depth} color_type={color_type}: {path}")
    channels = 3 if color_type == 2 else 4
    stride = int(width) * channels
    raw = zlib.decompress(bytes(idat))
    rows = np.zeros((int(height), stride), dtype=np.uint8)
    src = 0
    for y in range(int(height)):
        filter_type = raw[src]
        src += 1
        row = np.frombuffer(raw[src:src + stride], dtype=np.uint8).copy()
        src += stride
        prev = rows[y - 1] if y else np.zeros(stride, dtype=np.uint8)
        recon = row
        bpp = channels
        for x in range(stride):
            left = int(recon[x - bpp]) if x >= bpp else 0
            up = int(prev[x])
            up_left = int(prev[x - bpp]) if x >= bpp else 0
            if filter_type == 1:
                recon[x] = (int(recon[x]) + left) & 0xFF
            elif filter_type == 2:
                recon[x] = (int(recon[x]) + up) & 0xFF
            elif filter_type == 3:
                recon[x] = (int(recon[x]) + ((left + up) // 2)) & 0xFF
            elif filter_type == 4:
                recon[x] = (int(recon[x]) + _paeth(left, up, up_left)) & 0xFF
            elif filter_type != 0:
                raise ValueError(f"unsupported PNG filter {filter_type}: {path}")
        rows[y] = recon
    image = rows.reshape((int(height), int(width), channels))
    return image[:, :, :3].copy()


def _write_png_rgb(path: Path, image: np.ndarray) -> None:
    image = np.asarray(image, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected RGB uint8 image, got {image.shape}")
    height, width, _ = image.shape
    raw = bytearray()
    for y in range(height):
        raw.append(0)
        raw.extend(image[y].tobytes())

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
        )

    payload = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(bytes(raw), level=6))
        + chunk(b"IEND", b"")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _blend_pixels(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> None:
    if not np.any(mask):
        return
    src = np.asarray(color, dtype=np.float32)
    image[mask] = np.clip((1.0 - alpha) * image[mask].astype(np.float32) + alpha * src, 0, 255).astype(np.uint8)


def _draw_circle(image: np.ndarray, x: float, y: float, radius: int, color: tuple[int, int, int], alpha: float) -> None:
    h, w, _ = image.shape
    x0 = max(0, int(math.floor(x - radius)))
    x1 = min(w - 1, int(math.ceil(x + radius)))
    y0 = max(0, int(math.floor(y - radius)))
    y1 = min(h - 1, int(math.ceil(y + radius)))
    if x0 > x1 or y0 > y1:
        return
    yy, xx = np.ogrid[y0:y1 + 1, x0:x1 + 1]
    mask = (xx - float(x)) ** 2 + (yy - float(y)) ** 2 <= float(radius * radius)
    region = image[y0:y1 + 1, x0:x1 + 1]
    _blend_pixels(region, mask, color, alpha)


_DIGITS_3X5 = {
    "0": ("111", "101", "101", "101", "111"),
    "1": ("010", "110", "010", "010", "111"),
    "2": ("111", "001", "111", "100", "111"),
    "3": ("111", "001", "111", "001", "111"),
    "4": ("101", "101", "111", "001", "001"),
    "5": ("111", "100", "111", "001", "111"),
    "6": ("111", "100", "111", "101", "111"),
    "7": ("111", "001", "010", "010", "010"),
    "8": ("111", "101", "111", "101", "111"),
    "9": ("111", "101", "111", "001", "111"),
}


def _draw_small_digits(image: np.ndarray, x: int, y: int, text: str, color: tuple[int, int, int]) -> None:
    h, w, _ = image.shape
    cx = int(x)
    for char in str(text):
        bitmap = _DIGITS_3X5.get(char)
        if bitmap is None:
            cx += 4
            continue
        for row_index, row in enumerate(bitmap):
            for col_index, value in enumerate(row):
                if value != "1":
                    continue
                px = cx + col_index
                py = int(y) + row_index
                if 0 <= px < w and 0 <= py < h:
                    image[py, px] = color
        cx += 4


def generate_leap_candidate_overlay(
    *,
    rgb_path: Path,
    mask_path: Path,
    prediction: Path,
    intrinsics_path: Path,
    T_world_camera_path: Path,
    output_path: Path,
    query: str,
) -> Path | None:
    """Draw DGN2 LEAP root candidate positions without changing candidate data."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception as exc:
        Image = ImageDraw = ImageFont = None
        debug_write(f"LEAP overlay using stdlib PNG fallback: PIL unavailable: {exc}")

    try:
        mask = np.load(mask_path).astype(bool)
        K = np.asarray(np.load(intrinsics_path), dtype=np.float64)
        T_world_camera = np.asarray(np.load(T_world_camera_path), dtype=np.float64)
        T_camera_world = np.linalg.inv(T_world_camera)
        with np.load(prediction, allow_pickle=False) as z:
            roots_world = np.asarray(z["translation_world"], dtype=np.float64)
            order = np.asarray(z["target_score_descending_candidate_index"], dtype=np.int64)
            scores = np.asarray(z["score"], dtype=np.float64)
        if order.size == 0:
            return None
        title = (
            f"LEAP grasp root region | total target candidates={len(order)} | "
            f"top score={float(scores[int(order[0])]):.6f} | query={query}"
        )

        if Image is not None:
            rgb = Image.open(rgb_path).convert("RGBA")
            width, height = rgb.size
        else:
            rgb_np = _read_png_rgb(rgb_path)
            height, width, _ = rgb_np.shape
        roots = roots_world[order]
        hom = np.concatenate([roots, np.ones((len(roots), 1), dtype=np.float64)], axis=1)
        cam = (T_camera_world @ hom.T).T[:, :3]
        z = cam[:, 2]
        valid = z > 1e-6
        u = K[0, 0] * cam[:, 0] / np.maximum(z, 1e-9) + K[0, 2]
        v = K[1, 1] * cam[:, 1] / np.maximum(z, 1e-9) + K[1, 2]
        valid &= (u >= 0) & (u < width) & (v >= 0) & (v < height)

        if Image is None:
            image = rgb_np.copy()
            if mask.shape == (height, width):
                _blend_pixels(image, mask, (255, 0, 0), 0.35)
            # Dark title strip and visual legend. Full title is saved beside the PNG.
            image[:24, :, :] = (0.35 * image[:24, :, :]).astype(np.uint8)
            for idx in np.where(valid)[0]:
                _draw_circle(image, float(u[idx]), float(v[idx]), 2, (0, 220, 255), 0.35)
            for idx in np.where(valid[: min(100, len(order))])[0]:
                _draw_circle(image, float(u[idx]), float(v[idx]), 3, (255, 160, 0), 0.70)
            for idx in np.where(valid[: min(20, len(order))])[0]:
                _draw_circle(image, float(u[idx]), float(v[idx]), 5, (255, 255, 0), 0.90)
                _draw_small_digits(image, int(float(u[idx]) + 6), int(float(v[idx]) - 6), str(int(idx)), (255, 255, 255))
            _write_png_rgb(output_path, image)
            write_json(
                output_path.with_suffix(".metadata.json"),
                {
                    "title": title,
                    "total_target_candidates": int(len(order)),
                    "top_score": float(scores[int(order[0])]),
                    "query": query,
                    "projected_candidate_count": int(np.count_nonzero(valid)),
                    "note": "PNG generated with stdlib fallback; title metadata is stored here because PIL is unavailable.",
                },
            )
            return output_path

        overlay = Image.new("RGBA", rgb.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        if mask.shape == (height, width):
            mask_img = Image.fromarray((mask.astype(np.uint8) * 120), mode="L")
            mask_rgba = Image.new("RGBA", rgb.size, (255, 0, 0, 0))
            mask_rgba.putalpha(mask_img)
            overlay = Image.alpha_composite(overlay, mask_rgba)
            draw = ImageDraw.Draw(overlay)

        # All target candidates: small translucent cyan points.
        for idx in np.where(valid)[0]:
            x = float(u[idx])
            y = float(v[idx])
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(0, 220, 255, 90))

        # Top100: orange.
        for idx in np.where(valid[: min(100, len(order))])[0]:
            x = float(u[idx])
            y = float(v[idx])
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(255, 160, 0, 170))

        # Top20: larger yellow points with target_rank.
        font = ImageFont.load_default()
        for idx in np.where(valid[: min(20, len(order))])[0]:
            x = float(u[idx])
            y = float(v[idx])
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=(255, 255, 0, 230))
            draw.text((x + 6, y - 6), str(int(idx)), fill=(255, 255, 255, 255), font=font)

        composed = Image.alpha_composite(rgb, overlay)
        draw = ImageDraw.Draw(composed)
        draw.rectangle((0, 0, width, 24), fill=(0, 0, 0, 180))
        draw.text((8, 6), title, fill=(255, 255, 255, 255), font=font)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        composed.convert("RGB").save(output_path)
        return output_path
    except Exception as exc:
        debug_write(f"LEAP overlay failed: {type(exc).__name__}: {exc}")
        return None


def rfs_funnel_stats(rfs_runtime) -> dict:
    payload = rfs_runtime.to_jsonable()
    filter_path = payload.get("filter_json")
    if filter_path and Path(filter_path).is_file():
        try:
            data = load_json(Path(filter_path))
            candidate_count = int(data.get("candidate_count", payload.get("pass_count", 0) + payload.get("reject_count", 0)))
            pass_count = int(data.get("pass_count", payload.get("pass_count", 0)))
            reject_target = int(data.get("reject_target_reach_count", 0))
            reject_trajectory = int(data.get("reject_trajectory_space_count", 0))
            payload.update({
                "candidate_count": candidate_count,
                "target_reach_pass_count": int(candidate_count - reject_target),
                "trajectory_space_pass_count": int(pass_count),
                "pass_count": pass_count,
                "reject_count": int(candidate_count - pass_count),
                "reject_target_reach_count": reject_target,
                "reject_trajectory_space_count": reject_trajectory,
            })
        except Exception as exc:
            payload["stats_parse_error"] = f"{type(exc).__name__}: {exc}"
    report_path = payload.get("report_json")
    if report_path and Path(report_path).is_file():
        try:
            report = load_json(Path(report_path))
            endpoints = report.get("candidate_endpoints", {})
            trajectory = report.get("trajectory_space", {})
            payload.update({
                "grasp_coarse_ik_count": endpoints.get("grasp_direct_coarse_ik"),
                "pregrasp_coarse_ik_count": endpoints.get("pregrasp_direct_coarse_ik"),
                "support_pose_count": trajectory.get("support_pose_count"),
                "support_ik_reachable_count": trajectory.get("support_pose_ik_reachable_count"),
                "support_states_admitted_count": trajectory.get("support_pose_admitted_count", trajectory.get("support_pose_with_collision_free_ik_count")),
                "trajectory_branch_pass_count": trajectory.get("successful_branch_count"),
                "trajectory_branch_count": trajectory.get("anchor_count"),
                "observed_esdf_bypassed": trajectory.get("diagnostic_observed_esdf_bypassed"),
            })
        except Exception as exc:
            payload["report_parse_error"] = f"{type(exc).__name__}: {exc}"
    return payload


def open_rfs_diagnostics(rfs_runtime, cfg: dict) -> list[str]:
    output_dir = rfs_runtime.output_dir
    if not output_dir:
        return []
    root = Path(output_dir)
    opened = []
    for name in ("target_reach_region_overlay.png", "candidate_filter_overlay.png"):
        path = root / name
        if path.is_file():
            show_image_path(path, cfg)
            opened.append(str(path))
        else:
            debug_write(f"RFS diagnostic image missing: {path}")
    return opened


def route_failure_stage(route: dict) -> str:
    if route.get("status") == "PASS":
        return "PASS"
    return str(route.get("failed_stage") or "FLEXIBLE_ROUTE")


def print_route_diagnostics(item: dict, route: dict) -> None:
    print(f"\n    Candidate rank={item['target_rank']} candidate={item['candidate_index']}")
    summaries = route.get("stage_summaries") or []
    by_stage = {str(row.get("stage")): row for row in summaries if isinstance(row, dict)}
    pre = by_stage.get("pregrasp", {})
    pick = by_stage.get("pick_path", {})
    print(
        "      PREGRASP "
        f"targets={pre.get('target_count', 0)} raw={pre.get('raw_success_target_count', 0)} "
        f"reachable={pre.get('reachable_target_count', 0)} "
        f"accepted={pre.get('accepted_solution_count', 0)} nodes={pre.get('node_count', 0)}"
    )
    print(
        "      PRE/COVER "
        f"pairs={pick.get('pair_candidates', 0)} tested={pick.get('pairs_tested', 0)} "
        f"home_pre_fail={pick.get('home_pregrasp_fail', 0)} "
        f"home_pre_self={pick.get('home_pregrasp_self_collision_fail', 0)} "
        f"home_pre_esdf={pick.get('home_pregrasp_esdf_fail', 0)} "
        f"home_pre_pass={pick.get('home_pregrasp_pass', 0)} "
        f"pre_cover_fail={pick.get('pregrasp_cover_fail', 0)} "
        f"status={pick.get('status', 'NA')}"
    )
    for stage in ("lift", "transfer", "place", "retreat"):
        row = by_stage.get(stage, {})
        beam = by_stage.get(f"{stage}_beam", {})
        print(
            f"      {stage.upper():<8} "
            f"endpoint={row.get('target_count', 0)} "
            f"reachable={row.get('reachable_target_count', 0)} "
            f"nodes={row.get('node_count', 0)} "
            f"parents={beam.get('parent_route_count', 0)} "
            f"pairs={beam.get('possible_parent_node_pairs', 0)} "
            f"beam={beam.get('retained_beam_count', 0)}"
        )
    print(f"      FULL ROUTE = {route.get('status')} | reason={route.get('reason', '')}")


def print_funnel_summary(funnel: dict) -> None:
    batches = funnel.get("retarget_batches", [])
    first = batches[0] if batches else {}
    failures = Counter(funnel.get("flexible_route", {}).get("failure_stage_counts", {}))
    if not failures and first.get("failure_stage_counts"):
        failures.update(first["failure_stage_counts"])
    rows = {
        "FINALIZE": int(sum(int(b.get("finalize_reject", 0)) for b in batches)),
        "EXACT_COVER": int(funnel.get("exact_cover", {}).get("reject", 0)),
    }
    for key in ("PREGRASP", "HOME_TO_PRE", "PRE_TO_COVER", "LIFT", "TRANSFER", "PLACE", "RETREAT"):
        rows[key] = int(failures.get(key, 0))
    max_stage = "NA"
    if rows:
        max_stage = max(rows.items(), key=lambda item: item[1])[0]

    print("\n================ CANDIDATE FUNNEL ================")
    dgn = funnel.get("dgn2", {})
    print("[DGN2]")
    print(f"target candidates = {dgn.get('target_candidates', 0)}")
    rfs = funnel.get("rfs_v2", {})
    print("\n[RFS V2 - 粗可达性/粗路径排序]")
    print(
        f"target reach = {rfs.get('target_reach_pass_count', 'NA')}/"
        f"{rfs.get('candidate_count', dgn.get('target_candidates', 'NA'))}"
    )
    print(
        f"trajectory reach = {rfs.get('trajectory_space_pass_count', 'NA')}/"
        f"{rfs.get('candidate_count', dgn.get('target_candidates', 'NA'))}"
    )
    print(f"PASS = {rfs.get('pass_count', 'NA')}")
    print(f"REJECT/rescue = {rfs.get('reject_count', 'NA')}")
    if rfs.get("mode") == "priority_then_rescue":
        print("RFS REJECT is rescue tier; NOT a hard deletion")

    print("\n===== FIRST BATCH SURVIVAL SUMMARY =====")
    print(f"Input               {first.get('input_candidates', 0)}")
    print(f"Finalize PASS       {first.get('finalize_pass', 0)}")
    print(f"Exact COVER PASS    {first.get('exact_cover_pass', 0)}")
    print(f"Full Route PASS     {first.get('full_route_pass', 0)}")
    print("\nFailure breakdown:")
    for key in ("FINALIZE", "EXACT_COVER", "PREGRASP", "HOME_TO_PRE", "PRE_TO_COVER", "LIFT", "TRANSFER", "PLACE", "RETREAT"):
        print(f"{key:<20} {rows.get(key, 0)}")
    print(f"\n最大淘汰阶段 = {max_stage}")
    flex = funnel.get("flexible_route", {})
    hp = flex.get("home_pregrasp_path", {})
    if hp:
        print("\nHOME->PREGRASP path:")
        print(f"tested pairs          {hp.get('tested_pairs', 0)}")
        print(f"self-collision fail   {hp.get('self_collision_failures', 0)}")
        print(f"ESDF fail             {hp.get('esdf_failures', 0)}")
        print(f"PASS                  {hp.get('pass_count', 0)}")
        if hp.get("observed_esdf_bypassed"):
            print("HOME->PRE observed ESDF is DISABLED for this diagnostic run.")
        if hp.get("self_collision_bypassed"):
            print("HOME->PRE self collision is DISABLED for this diagnostic run.")
    print("==================================================")


def accumulate_home_pre_stats(funnel: dict, route: dict) -> None:
    target = funnel.setdefault("flexible_route", {}).setdefault(
        "home_pregrasp_path",
        {
            "tested_pairs": 0,
            "self_collision_failures": 0,
            "esdf_failures": 0,
            "pass_count": 0,
            "observed_esdf_bypassed": False,
            "self_collision_bypassed": False,
        },
    )
    for row in route.get("stage_summaries") or []:
        if not isinstance(row, dict) or row.get("stage") != "pick_path":
            continue
        target["tested_pairs"] += int(row.get("home_pregrasp_tested", 0))
        target["self_collision_failures"] += int(row.get("home_pregrasp_self_collision_fail", 0))
        target["esdf_failures"] += int(row.get("home_pregrasp_esdf_fail", 0))
        target["pass_count"] += int(row.get("home_pregrasp_pass", 0))
        target["observed_esdf_bypassed"] = bool(
            target.get("observed_esdf_bypassed", False)
            or row.get("home_pregrasp_esdf_bypassed", False)
        )
        target["self_collision_bypassed"] = bool(
            target.get("self_collision_bypassed", False)
            or row.get("home_pregrasp_self_collision_bypassed", False)
        )


def print_exact_cover_subfunnel(summary: dict) -> None:
    total = int(summary.get("input_candidates", 0))
    raw_targets = int(summary.get("raw_curobo_reachable_targets", 0))
    strict_targets = int(summary.get("strict_ik_targets", 0))
    post_collision_targets = int(summary.get("post_collision_targets", 0))
    final_targets = int(summary.get("final_exact_cover_pass_targets", 0))
    cover_esdf_bypassed = bool(summary.get("cover_esdf_bypassed", False))
    raw_solutions = int(summary.get("raw_success_solution_count", 0))
    strict_solutions = int(summary.get("strict_ik_accepted_solution_count", 0))
    collision_rejected = int(summary.get("collision_rejected_solution_count", 0))
    feasible_solutions = int(summary.get("feasible_solution_count", 0))
    print("\n[Exact COVER SUB-FUNNEL]")
    print(f"Input candidates              {total}")
    print(f"Raw cuRobo reachable          {raw_targets} / {total}")
    print(f"Strict 5mm/5deg/3deg IK       {strict_targets} / {total}")
    if cover_esdf_bypassed:
        print("COVER ESDF collision          BYPASSED")
    else:
        print(f"After COVER ESDF collision    {post_collision_targets} / {strict_targets}")
    print(f"Final Exact COVER PASS        {final_targets} / {total}")
    print("")
    print("Solutions:")
    print(f"raw success                   {raw_solutions}")
    print(f"strict accepted               {strict_solutions}")
    if cover_esdf_bypassed:
        print("collision rejected            0 (diagnostic bypass)")
    else:
        print(f"collision rejected            {collision_rejected}")
    print(f"feasible                      {feasible_solutions}")
    if cover_esdf_bypassed:
        print("DIAGNOSTIC ONLY:")
        print("Exact COVER ESDF collision filtering is disabled.")
    elif strict_solutions > 0 and feasible_solutions == 0:
        print("Exact COVER is being eliminated by collision filtering, not IK reachability.")
    elif strict_solutions == 0:
        print("Exact COVER is being eliminated by strict IK acceptance before collision filtering.")


def prompt_scene(project_root: Path, supplied: str | None) -> Path:
    if supplied:
        folder = Path(supplied).expanduser()
    else:
        print("\n请输入场景文件夹地址（文件夹内需直接包含 scene_manifest.json）")
        print("例如：/home/lin/Projects/DexGraspNet2_Wuji2/02_training_dataset/.../scenes/scene_0000")
        folder = Path(input("Scene folder > ").strip()).expanduser()
    folder = (project_root / folder).resolve() if not folder.is_absolute() else folder.resolve()
    manifest = folder / "scene_manifest.json"
    if not folder.is_dir() or not manifest.is_file():
        raise FileNotFoundError(f"场景目录必须包含 scene_manifest.json: {folder}")
    print(f"✓ 场景：{folder}")
    return folder


def load_robot_state(path: Path) -> tuple[np.ndarray, dict]:
    state = load_json(path)
    return (
        np.asarray(state["right_arm_q_current_rad"], dtype=np.float64),
        {str(key): float(value) for key, value in state["joint_positions_by_name"].items()},
    )


def world_from_base(project_root: Path) -> np.ndarray:
    layout = load_json(project_root / "08_dual_arm_scene_layout/config/manual_layout_calibrated.json")
    return np.asarray(
        layout["transforms"]["dual_arm_mount"]["Gf_local_to_world_row_major"],
        dtype=np.float64,
    ).T


def candidate_order(prediction: Path) -> tuple[list[dict], int]:
    with np.load(prediction, allow_pickle=False) as z:
        order = np.asarray(z["target_score_descending_candidate_index"], dtype=np.int64)
        score = np.asarray(z["score"], dtype=np.float64)
        graspness = np.asarray(z["graspness"], dtype=np.float64)
        log_prob = np.asarray(z["log_prob"], dtype=np.float64)
        total = int(len(score))
    rows = []
    for rank, index in enumerate(order):
        idx = int(index)
        rows.append({
            "target_rank": int(rank),
            "candidate_index": idx,
            "score": float(score[idx]),
            "graspness": float(graspness[idx]),
            "log_prob": float(log_prob[idx]),
        })
    return rows, total


def legacy_coarse_prefilter(
    *,
    client,
    project_root: Path,
    prediction: Path,
    q_current: np.ndarray,
    cfg: dict,
) -> tuple[list[dict], list[int], dict]:
    """Optional compatibility gate; default config bypasses it completely."""
    settings = cfg["coarse_ik_prefilter"]
    candidates, grasp_targets, pregrasp_targets, total = load_targets(
        project_root,
        prediction,
        float(settings.get("legacy_pregrasp_offset_m", 0.10)),
    )
    survivors = list(range(len(candidates)))
    report = {
        "enabled": True,
        "total_proposals": total,
        "target_candidates": len(candidates),
        "grasp_enabled": bool(settings.get("grasp_enabled", False)),
        "pregrasp_enabled": bool(settings.get("pregrasp_enabled", False)),
    }
    if bool(settings.get("grasp_enabled", False)):
        result = client.solve_ik(grasp_targets, q_current, select_chain=False)
        counts = [int(value) for value in result["accepted_per_target"]]
        survivors = [index for index in survivors if counts[index] > 0]
        report["grasp_survivors"] = len(survivors)
    if bool(settings.get("pregrasp_enabled", False)):
        if survivors:
            result = client.solve_ik(pregrasp_targets[survivors], q_current, select_chain=False)
            counts = [int(value) for value in result["accepted_per_target"]]
            survivors = [survivors[local] for local, count in enumerate(counts) if count > 0]
        else:
            survivors = []
        report["pregrasp_survivors"] = len(survivors)
    return candidates, survivors, report


def init_registry(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "schema_version": 2,
        "purpose": "session-local nominal-size placement centres",
        "placements": [],
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def commit_placement(path: Path, *, cycle: int, execution: dict, selected: dict) -> None:
    registry = load_json(path)
    centre = np.asarray(execution["final_object_position_world_m"][:2], dtype=np.float64)
    registry.setdefault("placements", []).append({
        "cycle": int(cycle),
        "candidate_index": int(selected["candidate_index"]),
        "target_rank": int(selected["target_rank"]),
        "target_segmentation_id": int(execution["target_segmentation_id"]),
        "center_world_xy_m": centre.tolist(),
        "actual_final_object_position_world_m": execution["final_object_position_world_m"],
        "committed_local": datetime.now().isoformat(timespec="seconds"),
    })
    path.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _print_route_summary(report: dict) -> None:
    for row in report.get("stage_summaries", []):
        stage = str(row.get("stage", "")).upper()
        if not stage:
            continue
        if "target_count" in row:
            print(
                f"    {stage:<10} 目标={row.get('target_count')} | "
                f"可达目标={row.get('reachable_target_count', row.get('solution_count', '—'))} | "
                f"IK节点={row.get('node_count', row.get('beam_count', '—'))}"
            )


def main() -> int:
    global VERBOSE, DEBUG_LOG
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--scene-folder")
    parser.add_argument("--planning-only", action="store_true")
    parser.add_argument("--sim-execute", action="store_true")
    parser.add_argument("--no-planner-collision-check", action="store_true")
    parser.add_argument(
        "--diagnostic-ignore-static-gate", action="store_true",
        help="兼容旧命令；V2 persistent执行器不再使用旧static gate。",
    )
    parser.add_argument(
        "--diagnostic-full-first-batch",
        action="store_true",
        help="诊断模式：完整评估第一个retarget batch并输出漏斗统计；不执行Isaac动作。",
    )
    parser.add_argument(
        "--diagnostic-disable-rfs-esdf",
        action="store_true",
        help="DIAGNOSTIC ONLY: disable observed RGB-D/ESDF environment rejection inside RFS V2 support-state screening.",
    )
    parser.add_argument(
        "--diagnostic-disable-cover-esdf",
        action="store_true",
        help="DIAGNOSTIC ONLY: disable observed RGB-D/ESDF collision filtering for Exact COVER.",
    )
    parser.add_argument(
        "--diagnostic-disable-home-pre-esdf",
        action="store_true",
        help="DIAGNOSTIC ONLY: disable observed RGB-D/ROI ESDF collision filtering for HOME->PREGRASP only; self-collision remains enabled.",
    )
    parser.add_argument(
        "--diagnostic-disable-home-pre-self-collision",
        action="store_true",
        help="DIAGNOSTIC ONLY: disable self-collision filtering for HOME->PREGRASP only.",
    )
    parser.add_argument("--isaac-headless", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    VERBOSE = bool(args.verbose)
    if args.planning_only and args.sim_execute:
        raise ValueError("--planning-only and --sim-execute are mutually exclusive")
    if args.diagnostic_full_first_batch and args.sim_execute:
        raise ValueError("--diagnostic-full-first-batch is planning-only and cannot be combined with --sim-execute")
    if (
        args.diagnostic_disable_rfs_esdf
        or args.diagnostic_disable_cover_esdf
        or args.diagnostic_disable_home_pre_esdf
        or args.diagnostic_disable_home_pre_self_collision
    ) and not args.diagnostic_full_first_batch:
        raise RuntimeError(
            "--diagnostic-disable-rfs-esdf, --diagnostic-disable-cover-esdf, "
            "--diagnostic-disable-home-pre-esdf, and "
            "--diagnostic-disable-home-pre-self-collision "
            "are allowed only with --diagnostic-full-first-batch"
        )
    if (
        args.diagnostic_disable_rfs_esdf
        or args.diagnostic_disable_cover_esdf
        or args.diagnostic_disable_home_pre_esdf
        or args.diagnostic_disable_home_pre_self_collision
    ):
        print("\n==================================================")
        print("DIAGNOSTIC COLLISION BYPASS ACTIVE")
        print(
            "RFS observed ESDF       : "
            f"{'DISABLED' if args.diagnostic_disable_rfs_esdf else 'ENABLED'}"
        )
        print(
            "Exact COVER observed ESDF: "
            f"{'DISABLED' if args.diagnostic_disable_cover_esdf else 'ENABLED'}"
        )
        print(
            "HOME->PRE observed ESDF  : "
            f"{'DISABLED' if args.diagnostic_disable_home_pre_esdf else 'ENABLED'}"
        )
        print(
            "HOME->PRE self collision : "
            f"{'DISABLED' if args.diagnostic_disable_home_pre_self_collision else 'ENABLED'}"
        )
        print("Physical execution      : DISABLED")
        print("==================================================")
    if args.diagnostic_ignore_static_gate:
        print("⚠ --diagnostic-ignore-static-gate 在V2中仅为旧命令兼容参数，不再参与筛选。")

    root = args.project_root.expanduser().resolve()
    cfg = load_json(args.config)
    scene_folder = prompt_scene(root, args.scene_folder)
    scene_manifest = scene_folder / "scene_manifest.json"

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_root = resolve(root, cfg["session_root"]) / stamp
    session_root.mkdir(parents=True, exist_ok=False)
    DEBUG_LOG = session_root / "debug.log"
    registry = session_root / "placement_registry.json"
    init_registry(registry)
    (session_root / "session.json").write_text(json.dumps({
        "schema_version": 2,
        "created_local": stamp,
        "architecture": cfg.get("architecture"),
        "source_scene_folder": str(scene_folder),
        "sim_execute": bool(args.sim_execute),
        "planner_collision_checks_disabled": bool(args.no_planner_collision_check),
        "diagnostic_full_first_batch": bool(args.diagnostic_full_first_batch),
        "diagnostic_collision_bypass": {
            "rfs_observed_esdf": bool(args.diagnostic_disable_rfs_esdf),
            "exact_cover_observed_esdf": bool(args.diagnostic_disable_cover_esdf),
            "home_pre_observed_esdf": bool(args.diagnostic_disable_home_pre_esdf),
            "home_pre_self_collision": bool(args.diagnostic_disable_home_pre_self_collision),
        },
        "coarse_ik_prefilter": cfg["coarse_ik_prefilter"],
        "flexible_ik": cfg["flexible_ik"],
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    persistent_config = resolve(root, cfg["persistent_isaac_config"])
    network_py = Path(cfg["network_python"])
    retarget_py = Path(cfg["retarget_python"])
    planner_py = Path(cfg["planner_python"])
    for path in (persistent_config, network_py, retarget_py, planner_py):
        if not path.is_file():
            raise FileNotFoundError(path)

    worker_cfg = WorkerConfig(
        startup_timeout_s=float(cfg.get("worker_startup_timeout_s", 180.0)),
        request_timeout_s=float(cfg.get("worker_request_timeout_s", 600.0)),
    )
    T_world_base = world_from_base(root)
    T_base_from_world = np.linalg.inv(T_world_base)

    print("\n============================================================")
    print("  Wuji2 语义灵巧抓取闭环 V2")
    print("  ✓ Isaac Sim 持续会话：一次启动，全程保留物理场景")
    print("  ✓ cuRobo 按轮启动，规划完成后释放 GPU")
    print("  ✓ COVER 精确 IK；其余阶段采用 6D 可行域批量 IK")
    print("  ✓ 执行前不再重复 IK / FK")
    print("============================================================")

    try:
        with PersistentIsaacClient(
            project_root=root,
            scene_manifest=scene_manifest,
            runtime_config=persistent_config,
            startup_timeout_s=float(cfg.get("isaac_startup_timeout_s", 300.0)),
            request_timeout_s=float(cfg.get("isaac_request_timeout_s", 300.0)),
            headless=bool(args.isaac_headless),
            verbose=VERBOSE,
            log_callback=debug_write,
        ) as isaac:
            print("✓ Isaac 持续场景已连接")

            cycle = 0
            while True:
                cycle += 1
                cycle_started = time.perf_counter()
                cycle_root = session_root / f"cycle_{cycle:03d}"
                capture_root = cycle_root / "capture"
                scratch_root = cycle_root / "scratch/final_planning"
                cycle_root.mkdir(parents=True, exist_ok=False)
                scratch_root.mkdir(parents=True, exist_ok=True)

                print(f"\n================ 第 {cycle:03d} 轮 ================")
                capture = isaac.capture(capture_root)
                rgb = Path(capture["rgb"])
                settled = Path(capture["settled_scene_manifest"])
                robot_state_path = Path(capture["robot_state"])
                show_async(cfg.get("show_rgb_command"), rgb=str(rgb))
                print(
                    f"[1] ✓ RGB-D 拍照完成 | HOME静置={float(capture['hold_s']):.1f}s "
                    f"| 有效深度={100.0*float(capture['valid_depth_fraction']):.1f}%"
                )

                print("\n你要抓什么东西？（例如 dog / red cup；输入“抓取完成”结束）")
                print("规划过程中 Ctrl+C = 取消当前目标并重新选择；输入“抓取完成” = 结束会话")
                query = input("Target > ").strip()
                stop_words = {str(value).lower() for value in cfg.get("stop_words", [])}
                if query.lower() in stop_words:
                    final_snapshot = session_root / "final_scene_manifest.json"
                    isaac.snapshot(final_snapshot)
                    print(f"\n✓ 抓取会话完成，最终场景已保存：{final_snapshot}")
                    return 0
                if not query:
                    print("⚠ 输入为空，本轮不执行规划；场景保持不变。")
                    continue
                target_slug = safe_slug(query)

                # 2) GroundingDINO + SAM
                gs_root = capture_root / "grounded_sam" / target_slug
                backend = cfg.get("grounded_sam_backend")
                if not backend:
                    raise RuntimeError("grounded_sam_backend is not configured")
                command = [str(x).format(project_root=root, rgb=rgb, text=query, output=gs_root) for x in backend]
                started = time.perf_counter()
                print("[2] GroundingDINO + SAM ...")
                run("GroundingDINO(text + RGB) -> SAM", command, cwd=root)
                gs_check = run("validate Grounded-SAM output", [
                    network_py, SCRIPTS / "validate_grounded_sam_output.py",
                    "--rgb", rgb, "--output-root", gs_root, "--query", query,
                ], cwd=root, capture_json=True)
                overlay = Path(gs_check["overlay"])
                show_async(cfg.get("show_overlay_command"), overlay=str(overlay))
                gs_result = load_json(gs_root / "result.json")
                print(
                    f"    ✓ 识别完成 | score={gs_result.get('grounding_score', gs_result.get('score', 'NA'))} "
                    f"| mask={gs_result.get('mask_pixels', gs_result.get('mask_area_px', 'NA'))} "
                    f"| {time.perf_counter()-started:.1f}s"
                )

                # 3) RGB-D -> official 40k input
                dgn_root = capture_root / "dgn2" / target_slug
                print("[3] 构建 DGN2 40k 场景点云 ...")
                run("RGB-D -> full-scene 40k + target membership", [
                    network_py, root / "08_dual_arm_scene_layout/scripts/08_build_target_network_input.py",
                    "--target", target_slug,
                    "--target-segmentation-id", str(int(cfg["dgn2_target_membership_id"])),
                    "--capture-root", capture_root,
                    "--mask", gs_root / "mask.npy",
                ], cwd=root)
                net_meta = load_json(dgn_root / "network_input.json")
                print(f"    ✓ 40k输入完成 | target_points={net_meta.get('sampled_target_point_count', 'NA')}")

                # 4) DGN2
                print("[4] DGN2 生成抓取候选 ...")
                started = time.perf_counter()
                run("Official DGN2 LEAP inference", [
                    network_py, root / "08_dual_arm_scene_layout/scripts/09_predict_official_leap_target.py",
                    "--target", target_slug,
                    "--rounds", str(int(cfg["dgn2_rounds"])),
                    "--input-root", dgn_root,
                ], cwd=root)
                prediction = dgn_root / "official_leap_1024_target_ranked.npz"
                candidates_plain, total_proposals = candidate_order(prediction)
                print(
                    f"    ✓ proposals={total_proposals} | 目标候选={len(candidates_plain)} "
                    f"| {time.perf_counter()-started:.1f}s"
                )
                funnel = {
                    "schema_version": 1,
                    "query": query,
                    "diagnostic_full_first_batch": bool(args.diagnostic_full_first_batch),
                    "diagnostic_collision_bypass": {
                        "rfs_observed_esdf": bool(args.diagnostic_disable_rfs_esdf),
                        "exact_cover_observed_esdf": bool(args.diagnostic_disable_cover_esdf),
                        "home_pre_observed_esdf": bool(args.diagnostic_disable_home_pre_esdf),
                        "home_pre_self_collision": bool(args.diagnostic_disable_home_pre_self_collision),
                    },
                    "dgn2": {
                        "total_proposals": int(total_proposals),
                        "target_candidates": int(len(candidates_plain)),
                    },
                    "rfs_v2": {},
                    "retarget_batches": [],
                    "exact_cover": {"tested": 0, "pass": 0, "reject": 0, "pass_rate": 0.0},
                    "flexible_route": {
                        "candidate_reports": [],
                        "full_route_pass_count": 0,
                        "full_route_fail_count": 0,
                        "failure_stage_counts": {},
                    },
                }
                leap_overlay = generate_leap_candidate_overlay(
                    rgb_path=rgb,
                    mask_path=gs_root / "mask.npy",
                    prediction=prediction,
                    intrinsics_path=capture_root / "intrinsics.npy",
                    T_world_camera_path=capture_root / "T_world_camera.npy",
                    output_path=dgn_root / "leap_grasp_candidate_region_overlay.png",
                    query=query,
                )
                if leap_overlay is not None:
                    funnel["dgn2"]["leap_grasp_candidate_region_overlay"] = str(leap_overlay)
                    print(f"    ✓ LEAP候选区域图：{leap_overlay}")
                    show_image_path(leap_overlay, cfg)
                else:
                    print("    ⚠ LEAP候选区域图生成失败；pipeline继续，详见debug.log")
                write_planning_funnel(cycle_root, funnel)
                rfs_runtime = run_candidate_rfs_v2(
                    project_root=root,
                    cycle_root=cycle_root,
                    query=query,
                    candidates=candidates_plain,
                    settings=cfg.get("candidate_rfs_v2", {}),
                    diagnostic_disable_observed_esdf=bool(args.diagnostic_disable_rfs_esdf),
                )
                rfs_priority_indices = list(rfs_runtime.ordered_indices)
                rfs_stats = rfs_funnel_stats(rfs_runtime)
                rfs_opened = open_rfs_diagnostics(rfs_runtime, cfg)
                rfs_stats["opened_diagnostic_images"] = rfs_opened
                funnel["rfs_v2"] = rfs_stats
                print("\n[RFS V2 - 粗可达性/粗路径排序]")
                print(
                    f"    GRASP coarse IK={rfs_stats.get('grasp_coarse_ik_count', 'NA')}/"
                    f"{rfs_stats.get('candidate_count', len(candidates_plain))} | "
                    f"PREGRASP coarse IK={rfs_stats.get('pregrasp_coarse_ik_count', 'NA')}/"
                    f"{rfs_stats.get('candidate_count', len(candidates_plain))}"
                )
                print(
                    f"    mode={rfs_stats.get('mode')} status={rfs_stats.get('status')} | "
                    f"target reach={rfs_stats.get('target_reach_pass_count', 'NA')}/"
                    f"{rfs_stats.get('candidate_count', len(candidates_plain))}"
                )
                print(
                    f"    trajectory reach={rfs_stats.get('trajectory_space_pass_count', 'NA')}/"
                    f"{rfs_stats.get('candidate_count', len(candidates_plain))} | "
                    f"PASS={rfs_stats.get('pass_count')} | REJECT/rescue={rfs_stats.get('reject_count')}"
                )
                print(
                    f"    support IK reachable={rfs_stats.get('support_ik_reachable_count', 'NA')}/"
                    f"{rfs_stats.get('support_pose_count', 'NA')} | "
                    f"support admitted={rfs_stats.get('support_states_admitted_count', 'NA')}/"
                    f"{rfs_stats.get('support_pose_count', 'NA')} | "
                    f"branches PASS={rfs_stats.get('trajectory_branch_pass_count', 'NA')}/"
                    f"{rfs_stats.get('trajectory_branch_count', 'NA')}"
                )
                if rfs_stats.get("mode") == "priority_then_rescue":
                    print("    RFS REJECT is rescue tier; NOT a hard deletion")
                write_planning_funnel(cycle_root, funnel)

                # simulation-only binding after semantic selection
                sim_binding = cycle_root / "sim_target.json"
                bind = run("simulation-only mask -> rigid-body binding", [
                    network_py, SCRIPTS / "resolve_sim_target.py",
                    "--capture-root", capture_root,
                    "--mask", gs_root / "mask.npy",
                    "--settled-manifest", settled,
                    "--output", sim_binding,
                ], cwd=root, capture_json=True)
                sim_target_id = int(bind["segmentation_id"])
                q_current, measured = load_robot_state(robot_state_path)

                try:
                    home_gate_cfg = cfg.get("home_pregrasp_collision_gate", {})
                    home_gate_enabled = bool(home_gate_cfg.get("enabled", True))
                    need_observed_map = home_gate_enabled or not args.no_planner_collision_check
                    if need_observed_map:
                        roi_depth_path, roi_depth_meta = prepare_roi_depth_for_esdf(capture_root)
                        map_report = {
                            "status": "PER_BATCH",
                            "depth_path": str(roi_depth_path),
                            "roi_metadata": str(roi_depth_meta),
                            "home_pregrasp_collision_gate": home_gate_enabled,
                            "full_planner_collision_check": not bool(args.no_planner_collision_check),
                            "batch_maps": [],
                        }
                        print(
                            "[5] ✓ ROI ESDF输入已准备 | "
                            f"ROI={list(WORKSPACE_ROI_XYXY)} | depth shape/K/T保持不变"
                        )
                    else:
                        map_report = {
                            "status": "SKIPPED",
                            "reason": "all planner collision checks disabled",
                        }
                        print("[5] ✓ 规划器碰撞检查：全部关闭（Isaac/PhysX仍开启）")

                    # Optional legacy approximate prefilter; OFF by default.
                    coarse_cfg = cfg["coarse_ik_prefilter"]
                    if bool(coarse_cfg.get("grasp_enabled")) or bool(coarse_cfg.get("pregrasp_enabled")):
                        raise RuntimeError(
                            "legacy coarse_ik_prefilter requires a cycle-wide cuRobo worker and is disabled "
                            "for the final per-batch worker architecture"
                        )
                    else:
                        candidates = candidates_plain
                        survivor_indices = list(range(len(candidates)))
                        coarse_report = {
                            "enabled": False,
                            "grasp_enabled": False,
                            "pregrasp_enabled": False,
                            "survivors": len(survivor_indices),
                        }
                        print(
                            f"[6] ✓ 旧粗 GRASP/PREGRASP IK：关闭 | {len(candidates)} 个目标候选直接进入真实 Wuji2"
                        )

                    allowed_survivors = {int(index) for index in survivor_indices}
                    survivor_indices = [
                        int(index) for index in rfs_priority_indices
                        if int(index) in allowed_survivors
                    ]
                    coarse_report["candidate_rfs_v2"] = rfs_runtime.to_jsonable()
                    print(
                        f"[RFS V2] production order applied | "
                        f"ordered survivors={len(survivor_indices)} | "
                        f"status={rfs_runtime.status}"
                    )

                    max_to_test = int(cfg.get("max_candidates_to_test", 0))
                    if max_to_test > 0:
                        survivor_indices = survivor_indices[:max_to_test]
                    retarget_chunk_size = int(cfg.get("retarget_chunk_size", 64))
                    total_batches = math.ceil(len(survivor_indices) / retarget_chunk_size)
                    selected = None
                    tested_cover = 0
                    retargeted = 0
                    exact_cover_pass_total = 0
                    full_route_pass_count = 0
                    full_route_fail_count = 0
                    failure_stage_counts: Counter[str] = Counter()

                    print("[7] Wuji2 重定向 + 精确 COVER + Flexible IK 搜索")
                    for chunk_index, start in enumerate(range(0, len(survivor_indices), retarget_chunk_size), start=1):
                        local_indices = survivor_indices[start:start + retarget_chunk_size]
                        chunk_items = []
                        for local_index in local_indices:
                            item = candidates[local_index]
                            rank = int(item["target_rank"])
                            idx = int(item["candidate_index"])
                            case_id = f"{cfg.get('candidate_case_prefix','closedloop')}_r{rank:04d}_cand{idx:04d}"
                            case_root = scratch_root / f"rank_{rank:04d}" / case_id
                            chunk_items.append({
                                "local_target_index": int(local_index),
                                "target_rank": rank,
                                "candidate_index": idx,
                                "official_score": float(item.get("score", item.get("official_score", float('nan')))),
                                "case_id": case_id,
                                "case_root": str(case_root),
                            })
                        if not chunk_items:
                            continue
                        chunk_dir = scratch_root / f"batch_{chunk_index:03d}"
                        chunk_dir.mkdir(parents=True, exist_ok=True)
                        items_json = chunk_dir / "items.json"
                        items_json.write_text(json.dumps(chunk_items, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                        batch_started = time.perf_counter()
                        batch_funnel = {
                            "batch_index": int(chunk_index),
                            "total_batches": int(total_batches),
                            "rank_range": [
                                int(chunk_items[0]["target_rank"]),
                                int(chunk_items[-1]["target_rank"]),
                            ],
                            "input_candidates": int(len(chunk_items)),
                            "retargeted": 0,
                            "finalize_pass": 0,
                            "finalize_reject": 0,
                            "exact_cover_tested": 0,
                            "exact_cover_pass": 0,
                            "exact_cover_reject": 0,
                            "full_route_pass": 0,
                            "full_route_fail": 0,
                            "failure_stage_counts": {},
                        }
                        print(
                            f"\n[Batch {chunk_index:02d}/{total_batches:02d}] "
                            f"ranks {batch_funnel['rank_range'][0]}..{batch_funnel['rank_range'][1]} | "
                            f"input={len(chunk_items)}"
                        )
                        build_report = run("batch build candidate cases", [
                            network_py, SCRIPTS / "batch_build_candidate_cases.py",
                            "--project-root", root,
                            "--prediction", prediction,
                            "--network-input", dgn_root / "network_input.npz",
                            "--capture-root", capture_root,
                            "--settled-manifest", settled,
                            "--sim-target-segmentation-id", str(sim_target_id),
                            "--items-json", items_json,
                            "--output", chunk_dir / "batch_build_report.json",
                        ], cwd=root, capture_json=True)
                        retarget_report = run("batch LEAP->Wuji2 retarget", [
                            retarget_py, SCRIPTS / "batch_retarget_cases.py",
                            "--items-json", items_json,
                            "--output", chunk_dir / "batch_retarget_report.json",
                        ], cwd=root, capture_json=True)
                        finalize_report = run("batch finalize Wuji2 + arm targets", [
                            network_py, SCRIPTS / "batch_finalize_candidate_cases.py",
                            "--items-json", items_json,
                            "--output", chunk_dir / "batch_finalize_report.json",
                        ], cwd=root, capture_json=True)
                        retargeted += len(chunk_items)
                        batch_funnel["retargeted"] = int(len(chunk_items))
                        batch_funnel["build_time_s"] = float((build_report or {}).get("wall_time_s", 0.0))
                        batch_funnel["retarget_time_s"] = float((retarget_report or {}).get("wall_time_s", 0.0))
                        batch_funnel["finalize_time_s"] = float((finalize_report or {}).get("wall_time_s", 0.0))
                        finalize_results = json.loads(
                            (chunk_dir / "batch_finalize_report.json").read_text(encoding="utf-8")
                        ).get("results", [])
                        failures_jsonl = chunk_dir / "flexible_route_failures.jsonl"
                        item_by_case = {
                            str(Path(item["case_root"]).resolve()): item
                            for item in chunk_items
                        }
                        with failures_jsonl.open("a", encoding="utf-8") as stream:
                            for row in finalize_results:
                                if row.get("status") == "PASS":
                                    continue
                                failure_stage_counts["FINALIZE"] += 1
                                item = item_by_case.get(str(Path(row.get("case_root", "")).resolve()))
                                stream.write(json.dumps({
                                    "target_rank": None if item is None else int(item["target_rank"]),
                                    "candidate_index": None if item is None else int(item["candidate_index"]),
                                    "failure_stage": "FINALIZE",
                                    "failure_reason": row.get("reason", row.get("error", "finalize failed")),
                                    "stage_summaries": row,
                                }, ensure_ascii=False) + "\n")
                        finalized_case_roots = {
                            str(Path(row["case_root"]).resolve())
                            for row in finalize_results
                            if row.get("status") == "PASS"
                        }
                        finalized_items = [
                            item for item in chunk_items
                            if str(Path(item["case_root"]).resolve()) in finalized_case_roots
                        ]
                        finalize_reject_count = int(finalize_report.get("reject_count", len(chunk_items) - len(finalized_items)))
                        batch_funnel["finalize_pass"] = int(len(finalized_items))
                        batch_funnel["finalize_reject"] = int(finalize_reject_count)
                        if not finalized_items:
                            batch_funnel["wall_s"] = float(time.perf_counter() - batch_started)
                            batch_funnel["failure_stage_counts"] = dict(Counter({"FINALIZE": finalize_reject_count}))
                            funnel["retarget_batches"].append(batch_funnel)
                            funnel["flexible_route"]["failure_stage_counts"] = dict(failure_stage_counts)
                            write_planning_funnel(cycle_root, funnel)
                            print(
                                f"    Batch {chunk_index:02d}/{total_batches:02d} ✓ 重定向={len(chunk_items)} | "
                                f"finalize PASS=0 REJECT={finalize_reject_count} | "
                                f"{time.perf_counter()-batch_started:.1f}s"
                            )
                            if args.diagnostic_full_first_batch:
                                print("    [DIAGNOSTIC] first batch complete: no finalized candidates.")
                                break
                            continue

                        print(
                            f"    [cuRobo Batch {chunk_index:02d}/{total_batches:02d}] start | "
                            f"GPU {gpu_memory_snapshot()}"
                        )
                        with CuroboWorkerClient(
                            root,
                            worker_config=worker_cfg,
                            seeds=int(cfg.get("gpu_ik_seeds", 48)),
                            batch_size=int(cfg.get("gpu_ik_batch_size", 512)),
                        ) as curobo:
                            if need_observed_map:
                                map_started = time.perf_counter()
                                batch_map = curobo.build_map(
                                    roi_depth_path,
                                    capture_root / "intrinsics.npy",
                                    capture_root / "T_world_camera.npy",
                                    gs_root / "mask.npy",
                                )
                                batch_map.update({
                                    "batch_index": int(chunk_index),
                                    "workspace_roi_xyxy": list(WORKSPACE_ROI_XYXY),
                                    "home_pregrasp_collision_gate": home_gate_enabled,
                                    "full_planner_collision_check": not bool(args.no_planner_collision_check),
                                    "build_wall_s": time.perf_counter() - map_started,
                                })
                                map_report["batch_maps"].append(batch_map)
                                print(
                                    f"      map ROI build {batch_map['build_wall_s']:.2f}s | "
                                    f"GPU {gpu_memory_snapshot()}"
                                )
                            else:
                                print(f"      map skipped | GPU {gpu_memory_snapshot()}")

                            cover_rows = screen_exact_cover_batch(
                                client=curobo,
                                case_roots=[Path(item["case_root"]) for item in finalized_items],
                                q_current=q_current,
                                measured=measured,
                                T_base_from_world=T_base_from_world,
                                T_world_base=T_world_base,
                                no_planner_collision_check=bool(args.no_planner_collision_check),
                                block_unknown=bool(cfg.get("block_unknown_space", False)),
                                solutions_per_candidate=int(cfg["flexible_ik"]["selection"]["cover_solutions_per_candidate"]),
                                diagnostic_disable_cover_esdf=bool(args.diagnostic_disable_cover_esdf),
                            )
                            passed_cover = [row for row in cover_rows if row["pass"]]
                            exact_subfunnel = summarize_exact_cover_subfunnel(cover_rows)
                            print_exact_cover_subfunnel(exact_subfunnel)
                            tested_cover += len(cover_rows)
                            exact_cover_pass_total += len(passed_cover)
                            batch_funnel["exact_cover_tested"] = int(len(cover_rows))
                            batch_funnel["exact_cover_pass"] = int(len(passed_cover))
                            batch_funnel["exact_cover_reject"] = int(len(cover_rows) - len(passed_cover))
                            batch_funnel["exact_cover_subfunnel"] = exact_subfunnel
                            cover_by_case = {str(Path(row["case_root"]).resolve()): row for row in cover_rows}
                            for item in finalized_items:
                                row = cover_by_case.get(str(Path(item["case_root"]).resolve()))
                                if row is not None and row.get("pass"):
                                    continue
                                failure_stage_counts["EXACT_COVER"] += 1
                                with failures_jsonl.open("a", encoding="utf-8") as stream:
                                    stream.write(json.dumps({
                                        "target_rank": int(item["target_rank"]),
                                        "candidate_index": int(item["candidate_index"]),
                                        "failure_stage": "EXACT_COVER",
                                        "failure_reason": None if row is None else row.get("reason", "Exact COVER failed"),
                                        "stage_summaries": None if row is None else row,
                                    }, ensure_ascii=False) + "\n")
                            print(
                                f"    Batch {chunk_index:02d}/{total_batches:02d} ✓ 重定向={len(chunk_items)} | "
                                f"finalize PASS={len(finalized_items)} REJECT={finalize_reject_count} | "
                                f"精确COVER可达={len(passed_cover)} | {time.perf_counter()-batch_started:.1f}s"
                            )

                            # Preserve official DGN2 order inside the batch.
                            by_case = {str(Path(item["case_root"]).resolve()): item for item in finalized_items}
                            for cover_row in passed_cover:
                                item = by_case[cover_row["case_root"]]
                                route_started = time.perf_counter()
                                route = plan_flexible_route(
                                    client=curobo,
                                    project_root=root,
                                    case_root=Path(item["case_root"]),
                                    cover_solutions=cover_row["cover_solutions"],
                                    q_current=q_current,
                                    measured=measured,
                                    placement_registry=registry,
                                    config=cfg,
                                    no_planner_collision_check=bool(args.no_planner_collision_check),
                                    block_unknown=bool(cfg.get("block_unknown_space", False)),
                                    diagnostic_disable_home_pre_esdf=bool(args.diagnostic_disable_home_pre_esdf),
                                    diagnostic_disable_home_pre_self_collision=bool(args.diagnostic_disable_home_pre_self_collision),
                                )
                                route["diagnostic_wall_s"] = float(time.perf_counter() - route_started)
                                accumulate_home_pre_stats(funnel, route)
                                route_report = {
                                    "target_rank": int(item["target_rank"]),
                                    "candidate_index": int(item["candidate_index"]),
                                    "official_score": float(item["official_score"]),
                                    "status": str(route.get("status")),
                                    "failed_stage": None if route.get("status") == "PASS" else route_failure_stage(route),
                                    "reason": route.get("reason"),
                                    "wall_s": route["diagnostic_wall_s"],
                                    "stage_summaries": route.get("stage_summaries", []),
                                }
                                funnel["flexible_route"]["candidate_reports"].append(route_report)
                                print_route_diagnostics(item, route)
                                if route.get("status") == "PASS":
                                    full_route_pass_count += 1
                                    batch_funnel["full_route_pass"] += 1
                                    selected = {
                                        "target_rank": int(item["target_rank"]),
                                        "candidate_index": int(item["candidate_index"]),
                                        "official_score": float(item["official_score"]),
                                        "case_root": str(Path(item["case_root"]).resolve()),
                                        "route": route,
                                    }
                                    print(
                                        f"    ✓ Flexible Route PASS | rank={selected['target_rank']} "
                                        f"candidate={selected['candidate_index']} | {time.perf_counter()-route_started:.2f}s"
                                    )
                                    _print_route_summary(route)
                                    if not args.diagnostic_full_first_batch:
                                        break
                                    continue
                                full_route_fail_count += 1
                                batch_funnel["full_route_fail"] += 1
                                stage = route_failure_stage(route)
                                failure_stage_counts[stage] += 1
                                with failures_jsonl.open("a", encoding="utf-8") as stream:
                                    stream.write(json.dumps({
                                        "target_rank": int(item["target_rank"]),
                                        "candidate_index": int(item["candidate_index"]),
                                        "failure_stage": route.get("failed_stage", "FLEXIBLE_ROUTE"),
                                        "failure_reason": route.get("reason"),
                                        "stage_summaries": route.get("stage_summaries"),
                                    }, ensure_ascii=False) + "\n")
                                if VERBOSE:
                                    print(
                                        f"    ✗ route rank={item['target_rank']} cand={item['candidate_index']}: "
                                        f"{route.get('reason')}"
                                    )
                            batch_funnel["wall_s"] = float(time.perf_counter() - batch_started)
                            batch_funnel["failure_stage_counts"] = dict(failure_stage_counts)
                            funnel["retarget_batches"].append(batch_funnel)
                            funnel["exact_cover"] = {
                                "tested": int(tested_cover),
                                "pass": int(exact_cover_pass_total),
                                "reject": int(tested_cover - exact_cover_pass_total),
                                "pass_rate": float(exact_cover_pass_total / tested_cover) if tested_cover else 0.0,
                                "latest_batch_subfunnel": exact_subfunnel,
                            }
                            funnel["flexible_route"]["full_route_pass_count"] = int(full_route_pass_count)
                            funnel["flexible_route"]["full_route_fail_count"] = int(full_route_fail_count)
                            funnel["flexible_route"]["failure_stage_counts"] = dict(failure_stage_counts)
                            write_planning_funnel(cycle_root, funnel)
                        print(
                            f"    [cuRobo Batch {chunk_index:02d}/{total_batches:02d}] closed | "
                            f"GPU {gpu_memory_snapshot()}"
                        )
                        if args.diagnostic_full_first_batch:
                            print("    [DIAGNOSTIC] first batch fully evaluated; stopping before Isaac execution.")
                            break
                        if selected is not None:
                            break

                    planning_result = {
                        "schema_version": 2,
                        "status": (
                            "DIAGNOSTIC_FIRST_BATCH_COMPLETE"
                            if args.diagnostic_full_first_batch
                            else "PASS"
                            if selected is not None
                            else "FAIL"
                        ),
                        "architecture": cfg.get("architecture"),
                        "query": query,
                        "total_proposals": total_proposals,
                        "target_candidates": len(candidates),
                        "retargeted_candidate_count": retargeted,
                        "exact_cover_tested": tested_cover,
                        "coarse_prefilter": coarse_report,
                        "map": map_report,
                        "selected": selected,
                        "planning_funnel": str(cycle_root / "planning_funnel.json"),
                        "diagnostic_full_first_batch": bool(args.diagnostic_full_first_batch),
                        "planning_wall_s": time.perf_counter() - cycle_started,
                    }
                    planning_path = cycle_root / "planning_result.json"
                    planning_path.write_text(json.dumps(planning_result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                    funnel["planning_result"] = str(planning_path)
                    funnel["planning_wall_s"] = float(planning_result["planning_wall_s"])
                    funnel_path = write_planning_funnel(cycle_root, funnel)
                    print_funnel_summary(funnel)
                    print(f"planning_funnel.json = {funnel_path}")
                except KeyboardInterrupt:
                    planning_result = {
                        "schema_version": 2,
                        "status": "CANCELLED",
                        "architecture": cfg.get("architecture"),
                        "query": query,
                        "planning_wall_s": time.perf_counter() - cycle_started,
                    }
                    planning_path = cycle_root / "planning_result.json"
                    planning_path.write_text(json.dumps(planning_result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                    print(f"[CANCEL] 已取消当前目标：{query}")
                    print("✓ cuRobo 已释放")
                    print("✓ Isaac 会话继续保留，机械臂仍在 HOME")
                    continue

                if args.diagnostic_full_first_batch:
                    print("✓ diagnostic-full-first-batch 完成；未执行 Isaac 物理动作。")
                    return 0

                if selected is None:
                    print(f"\n✗ 本轮未找到完整可行路线；场景保持原样，可重新描述目标或继续尝试。")
                    print(f"  详细日志：{DEBUG_LOG}")
                    # Persistent Isaac is still paused at the captured state.
                    # The next cycle will only perform the configured 1 s HOME
                    # hold + fresh RGB-D capture; it will NOT reload the scene.
                    continue

                print("\n---------------- 规划结果 ----------------")
                print(f"✓ target rank : {selected['target_rank']}")
                print(f"✓ candidate   : {selected['candidate_index']}")
                print(f"✓ route plan  : {selected['route']['output_npz']}")
                print(f"✓ planning    : {planning_result['planning_wall_s']:.1f}s")
                print("------------------------------------------")

                if args.planning_only or not args.sim_execute:
                    print("✓ Planning-only 完成；未执行物理抓取。")
                    return 0

                print("[8] 同一 Isaac 场景直接执行（不重复加载，不二次 IK）")
                execution_root = cycle_root / "execution"
                execution = isaac.execute(
                    case_root=selected["case_root"],
                    plan_npz=selected["route"]["output_npz"],
                    output_dir=execution_root,
                    target_segmentation_id=sim_target_id,
                )
                execution_status = str(execution.get("status", ""))
                if execution_status == "RECOVERED_FAIL":
                    print("✗ 物理执行失败，但 runtime 已完成恢复；不提交 placement，进入下一轮。")
                    print(f"  failure_stage   : {execution.get('failure_stage')}")
                    print(f"  failure_type    : {execution.get('failure_type')}")
                    print(f"  failure_reason  : {execution.get('failure_reason')}")
                    print(f"  recovery_status : {execution.get('recovery_status')}")
                    print(f"  report          : {execution.get('report')}")
                    continue
                if execution_status != "PASS":
                    print(f"✗ 物理执行失败：{execution.get('report')}")
                    return 3
                commit_placement(registry, cycle=cycle, execution=execution, selected=selected)
                print(
                    f"✓ 物理执行 PASS | 抬升={float(execution['max_object_lift_mm']):.1f}mm "
                    f"| 放置中心在绿色区域={execution['final_object_center_inside_green_zone']}"
                )
                print("✓ 机械臂已回 HOME；下一轮拍照前仅静置 1.0s，场景不会重新加载。")
                print(f"✓ 本轮总耗时={time.perf_counter()-cycle_started:.1f}s")

    except KeyboardInterrupt:
        print("\n[STOP] 用户中断")
        return 130
    except Exception as exc:
        debug_write(traceback_text := f"{type(exc).__name__}: {exc}")
        print(f"\n✗ ERROR: {traceback_text}")
        print(f"详细日志：{DEBUG_LOG}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
