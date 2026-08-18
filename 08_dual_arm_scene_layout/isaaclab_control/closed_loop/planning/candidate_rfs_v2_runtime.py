from __future__ import annotations

"""Production-facing wrapper for the candidate-centric RFS V2 pre-retarget filter.

This module intentionally contains no cuRobo/torch imports.  The production
orchestrator can import it from the Isaac-Lab Python environment; the actual
RFS V2 standalone backend is launched in the dedicated ``curobo_v2`` conda
environment and exits before the production cuRobo worker starts.

The first integration mode is deliberately conservative:
    priority_then_rescue

RFS-PASS candidates are tried first, preserving original DGN2 target-rank order.
If they are exhausted without a full route, original RFS-rejected candidates are
then appended in their original DGN2 order.  Therefore RFS V2 accelerates the
common case without replacing downstream exact COVER IK, Flexible Route, or
Isaac/PhysX verification, and without permanently deleting candidates during
this first production rollout.
"""

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import time
from typing import Iterable


@dataclass
class RfsV2ProductionResult:
    status: str
    ordered_indices: list[int]
    pass_indices: list[int]
    reject_indices: list[int]
    pass_count: int
    reject_count: int
    wall_time_s: float
    filter_json: str | None
    report_json: str | None
    output_dir: str | None
    mode: str
    fallback_reason: str | None = None

    def to_jsonable(self) -> dict:
        return {
            "status": self.status,
            "mode": self.mode,
            "pass_count": self.pass_count,
            "reject_count": self.reject_count,
            "wall_time_s": self.wall_time_s,
            "filter_json": self.filter_json,
            "report_json": self.report_json,
            "output_dir": self.output_dir,
            "fallback_reason": self.fallback_reason,
        }


def _resolve(project_root: Path, value: str | Path) -> Path:
    p = Path(value).expanduser()
    return p.resolve() if p.is_absolute() else (project_root / p).resolve()


def _tee_process(cmd: list[str], *, cwd: Path) -> tuple[int, str]:
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    lines: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    return proc.wait(), "".join(lines)


def _validate_and_order(
    candidates: list[dict],
    filter_payload: dict,
    mode: str,
) -> tuple[list[int], list[int], list[int]]:
    rows = filter_payload.get("rows")
    if not isinstance(rows, list):
        raise RuntimeError("RFS V2 filter JSON has no rows list")

    # Candidate list is production DGN2 target-rank order.  Validate every row
    # against that contract so a stale/mismatched filter can never silently
    # reorder the wrong grasp.
    by_rank: dict[int, dict] = {}
    for row in rows:
        rank = int(row["target_rank"])
        if rank in by_rank:
            raise RuntimeError(f"Duplicate target_rank in RFS filter: {rank}")
        by_rank[rank] = row

    if len(by_rank) != len(candidates):
        raise RuntimeError(
            f"RFS/candidate count mismatch: filter={len(by_rank)} "
            f"production={len(candidates)}"
        )

    pass_indices: list[int] = []
    reject_indices: list[int] = []
    for local_index, item in enumerate(candidates):
        rank = int(item["target_rank"])
        cand = int(item["candidate_index"])
        row = by_rank.get(rank)
        if row is None:
            raise RuntimeError(f"RFS filter missing target_rank={rank}")
        if int(row["candidate_index"]) != cand:
            raise RuntimeError(
                f"RFS candidate mismatch at rank={rank}: "
                f"filter={row['candidate_index']} production={cand}"
            )
        if str(row.get("status", "")).upper() == "PASS":
            pass_indices.append(local_index)
        else:
            reject_indices.append(local_index)

    if mode == "hard_filter":
        ordered = list(pass_indices)
    elif mode == "priority_then_rescue":
        ordered = list(pass_indices) + list(reject_indices)
    else:
        raise ValueError(f"Unsupported candidate_rfs_v2 mode: {mode}")

    return ordered, pass_indices, reject_indices


def run_candidate_rfs_v2(
    *,
    project_root: Path,
    cycle_root: Path,
    query: str,
    candidates: list[dict],
    settings: dict,
) -> RfsV2ProductionResult:
    """Run standalone candidate-centric RFS V2 and return production candidate order.

    On the first rollout, ``fallback_on_error`` should remain true and
    ``mode=priority_then_rescue``.  This keeps the original DGN2 order as a rescue
    tier if RFS fails or if all RFS-PASS candidates fail downstream.
    """
    project_root = Path(project_root).expanduser().resolve()
    cycle_root = Path(cycle_root).expanduser().resolve()
    all_indices = list(range(len(candidates)))

    enabled = bool(settings.get("enabled", True))
    mode = str(settings.get("mode", "priority_then_rescue"))
    fallback_on_error = bool(settings.get("fallback_on_error", True))
    minimum_pass = int(settings.get("minimum_pass_candidates", 1))

    if not enabled:
        return RfsV2ProductionResult(
            status="DISABLED",
            ordered_indices=all_indices,
            pass_indices=[],
            reject_indices=all_indices,
            pass_count=0,
            reject_count=len(all_indices),
            wall_time_s=0.0,
            filter_json=None,
            report_json=None,
            output_dir=None,
            mode=mode,
        )

    script = _resolve(
        project_root,
        settings.get(
            "script",
            "08_dual_arm_scene_layout/isaaclab_control/closed_loop/"
            "rfs_prototype/04_candidate_centric_rfs_v2.py",
        ),
    )
    bridge = _resolve(
        project_root,
        settings.get(
            "bridge_npz",
            "08_dual_arm_scene_layout/isaaclab_control/closed_loop/"
            "rfs_prototype/calibration_production/bridge_calibration_bottle512.npz",
        ),
    )
    conda_exe = Path(
        settings.get("conda_exe", "/home/lin/miniconda3/bin/conda")
    ).expanduser().resolve()
    conda_env = str(settings.get("conda_env", "curobo_v2"))
    output_dir = cycle_root / str(
        settings.get("output_subdir", "rfs_candidate_centric_v2_runtime")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    required = [script, bridge, conda_exe]
    for path in required:
        if not path.is_file():
            reason = f"required RFS V2 file missing: {path}"
            if fallback_on_error:
                print(f"[RFS V2 FALLBACK] {reason}")
                return RfsV2ProductionResult(
                    status="FALLBACK",
                    ordered_indices=all_indices,
                    pass_indices=[],
                    reject_indices=all_indices,
                    pass_count=0,
                    reject_count=len(all_indices),
                    wall_time_s=0.0,
                    filter_json=None,
                    report_json=None,
                    output_dir=str(output_dir),
                    mode=mode,
                    fallback_reason=reason,
                )
            raise FileNotFoundError(path)

    cmd = [
        str(conda_exe),
        "run",
        "--no-capture-output",
        "-n",
        conda_env,
        "python",
        str(script),
        "--project-root",
        str(project_root),
        "--cycle-root",
        str(cycle_root),
        "--query",
        str(query),
        "--bridge-npz",
        str(bridge),
        "--output-dir",
        str(output_dir),
    ]

    print(
        f"[RFS V2] candidate-centric pre-retarget filter | "
        f"candidates={len(candidates)} | mode={mode}"
    )
    started = time.perf_counter()
    code, output = _tee_process(cmd, cwd=project_root)
    wall = time.perf_counter() - started

    filter_path = output_dir / "candidate_centric_rfs_v2_filter.json"
    report_path = output_dir / "candidate_centric_rfs_v2_report.json"

    if code != 0:
        reason = (
            f"RFS V2 backend exited with code {code}; "
            f"tail={' | '.join(output.splitlines()[-8:])}"
        )
        if fallback_on_error:
            print(f"[RFS V2 FALLBACK] {reason}")
            return RfsV2ProductionResult(
                status="FALLBACK",
                ordered_indices=all_indices,
                pass_indices=[],
                reject_indices=all_indices,
                pass_count=0,
                reject_count=len(all_indices),
                wall_time_s=wall,
                filter_json=str(filter_path) if filter_path.exists() else None,
                report_json=str(report_path) if report_path.exists() else None,
                output_dir=str(output_dir),
                mode=mode,
                fallback_reason=reason,
            )
        raise RuntimeError(reason)

    if not filter_path.is_file():
        reason = f"RFS V2 backend returned success but filter is missing: {filter_path}"
        if fallback_on_error:
            print(f"[RFS V2 FALLBACK] {reason}")
            return RfsV2ProductionResult(
                status="FALLBACK",
                ordered_indices=all_indices,
                pass_indices=[],
                reject_indices=all_indices,
                pass_count=0,
                reject_count=len(all_indices),
                wall_time_s=wall,
                filter_json=None,
                report_json=str(report_path) if report_path.exists() else None,
                output_dir=str(output_dir),
                mode=mode,
                fallback_reason=reason,
            )
        raise RuntimeError(reason)

    try:
        payload = json.loads(filter_path.read_text(encoding="utf-8"))
        ordered, passed, rejected = _validate_and_order(candidates, payload, mode)
        if len(passed) < minimum_pass:
            raise RuntimeError(
                f"RFS V2 returned only {len(passed)} PASS candidates; "
                f"minimum_pass_candidates={minimum_pass}"
            )
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        if fallback_on_error:
            print(f"[RFS V2 FALLBACK] filter validation failed: {reason}")
            return RfsV2ProductionResult(
                status="FALLBACK",
                ordered_indices=all_indices,
                pass_indices=[],
                reject_indices=all_indices,
                pass_count=0,
                reject_count=len(all_indices),
                wall_time_s=wall,
                filter_json=str(filter_path),
                report_json=str(report_path) if report_path.exists() else None,
                output_dir=str(output_dir),
                mode=mode,
                fallback_reason=reason,
            )
        raise

    print(
        f"[RFS V2] PASS={len(passed)}/{len(candidates)} | "
        f"REJECT={len(rejected)} | fast tier first | wall={wall:.1f}s"
    )
    if mode == "priority_then_rescue":
        print(
            "[RFS V2] rescue tier enabled: if all RFS-PASS candidates fail "
            "downstream, original rejected candidates remain available."
        )

    return RfsV2ProductionResult(
        status="PASS",
        ordered_indices=ordered,
        pass_indices=passed,
        reject_indices=rejected,
        pass_count=len(passed),
        reject_count=len(rejected),
        wall_time_s=wall,
        filter_json=str(filter_path),
        report_json=str(report_path) if report_path.exists() else None,
        output_dir=str(output_dir),
        mode=mode,
    )
