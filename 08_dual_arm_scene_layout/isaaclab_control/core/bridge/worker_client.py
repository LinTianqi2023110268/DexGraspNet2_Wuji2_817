from __future__ import annotations

from pathlib import Path
import json
import os
import queue
import subprocess
import threading
import time


def _jsonable(value):
    try:
        import numpy as np
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.floating, np.integer, np.bool_)):
            return value.item()
    except Exception:
        pass
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value

from ..config import WorkerConfig

_PROTOCOL = "__CUROBO_CORE__"


def _worker_subprocess_env(parent: dict[str, str] | None = None) -> dict[str, str]:
    """Isolate the cuRobo worker from Isaac Sim/Kit Python and shared libraries."""
    env = dict(os.environ if parent is None else parent)
    for name in (
        "PYTHONPATH",
        "PYTHONHOME",
        "LD_LIBRARY_PATH",
        "ISAAC_PATH",
        "ISAACLAB_PATH",
        "CARB_APP_PATH",
        "EXP_PATH",
        "OV_PACKAGE_ROOT",
    ):
        env.pop(name, None)
    # Base-conda plugins can otherwise import pydantic from Kit's pip archive
    # before `conda run` activates curobo_v2.
    env["CONDA_NO_PLUGINS"] = "true"
    return env


class CuroboWorkerClient:
    """Persistent cross-conda cuRobo worker client.

    Designed to be imported by ``isaaclab22_sim50`` without importing torch/curobo.
    The worker itself runs in ``curobo_v2`` and stays alive, so cuRobo initialization
    and CUDA warm-up are paid once rather than once per IK request.
    """

    def __init__(
        self,
        project_root: Path | str,
        worker_config: WorkerConfig | None = None,
        device: str = "cuda:0",
        seeds: int = 48,
        batch_size: int = 64,
    ):
        self.project_root = Path(project_root).expanduser().resolve()
        self.cfg = worker_config or WorkerConfig()
        conda_exe = self.cfg.conda_exe or os.environ.get("CUROBO_CONDA_EXE")
        if not conda_exe:
            conda_exe = str(Path.home() / "miniconda3/bin/conda")
        self.conda_exe = Path(conda_exe).expanduser()
        if not self.conda_exe.is_file():
            raise FileNotFoundError(
                f"conda executable not found: {self.conda_exe}; set CUROBO_CONDA_EXE"
            )
        self.worker_path = (
            self.project_root
            / "08_dual_arm_scene_layout/isaaclab_control/core/bridge/curobo_worker.py"
        )
        if not self.worker_path.is_file():
            raise FileNotFoundError(self.worker_path)
        cmd = [
            str(self.conda_exe), "run", "--no-capture-output", "-n", self.cfg.conda_env,
            "python", "-u", str(self.worker_path),
            "--project-root", str(self.project_root),
            "--device", device,
            "--seeds", str(int(seeds)),
            "--batch-size", str(int(batch_size)),
            "--stdio",
        ]
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=_worker_subprocess_env(),
        )
        self._responses: queue.Queue[dict] = queue.Queue()
        self._stdout_thread = threading.Thread(target=self._stdout_loop, daemon=True)
        self._stderr_thread = threading.Thread(target=self._stderr_loop, daemon=True)
        self._stdout_thread.start()
        self._stderr_thread.start()
        self.request({"op": "ping"}, timeout=self.cfg.startup_timeout_s)

    def _stdout_loop(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            line = line.rstrip("\n")
            if line.startswith(_PROTOCOL):
                try:
                    self._responses.put(json.loads(line[len(_PROTOCOL):]))
                except Exception as exc:
                    self._responses.put({"ok": False, "error": f"protocol decode: {exc}"})
            elif line:
                print(f"[curobo-worker] {line}")

    def _stderr_loop(self) -> None:
        assert self.proc.stderr is not None
        for line in self.proc.stderr:
            line = line.rstrip("\n")
            if line:
                print(f"[curobo-worker:stderr] {line}")

    def request(self, payload: dict, timeout: float | None = None) -> dict:
        proc = getattr(self, "proc", None)
        if proc is None:
            raise RuntimeError("cuRobo worker is closed")
        if proc.poll() is not None:
            raise RuntimeError(f"cuRobo worker exited with code {proc.returncode}")
        assert proc.stdin is not None
        proc.stdin.write(json.dumps(_jsonable(payload), separators=(",", ":")) + "\n")
        proc.stdin.flush()
        timeout = self.cfg.request_timeout_s if timeout is None else timeout
        try:
            response = self._responses.get(timeout=timeout)
        except queue.Empty as exc:
            raise TimeoutError(f"cuRobo worker request timed out after {timeout}s") from exc
        if not response.get("ok", False):
            raise RuntimeError(response.get("error", "unknown cuRobo worker error"))
        return response

    def solve_ik(
        self,
        target_matrices_base,
        q_reference_rad,
        select_chain: bool = True,
        collision_context: dict | None = None,
        acceptance_policy: dict | None = None,
    ) -> dict:
        payload = {
            "op": "solve_ik",
            "targets": target_matrices_base,
            "q_reference_rad": q_reference_rad,
            "select_chain": bool(select_chain),
        }
        if collision_context is not None:
            payload["collision_context"] = collision_context
        if acceptance_policy is not None:
            payload["acceptance_policy"] = acceptance_policy
        return self.request(payload)

    def solve_ik_groups(
        self,
        target_matrices_base,
        q_reference_rad,
        group_sizes,
        select_chain: bool = True,
        collision_context: dict | None = None,
        acceptance_policy: dict | None = None,
    ) -> dict:
        """Solve multiple ordered waypoint groups in one worker/GPU request.

        ``target_matrices_base`` is flattened as [sum(group_sizes),4,4].  The
        worker runs one batched cuRobo solve over all poses, then performs
        continuity-based branch selection independently inside each group.  This
        is the production screening primitive for candidate chunks.
        """
        payload = {
            "op": "solve_ik_groups",
            "targets": target_matrices_base,
            "q_reference_rad": q_reference_rad,
            "group_sizes": [int(x) for x in group_sizes],
            "select_chain": bool(select_chain),
        }
        if collision_context is not None:
            payload["collision_context"] = collision_context
        if acceptance_policy is not None:
            payload["acceptance_policy"] = acceptance_policy
        return self.request(payload)

    def check_self_collision(self, joint_positions_by_name: dict) -> dict:
        return self.request({
            "op": "check_self_collision",
            "joint_positions_by_name": joint_positions_by_name,
        })

    def check_joint_path(
        self,
        q_nodes_rad,
        joint_positions_by_name: dict,
        *,
        joint_positions_by_node=None,
        T_world_base=None,
        phases=None,
        margin_m: float = 0.0,
        path_max_joint_step_rad=None,
        check_observed_map: bool = False,
        check_self_collision: bool = False,
    ) -> dict:
        payload = {
            "op": "check_joint_path",
            "q_nodes_rad": q_nodes_rad,
            "joint_positions_by_name": joint_positions_by_name,
            "joint_positions_by_node": joint_positions_by_node,
            "T_world_base": T_world_base,
            "phases": phases,
            "margin_m": float(margin_m),
            "check_observed_map": bool(check_observed_map),
            "check_self_collision": bool(check_self_collision),
        }
        if path_max_joint_step_rad is not None:
            payload["path_max_joint_step_rad"] = float(path_max_joint_step_rad)
        return self.request(payload)

    def diagnose_ik_collisions(
        self,
        target_matrices_base,
        q_reference_rad,
        collision_context: dict,
        top_k: int = 5,
    ) -> dict:
        return self.request({
            "op": "diagnose_ik_collisions",
            "targets": target_matrices_base,
            "q_reference_rad": q_reference_rad,
            "collision_context": collision_context,
            "top_k": int(top_k),
        })

    def coarse_prefilter(
        self,
        target_matrices_base,
        q_reference_rad,
        *,
        joint_positions_by_name: dict,
        T_world_base,
        phase: str = "pregrasp",
        margin_m: float = 0.0,
        arm_link_prefixes=None,
    ) -> dict:
        return self.request({
            "op": "coarse_prefilter",
            "targets": target_matrices_base,
            "q_reference_rad": q_reference_rad,
            "joint_positions_by_name": joint_positions_by_name,
            "T_world_base": T_world_base,
            "phase": phase,
            "margin_m": float(margin_m),
            "arm_link_prefixes": arm_link_prefixes,
        })

    def coarse_approach_prefilter(
        self,
        pregrasp_matrices_base,
        grasp_matrices_base,
        q_reference_rad,
        *,
        joint_positions_by_name: dict,
        T_world_base,
        margin_m: float = 0.0,
        path_max_joint_step_rad=None,
        arm_link_prefixes=None,
    ) -> dict:
        payload = {
            "op": "coarse_approach_prefilter",
            "pregrasp_targets": pregrasp_matrices_base,
            "grasp_targets": grasp_matrices_base,
            "q_reference_rad": q_reference_rad,
            "joint_positions_by_name": joint_positions_by_name,
            "T_world_base": T_world_base,
            "margin_m": float(margin_m),
            "arm_link_prefixes": arm_link_prefixes,
        }
        if path_max_joint_step_rad is not None:
            payload["path_max_joint_step_rad"] = float(path_max_joint_step_rad)
        return self.request(payload)

    def build_map(self, depth_path, intrinsics_path, T_world_camera_path, target_mask_path=None) -> dict:
        return self.request({
            "op": "build_map",
            "depth_path": str(depth_path),
            "intrinsics_path": str(intrinsics_path),
            "T_world_camera_path": str(T_world_camera_path),
            "target_mask_path": None if target_mask_path is None else str(target_mask_path),
        })

    def query_spheres(self, centers_world, radii_m, phase: str, margin_m: float = 0.0) -> dict:
        return self.request({
            "op": "query_spheres",
            "centers_world": centers_world,
            "radii_m": radii_m,
            "phase": phase,
            "margin_m": float(margin_m),
        })

    def robot_spheres(self, joint_positions_by_name: dict, T_world_base=None) -> dict:
        return self.request({
            "op": "robot_spheres",
            "joint_positions_by_name": joint_positions_by_name,
            "T_world_base": T_world_base,
        })

    def check_robot_state(
        self,
        joint_positions_by_name: dict,
        T_world_base,
        phase: str,
        margin_m: float = 0.0,
    ) -> dict:
        """Check cuRobo robot collision spheres against the current observed map."""
        return self.request({
            "op": "check_robot_state",
            "joint_positions_by_name": joint_positions_by_name,
            "T_world_base": T_world_base,
            "phase": phase,
            "margin_m": float(margin_m),
        })

    def close(self) -> None:
        proc = getattr(self, "proc", None)
        if proc is None:
            return
        if proc.poll() is None:
            try:
                self.request({"op": "shutdown"}, timeout=5.0)
            except Exception:
                pass
            try:
                proc.terminate()
                proc.wait(timeout=5.0)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        self.proc = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
