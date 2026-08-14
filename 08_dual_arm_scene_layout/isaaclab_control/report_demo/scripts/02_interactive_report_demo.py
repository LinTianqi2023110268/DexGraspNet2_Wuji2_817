#!/usr/bin/env python3
"""Terminal-driven report replay followed by an optional live Isaac Lab run."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEMO_ROOT = PROJECT_ROOT / "08_dual_arm_scene_layout/isaaclab_control/report_demo"
CAPTURE_ROOT = PROJECT_ROOT / "08_dual_arm_scene_layout/captures/live_dynamic_scene0000"
GENERATED = DEMO_ROOT / "assets/generated"
RUNNER = PROJECT_ROOT / "08_dual_arm_scene_layout/isaaclab_control/run_report_demo_20s_dog_candidate3800.sh"
VIEWER = DEMO_ROOT / "scripts/03_fixed_stage_viewer.py"
VIEWER_GEOMETRIES = (
    "510x390+10+20",
    "510x390+530+20",
    "510x390+10+430",
)


TARGETS = {
    "1": ("dog", "FULL PIPELINE READY; physically verified PASS"),
    "2": ("ashtray", "perception/network ready; no executable arm path"),
    "3": ("hammer", "perception ready; no target grasp proposal"),
}
ALIASES = {
    "dog": "1",
    "toy dog": "1",
    "狗": "1",
    "玩具狗": "1",
    "ashtray": "2",
    "烟灰缸": "2",
    "hammer": "3",
    "锤子": "3",
}


def wait(message: str) -> None:
    input(f"\n{message}  [Press Enter]")


class FixedStageWindow:
    """Keep three equal-sized report windows in fixed non-overlapping slots."""

    def __init__(self) -> None:
        self.processes: list[subprocess.Popen] = []

    def close(self) -> None:
        for process in self.processes:
            if process.poll() is not None:
                continue
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
        self.processes.clear()

    def show(self, title: str, *paths: Path) -> None:
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(path)
        slot = len(self.processes)
        if slot >= len(VIEWER_GEOMETRIES):
            raise RuntimeError("The fixed three-window report layout is already full")
        process = subprocess.Popen(
            [
                sys.executable,
                str(VIEWER),
                "--title",
                title,
                "--geometry",
                VIEWER_GEOMETRIES[slot],
                *[str(path) for path in paths],
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        self.processes.append(process)


def select_target() -> str:
    print("\nAvailable objects in current captured scene:")
    for key, (name, status) in TARGETS.items():
        print(f"  {key}. {name:8s} | {status}")
    answer = input("\nWhat object should be grasped? ").strip().lower()
    key = ALIASES.get(answer, answer)
    if key not in TARGETS:
        raise SystemExit(f"Unknown target: {answer!r}")
    return TARGETS[key][0]


def show_cached_pipeline(target: str, window: FixedStageWindow) -> bool:
    print("\n[CACHE REPLAY, NOT NEW INFERENCE]")
    print("Source: official test scene_0000, gravity-settled and recaptured from top camera")
    print(f"Target query: {target}")

    window.show(
        "Stage 1 - Top-camera RGB and depth",
        CAPTURE_ROOT / "rgb.png",
        CAPTURE_ROOT / "depth_preview.png",
    )
    wait("RGB and depth capture loaded")

    overlay = CAPTURE_ROOT / f"grounded_sam/{target}/overlay.png"
    window.show(f"Stage 2 - GroundingDINO + SAM: {target}", overlay)
    wait(f"GroundingDINO + SAM result loaded for {target}")

    if target == "hammer":
        print("[STOPPED AT NETWORK GATE] Target mask exists, but no target grasp proposal was produced.")
        return False

    if target == "ashtray":
        print("[STOPPED AT ARM GATE] Network candidates exist, but no full executable arm path is verified.")
        return False

    window.show("Stage 3 - Camera-view 40,000-point scene cloud", GENERATED / "04_network_point_cloud.png")
    wait("Camera-view scene cloud loaded; selected target points use a different color")

    print("\n[DGN2 RESULT]")
    print("  proposals                : 8192")
    print("  target-seed proposals     : 7688")
    print("  collision-valid           : 6614")
    print("  coarse arm-reachable      : 30")
    print("  exact IK + path-valid     : candidate 3800")
    print("  official target rank      : 7")
    print("  official score            : 43.876648")
    print("  selected log_prob         : 19.516697")
    print("  selected graspness        : 4.871991")
    print("  verified 20 s result      : PASS, 17.78 s, lift 123.20 mm")
    return True


def main() -> None:
    print("=" * 76)
    print("DexGraspNet 2.0 -> Wuji2 -> dual-arm interactive report demo")
    print("=" * 76)
    window = FixedStageWindow()
    target = select_target()
    if not show_cached_pipeline(target, window):
        window.close()
        return

    answer = input(
        "\nPress Enter to start the live Isaac Lab 20-second grasp "
        "(type n to cancel): "
    ).strip().lower()
    if answer in {"n", "no"}:
        print("Simulation not started. Report windows remain open for inspection.")
        print("Close the three image windows manually when finished.")
        return
    if answer not in {"", "y", "yes"}:
        print(f"Unrecognized answer {answer!r}; simulation was not started.")
        print("Report windows remain open for inspection.")
        return

    window.close()
    print("\n[START CONFIRMED] Closing report windows and launching Isaac Lab now.")
    print("Keep this terminal open: startup logs and live state telemetry appear here.")
    print("Timing contract: 17.78 s action; Kit startup and initial 3 s settle excluded.")
    os.execv("/bin/bash", ["bash", str(RUNNER)])


if __name__ == "__main__":
    main()
