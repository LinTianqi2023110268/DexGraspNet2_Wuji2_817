#!/usr/bin/env python3
"""Replay or export one recorded physical grasp.

This program never re-solves IK and never advances physics.  It writes the
recorded *actual* joint/object states.  Normal replay uses a monotonic wall
clock.  ``--export-mp4`` instead sends every recorded frame directly through
Isaac Sim's native viewport capture and H.264 encoder; it never records the
desktop or application UI.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REPLAY = (
    PROJECT_ROOT / "08_dual_arm_scene_layout/isaaclab_control/outputs/"
    "full_pick_place_25s_dog_candidate3800/physical_replay_30fps.npz"
)
DEFAULT_VIDEO = (
    PROJECT_ROOT / "08_dual_arm_scene_layout/isaaclab_control/outputs/"
    "full_pick_place_25s_dog_candidate3800/videos/"
    "full_pick_place_replay_24.85s.mp4"
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument("--ready-file", type=Path, default=None)
    parser.add_argument("--start-file", type=Path, default=None)
    parser.add_argument("--done-file", type=Path, default=None)
    parser.add_argument("--export-mp4", action="store_true")
    parser.add_argument("--video-output", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--video-width", type=int, default=1920)
    parser.add_argument("--video-height", type=int, default=1080)
    parser.add_argument("--video-fps", type=float, default=15.0)
    parser.add_argument("--video-frame-stride", type=int, default=2)
    parser.add_argument("--video-bitrate", type=int, default=16_777_216)
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


ARGS = parse_arguments()
APP_LAUNCHER = AppLauncher(ARGS)
SIMULATION_APP = APP_LAUNCHER.app


import torch  # noqa: E402
import isaacsim.core.utils.stage as stage_utils  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage  # noqa: E402
from isaacsim.core.utils.viewports import set_camera_view  # noqa: E402
from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402


def find_one_rigid_prim(prefix: str) -> Usd.Prim:
    stage = get_current_stage()
    matches = [
        prim for prim in stage.Traverse()
        if str(prim.GetPath()).startswith(prefix + "/") and prim.HasAPI(UsdPhysics.RigidBodyAPI)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one rigid body under {prefix}, got {[str(x.GetPath()) for x in matches]}")
    return matches[0]


def configure_view(metadata: dict) -> None:
    view = metadata.get("viewer_camera", {})
    stage = get_current_stage()
    for path in view.get("hide_prims", []):
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            UsdGeom.Imageable(prim).MakeInvisible()
    target = np.asarray(view.get("target_world_m", [0.0, -0.145, 0.50]), dtype=np.float64)
    yaw = math.radians(float(view.get("yaw_about_world_z_deg", -90.0)))
    distance = float(view.get("horizontal_distance_m", 1.45))
    eye = target + np.asarray([
        distance * math.cos(yaw), distance * math.sin(yaw),
        float(view.get("height_above_target_m", 0.75)),
    ])
    set_camera_view(eye=eye, target=target, camera_prim_path="/OmniverseKit_Persp")


def format_eta(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


class ExportProgressWindow:
    """Small non-modal UI; viewport-only capture will not include it."""

    def __init__(self, total_frames: int):
        import omni.ui as ui

        self._total_frames = total_frames
        self._window = ui.Window(
            "DGN2 MP4 Export Progress",
            width=560,
            height=155,
            visible=True,
            flags=ui.WINDOW_FLAGS_NO_COLLAPSE | ui.WINDOW_FLAGS_NO_RESIZE,
        )
        with self._window.frame:
            with ui.VStack(spacing=8):
                self._stage = ui.Label("Preparing native 3D viewport capture...", height=24)
                self._progress = ui.ProgressBar(height=24)
                self._detail = ui.Label(
                    f"0/{total_frames} | 0.0% | ETA --:--",
                    height=24,
                )

    def update_capture(self, completed: int, average: float, eta: float) -> None:
        percent = 100.0 * completed / self._total_frames
        self._stage.text = "Capturing 3D viewport frames (1920x1080)"
        self._progress.model.set_value(completed / self._total_frames)
        self._detail.text = (
            f"{completed}/{self._total_frames} | {percent:.1f}% | "
            f"{average:.3f} s/frame | ETA {format_eta(eta)}"
        )

    def set_encoding(self) -> None:
        self._stage.text = "Encoding H.264 MP4 with omni.videoencoding..."
        self._progress.model.set_value(1.0)
        self._detail.text = f"{self._total_frames}/{self._total_frames} frames captured"

    def set_complete(self, output: Path, size_mb: float) -> None:
        self._stage.text = "MP4 export complete"
        self._progress.model.set_value(1.0)
        self._detail.text = f"{output.name} | {size_mb:.2f} MB"


def export_viewport_mp4(show_frame, trajectory_frame_count: int) -> tuple[Path, int]:
    """Capture viewport PNGs and encode them with Kit's native H.264 plugin.

    This deliberately does not call ``CaptureExtension.start()``: that API
    owns the animation timeline, whereas this replay writes external recorded
    states through an Isaac Lab tensor view.  The lower-level official
    viewport capture and ``omni.videoencoding`` APIs do not reset the timeline.
    """
    import omni.kit.app
    import omni.kit.renderer_capture

    extension_manager = omni.kit.app.get_app().get_extension_manager()
    extension_manager.set_extension_enabled_immediate("omni.videoencoding", True)
    extension_manager.set_extension_enabled_immediate("omni.kit.capture.viewport", True)

    from omni.kit.capture.viewport.video_generation import VideoGenerationHelper
    import omni.kit.viewport.utility as viewport_utility
    import carb

    output = ARGS.video_output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.unlink(missing_ok=True)
    frames_dir = output.parent / f"{output.stem}_frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True)

    viewport = viewport_utility.get_active_viewport()
    if viewport is None:
        raise RuntimeError("Isaac Sim has no active viewport for native capture")

    if ARGS.video_frame_stride < 1:
        raise ValueError("--video-frame-stride must be >= 1")
    source_indices = list(range(0, trajectory_frame_count, ARGS.video_frame_stride))
    frame_count = len(source_indices)
    progress_window = ExportProgressWindow(frame_count)
    viewport.resolution = (ARGS.video_width, ARGS.video_height)
    viewport.resolution_scale = 1.0
    for _ in range(3):
        SIMULATION_APP.update()

    print(
        f"[NATIVE CAPTURE START] source={trajectory_frame_count} | "
        f"stride={ARGS.video_frame_stride} | output={frame_count} frames | "
        f"{ARGS.video_width}x{ARGS.video_height} | {ARGS.video_fps:.2f} FPS"
    )
    capture_start = time.monotonic()
    renderer_capture = omni.kit.renderer_capture.acquire_renderer_capture_interface()
    for output_index, source_index in enumerate(source_indices):
        show_frame(source_index)
        frame_path = frames_dir / f"{output.stem}.{output_index:04d}.png"
        viewport_utility.capture_viewport_to_file(viewport, file_path=str(frame_path))
        deadline = time.monotonic() + 30.0
        while not frame_path.is_file():
            if time.monotonic() > deadline:
                raise TimeoutError(f"Viewport PNG capture timed out: {frame_path}")
            SIMULATION_APP.update()
        renderer_capture.wait_async_capture()

        completed = output_index + 1
        elapsed = time.monotonic() - capture_start
        average = elapsed / completed
        eta = average * (frame_count - completed)
        progress_window.update_capture(completed, average, eta)
        if completed % 20 == 0 or completed == frame_count:
            percent = 100.0 * completed / frame_count
            print(
                f"[{completed}/{frame_count}] {percent:5.1f}% "
                f"ETA {format_eta(eta)} | {average:.3f} s/frame",
                flush=True,
            )

    settings = carb.settings.get_settings()
    settings.set("/exts/omni.videoencoding/bitrate", ARGS.video_bitrate)
    settings.set("/exts/omni.videoencoding/iframeinterval", 60)
    settings.set("/exts/omni.videoencoding/preset", "PRESET_DEFAULT")
    settings.set("/exts/omni.videoencoding/profile", "H264_PROFILE_HIGH")
    settings.set("/exts/omni.videoencoding/rcMode", "RC_VBR")
    settings.set("/exts/omni.videoencoding/rcTargetQuality", 0)
    settings.set("/exts/omni.videoencoding/videoFullRangeFlag", False)

    encoder = VideoGenerationHelper()
    if not encoder.generating_video(
        str(output), str(frames_dir), output.stem, ".####", 0,
        frame_count, ARGS.video_fps,
    ):
        raise RuntimeError("omni.videoencoding refused to start H.264 encoding")
    progress_window.set_encoding()
    print("[NATIVE ENCODING] H.264 MP4 encoding started", flush=True)
    deadline = time.monotonic() + 300.0
    while not encoder.encoding_done:
        if time.monotonic() > deadline:
            raise TimeoutError("Native H.264 encoding exceeded 300 seconds")
        SIMULATION_APP.update()

    if not output.is_file():
        raise RuntimeError(f"Native encoding completed but MP4 was not found: {output}")
    if output.stat().st_size == 0:
        raise RuntimeError(f"Native capture produced an empty MP4: {output}")
    size_mb = output.stat().st_size / 1e6
    progress_window.set_complete(output, size_mb)
    SIMULATION_APP.update()
    print(f"[NATIVE CAPTURE COMPLETE] {output} ({size_mb:.2f} MB)")
    # PNGs are only a transient input to the native encoder.  Keep them when
    # capture/encoding raises an exception for diagnosis, but remove them once
    # a non-empty MP4 has been confirmed so repeated exports stay tidy.
    shutil.rmtree(frames_dir)
    print(f"[INTERMEDIATE FRAMES REMOVED] {frames_dir}")
    return output, frame_count


def run() -> None:
    replay_path = ARGS.replay.resolve()
    if not replay_path.is_file():
        raise FileNotFoundError(
            f"Replay does not exist yet: {replay_path}\n"
            "Run the physical full-pipeline once with the recorder enabled first."
        )
    with np.load(replay_path, allow_pickle=False) as archive:
        time_s = np.asarray(archive["time_s"], dtype=np.float64)
        states = np.asarray(archive["state"]).astype(str)
        joint_q = np.asarray(archive["joint_position_rad"], dtype=np.float32)
        object_poses = np.asarray(archive["object_pose_world_wxyz"], dtype=np.float32)
        metadata = json.loads(str(np.asarray(archive["metadata_json"]).item()))
    if not (len(time_s) == len(states) == joint_q.shape[0] == object_poses.shape[0]):
        raise RuntimeError("Replay arrays have inconsistent frame counts")

    if not stage_utils.open_stage(metadata["stage"]):
        raise RuntimeError(f"Cannot open replay stage: {metadata['stage']}")
    stage = get_current_stage()
    duplicate = stage.GetPrimAtPath("/World/Layout/TableAssembly/TestScene0000")
    if duplicate.IsValid():
        stage.RemovePrim(duplicate.GetPath())
    UsdGeom.Xform.Define(stage, "/World/TaskObjects")

    object_wrappers: list[RigidObject] = []
    for record in metadata["objects"]:
        root = record["reference_root_path"]
        add_reference_to_stage(record["simulation_usd"], root)
        rigid = find_one_rigid_prim(root)
        object_wrappers.append(RigidObject(RigidObjectCfg(prim_path=str(rigid.GetPath()), spawn=None)))

    simulation = SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 120.0, device=ARGS.device))
    robot = Articulation(ArticulationCfg(
        prim_path=metadata["robot_prim"], spawn=None,
        actuators={"replay_only": ImplicitActuatorCfg(
            joint_names_expr=[".*"], stiffness=None, damping=None,
            effort_limit_sim=None, velocity_limit_sim=None,
        )},
    ))
    simulation.reset()
    configure_view(metadata)
    if list(robot.joint_names) != list(metadata["joint_names"]):
        raise RuntimeError("Replay robot joint order no longer matches the recorded order")

    zero_velocity = torch.zeros((1, robot.num_joints), device=robot.device)

    def show_frame(index: int) -> None:
        q = torch.as_tensor(joint_q[index], device=robot.device).reshape(1, -1)
        robot.write_joint_state_to_sim(q, zero_velocity)
        for object_index, obj in enumerate(object_wrappers):
            pose = torch.as_tensor(
                object_poses[index, object_index], device=obj.device
            ).reshape(1, 7)
            obj.write_root_pose_to_sim(pose)
        simulation.render()

    duration = float(time_s[-1])
    show_frame(0)
    if ARGS.export_mp4:
        output, encoded_frames = export_viewport_mp4(show_frame, len(time_s))
        print(
            f"[VIDEO SUMMARY] trajectory={duration:.2f}s | "
            f"encoded_frames={encoded_frames} | nominal_fps={ARGS.video_fps:.2f} | "
            f"video_duration={encoded_frames / ARGS.video_fps:.3f}s | {output}"
        )
        return
    if ARGS.ready_file is not None:
        ARGS.ready_file.resolve().parent.mkdir(parents=True, exist_ok=True)
        ARGS.ready_file.resolve().touch()
        print(f"[REPLAY READY FILE] {ARGS.ready_file.resolve()}")
    if ARGS.start_file is not None:
        start_file = ARGS.start_file.resolve()
        print(f"[REPLAY WAITING] create {start_file} to begin")
        while SIMULATION_APP.is_running() and not start_file.exists():
            # Do not call app.update() while the renderer is held at frame 1.
            # On this Isaac Sim 5.0 + 580.159.03 workstation that re-entrant
            # update path can crash in libGLX_nvidia.  A short passive wait is
            # sufficient because the first frame has already been rendered.
            time.sleep(0.02)
    print(f"[REPLAY READY] frames={len(time_s)}; recorded={metadata['record_fps']:.1f} FPS")
    print(f"[REPLAY START] wall-clock duration={duration:.2f} s; physics disabled")
    start = time.perf_counter()
    last_index = -1
    last_state = ""
    rendered = 0
    while SIMULATION_APP.is_running():
        elapsed = min(time.perf_counter() - start, duration)
        index = int(np.searchsorted(time_s, elapsed, side="right") - 1)
        index = max(0, min(index, len(time_s) - 1))
        if index != last_index:
            show_frame(index)
            rendered += 1
            last_index = index
            if states[index] != last_state:
                last_state = states[index]
                print(f"[{elapsed:6.2f}s] STATE: {last_state}")
            print(
                f"\rREPLAY {elapsed:6.2f}/{duration:.2f}s | frame {index + 1:4d}/{len(time_s)}",
                end="", flush=True,
            )
        else:
            time.sleep(0.001)
        if elapsed >= duration:
            break
    wall = time.perf_counter() - start
    print(f"\n[REPLAY COMPLETE] wall={wall:.2f}s; rendered={rendered}; skipped={len(time_s)-rendered}")
    if ARGS.done_file is not None:
        ARGS.done_file.resolve().parent.mkdir(parents=True, exist_ok=True)
        ARGS.done_file.resolve().touch()


def main() -> int:
    try:
        run()
        return 0
    except Exception as error:
        print(f"[REPLAY ERROR] {type(error).__name__}: {error}")
        return 1
    finally:
        SIMULATION_APP.close()


if __name__ == "__main__":
    raise SystemExit(main())
