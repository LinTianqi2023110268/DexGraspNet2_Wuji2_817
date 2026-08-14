"""Isaac Lab command-line smoke test for one deterministic scene-0000 RGB-D capture.

This stage deliberately stops before GroundingDINO, SAM, grasp inference or arm
motion.  It reuses the already-reviewed Isaac Sim scene-import and camera-capture
implementations, then validates their files as one command-line pipeline step.
"""

from __future__ import annotations

import argparse
import json
import os
import runpy
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[3]
LAYOUT_ROOT = PROJECT_ROOT / "08_dual_arm_scene_layout"
DEFAULT_STAGE = LAYOUT_ROOT / "scenes/manual_layout_calibrated_mass_fixed.usda"
DEFAULT_OUTPUT = LAYOUT_ROOT / "captures/isaaclab_scene0000_smoke"
IMPORT_SCRIPT = LAYOUT_ROOT / "scripts/06b_import_test_scene0000_into_source_zone.py"
CAPTURE_SCRIPT = LAYOUT_ROOT / "scripts/07_capture_single_rgbd.py"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, default=DEFAULT_STAGE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--scene-manifest", type=Path, default=None,
        help="Optional settled scene manifest; defaults to the original test scene.",
    )
    parser.add_argument("--timeout-s", type=float, default=120.0)
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


ARGS = parse_arguments()
APP_LAUNCHER = AppLauncher(ARGS)
SIMULATION_APP = APP_LAUNCHER.app


import isaacsim.core.utils.stage as stage_utils  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sensors.camera import Camera, CameraCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402


def validate_capture(output: Path) -> dict:
    required = (
        "rgb.png",
        "depth_m.npy",
        "depth_preview.png",
        "intrinsics.npy",
        "T_world_camera.npy",
        "capture_manifest.json",
        "scene_snapshot.json",
    )
    missing = [name for name in required if not (output / name).is_file()]
    if missing:
        raise RuntimeError(f"RGB-D capture is incomplete; missing: {missing}")

    manifest = json.loads((output / "capture_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "single_rgbd_capture_complete":
        raise RuntimeError(f"Unexpected capture status: {manifest.get('status')}")
    if manifest.get("test_scene", {}).get("object_count") != 6:
        raise RuntimeError("Capture snapshot does not contain all six scene-0000 objects")
    if not manifest.get("grounded_sam_compatible", False):
        raise RuntimeError("Capture did not declare the GroundedSAM data contract")
    return manifest


def main() -> Path:
    stage_path = ARGS.stage.resolve()
    output = ARGS.output.resolve()
    if not stage_path.is_file():
        raise FileNotFoundError(stage_path)
    if not IMPORT_SCRIPT.is_file() or not CAPTURE_SCRIPT.is_file():
        raise FileNotFoundError("Scene import or RGB-D capture implementation is missing")

    print("[STAGE 00] Opening calibrated mass-fixed scene")
    if not stage_utils.open_stage(str(stage_path)):
        raise RuntimeError(f"Could not open stage: {stage_path}")
    for _ in range(4):
        SIMULATION_APP.update()

    print("[STAGE 01] Importing deterministic test scene 0000 into SourceZone")
    if ARGS.scene_manifest is not None:
        scene_manifest = ARGS.scene_manifest.resolve()
        if not scene_manifest.is_file():
            raise FileNotFoundError(scene_manifest)
        os.environ["DGN2_SCENE_MANIFEST"] = str(scene_manifest)
        os.environ["DGN2_SCENE_IMPORT_REPORT"] = str(output / "scene_import_report.json")
    try:
        runpy.run_path(str(IMPORT_SCRIPT), run_name="__main__")
    except BaseException:
        print("[STAGE 01 ERROR] static reconstruction failed", flush=True)
        traceback.print_exc()
        raise
    print("[STAGE 01 PASS] settled scene reconstructed for camera capture", flush=True)
    for _ in range(4):
        SIMULATION_APP.update()

    print("[STAGE 02] Initializing the official Isaac Lab Camera sensor")
    os.environ["DGN2_RGBD_OUTPUT_DIR"] = str(output)
    os.environ["DGN2_RGBD_LIBRARY_MODE"] = "1"
    capture_module = runpy.run_path(str(CAPTURE_SCRIPT), run_name="dgn2_rgbd_capture_module")
    stage = capture_module["omni"].usd.get_context().get_stage()
    capture_module["validate_test_scene"](stage)
    capture_module["hide_debug_visuals"](stage)
    capture_module["carb"].settings.get_settings().set_bool("/app/viewport/show/camera", False)
    intrinsic, world_from_camera, camera_model = capture_module["camera_calibration"](stage)
    np = capture_module["np"]
    Image = capture_module["Image"]
    camera_path = capture_module["CAMERA_PATH"]
    width, height = capture_module["WIDTH"], capture_module["HEIGHT"]
    simulation = SimulationContext(
        sim_utils.SimulationCfg(dt=1.0 / 120.0, render_interval=1, device=ARGS.device)
    )
    camera = Camera(
        CameraCfg(
            prim_path=camera_path,
            update_period=0.0,
            width=width,
            height=height,
            data_types=["rgb", "distance_to_image_plane"],
            update_latest_camera_pose=False,
            spawn=None,
        )
    )
    simulation.reset()

    print("[STAGE 03] Rendering and reading one aligned RGB-D frame")
    # The first RTX frames initialize render products and annotator buffers.
    for _ in range(12):
        simulation.step(render=True)
        camera.update(dt=simulation.get_physics_dt(), force_recompute=True)

    rgb = camera.data.output["rgb"][0].detach().cpu().numpy()[..., :3].astype(np.uint8, copy=False)
    depth = (
        camera.data.output["distance_to_image_plane"][0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    depth = np.squeeze(depth)
    if rgb.shape != (height, width, 3) or depth.shape != (height, width):
        raise RuntimeError(f"Unexpected RGB/depth shapes: {rgb.shape}, {depth.shape}")

    output.mkdir(parents=True, exist_ok=True)
    preview, depth_stats = capture_module["make_depth_preview"](depth)
    Image.fromarray(rgb, mode="RGB").save(output / "rgb.png")
    Image.fromarray(preview, mode="L").save(output / "depth_preview.png")
    np.save(output / "depth_m.npy", depth)
    np.save(output / "intrinsics.npy", intrinsic)
    np.save(output / "T_world_camera.npy", world_from_camera)

    capture_module["OUTPUT_DIR"] = output
    snapshot = capture_module["write_scene_snapshot"](stage)
    manifest = {
        "schema_version": 1,
        "status": "single_rgbd_capture_complete",
        "capture_backend": "Isaac Lab 2.2 CameraCfg/Camera",
        "camera_prim": camera_path,
        "resolution_wh": [width, height],
        "rgb": {"file": "rgb.png", "dtype": "uint8", "shape": list(rgb.shape)},
        "depth": {
            "file": "depth_m.npy",
            "dtype": "float32",
            "shape": list(depth.shape),
            "meaning": "distance_to_image_plane in metres",
            **depth_stats,
        },
        "intrinsics": {"file": "intrinsics.npy", "K": intrinsic.tolist()},
        "extrinsics": {
            "file": "T_world_camera.npy",
            "matrix": world_from_camera.tolist(),
            "convention": "OpenCV camera-to-world: +x right, +y down, +z forward",
        },
        "camera_model": camera_model,
        "scene_snapshot": "scene_snapshot.json",
        "stage_identifier": snapshot["stage_identifier"],
        "grounded_sam_compatible": True,
        "test_scene": snapshot["test_scene"],
    }
    if ARGS.scene_manifest is not None:
        # Keep the perception products bound to the exact post-settle object
        # poses.  Downstream point-cloud cropping, collision filtering and
        # grasp-case freezing must never silently fall back to the original
        # pre-physics dataset manifest.
        manifest["settled_scene_manifest"] = str(ARGS.scene_manifest.resolve())
    (output / "capture_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    manifest = validate_capture(output)
    print("\n[ISAAC LAB RGBD SMOKE PASS]")
    print("output:", output)
    print("objects:", manifest["test_scene"]["object_count"])
    print("rgb shape:", manifest["rgb"]["shape"])
    print("depth shape:", manifest["depth"]["shape"])
    print("valid depth fraction:", round(manifest["depth"]["valid_fraction"], 6))
    print("GroundedSAM interface: PASS")
    return output / "capture_manifest.json"


try:
    RESULT = main()
finally:
    SIMULATION_APP.close()
