# `isaaclab_control/core` replacement package

This package is intentionally small and layered.  It is **not** a new parallel project.
Copy the supplied `core/` directory directly under:

`08_dual_arm_scene_layout/isaaclab_control/core/`

## What it replaces in production

- CPU SciPy/Pinocchio production IK -> `core/ik/curobo_gpu_ik.py`
- "take whichever IK solution cuRobo returns" -> `core/ik/ik_solution_selector.py`
- complete-mesh / simulation-only scene collision as the production truth ->
  `core/perception_collision/` based on the current RGB-D observation.

The old CPU IK and historical collision scripts should be retained under diagnostics/history
for regression evidence, not imported by the production runtime.

## Environment contract

Keep the environments isolated:

- `isaaclab22_sim50`: Isaac Sim / Isaac Lab runtime.  It may import
  `core.bridge.worker_client`, which uses only the Python standard library.
- `curobo_v2`: cuRobo GPU IK + Mapper/TSDF/ESDF.  It runs
  `core/bridge/curobo_worker.py` persistently.
- `wuji2_factory`: DGN2 / retargeting pipeline.  Do not install cuRobo here.

The bridge defaults to `~/miniconda3/bin/conda` and conda env `curobo_v2`.  Override with
`CUROBO_CONDA_EXE` if needed.  A one-time `build_robot_collision_model.py` tool uses
cuRobo's official `RobotBuilder` to fit collision spheres from the existing combined URDF;
the generated YAML is stored under `core/generated/` and never edits the vendor robot files.

## IK contract

Default production acceptance remains:

- position error <= 5 mm
- orientation error <= 5 deg
- inner joint-limit margin >= 3 deg
- physical URDF limits are shrunk inward by 0.01 rad before margin measurement
- right-arm joint order is exactly `arm_r_joint_1 ... arm_r_joint_7`
- initial/offline reference posture is `[50,-70,0,40,35,0,25] deg`
- online runtime should pass the measured Isaac Lab `q_current` as the first reference

The solver requests many cuRobo seeds and preserves all accepted solutions.  Selection is
then performed separately: nearest normalized joint-space branch to `q_reference` first,
then larger joint-limit margin, then smaller pose error.  For ordered waypoints the selected
solution of PREGRASP becomes the reference for COVER, then GRASP, SQUEEZE, LIFT, etc.

This keeps the already-validated cuRobo GPU behavior but fixes branch discontinuity.

## Observed-scene collision contract

The mapper uses cuRobo V2's native `Mapper -> TSDF -> compute_esdf()` APIs.  It auto-fits
the mapping volume to the current valid depth frame, with safety caps to catch a wrong depth
scale or camera transform.

If a GroundedSAM target mask is available, the frame is split into two layers:

1. non-target observed scene ESDF (table + other observed obstacles),
2. target observed surface ESDF.

A small dilation removes target-boundary pixels from the non-target layer, reducing mask
leakage.  The target is a blocking obstacle in PREGRASP/APPROACH/COVER and an intentional
contact layer in GRASP/SQUEEZE/LIFT.  In the first version the target contact allowance is
phase-wide; when Wuji2 collision-link groups are wired in, narrow it to finger contact links.

### Single-view unknown space

`SingleViewVisibility` deliberately does **not** interpret "no point" as free.  A query sphere
is known-free only if the entire sphere lies in front of the measured depth along its camera
ray.  Points behind the first measured surface are `OCCLUDED_UNKNOWN`; outside-FOV or invalid
depth is `UNKNOWN`.  The collision result returns `unknown` separately from collision.

This first version assumes the camera frame does not contain the robot/hand, per the current
project decision.  Robot self-observation removal is intentionally not implemented yet.

For robot-vs-observed-scene checks, `CuroboRobotSphereModel` computes cuRobo collision spheres
from the measured named joint state and transforms them with `T_world_base`; the worker exposes
`check_robot_state(...)` so the Isaac Lab process does not need to import cuRobo.  At runtime,
provide all measured active joint positions by name whenever available.  Target-contact allowance
is still phase-wide in this first version and must later be narrowed to intentional finger links.

The production worker applies this check to every IK-accepted solution before
continuity selection. It preserves the full audited solution set and reports
`ik_pass`, `observed_scene_collision_pass`, and `unknown_space_exposure` separately.
Continuous-path observed collision is not implemented in this revision, so runtime
reports `path_pass=null` rather than fabricating a PASS.

## cuRobo API choices

The code follows the current V2 public examples:

- `from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg`
- `RobotCfg.from_basic(...)`
- `solve_pose(..., return_seeds=N)`
- `from curobo.perception import Mapper, MapperCfg`
- `CameraObservation` + `Mapper.integrate(...)` + `Mapper.compute_esdf()`

ESDF trilinear querying follows the coordinate reorder used by cuRobo's official
`volumetric_mapping.py` example (`grid_sample` coordinates z,y,x).

## Tests in this archive

The archive includes pure-Python/NumPy tests for branch selection, RGB-D geometry, mask
dilation, and single-view unknown-space classification.  They do not require a GPU.

The actual cuRobo/Mapper smoke test must be run on the user's Ubuntu machine because this
ChatGPT execution container does not have the user's `curobo_v2` environment or RTX GPU.
Use the commands in `INTEGRATION_GUIDE.md`.
