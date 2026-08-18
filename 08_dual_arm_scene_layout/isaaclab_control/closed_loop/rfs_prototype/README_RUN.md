# RFS Prototype V1 — standalone, no Isaac

Purpose: validate a coarse pre-retarget **Reachable Free-Space (RFS)** region on the existing bottle cycle before touching the production closed-loop pipeline.

This prototype deliberately does **not** use OMPL / MoveIt / ROS and does **not** start Isaac Sim.

## Files

- `01_calibrate_leap_wuji_bridge.py`
  - Uses already-retargeted candidates to measure the transformation variation from raw DGN2 LEAP root to final Wuji2 wrist.
  - Writes `bridge_calibration.json/.npz`.
  - This determines the conservative inflation used before retargeting.

- `02_build_rfs_prototype.py`
  - Reuses the project's existing cuRobo V2 modules:
    - `CuroboGpuIK`
    - `CuroboRGBDMapper`
    - `CuroboRobotSphereModel`
  - Builds a spatial grid spanning HOME to the SAM target neighborhood.
  - Tests several task-relevant flange orientations per voxel.
  - Classifies voxels as:
    - `UNREACHABLE`
    - `REACHABLE_BUT_POINTCLOUD_BLOCKED`
    - `FREE_BUT_NOT_HOME_CONNECTED`
    - `HOME_CONNECTED_RFS`
  - Projects the result back to the RGB camera as `rfs_overlay.png`.
  - Coarse-filters DGN2 LEAP candidates only by their inflated distance to the HOME-connected RFS. It preserves original DGN2 score order.
  - Exact post-retarget COVER IK remains mandatory.

## Current bottle cycle

```bash
CYCLE=/home/lin/Projects/DexGraspNet2_Wuji2/08_dual_arm_scene_layout/isaaclab_control/outputs/closed_loop_sessions/20260818_164056/cycle_001
```

Copy the two Python files into a temporary project folder, for example:

```bash
mkdir -p /home/lin/Projects/DexGraspNet2_Wuji2/08_dual_arm_scene_layout/isaaclab_control/closed_loop/rfs_prototype
cp ~/下载/01_calibrate_leap_wuji_bridge.py \
   ~/下载/02_build_rfs_prototype.py \
   /home/lin/Projects/DexGraspNet2_Wuji2/08_dual_arm_scene_layout/isaaclab_control/closed_loop/rfs_prototype/
```

### Step 1 — bridge calibration

Run in ordinary project Python; only NumPy is needed:

```bash
python /home/lin/Projects/DexGraspNet2_Wuji2/08_dual_arm_scene_layout/isaaclab_control/closed_loop/rfs_prototype/01_calibrate_leap_wuji_bridge.py \
  --cycle-root "$CYCLE" \
  --query bottle
```

Expected outputs:

```text
$CYCLE/rfs_prototype/bridge_calibration.json
$CYCLE/rfs_prototype/bridge_calibration.npz
```

For the uploaded 512 bottle cases, the already-computed reference values are approximately:

```text
translation residual p50 = 6.8 mm
translation residual p95 = 18.0 mm
translation residual p99 = 27.7 mm
translation residual max = 32.5 mm
rotation residual p50 = 3.22 deg
rotation residual p95 = 7.72 deg
rotation residual p99 = 9.10 deg
rotation residual max = 11.36 deg
recommended coarse position inflation = 40 mm
recommended orientation buffer = 12 deg
```

This means pre-retarget LEAP-root filtering is plausible, but must remain conservative.

### Step 2 — RFS build

Run inside `curobo_v2`:

```bash
~/miniconda3/bin/conda run --no-capture-output -n curobo_v2 \
python /home/lin/Projects/DexGraspNet2_Wuji2/08_dual_arm_scene_layout/isaaclab_control/closed_loop/rfs_prototype/02_build_rfs_prototype.py \
  --cycle-root "$CYCLE" \
  --query bottle \
  --cover-diagnostic-json /home/lin/下载/bottle_cover_ik_diag_first8.json
```

Default prototype settings are intentionally conservative/lightweight:

```text
coarse IK seeds                24
coarse joint-margin gate        0 deg   (exact downstream gate stays 3 deg)
initial grid step              50 mm
max spatial voxels           2200
orientation bins                8
ESDF collision margin           5 mm
moving collision geometry     arm_r_* only
unknown single-view space     reported, not blocked
HOME connectivity             6-neighbor spatial graph + local q continuity + intermediate q collision samples
```

Expected outputs:

```text
$CYCLE/rfs_prototype/rfs_map.npz
$CYCLE/rfs_prototype/rfs_report.json
$CYCLE/rfs_prototype/dgn2_rfs_filter.json
$CYCLE/rfs_prototype/rfs_overlay.png
```

## What counts as success for V1

Do not integrate into the production loop yet. First inspect:

1. `rfs_overlay.png`: a plausible green HOME-connected region should appear from the right-arm HOME side toward the bottle neighborhood.
2. Point-cloud obstacles should carve out/block parts of the otherwise reachable region.
3. In `rfs_report.json -> exact_cover_diagnostic_validation`:
   - the known exact-COVER PASS candidate must be retained;
   - a substantial fraction of the previous `NO_RAW_CUROBO_SUCCESS` candidates should be rejected.

If the known exact-COVER PASS is rejected, V1 is too aggressive. Increase the bridge/grid inflation or refine the grid before any production integration.

## Important semantics

RFS V1 is **not** a final motion plan. It is a coarse front-end region:

```text
DGN2 LEAP candidates
    -> inflated RFS coarse filter
    -> LEAP->Wuji2 retarget
    -> exact COVER IK
    -> Flexible Route / precise collision verification
    -> Isaac execution
```

Do not delete the downstream exact IK or final collision verification.
