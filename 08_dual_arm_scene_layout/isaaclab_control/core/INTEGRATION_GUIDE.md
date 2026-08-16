# Integration and validation guide

Run all commands from the project root:

```bash
cd ~/Projects/DexGraspNet2_Wuji2
```

## 1. Copy this archive into the project

The zip already contains the correct relative path.  Extract it at the project root, or let
Codex copy only `08_dual_arm_scene_layout/isaaclab_control/core/`.

Do **not** delete old files yet.

## 2. Read-only environment probe

```bash
conda activate curobo_v2
python 08_dual_arm_scene_layout/isaaclab_control/core/tools/probe_environment.py \
  --project-root ~/Projects/DexGraspNet2_Wuji2
```

Expected: cuRobo version, PyTorch version, CUDA=True, RTX GPU name, Mapper import OK, IK import OK.

## 3. Pure tests

These can run in any environment with NumPy:

```bash
PYTHONPATH="$PWD/08_dual_arm_scene_layout/isaaclab_control" \
python -m unittest discover \
  -s 08_dual_arm_scene_layout/isaaclab_control/core/tests \
  -v
```

## 4. Generate the cuRobo robot collision-sphere model once

Run inside `curobo_v2`; this reads the existing URDF/meshes and writes only under `core/generated/`:

```bash
conda activate curobo_v2
python 08_dual_arm_scene_layout/isaaclab_control/core/tools/build_robot_collision_model.py \
  --project-root ~/Projects/DexGraspNet2_Wuji2 \
  --compute-metrics
```

The expected output file is:

```text
08_dual_arm_scene_layout/isaaclab_control/core/generated/dual_arm_right_wuji2_curobo.yml
```

Do not modify the vendor URDF to make the sphere fit pass.  If the installed cuRobo V2 reports a
concrete package-path/API mismatch, fix only that adapter and rerun.

## 5. Persistent worker smoke test from Isaac Lab environment

```bash
conda activate isaaclab22_sim50
python 08_dual_arm_scene_layout/isaaclab_control/core/tools/smoke_test_worker.py \
  --project-root ~/Projects/DexGraspNet2_Wuji2
```

Expected: worker `ping` reply with cuRobo version and seven right-arm joint names.  This proves
the two environments stay separate while the Isaac Lab process can use the cuRobo service.

## 6. First production IK regression

Codex should wire the exact five flange targets of a known case to `CuroboWorkerClient.solve_ik`.
Use the current runtime/measured right-arm q as `q_reference_rad`; when unavailable in an offline
tool, use:

```text
[50, -70, 0, 40, 35, 0, 25] deg
```

Required acceptance remains 5 mm / 5 deg / 3 deg inner margin.  Compare candidate3800 first,
then candidate34 if available, then Top-20, before replacing any legacy production path.

## 7. RGB-D map smoke test

Use the same files already produced by the capture pipeline:

```text
depth_m.npy
intrinsics.npy
T_world_camera.npy
grounded_sam/<target>/mask.npy
```

Before runtime integration, the mapper can also be exercised directly in `curobo_v2`:

```bash
python 08_dual_arm_scene_layout/isaaclab_control/core/tools/smoke_test_rgbd_map.py \
  --project-root ~/Projects/DexGraspNet2_Wuji2 \
  --depth /ABS/PATH/depth_m.npy \
  --intrinsics /ABS/PATH/intrinsics.npy \
  --camera-pose /ABS/PATH/T_world_camera.npy \
  --target-mask /ABS/PATH/grounded_sam/<target>/mask.npy
```

Then call worker `build_map`, and either `query_spheres` for diagnostic spheres or
`check_robot_state` with the measured named joint state and `T_world_base`.

For a diagnostic sphere test:

- one well in front of observed geometry -> no collision, observed-free;
- one centered on a table/object surface -> collision;
- one behind an observed surface -> `unknown=True`.

Do not treat `unknown=True` as a collision-free certificate.

## 8. Only after regression passes

Move legacy production-only implementations to history/diagnostics, update imports, README and
launchers, then run `git diff --check`, `python -m py_compile` on changed Python files, and the
existing project smoke tests before commit/push.
