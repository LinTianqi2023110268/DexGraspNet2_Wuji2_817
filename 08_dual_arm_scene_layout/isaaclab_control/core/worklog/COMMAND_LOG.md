# Route-C V2 command log

## 2026-08-16 15:15 +08:00 — Initial read-only Git baseline

- Purpose: verify the required working directory and preserve the user's existing work.
- Conda environment: shell/base context; no project environment activated.
- Working directory: `/home/lin/Projects/DexGraspNet2_Wuji2`
- Command:

  ```bash
  pwd
  git status --short
  git diff --stat
  git diff
  ```

- Exit code: `0`
- Key output: correct project root; `core/`, three diagnostic scripts, `outputs/ik_failure_diagnosis/`, `MANIFEST.txt`, and `SELF_TEST_REPORT.txt` are untracked; tracked diff is empty.
- Conclusion: preserve all existing untracked paths and avoid destructive cleanup.

## 2026-08-16 15:16 +08:00 — Read integration and skill instructions

- Purpose: read all user-designated fact sources plus the two applicable local skill files.
- Conda environment: shell/base context.
- Working directory: `/home/lin/Projects/DexGraspNet2_Wuji2`
- Command:

  ```bash
  cat .agents/skills/isaaclab22-manipulator-control/SKILL.md
  cat .agents/skills/isaacsim5-manipulator-control/SKILL.md
  cat 08_dual_arm_scene_layout/isaaclab_control/core/CODEX_INSTRUCTION.md
  cat 08_dual_arm_scene_layout/isaaclab_control/core/INTEGRATION_GUIDE.md
  cat 08_dual_arm_scene_layout/isaaclab_control/core/README.md
  ```

- Exit code: `0`
- Key output: confirmed two-conda worker architecture, fixed IK thresholds, observed-scene Mapper/TSDF/ESDF contract, continuity selection, and static-stability gate.
- Conclusion: integrate supplied code with minimal compatibility changes; do not redesign or alter vendor assets.

## 2026-08-16 15:17 +08:00 — Skill reference and file inventory check

- Purpose: load skill-linked references and inventory relevant source files.
- Conda environment: shell/base context.
- Working directory: `/home/lin/Projects/DexGraspNet2_Wuji2`
- Command:

  ```bash
  cat .agents/skills/isaaclab22-manipulator-control/reference.md
  cat .agents/skills/isaaclab22-manipulator-control/evaluations.md
  cat .agents/skills/isaacsim5-manipulator-control/reference.md
  cat .agents/skills/isaacsim5-manipulator-control/evaluations.md
  rg --files 08_dual_arm_scene_layout/isaaclab_control/core
  ```

- Exit code: `127`
- Key output: all four referenced files are absent; `rg` is not installed.
- Conclusion: use the fully read skill bodies and fall back to `find`/`grep` for source audit.

## 2026-08-16 15:17 +08:00 — Fallback source inventory

- Purpose: inventory core, runtime, tools, and diagnostics using available utilities.
- Conda environment: shell/base context.
- Working directory: `/home/lin/Projects/DexGraspNet2_Wuji2`
- Command:

  ```bash
  find 08_dual_arm_scene_layout/isaaclab_control/core -type f -print | sort
  find 08_dual_arm_scene_layout/isaaclab_control/runtime 08_dual_arm_scene_layout/isaaclab_control/tools 08_dual_arm_scene_layout/isaaclab_control/diagnostics -type f -print | sort
  ```

- Exit code: `0`
- Key output: supplied core includes IK, worker bridge, RGB-D mapper, ESDF collision, robot spheres, tests, and a generated robot YAML; legacy runtime/tool paths remain present.
- Conclusion: proceed to content-level writer/consumer and production import audit.

## 2026-08-16 15:20 +08:00 — A. cuRobo environment probe

- Purpose: verify the frozen `curobo_v2` environment and required current APIs.
- Conda environment: `curobo_v2`.
- Working directory: `/home/lin/Projects/DexGraspNet2_Wuji2`
- Command:

  ```bash
  conda run -n curobo_v2 python 08_dual_arm_scene_layout/isaaclab_control/core/tools/probe_environment.py --project-root /home/lin/Projects/DexGraspNet2_Wuji2
  ```

- Exit code: `0`
- Key output: Python `3.11.15`; PyTorch `2.13.0+cu130`; cuRobo `0.8.0.post1.dev42`; Mapper, InverseKinematics, Kinematics, and RobotBuilder imports all OK; `cuda_available=False`.
- Conclusion: API imports pass, but this process currently has no usable CUDA device. GPU-dependent acceptance cannot be claimed unless a later run sees CUDA.

## 2026-08-16 15:21 +08:00 — B. Core pure tests, initial attempt

- Purpose: run supplied NumPy-only core tests with the required package path.
- Conda environment: base interpreter context.
- Working directory: `/home/lin/Projects/DexGraspNet2_Wuji2`
- Command:

  ```bash
  PYTHONPATH="$PWD/08_dual_arm_scene_layout/isaaclab_control" python -m unittest discover -s 08_dual_arm_scene_layout/isaaclab_control/core/tests -v
  ```

- Exit code: `1`
- Key output: all five test modules fail during collection with `ModuleNotFoundError: No module named 'numpy'`; interpreter is base Python 3.14.
- Conclusion: environment-only failure. Do not install into base; rerun unchanged tests with the existing `curobo_v2` interpreter.

## 2026-08-16 15:22 +08:00 — B. Core pure tests, correct environment

- Purpose: validate branch selection, RGB-D geometry, ESDF query, unknown-space semantics, and worker serialization.
- Conda environment: `curobo_v2`.
- Working directory: `/home/lin/Projects/DexGraspNet2_Wuji2`
- Command:

  ```bash
  PYTHONPATH="$PWD/08_dual_arm_scene_layout/isaaclab_control" conda run -n curobo_v2 python -m unittest discover -s 08_dual_arm_scene_layout/isaaclab_control/core/tests -v
  ```

- Exit code: `0`
- Key output: 11 tests run; all passed.
- Conclusion: supplied pure logic passes unchanged in the correct existing environment.

## 2026-08-16 15:24 +08:00 — C. Robot collision model build attempt

- Purpose: generate or validate the cuRobo collision-sphere robot model without touching vendor assets.
- Conda environment: `curobo_v2`.
- Working directory: `/home/lin/Projects/DexGraspNet2_Wuji2`
- Command:

  ```bash
  conda run -n curobo_v2 python 08_dual_arm_scene_layout/isaaclab_control/core/tools/build_robot_collision_model.py --project-root /home/lin/Projects/DexGraspNet2_Wuji2 --compute-metrics
  ```

- Exit code: `1`
- Key output: builder refused to overwrite the existing generated YAML and requested `--force` only after review.
- Conclusion: safe expected refusal. Existing YAML and alias adapter are present; inspect and load-test them before any rebuild.

## 2026-08-16 15:25 +08:00 — C. Existing collision model review

- Purpose: verify the generated model's provenance and fixed robot contract.
- Conda environment: shell/base context; read-only inspection.
- Working directory: `/home/lin/Projects/DexGraspNet2_Wuji2`
- Command:

  ```bash
  sha256sum 08_dual_arm_scene_layout/isaaclab_control/core/generated/dual_arm_right_wuji2_curobo.yml
  grep -nE 'urdf|base_link|arm_r_joint|collision_spheres|asset_alias' 08_dual_arm_scene_layout/isaaclab_control/core/generated/dual_arm_right_wuji2_curobo.yml
  ls -la 08_dual_arm_scene_layout/isaaclab_control/core/generated/asset_aliases
  ```

- Exit code: `0`
- Key output: SHA-256 `f743b132fe15c8e66b6f31c6e7aad5b161d041e31c51dd12b77fe1275c313f3c`; base `arm_base_link`; right-arm order J1..J7; tool frames include `arm_r_link_tf`; YAML points to the official combined URDF; both package aliases are symlinks into the vendor description tree.
- Conclusion: contract and non-invasive asset adapter are present; GPU load validation remains pending.

## 2026-08-16 15:28 +08:00 — D. Persistent worker smoke, restricted execution channel

- Purpose: ping the persistent `curobo_v2` worker from `isaaclab22_sim50`.
- Conda environment: client `isaaclab22_sim50`; worker `curobo_v2`.
- Working directory: `/home/lin/Projects/DexGraspNet2_Wuji2`
- Command:

  ```bash
  conda run -n isaaclab22_sim50 python 08_dual_arm_scene_layout/isaaclab_control/core/tools/smoke_test_worker.py --project-root /home/lin/Projects/DexGraspNet2_Wuji2
  ```

- Exit code: `1`
- Key output: client timed out after 120 s; delayed worker stderr reports `IKSolverUnavailable: CUDA is not visible to PyTorch in curobo_v2`.
- Conclusion: failure is tied to the restricted execution channel's GPU visibility, not proof that the machine lacks a GPU. Retest outside the sandbox.

## 2026-08-16 15:31 +08:00 — GPU visibility diagnosis

- Purpose: distinguish repository/environment errors from execution-channel device isolation.
- Conda environment: `curobo_v2` plus host driver utility.
- Working directory: `/home/lin/Projects/DexGraspNet2_Wuji2`
- Commands:

  ```bash
  nvidia-smi -L
  conda run -n curobo_v2 python -c "import torch; ..."
  ```

- Exit codes: restricted channel `1`; approved host channel `0`.
- Key output: restricted channel cannot communicate with the driver and reports zero devices; host channel reports `NVIDIA GeForce RTX 4070 Laptop GPU`, PyTorch `2.13.0+cu130`, CUDA `13.0`, `cuda_available=True`, device count `1`.
- Conclusion: the GPU is healthy and available on the host. All GPU-dependent acceptance commands must run through the host execution channel.

## 2026-08-16 15:33 +08:00 — A. Formal host-channel cuRobo probe

- Purpose: record the authoritative environment probe with real GPU access.
- Conda environment: `curobo_v2`.
- Working directory: `/home/lin/Projects/DexGraspNet2_Wuji2`
- Command:

  ```bash
  conda run -n curobo_v2 python 08_dual_arm_scene_layout/isaaclab_control/core/tools/probe_environment.py --project-root /home/lin/Projects/DexGraspNet2_Wuji2
  ```

- Exit code: `0`
- Key output: Python `3.11.15`; PyTorch `2.13.0+cu130`; CUDA true; cuRobo `0.8.0.post1.dev42`; GPU `NVIDIA GeForce RTX 4070 Laptop GPU`; Mapper, IK, Kinematics, and RobotBuilder imports OK.
- Conclusion: A. environment probe PASS on the authoritative host execution channel.

## 2026-08-16 15:34 +08:00 — D. Persistent worker host-channel smoke

- Purpose: validate isolated cross-conda IPC from Isaac Lab to persistent cuRobo.
- Conda environment: client `isaaclab22_sim50`; worker `curobo_v2`.
- Working directory: `/home/lin/Projects/DexGraspNet2_Wuji2`
- Command:

  ```bash
  conda run -n isaaclab22_sim50 python 08_dual_arm_scene_layout/isaaclab_control/core/tools/smoke_test_worker.py --project-root /home/lin/Projects/DexGraspNet2_Wuji2
  ```

- Exit code: `0`
- Key output: pong from cuRobo `0.8.0.post1.dev42` with right-arm joints `arm_r_joint_1` through `arm_r_joint_7` in exact order.
- Conclusion: D. persistent worker smoke PASS; environment isolation and IPC architecture are valid.

## 2026-08-16 15:39 +08:00 — E. candidate3800 exact GPU IK, initial client attempt

- Purpose: solve the five known exact flange targets through the persistent worker.
- Conda environment: bare `isaaclab22_sim50` client before AppLauncher; worker not reached.
- Working directory: `/home/lin/Projects/DexGraspNet2_Wuji2`
- Command: `conda run -n isaaclab22_sim50 python -c '<load case NPZ and call CuroboWorkerClient.solve_ik>'`
- Exit code: `1`
- Key output: `ModuleNotFoundError: No module named 'numpy'` while the bare client tried to read the NPZ.
- Conclusion: this is not an IK/IPC failure. The real Isaac runtime obtains NumPy after AppLauncher; for standalone exact-case regression, use the existing `curobo_v2` NumPy without changing environments.

## 2026-08-16 15:41 +08:00 — E. candidate3800 exact five-stage GPU IK

- Purpose: verify current cuRobo V2 solver API, returned solution shapes, fixed thresholds, and continuity selection on a known case.
- Conda environment: `curobo_v2` client and persistent `curobo_v2` worker.
- Working directory: `/home/lin/Projects/DexGraspNet2_Wuji2`
- Command: `conda run -n curobo_v2 python -c '<load candidate3800 exact base-frame targets and call CuroboWorkerClient.solve_ik>'`
- Exit code: `0`
- Key output: accepted solutions per PREGRASP/COVER/GRASP/SQUEEZE/LIFT = `32/34/34/34/31` out of 48; raw success = `33/37/37/37/31`; every stage selected continuously; solve time `1.989 s`; selected position/orientation errors are far below fixed limits and minimum selected inner margin is `0.4959 rad` (~28.41 deg).
- Conclusion: E. known candidate3800 GPU IK PASS. Multi-solution preservation inside the solver and q-reference chaining work with installed cuRobo V2.

## 2026-08-16 15:42 +08:00 — C. Existing robot collision model GPU load

- Purpose: validate that the reviewed YAML and asset aliases load in installed cuRobo and produce collision geometry.
- Conda environment: `curobo_v2` client and persistent worker.
- Working directory: `/home/lin/Projects/DexGraspNet2_Wuji2`
- Command: `conda run -n curobo_v2 python -c '<call CuroboWorkerClient.robot_spheres at the reference posture>'`
- Exit code: `0`
- Key output: 35 active joints and 233 collision spheres returned.
- Conclusion: C. robot collision model build/load PASS using the existing generated YAML; no overwrite or vendor change is needed.

## 2026-08-16 15:45 +08:00 — F/G/H. Top-20 five-stage coarse GPU regression

- Purpose: run fixed-threshold pre-cleanup regression covering candidate3800, candidate34, and score-ranked Top-20.
- Conda environment: `curobo_v2`.
- Working directory: `/home/lin/Projects/DexGraspNet2_Wuji2`
- Command:

  ```bash
  conda run -n curobo_v2 python 08_dual_arm_scene_layout/isaaclab_control/diagnostics/scripts/GPT_cuRoboV2_GPU全场景IK对比.py --reference-case 06_leap_to_wuji2_final_pipeline/01_cases/live_dynamic_scene0000_dog_candidate3800 --target dog --capture-target 08_dual_arm_scene_layout/captures/live_dynamic_scene0000/dgn2/dog --scope valid --stages pregrasp,cover,grasp,squeeze,lift --order score --limit 20 --seeds 48 --batch-size 64 --output-root 08_dual_arm_scene_layout/isaaclab_control/core/worklog/raw/top20_preintegration
  ```

- Exit code: `0`
- Key output: 8/20 candidates PASS all five stages; candidate3800 PASS (worst-stage margin 32.49 deg); candidate34 PASS (worst-stage margin 28.41 deg); Top-10 4/10; throughput 1812.55 candidates/s; no CPU/GPU classification mismatches among these 20.
- Raw output: `core/worklog/raw/top20_preintegration/curobo_gpu_scene_scan.json`, `.csv`, and `cpu_vs_curobo_comparison.json`.
- Conclusion: F candidate3800 PASS; G candidate34 PASS; H Top-20 8/20 PASS. This is coarse pure-IK evidence only, not exact collision/path/physics reachability.

## 2026-08-16 15:48 +08:00 — RGB-D Mapper smoke, initial attempt

- Purpose: build separate non-target and GroundedSAM target TSDF/ESDF layers from the formal capture.
- Conda environment: `curobo_v2`.
- Working directory: `/home/lin/Projects/DexGraspNet2_Wuji2`
- Command: `conda run -n curobo_v2 python core/tools/smoke_test_rgbd_map.py ...depth_m.npy ...intrinsics.npy ...T_world_camera.npy ...grounded_sam/dog/mask.npy`
- Exit code: `1`
- Key output: installed cuRobo 0.8.x camera integrator dereferenced `rgb_images.shape`; supplied depth-only `CameraObservation.rgb_image` was `None`.
- Conclusion: concrete installed-API mismatch; official local tests use uint8 `BxHxWx3` RGB. Add only a zero-valued placeholder plus resolution metadata.

## 2026-08-16 15:50 +08:00 — RGB-D Mapper smoke after minimal compatibility fix

- Purpose: verify native Mapper -> TSDF -> `compute_esdf()` for scene and target layers.
- Conda environment: `curobo_v2`.
- Working directory: `/home/lin/Projects/DexGraspNet2_Wuji2`
- Command: same formal capture smoke command as above.
- Exit code: `0`
- Key output: PASS; depth 720x1280; 407,537 valid pixels; fitted extent `[1.0604, 0.7002, 0.4318] m`; scene and target ESDF both shape `[54,36,22]`, voxel size `0.02 m`.
- Conclusion: native observed-scene and target-layer TSDF/ESDF construction works with installed cuRobo 0.8.x.

## 2026-08-16 15:53 +08:00 — ESDF semantic and robot-sphere diagnostics

- Purpose: verify observed-free/surface/occluded semantics and robot-state collision queries.
- Conda environment: `curobo_v2` persistent worker.
- Working directory: `/home/lin/Projects/DexGraspNet2_Wuji2`
- Command: worker `build_map`, diagnostic `query_spheres`, then `check_robot_state` with `T_world_base`.
- Exit code: `0`
- Key output: farther front sphere is `OBSERVED_FREE`; surface sphere is near-surface and collides when its radius exceeds ESDF distance; behind-surface sphere is `OCCLUDED_UNKNOWN`; reference robot has 0 blocking collisions and 233/233 unknown spheres because the current camera view does not observe the robot.
- Conclusion: collision and unknown are separate. Observed-safe must not be reported as guaranteed collision-free.

## 2026-08-16 15:58 +08:00 — Collision-aware candidate3800 multi-solution IK

- Purpose: hard-filter every IK-accepted solution with phase-aware scene/target ESDF before continuity selection.
- Conda environment: `curobo_v2` persistent worker.
- Working directory: `/home/lin/Projects/DexGraspNet2_Wuji2`
- Command: worker `build_map` followed by collision-aware `solve_ik` for five exact candidate3800 targets, 48 seeds.
- Exit code: `0`
- Key output: IK accepted `32/34/34/34/31`; collision-feasible `32/34/34/34/31`; selected chain observed-collision PASS at every stage; unknown exposure true at every stage (167–233 of 233 robot spheres unknown).
- Conclusion: IK and observed-scene collision gates PASS, while single-view unknown exposure remains explicitly unresolved and cannot be called guaranteed safe.

## 2026-08-16 16:02 +08:00 — J. Isaac Lab dry-run compatibility iterations

- Purpose: launch the formal runtime in `isaaclab22_sim50`, settle only, read q_current, and execute Route-C V2 planning without action.
- Conda environment: Isaac runtime `isaaclab22_sim50`; worker `curobo_v2`.
- Working directory: `/home/lin/Projects/DexGraspNet2_Wuji2`
- Command: `bash runtime/launchers/run_full_pick_place_25s_dog_candidate3800.sh --headless --dry-run`
- Initial outcomes: missing local Isaac Lab source path; then missing lightweight Isaac Lab dependencies; then missing archived-case `leap_selected_rank0.npz`; then worker inherited Kit `PYTHONPATH` and polluted base conda plugins.
- Fixes: launcher-level local Isaac Lab 2.2 source path; minimal fixed-environment dependencies only (`setuptools<81`, `numpy<2`, `flatdict`, `gymnasium==0.29.0`, `prettytable`, `toml`, `hidapi`, `trimesh`, `h5py`); use `case.json` candidate metadata rather than legacy collision-filter output; sanitize worker subprocess Python/Isaac/LD paths and disable base conda plugins.
- Conclusion: each failure was isolated; no CUDA/PyTorch/Isaac Sim/Isaac Lab version upgrade or environment rebuild occurred.

## 2026-08-16 16:08 +08:00 — J. Route-C V2 Isaac Lab dry-run

- Purpose: validate the full production planning call chain after real PhysX settle, without entering any motion state.
- Conda environment: Isaac runtime `isaaclab22_sim50`; worker `curobo_v2`.
- Working directory: `/home/lin/Projects/DexGraspNet2_Wuji2`
- Command: same formal `--headless --dry-run` launcher command.
- Exit/result: planning completed and report was written; Kit shutdown then hung and required interrupt after completion.
- Key output: fixed-base 35-DOF audit PASS; q_current deg `[49.058,-69.498,-0.798,39.175,33.535,1.191,25.523]`; PREGRASP through RETREAT all IK threshold PASS and observed-collision PASS; all stages unknown exposure true; report at `outputs/full_pick_place_25s_dog_candidate3800/route_c_v2_planning.json`.
- Conclusion: J. runtime dry-run planning PASS, with a separate headless Kit shutdown-hang limitation and explicit unknown exposure.

## 2026-08-16 16:12 +08:00 — K prerequisite: current ft04 static stability gate

- Purpose: enforce the 10-second gravity-on, IK-off, no-motion stability gate before any action smoke.
- Conda environment: `isaaclab22_sim50`.
- Working directory: `/home/lin/Projects/DexGraspNet2_Wuji2`
- Command: `bash diagnostics/launchers/run_initial_stability.sh --config core/worklog/static_gate_ft04.json --headless`
- Exit/result: formal report `FAIL`; Kit shutdown hung after report and was interrupted.
- Key output: right arm FAIL (max target error `1.465 deg`, max speed `0.0877 rad/s`); Wuji2 FAIL (max target error `0.321 deg`, max speed `0.0398 rad/s`); flange PASS; wrist PASS. Report: `core/worklog/raw/static_gate_ft04_current/report.json`.
- Conclusion: K physical/action smoke is locked and was not run. No threshold, gain, effort, asset, or drive tuning was changed.

## 2026-08-16 16:14 +08:00 — Legacy production-path archive

- Purpose: remove old SciPy/Pinocchio and complete-mesh collision implementations from active runtime/tools while preserving provenance.
- Conda environment: shell/base context.
- Working directory: `/home/lin/Projects/DexGraspNet2_Wuji2`
- Command: move `runtime_rebase_ik.py` plus tools `04`–`09` and `12`–`14` to `history/legacy_route_c_cpu_mesh/`.
- Exit code: `0`
- Key output: files moved intact; historical outputs untouched.
- Conclusion: active production runtime no longer imports/calls SciPy `least_squares`, old coarse reachability, or complete-mesh path collision.

## 2026-08-16 15:50 +08:00 — Final local regression and audit

- Purpose: re-run the post-integration regression, core tests, production-path scan, syntax compilation, and final Git checks without staging or publishing.
- Conda environment: `curobo_v2` for core/GPU work; shell for read-only Git audit.
- Working directory: `/home/lin/Projects/DexGraspNet2_Wuji2`
- Commands:

  ```bash
  conda run -n curobo_v2 python -m unittest discover -s 08_dual_arm_scene_layout/isaaclab_control/core/tests -t 08_dual_arm_scene_layout/isaaclab_control
  conda run -n curobo_v2 python 08_dual_arm_scene_layout/isaaclab_control/diagnostics/scripts/GPT_cuRoboV2_GPU全场景IK对比.py --reference-case 06_leap_to_wuji2_final_pipeline/01_cases/live_dynamic_scene0000_dog_candidate3800 --target dog --capture-target 08_dual_arm_scene_layout/captures/live_dynamic_scene0000/dgn2/dog --scope valid --stages pregrasp,cover,grasp,squeeze,lift --order score --limit 20 --seeds 48 --batch-size 64 --output-root 08_dual_arm_scene_layout/isaaclab_control/core/worklog/raw/top20_postintegration
  git status --short
  git diff --check
  git diff --stat
  git diff
  ```

- Exit code: `0` for unit tests, GPU regression, diff check, and compilation audit.
- Key output: 15/15 tests PASS; post-integration Top-20 remains 8/20 PASS with candidate3800 and candidate34 PASS; no active runtime/core legacy IK/collision import; tracked diff stat is 23 files, 177 insertions, 2051 deletions (untracked new core/archive files are not included by Git until staging, which was intentionally not done).
- Conclusion: local audit is clean. No vendor, DGN2, URDF, USD, acceptance-threshold, Git index, commit, remote, or GitHub mutation occurred.

## 2026-08-16 15:50 +08:00 — Isaac runtime environment version audit

- Purpose: record the minimal fixed-environment dependencies used by the real Isaac Lab runtime.
- Conda environment: `isaaclab22_sim50`.
- Working directory: `/home/lin/Projects/DexGraspNet2_Wuji2`
- Command: `conda run -n isaaclab22_sim50 python -c '<print Python and dependency versions>'`
- Exit code: `0`
- Key output: Python `3.11.15`; NumPy `1.26.4`; Gymnasium `0.29.0`; flatdict `4.0.1`; prettytable `3.3.0`; trimesh `5.0.0`; h5py `3.16.0`.
- Conclusion: only lightweight compatibility dependencies were added; CUDA, driver, PyTorch, Isaac Sim, and Isaac Lab were not upgraded or rebuilt.
