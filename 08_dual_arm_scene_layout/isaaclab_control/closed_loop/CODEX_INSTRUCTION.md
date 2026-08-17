# Codex task: complete the one-command closed loop

Work only in:

```text
/home/lin/Projects/DexGraspNet2_Wuji2
```

Base branch/worktree state is the user's `route-c-v2-curobo-esdf` checkpoint.
A patch directory has been copied into the project.  Read this file and all
files under:

```text
08_dual_arm_scene_layout/isaaclab_control/closed_loop/
```

before editing.

## User-visible target

Exactly one command:

```bash
cd ~/Projects/DexGraspNet2_Wuji2
./run_closed_loop.sh
```

The interaction must be:

1. Prompt for a **scene folder path**.  The input format is a directory that
   directly contains `scene_manifest.json`.  Print one concrete example.
2. Start the selected simulation scene, gravity-settle it, capture aligned RGB-D,
   and open `rgb.png`.
3. Ask free text:
   `你要抓什么东西？`
   Do NOT print simulator object names as semantic suggestions.
4. Run real GroundingDINO using **only RGB + the user text**.  It must not use
   scene_manifest/object_code/segmentation/USD/3-D object pose to decide the object.
5. Run SAM from the DINO box and save `mask.npy`, `overlay.png`, `result.json`.
6. Build the existing full-scene 40k RGB-D input with target membership using
   `08_build_target_network_input.py`.
7. Run the existing untouched official LEAP DGN2 inference using
   `09_predict_official_leap_target.py`.
8. Sort target candidates by the untouched official score
   `log_prob + 5*graspness`.
9. Test candidates in that exact descending score order.  For each candidate:
   - compose official LEAP waypoints;
   - run the existing reviewed LEAP->Wuji2 GRASP/root/SQUEEZE/final-waypoint stages;
   - build 5-stage flange targets;
   - allocate a placement slot with the existing `placement_allocator.py`;
   - build TRANSFER/PLACE/RELEASE/RETREAT Cartesian targets with
     `closed_loop/scripts/build_cartesian_route.py`;
   - run the cuRobo full gate.
   Select the FIRST fully feasible candidate.  That is the highest-scoring
   feasible candidate.  Do not replace the DGN2 rank with IK margin or a new score.
10. Execute pick -> place -> release -> retreat -> HOME.
11. Commit the placement registry only on physical PASS.
12. Persist final object poses as the next scene manifest.
13. Capture a fresh RGB-D frame and ask for another free-text target.
14. Repeat until the user types q/quit/exit.

## Environment isolation is mandatory

Keep:
- `isaaclab22_sim50`: Isaac Lab / Isaac Sim only.
- `curobo_v2`: cuRobo only, through the existing persistent worker.
- `graspnet2.0`: official DGN2 and, if compatible, the local GroundingDINO/SAM backend.
- existing Wuji retarget environment for LEAP->Wuji2.

Do not install cuRobo into Isaac Lab or retargeting.  Do not upgrade CUDA,
PyTorch, Isaac Sim, Isaac Lab, or drivers.

## Concrete missing items you must implement

### A. GroundingDINO + SAM backend

The repository currently contains cached Grounded-SAM products but no checked-in
live backend script.  Inspect the user's local installed environments and existing
model/checkpoint directories.  Wire
`closed_loop/scripts/grounded_sam_backend_template.py` or a replacement backend
with this CLI contract:

```text
--image <rgb.png>
--text <free text>
--output <folder>
```

Outputs:
- `mask.npy` bool HxW
- `overlay.png`
- `result.json`

`result.json` must include:
- query
- grounding_score
- box_xyxy
- sam_predicted_iou (if backend exposes it)
- backend/model identifiers

GroundingDINO input is RGB+text only.  Scene metadata may be used only later for
simulation-only rigid-body binding.

Write the final command array into
`closed_loop/config/closed_loop.json -> grounded_sam_backend`.

### B. Capture must emit measured robot state

Extend
`isaaclab_control/tools/11_settle_and_capture_dynamic_scene.py`
without changing physics parameters.  After settle/capture, write:

```text
<output>/robot_state.json
```

with:
- all 35 `joint_positions_by_name`
- `right_arm_q_current_rad` in exact J1..J7 order
- joint names
- fixed-base audit
- timestamp

This is measured state, not the nominal [50,-70,0,40,35,0,25] fallback.

### C. cuRobo self collision

The generated robot YAML already contains fitted spheres and a self-collision
ignore matrix.  Extend the existing cuRobo worker so each accepted candidate is
checked for robot self collision.  Return:

```json
"self_collision_pass": true|false
```

Do not change vendor URDF/USD or hide failures by altering spheres/thresholds.

### D. Continuous path collision

Current Route-C V2 checks waypoint states only.  Add continuous/interpolated
joint-path checking between every consecutive selected waypoint, including
HOME/q_current -> PREGRASP and the full route through RETREAT/HOME as applicable.

Use the selected q chain and sufficient interpolation based on maximum joint
step; check robot spheres against observed scene ESDF at every sample and check
self collision at every sample.

Return:

```json
"path_pass": true|false
```

with per-segment sample counts and first failing sample if any.

Do not fabricate PASS for unknown/unimplemented path checks.

### E. Unknown policy

Preserve `unknown_space_exposure` separately.  The current camera does not fully
observe the robot.  Do not silently call unknown free.  `closed_loop.json`
currently uses `block_unknown_space=false` so unknown is reported but not a hard
gate.  If you later add a static calibrated workcell layer or a view that
certifies the swept volume, this can be tightened deliberately.

### F. Remove legacy mesh collision from the new decision chain

Do not call as production gates:
- `08_dual_arm_scene_layout/scripts/10_filter_target_pregrasp_collision.py`
- archived SciPy/Pinocchio/full-mesh Route-C validators.

Keep them for history/regression only.

### G. Generic case generation

Use `closed_loop/scripts/build_candidate_case.py`.  Fix only concrete API/shape
mismatches against the installed official DGN2 code.  It must work for arbitrary
free-text targets resolved by the visual mask, not only `ashtray` or `dog`.

### H. Placement and repeated cycles

Reuse `runtime/scripts/placement_allocator.py`.
`build_cartesian_route.py` writes a compatibility
`07_arm_execution/full_arm_waypoint_ik.npz`, but it contains Cartesian route
targets only; no CPU IK.

After a physical PASS the existing runtime should commit
`placement_registry.json`.  Verify the second cycle receives a different free
placement slot.

Use `build_next_scene_manifest.py` to persist ALL final object poses so restarting
Isaac between cycles reproduces the post-placement state.  This is acceptable for
the first closed-loop implementation and avoids mixing perception dependencies
inside one long-lived Kit interpreter.

### I. Runtime genericization

The current full-pick runtime is dog/candidate3800-oriented by launcher/config.
Make it generic through existing CLI/config inputs:
- arbitrary `--case-root`
- per-cycle capture_root
- target key
- per-cycle output directory

Do not hard-code candidate3800 in the new closed-loop path.

The runtime may recompute cuRobo planning for defense-in-depth, but it must not
fall back to old CPU IK or mesh collision.

### J. Fail-closed motion permission

Physical action MUST NOT start unless all required gates are explicit PASS:
- fresh static stability gate
- IK pass
- observed ESDF collision pass
- self-collision pass
- continuous-path pass
- selected candidate exists

The current static gate is not cleared.  Do not set
`execution_enabled=true` until a fresh 10 s static test passes the frozen limits.
Do not relax thresholds to make it pass.

When all gates are genuinely passing, set
`closed_loop/config/closed_loop.json`:

```json
"execution_enabled": true
```

### K. RGB display and interaction

`rgb.png` must open after every capture.  After Grounded-SAM, open `overlay.png`.
The terminal prompt must remain free text and must not enumerate scene labels.

If DINO finds no target, do not run DGN2; ask for another phrase.
If DGN2 has no fully feasible candidate, report that and return to the text prompt
or resample according to a clearly logged policy.

## Validation order

Do not start with physical motion.

1. `python -m py_compile` all new Python files.
2. Unit tests for:
   - path validation / scene-folder parsing
   - candidate score ordering
   - next-scene-manifest pose persistence
   - placement second-cycle slot differs from first after registry commit.
3. One `scene_0000`, target `dog`, planning-only loop:
   RGB -> DINO -> SAM -> 40k -> DGN2 -> retarget -> candidate gate.
4. Verify the selected candidate is the first feasible candidate in descending
   official DGN2 score.
5. Verify no legacy mesh/CPU IK code is imported.
6. Fresh static stability test.
7. Only after all hard gates pass, one physical dog cycle.
8. Then a two-cycle test with two different targets; verify:
   - HOME after each cycle;
   - second RGB is freshly captured;
   - second placement does not overlap the first;
   - final scene state persists across the process restart.

## Logging

Keep:
- `core/worklog/COMMAND_LOG.md`
- `core/worklog/SESSION_SUMMARY.md`

Add a concise closed-loop section.  Large raw outputs stay local and untracked.

## Git safety

Do not use `git add .` or `git add -A`.
Do not upload:
- training dataset
- captures
- 01_cases generated data
- outputs
- npy/npz replay/capture blobs
- model weights
- raw logs

Do not commit/push until the user reviews the final local report.

End with:

```text
CLOSED_LOOP_LOCAL_REPORT
- files changed
- Grounded-SAM backend/model paths
- environments used
- dog planning-only result
- self collision result
- continuous path result
- static gate result
- physical smoke result (if allowed)
- two-cycle placement result (if allowed)
- remaining blockers
- git status
```

Then wait for user review.
