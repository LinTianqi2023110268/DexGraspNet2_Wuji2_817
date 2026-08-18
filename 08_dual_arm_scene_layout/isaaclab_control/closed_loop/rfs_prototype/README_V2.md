# Candidate-centric RFS V2

## 1. This version's semantic contract

This version intentionally changes the **upper-level definition** from RFS V1.

The object being filtered is always a **DexGraspNet2 LEAP-hand grasp candidate**.  RGB-D / point-cloud samples are never classified as reachable or unreachable.

For every DGN2 LEAP candidate, V2 builds:

1. `LEAP GRASP root 6D`
2. official `LEAP PREGRASP root 6D` using DexGraspNet2's 0.10 m root retreat
3. an approximate right-arm flange GRASP/PREGRASP pair using the previously calibrated mean LEAP -> final Wuji2 wrist bridge

The bridge is only an **arm-query approximation before retargeting**.  Exact post-retarget COVER IK remains mandatory.

## 2. Two spaces used by the first filter

### A. Target Reach Region

- Query subject: DGN2 **LEAP GRASP root 6D** candidates.
- Each LEAP candidate is mapped to an approximate arm-flange 6D target.
- cuRobo coarse IK is run in one batched pass.
- Candidates with direct coarse IK define reference samples of the target reach region.
- The region is conservatively inflated by the calibrated LEAP -> final-Wuji2 bridge uncertainty (`40 mm / 12 deg` for the current bottle calibration unless calibration output changes).

This is a LEAP-candidate reach region, not a point-cloud reachability map.

### B. HOME -> PREGRASP -> GRASP Rough Trajectory Space

- A diverse subset of LEAP candidates is chosen as corridor anchors.
- For each anchor, V2 samples **multiple route tubes**, not one path:
  - HOME -> PREGRASP direct tube
  - HOME -> PREGRASP lifted/over-table alternative tube
  - PREGRASP -> GRASP local approach tube
- Every tube contains multiple cross-section samples and multiple IK solutions.
- Full right-arm collision spheres are tested against the **non-target RGB-D ESDF**.
- Layered q-space connectivity is propagated from measured HOME.
- The union of successful connected branches is the rough trajectory space.

The SAM target layer is deliberately not treated as an obstacle in this coarse pre-retarget trajectory space because it is the destination.  Non-target observed geometry remains an obstacle.  Exact hand/target contact handling stays downstream.

## 3. First-filter rule

A candidate passes only if:

```text
TARGET_REACH_REGION
AND
HOME -> PREGRASP -> GRASP ROUGH_TRAJECTORY_SPACE
```

The original DGN2 score order is preserved among PASS candidates.

Possible reasons:

```text
PASS
REJECT_TARGET_REACH_REGION
REJECT_NO_HOME_PREGRASP_GRASP_TRAJECTORY_SPACE
```

## 4. What V2 does NOT replace

V2 does **not** replace:

- LEAP -> Wuji2 retargeting
- exact post-retarget COVER IK
- final motion planning / exact collision verification
- Isaac Sim / PhysX execution

V2 does not start Isaac Sim and does not modify production closed-loop files.

## 5. Install in the project

The archive should be extracted into the existing prototype folder:

```bash
cd /home/lin/Projects/DexGraspNet2_Wuji2

tar -xzf ~/下载/rfs_candidate_centric_v2.tar.gz \
-C 08_dual_arm_scene_layout/isaaclab_control/closed_loop/rfs_prototype \
--strip-components=1
```

Expected new files:

```text
04_candidate_centric_rfs_v2.py
README_V2.md
```

## 6. Codex task

Codex should only perform static/path/API compatibility checks.  It should **not** redesign the algorithm and should not try to use sandbox GPU.

Suggested prompt:

```text
ChatGPT has completed the candidate-centric RFS V2 implementation at:

08_dual_arm_scene_layout/isaaclab_control/closed_loop/rfs_prototype/04_candidate_centric_rfs_v2.py

Semantic contract is fixed:
1. reachability subject is DGN2 LEAP root 6D candidates;
2. first space is the LEAP target reach region;
3. second space is the union of rough HOME->PREGRASP->GRASP arm trajectory corridors;
4. RGB-D/ESDF is only obstacle geometry; never classify point-cloud points as reachable/unreachable;
5. this is pre-retarget coarse filtering only.

Your task is limited to:
- py_compile/static checks;
- import path compatibility;
- current local API name compatibility;
- project-internal path compatibility;
- very small compatibility fixes only.

Do not modify algorithm, thresholds, math, filtering semantics, corridor construction, or production closed-loop code.
Do not start Isaac Sim.
Do not attempt sandbox GPU execution.
If an algorithm change appears necessary, stop and report it instead of changing it.

Finally report only:
- py_compile PASS/FAIL
- import/path/API issues
- exact lines changed, if any
- whether it is ready for local curobo_v2 GPU execution
```

## 7. Local GPU run

After Codex static check, run in a normal Ubuntu terminal:

```bash
cd /home/lin/Projects/DexGraspNet2_Wuji2

/home/lin/miniconda3/bin/conda run --no-capture-output -n curobo_v2 \
python \
08_dual_arm_scene_layout/isaaclab_control/closed_loop/rfs_prototype/04_candidate_centric_rfs_v2.py \
--cycle-root \
/home/lin/Projects/DexGraspNet2_Wuji2/08_dual_arm_scene_layout/isaaclab_control/outputs/closed_loop_sessions/20260818_164056/cycle_001 \
--query bottle \
--cover-diagnostic-json \
/home/lin/Projects/DexGraspNet2_Wuji2/08_dual_arm_scene_layout/isaaclab_control/closed_loop/rfs_prototype/bottle_cover_ik_diag_first8.json
```

## 8. Outputs

Default output folder:

```text
cycle_001/rfs_prototype/v2_candidate_centric/
```

Files:

```text
candidate_centric_rfs_v2_report.json
candidate_centric_rfs_v2_filter.json
candidate_centric_rfs_v2_map.npz

target_reach_region_overlay.png
trajectory_space_overlay.png
candidate_filter_overlay.png
```

### `target_reach_region_overlay.png`

Shows the target-side **LEAP candidate reach region**.  It does not draw a scene-wide reachable/unreachable point cloud.

### `trajectory_space_overlay.png`

Shows the union of successful connected arm corridor graph edges from HOME -> PREGRASP -> GRASP.  This is a rough path **space**, not one execution trajectory.

### `candidate_filter_overlay.png`

Shows DGN2 LEAP PREGRASP->GRASP candidate pairs:

- green: first-filter PASS
- red: rejected by target reach region
- orange: target-reachable but not covered by the rough HOME->PREGRASP->GRASP trajectory space
- cyan ring: known exact-COVER PASS from the old diagnostic, when available

## 9. Acceptance criteria for this bottle prototype

Do not integrate into production yet.  First inspect:

1. `known exact COVER target retained` should remain `1/1`.
2. Inspect whether the known rank447/candidate6559 pair lies in a visually plausible trajectory corridor.
3. Check how many of the 510 historical `NO_RAW_CUROBO_SUCCESS` candidates are rejected by:
   - target reach region
   - trajectory space additionally
   - overall first filter
4. The camera overlay should show a coherent corridor from HOME toward candidate PREGRASP/GRASP regions, not a scene-wide cloud of red/green dots.

