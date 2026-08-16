# Route-C V2 session summary

## Current phase

Local implementation and audit complete; awaiting user review before any GitHub action.

## Completed

- Confirmed the project root is `/home/lin/Projects/DexGraspNet2_Wuji2`.
- Captured the initial Git status and diff; tracked diff is empty.
- Read `CODEX_INSTRUCTION.md`, `INTEGRATION_GUIDE.md`, and `README.md` completely.
- Read both available Isaac Lab / Isaac Sim skill instruction files completely.
- Added the minimal repository-level `AGENTS.md` requested for durable rules.
- Completed the `curobo_v2` probe; all required cuRobo V2 imports succeed.
- Formal host-channel environment probe sees the RTX 4070 and CUDA successfully.
- Core pure tests pass: 15/15 after adding runtime-math, collision-payload, and worker-environment regressions.
- Persistent worker ping passes from `isaaclab22_sim50` to `curobo_v2` with exact right-arm joint order.
- candidate3800 exact five-stage GPU IK passes with 31–34 accepted solutions per target and continuous chained selection.
- Existing generated robot YAML loads on GPU and returns 233 spheres for all 35 active joints.
- Pre-integration Top-20 five-stage coarse GPU regression passes 8/20; candidate3800 and candidate34 both pass.
- Formal RGB-D + GroundedSAM mask builds separate scene/target ESDF layers after one minimal cuRobo 0.8.x RGB-placeholder compatibility fix.
- Worker now supports per-solution phase-aware observed-collision hard filtering before q-reference continuity selection and preserves the audited solution sets.
- candidate3800 collision-aware exact solve passes observed collision at all five stages, but every selected state has single-view unknown exposure.
- Formal Isaac Lab dry-run passes the complete q_current -> worker -> GPU IK -> ESDF planning chain for nine stages.
- Archived the former runtime SciPy IK and old reachability/complete-mesh collision tool cohort under history.
- Post-integration Top-20 regression is unchanged at 8/20 PASS; candidate3800 and candidate34 remain PASS.
- Final `git diff --check` and Python compile/import audit pass; no vendor, DGN2, URDF, or USD file was modified.

## Current conclusion

- `core/` and several diagnostic/result paths are currently untracked user work and must be preserved.
- The skill files' referenced `reference.md` and `evaluations.md` files are absent; work continues from the complete skill bodies and repository source.
- `rg` is unavailable, so repository audits use `find` and `grep`.
- Installed versions are API-compatible. CUDA is unavailable only in the restricted execution channel; the host channel sees the RTX 4070 correctly.
- Production runtime no longer imports SciPy/Pinocchio IK or consumes the old path-collision report as a gate.

## Current blockers

Physical/action smoke is blocked by the unchanged ft04 static stability thresholds: right arm and Wuji2 both FAIL. GPU-dependent commands require the host execution channel. Continuous interpolated-path ESDF checking and link-scoped intentional finger/target contact remain future work.

## Next step

Wait for user review. Do not add, commit, push, or run physical motion while the static gate is FAIL.

## Modified files

- New/updated core production implementation: `bridge/curobo_worker.py`, `bridge/worker_client.py`, `perception_collision/rgbd_mapper.py`, `runtime_math.py`, tests, generated robot YAML/asset aliases, READMEs, and worklogs.
- Runtime replacement: `runtime/scripts/10_run_full_pick_place.py`, runtime launchers, visible launcher, and runtime README.
- Environment-entry fixes: five active diagnostics launchers now activate `isaaclab22_sim50` and expose local Isaac Lab sources.
- Documentation: `isaaclab_control/README.md`, `tools/README.md`, `history/INDEX.md`, root `AGENTS.md`, and archive README.
- Archived intact: `runtime_rebase_ik.py` and tools `04`-`09`, `12`-`14` under `history/legacy_route_c_cpu_mesh/`.
- Generated evidence: post/pre-integration Top-20 reports, current ft04 report, and `route_c_v2_planning.json`.
- Preserved pre-existing untracked diagnostics/results/manifest files; they were not edited or deleted.

## Test results

- A. Environment probe: PASS on host channel (CUDA true, RTX 4070 Laptop GPU); restricted-channel false result retained only as execution-channel diagnosis.
- B. Core unit tests: PASS (15/15) in `curobo_v2`; the base-interpreter attempt failed only because base has no NumPy.
- C. Robot model: PASS; existing YAML loads on GPU (35 joints, 233 spheres).
- D. Persistent worker: PASS from `isaaclab22_sim50`.
- E. candidate3800 exact GPU IK: PASS; accepted counts `32/34/34/34/31`, solve time `1.989 s`.
- F. candidate3800 coarse five-stage: PASS.
- G. candidate34 coarse five-stage: PASS.
- H. Top-20 coarse five-stage: 8/20 PASS (Top-10 4/10); limitation: pure coarse IK only.
- I. Post-integration Top-20 regression: unchanged at 8/20 PASS; candidate3800 and candidate34 remain PASS.
- RGB-D Mapper: PASS; 407,537 valid pixels, separate scene/target ESDF grids.
- J. Isaac Lab dry-run: planning PASS after real settle; q_current read and nine continuous stages selected; headless Kit shutdown hang remains.
- K. Physical smoke: NOT RUN because the current 10-second static gate FAILS (right arm FAIL, Wuji2 FAIL, flange/wrist PASS).
- Final checks: `git diff --check` PASS; related Python compilation PASS; production-path grep found no active legacy IK/collision imports.

## Current environment versions

- Python `3.11.15`
- PyTorch `2.13.0+cu130`
- cuRobo `0.8.0.post1.dev42`
- CUDA available: `True` on the host execution channel (`False` only in the restricted sandbox channel)
- Isaac runtime conda: Python `3.11.15`, NumPy `1.26.4`, Gymnasium `0.29.0`; Isaac Sim `5.0`, local Isaac Lab `2.2.x` source tree
