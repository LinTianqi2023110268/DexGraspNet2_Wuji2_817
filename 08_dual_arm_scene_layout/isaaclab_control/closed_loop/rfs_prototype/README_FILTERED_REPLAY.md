# RFS V2 Filtered Offline Replay + Timing Profile

This package is the next experiment after candidate-centric RFS V2.

## What it does

It takes the first 64 `PASS` rows from:

`cycle_001/rfs_prototype/v2_candidate_centric/candidate_centric_rfs_v2_filter.json`

and preserves original DGN2 score/rank order.

Then it runs, offline:

1. candidate-case build
2. LEAP -> Wuji2 retarget, with independent timing for:
   - `01_retarget_grasp_official.py`
   - `02_align_root6d.py`
   - `03_retarget_squeeze_official.py`
3. Wuji2/final arm-target finalization
4. Exact COVER batch IK
5. Flexible Route for every Exact-COVER-PASS candidate

It does **not** start Isaac Sim and does **not** modify production closed-loop code.

## Files

- `05_filtered_offline_replay_profile.py`: master experiment
- `05_profile_retarget_stages.py`: timing-only wrapper for current retarget 01/02/03

## Hard bottle validation

For the current bottle replay, run with:

`--expected-rank 447 --expected-candidate-index 6559`

The script will stop with an error unless that known Exact-COVER positive is:
- present in the first 64 filtered PASS candidates;
- finalized successfully;
- Exact-COVER PASS again.

## Recommended local command

Run with the current planner Python, not the Codex sandbox:

```bash
cd /home/lin/Projects/DexGraspNet2_Wuji2

/home/lin/miniconda3/envs/isaaclab22_sim50/bin/python \
08_dual_arm_scene_layout/isaaclab_control/closed_loop/rfs_prototype/05_filtered_offline_replay_profile.py \
--cycle-root \
/home/lin/Projects/DexGraspNet2_Wuji2/08_dual_arm_scene_layout/isaaclab_control/outputs/closed_loop_sessions/20260818_164056/cycle_001 \
--query bottle \
--expected-rank 447 \
--expected-candidate-index 6559
```

Default planner collision semantics match the user's current production launch with
`--no-planner-collision-check`; the current project code still keeps its mandatory
HOME->PREGRASP observed-map safety gate.

Do not add `--full-planner-collision-check` for this first timing replay unless you
intentionally want to compare a different planner policy.

## Main output

`cycle_001/rfs_prototype/v2_filtered_offline_replay_top64/filtered_offline_replay_profile.json`

The terminal also prints:

- build time
- 01 retarget time
- 02 root alignment time
- 03 squeeze retarget time
- retarget total
- finalize time
- cuRobo RGB-D map time
- Exact COVER time
- Flexible Route total
- total wall time
