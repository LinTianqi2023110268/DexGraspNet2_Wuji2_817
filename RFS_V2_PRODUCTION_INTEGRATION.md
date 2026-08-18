# Candidate-centric RFS V2 — first production integration

This package contains the complex production-facing wrapper.  Codex only needs
to make small edits in `orchestrator.py` and `config/closed_loop.json`.

## Why this is qualified for a first rollout

Bottle validation:
- DGN2 LEAP candidates: 6195
- RFS V2 first-filter PASS: 865
- known exact-COVER positive rank447/candidate6559 retained
- first 64 RFS-PASS candidates: 14 exact-COVER PASS
- among those 14, Flexible Route:
  - 2 full PASS (rank810, rank813)
  - 8 fail at TRANSFER
  - 4 fail at PLACE
- none of the 12 route failures fail at the target reach / PREGRASP / COVER stage.

Therefore the candidate-centric front-end is doing the job it was designed for.
The first rollout MUST keep downstream exact COVER IK, Flexible Route, and
Isaac/PhysX unchanged.

## First rollout policy

Use:
`mode = "priority_then_rescue"`

Meaning:
1. RFS-PASS candidates are tried first, preserving DGN2 target-rank order.
2. If those are exhausted, RFS-rejected candidates remain as a rescue tier.
3. If the RFS backend errors, the original DGN2 order is restored automatically.

This avoids a hard completeness regression while we validate more objects.

## Minimal Codex edits

### 1) orchestrator.py import

Add:

```python
from planning.candidate_rfs_v2_runtime import run_candidate_rfs_v2
```

### 2) after DGN2 candidate generation and before the normal cuRobo worker loop

After:

```python
prediction = dgn_root / "official_leap_1024_target_ranked.npz"
candidates_plain, total_proposals = candidate_order(prediction)
```

run:

```python
rfs_runtime = run_candidate_rfs_v2(
    project_root=root,
    cycle_root=cycle_root,
    query=query,
    candidates=candidates_plain,
    settings=cfg.get("candidate_rfs_v2", {}),
)
rfs_priority_indices = list(rfs_runtime.ordered_indices)
```

Do not remove GroundedSAM, DGN2, sim-target binding, or any downstream gate.

### 3) after legacy coarse-prefilter selection is formed

Immediately after the existing `coarse_cfg` if/else creates
`candidates` and `survivor_indices`, apply:

```python
if candidates is not candidates_plain:
    # Current default has legacy coarse filter OFF.  If someone enables it later,
    # preserve only indices that survived legacy filtering, but keep RFS ordering.
    allowed = set(int(i) for i in survivor_indices)
    survivor_indices = [i for i in rfs_priority_indices if i in allowed]
else:
    survivor_indices = list(rfs_priority_indices)

coarse_report["candidate_rfs_v2"] = rfs_runtime.to_jsonable()
```

If Codex prefers not to use object identity, it may instead apply the same
intersection logic unconditionally:

```python
allowed = set(int(i) for i in survivor_indices)
survivor_indices = [i for i in rfs_priority_indices if i in allowed]
coarse_report["candidate_rfs_v2"] = rfs_runtime.to_jsonable()
```

That is safer and recommended.

### 4) config/closed_loop.json

Add a top-level object:

```json
"candidate_rfs_v2": {
  "enabled": true,
  "mode": "priority_then_rescue",
  "fallback_on_error": true,
  "minimum_pass_candidates": 1,
  "conda_exe": "/home/lin/miniconda3/bin/conda",
  "conda_env": "curobo_v2",
  "script": "08_dual_arm_scene_layout/isaaclab_control/closed_loop/rfs_prototype/04_candidate_centric_rfs_v2.py",
  "bridge_npz": "08_dual_arm_scene_layout/isaaclab_control/closed_loop/rfs_prototype/calibration_production/bridge_calibration_bottle512.npz",
  "output_subdir": "rfs_candidate_centric_v2_runtime"
}
```

Do not change:
- `coarse_ik_prefilter` (leave legacy filter OFF)
- exact COVER settings
- Flexible Route
- HOME->PREGRASP mandatory collision gate
- Isaac execution
- `retarget_chunk_size=64`

## Expected bottle behavior after integration

The old unfiltered order encountered full-route candidates much later.
With RFS priority, rank810/candidate6171 and rank813/candidate6074 are both
inside the first 64 RFS-PASS candidates and both passed the offline full
Flexible Route replay.  Therefore the bottle scene should normally find a full
route in RFS fast-tier Batch 1, subject to the same production runtime state.

## First full-flow test

Use the user's normal production command:

```bash
cd /home/lin/Projects/DexGraspNet2_Wuji2
./run_closed_loop.sh --sim-execute --no-planner-collision-check
```

Expected new log block after DGN2:

```text
[RFS V2] candidate-centric pre-retarget filter ...
...
[RFS V2] PASS=.../... | REJECT=... | fast tier first
[RFS V2] rescue tier enabled ...
```

Then the normal Wuji2 retarget / Exact COVER / Flexible Route / Isaac stages
continue unchanged.

If the RFS subprocess fails, the log should explicitly show
`[RFS V2 FALLBACK]` and the old DGN2 order should continue.
