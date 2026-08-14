# Force-adjusted legacy-geometry training dataset

This is an isolated A/B dataset variant. It answers one question only: what
happens when DexGraspNet2-Wuji2 learns the Wuji2 1.0 force-adjusted 20-joint
target instead of the optimizer output?

## The only changed variable

- old network target: `pre_force_joint_positions_rad` (`q_opt`)
- new network target: `joint_positions_rad` (`q_force_adjusted`)
- retained provenance: `qpos_pre_force`, `force_adjustment_delta`
- invariant identity: `qpos_pre_force + force_adjustment_delta == qpos`

Scene poses, 256 camera views, 40000 points/view, object scales, legacy Wuji2
URDF geometry, six-direction eligibility, collision filtering, path filtering,
reference-point construction and graspness equations remain unchanged.

`joint_positions_rad` is the commanded force-adjusted target that passed the
Wuji2 1.0 six-direction tests. It is not a measured final joint state.

## Why the legacy URDF is intentional

The source optimizer and six-direction validation used the legacy Wuji2 URDF.
Keeping it here makes the experiment a strict single-variable A/B comparison.
Isaac Sim deployment remains based on the official Hand2 USD; the already
audited fixed transform converts legacy `r_base_link` poses to official
`r_wrist` poses at deployment.

## Commands

Initialize one acceptance scene:

```bash
python 02_training_dataset/variants/force_adjusted_legacy_v1/initialize_dataset.py --scene 17
python 02_training_dataset/code/prepare_wuji2_training_dataset.py \
  --config 02_training_dataset/config/wuji2_train60_100seminal_256view_force_adjusted_legacy_v1.json \
  --scene 17 --device cuda:0
```

Create the two Trimesh comparison files:

```bash
python trimesh/view_force_adjusted_training_ab.py --scene 17 --object-id 2
```

Audit the single-variable contract:

```bash
python 02_training_dataset/variants/force_adjusted_legacy_v1/check_contract.py --scene 17
```

After acceptance, initialize and preprocess all 100 scenes:

```bash
python 02_training_dataset/variants/force_adjusted_legacy_v1/initialize_dataset.py --all
python 02_training_dataset/code/prepare_wuji2_training_dataset.py \
  --config 02_training_dataset/config/wuji2_train60_100seminal_256view_force_adjusted_legacy_v1.json \
  --device cuda:0
```

The resumable command (recommended for long runs) is:

```bash
python 02_training_dataset/variants/force_adjusted_legacy_v1/prepare_remaining.py --device cuda:0
```

The resumable command intentionally stops after Stage03 by default. Those four
stages run once per scene and perform no Isaac simulation. Stage04 is the
separate 256-view mapping needed only immediately before network training:

```bash
python 02_training_dataset/variants/force_adjusted_legacy_v1/prepare_remaining.py \
  --device cuda:0 --stop-after-stage 04_single_view
```

Read-only live monitor:

```bash
python 02_training_dataset/code/live_label_generation_monitor.py \
  --config 02_training_dataset/config/wuji2_train60_100seminal_256view_force_adjusted_legacy_v1.json
```

For the current per-scene-only run, use the dedicated monitor:

```bash
python 02_training_dataset/variants/force_adjusted_legacy_v1/live_scene_filter_monitor.py
```
