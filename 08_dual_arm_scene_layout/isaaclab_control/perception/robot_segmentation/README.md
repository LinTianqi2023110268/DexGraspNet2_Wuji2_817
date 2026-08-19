# Robot depth segmentation adapter

Standalone cuRobo V2 `RobotSegmenter` adapter for existing persistent-capture
folders.  This is not connected to the baseline planner yet.

Input capture folder must contain:

- `depth_m.npy`
- `intrinsics.npy`
- `T_world_camera.npy`
- `robot_state.json`

Output is written to `capture/planning/`:

- `robot_mask.npy`
- `robot_mask.png`
- `filtered_depth.npy`
- `filtered_depth_preview.png`
- `robot_segmentation_report.json`

Run in the cuRobo environment:

```bash
cd /home/lin/Projects/DexGraspNet2_Wuji2
conda run -n curobo_v2 python \
  08_dual_arm_scene_layout/isaaclab_control/perception/robot_segmentation/run_robot_segmenter_capture.py \
  --capture-dir /path/to/capture
```

Coordinate contract:

- capture `T_world_camera.npy` maps camera coordinates into layout world;
- layout JSON provides `T_world_base` from `transforms.dual_arm_mount`;
- the adapter sends `T_base_camera = inv(T_world_base) @ T_world_camera` to
  `RobotSegmenter`, because cuRobo transforms projected camera points into the
  robot/base frame before distance checks.
