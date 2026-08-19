# robot_segmentation_v1

cuRobo V2 RobotSegmenter wrapper.

Purpose:
- remove robot pixels from depth before ESDF planning
- keep existing interpolation baseline unchanged
- provide future MotionPlanner input

Outputs:
- robot_mask
- filtered_depth
