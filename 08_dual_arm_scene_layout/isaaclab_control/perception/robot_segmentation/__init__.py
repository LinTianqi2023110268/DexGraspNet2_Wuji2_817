"""cuRobo RobotSegmenter adapter for capture-time robot depth filtering."""

from .curobo_robot_segmenter import RobotDepthCleaner, run_capture_robot_segmentation

__all__ = ["RobotDepthCleaner", "run_capture_robot_segmentation"]
