from __future__ import annotations

import numpy as np
import torch
from curobo.perception import RobotSegmenter
from curobo.types import CameraObservation, JointState, Pose


class RobotDepthCleaner:
    """cuRobo V2 robot depth segmentation wrapper.

    This module only cleans depth input. It does not modify planning or execution.
    """

    def __init__(self, robot_file, distance_threshold=0.05):
        self.segmenter = RobotSegmenter.from_robot_file(
            robot_file=robot_file,
            distance_threshold=distance_threshold,
        )

    def remove_robot(self, depth, intrinsics, T_world_camera, joint_positions):
        device = next(self.segmenter.parameters()).device if hasattr(self.segmenter, 'parameters') else torch.device('cuda:0')
        depth_t = torch.as_tensor(depth, dtype=torch.float32, device=device)
        K_t = torch.as_tensor(intrinsics, dtype=torch.float32, device=device)
        T_t = torch.as_tensor(T_world_camera, dtype=torch.float32, device=device)
        q_t = torch.as_tensor(joint_positions, dtype=torch.float32, device=device).reshape(1, -1)

        obs = CameraObservation(
            name='camera',
            depth_image=depth_t,
            intrinsics=K_t,
            pose=Pose.from_matrix(T_t.unsqueeze(0)),
        )
        state = JointState.from_position(q_t)
        mask, filtered = self.segmenter.get_robot_mask(obs, state)
        return {
            'robot_mask': mask.detach().cpu().numpy(),
            'filtered_depth': filtered.detach().cpu().numpy(),
        }
