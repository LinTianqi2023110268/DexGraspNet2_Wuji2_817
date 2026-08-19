from __future__ import annotations


class RouteBMotionPlannerAdapter:
    """
    Route B interface.

    Existing code keeps ownership of:
    - PREGRASP relaxed IK
    - COVER strict IK
    - Wuji2 grasp/squeeze
    - Isaac execution

    cuRobo owns:
    - current -> PREGRASP trajectory
    - grasp -> lift -> transfer -> place trajectory
    """

    def __init__(self, config):
        self.config = config
        self.planner = None

    def build_pick_scene(
        self,
        filtered_depth_path,
        intrinsics_path,
        camera_pose_path,
    ):
        """
        Robot-cleaned depth:
        depth -> Mapper -> ESDF -> cuRobo Scene
        """
        raise NotImplementedError

    def plan_current_to_pregrasp(
        self,
        q_current,
        pregrasp_pose,
        scene,
    ):
        """
        Generate collision-free approach trajectory.

        Output:
        {
          time_s,
          q_rad,
          qd_rad_s,
          qdd_rad_s2,
          phase
        }
        """
        raise NotImplementedError

    def build_carry_scene(
        self,
        filtered_depth_path,
        attached_object,
    ):
        """
        Remove target object from world scene.
        Attach target geometry to robot.
        Build carry ESDF.
        """
        raise NotImplementedError

    def plan_grasp_to_place(
        self,
        q_cover,
        place_pose,
        carry_scene,
        attached_object,
    ):
        """
        Generate:
        COVER -> LIFT -> TRANSFER -> PLACE

        collision-free trajectory.
        """
        raise NotImplementedError
