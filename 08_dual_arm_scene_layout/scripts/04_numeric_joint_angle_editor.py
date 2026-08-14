"""Isaac Sim 5.0 Script Editor：用数字精确编辑整机35个关节角度。

使用前提：
1. 已运行 ``01_create_manual_layout.py``；
2. 已运行 ``03_open_physics_inspector.py``，并在Physics Inspector中选择
   ``/World/Layout/DualArmMount/DualArm/root_joint``；
3. 时间线保持停止，不要点击Play。

输入单位是度。任意一个滑块或数字框变化时，脚本都会把全部35个关节作为
一个完整姿态原子性提交：35个JointState位置同时锁定、35个速度清零、35个
Drive Target同步，然后只触发一次Physics Inspector准静态刷新。这样调整一个
关节时，其余关节不会被旧驱动目标或残余速度带动。脚本不修改上游双臂模型
或官方Wuji2 Hand 2 USD，也不修改刚度、阻尼、力矩和关节限位。
"""

import builtins
import math

import omni.kit.app
import omni.kit.commands
import omni.ui as ui
import omni.usd
from pxr import PhysxSchema, Sdf, UsdPhysics


ROBOT_PATH = "/World/Layout/DualArmMount/DualArm"
ARTICULATION_ROOT_PATH = f"{ROBOT_PATH}/root_joint"
WINDOW_STATE_KEY = "DGN2_NUMERIC_JOINT_EDITOR_STATE"

# 显式顺序比依赖USD遍历顺序更容易核对，也能及时发现装配拓扑被意外改变。
JOINT_GROUPS = (
    ("RIGHT ARM (7)", tuple(f"arm_r_joint_{i}" for i in range(1, 8))),
    ("LEFT ARM (7)", tuple(f"arm_l_joint_{i}" for i in range(1, 8))),
    ("LEFT GRIPPER (1)", ("arm_l_joint_finger",)),
    (
        "WUJI2 THUMB (4)",
        ("r_thumb_cmc_flex", "r_thumb_cmc_abd", "r_thumb_mcp", "r_thumb_ip"),
    ),
    (
        "WUJI2 INDEX (4)",
        (
            "r_index_finger_mcp_flex",
            "r_index_finger_mcp_abd",
            "r_index_finger_pip",
            "r_index_finger_dip",
        ),
    ),
    (
        "WUJI2 MIDDLE (4)",
        (
            "r_middle_finger_mcp_flex",
            "r_middle_finger_mcp_abd",
            "r_middle_finger_pip",
            "r_middle_finger_dip",
        ),
    ),
    (
        "WUJI2 RING (4)",
        (
            "r_ring_finger_mcp_flex",
            "r_ring_finger_mcp_abd",
            "r_ring_finger_pip",
            "r_ring_finger_dip",
        ),
    ),
    (
        "WUJI2 PINKY (4)",
        ("r_pinky_mcp_flex", "r_pinky_mcp_abd", "r_pinky_pip", "r_pinky_dip"),
    ),
)


def _enable_physx_support_ui():
    """启用Physics Inspector后端，使暂停状态下的JointState立即刷新外观。"""
    manager = omni.kit.app.get_app().get_extension_manager()
    manager.set_extension_enabled_immediate("omni.physx.supportui", True)


def _find_revolute_joints(stage):
    robot = stage.GetPrimAtPath(ROBOT_PATH)
    if not robot.IsValid():
        raise RuntimeError(
            f"Robot not found at {ROBOT_PATH}. Run 01_create_manual_layout.py first."
        )

    by_name = {}
    for prim in stage.Traverse():
        if not str(prim.GetPath()).startswith(ROBOT_PATH + "/"):
            continue
        if prim.IsA(UsdPhysics.RevoluteJoint):
            name = prim.GetName()
            if name in by_name:
                raise RuntimeError(f"Duplicate revolute joint name: {name}")
            by_name[name] = prim

    expected = [name for _, names in JOINT_GROUPS for name in names]
    missing = [name for name in expected if name not in by_name]
    if missing:
        raise RuntimeError(f"Expected joints missing from composed robot: {missing}")
    if len(expected) != 35:
        raise RuntimeError(f"Editor contract contains {len(expected)} joints, expected 35")
    return {name: by_name[name] for name in expected}


def _limits_deg(prim):
    joint = UsdPhysics.RevoluteJoint(prim)
    lower = joint.GetLowerLimitAttr().Get()
    upper = joint.GetUpperLimitAttr().Get()
    lower = -360.0 if lower is None or not math.isfinite(float(lower)) else float(lower)
    upper = 360.0 if upper is None or not math.isfinite(float(upper)) else float(upper)
    return lower, upper


def _read_deg(prim):
    # Physics Inspector编辑后的值优先来自JointStateAPI。
    state = PhysxSchema.JointStateAPI.Get(prim, "angular")
    if state:
        value = state.GetPositionAttr().Get()
        if value is not None:
            return float(value)

    # 尚未建立JointState时，使用关节驱动目标作为初始显示值。
    drive = UsdPhysics.DriveAPI.Get(prim, "angular")
    if drive:
        value = drive.GetTargetPositionAttr().Get()
        if value is not None:
            return float(value)
    return 0.0


def _validate_deg(prim, value_deg):
    lower, upper = _limits_deg(prim)
    if value_deg < lower - 1.0e-6 or value_deg > upper + 1.0e-6:
        raise ValueError(
            f"{prim.GetName()}={value_deg:.3f} deg exceeds [{lower:.3f}, {upper:.3f}] deg"
        )


def _prepare_all_joint_controls(joints):
    """一次性准备全部JointState和Drive，操作途中不再改变USD结构。"""
    state_added = []
    drive_added = []
    for name, prim in joints.items():
        initial = _read_deg(prim)
        if prim.HasAPI(PhysxSchema.JointStateAPI, "angular"):
            state_api = PhysxSchema.JointStateAPI.Get(prim, "angular")
        else:
            state_api = PhysxSchema.JointStateAPI.Apply(prim, "angular")
            state_added.append(name)
        position_attr = state_api.GetPositionAttr()
        if not position_attr:
            position_attr = state_api.CreatePositionAttr()
        if position_attr.Get() is None:
            position_attr.Set(float(initial))
        velocity_attr = state_api.GetVelocityAttr()
        if not velocity_attr:
            velocity_attr = state_api.CreateVelocityAttr()
        if velocity_attr.Get() is None:
            velocity_attr.Set(0.0)

        if prim.HasAPI(UsdPhysics.DriveAPI, "angular"):
            drive_api = UsdPhysics.DriveAPI.Get(prim, "angular")
        else:
            drive_api = UsdPhysics.DriveAPI.Apply(prim, "angular")
            drive_added.append(name)
        target_attr = drive_api.GetTargetPositionAttr()
        if not target_attr:
            target_attr = drive_api.CreateTargetPositionAttr()
        if target_attr.Get() is None:
            target_attr.Set(float(initial))

    if state_added or drive_added:
        print(
            "ONE-TIME PREPARATION: "
            f"JointStateAPI added={state_added}; DriveAPI added={drive_added}. "
            "Click Re-Enable authoring once in Physics Inspector if requested.",
            flush=True,
        )
    return state_added, drive_added


def _collect_and_validate_model_values(joints, models):
    """先检查完整35维姿态，任何一维越界时都不写USD。"""
    values = {}
    for name, model in models.items():
        value = float(model.as_float)
        _validate_deg(joints[name], value)
        values[name] = value
    if set(values) != set(joints):
        missing = sorted(set(joints) - set(values))
        raise RuntimeError(f"Incomplete 35-DOF editor model; missing={missing}")
    return values


def _apply_full_static_pose(joints, values, trigger_name):
    """锁定完整35维姿态，并只用一个ChangeProperty触发一次准静态刷新。

    直接对其余属性Set是有意设计：它们必须在Inspector求解前已经同时就位，
    但不能各自触发求解。最后只对trigger关节发一个ChangeProperty命令，
    Physics Inspector收到该命令后进行一次authoring step，看到的是完整新姿态。
    """
    if trigger_name not in joints:
        raise KeyError(f"Unknown trigger joint: {trigger_name}")

    # 再次独立校验，保证该函数也可被按钮安全调用。
    for name, value in values.items():
        _validate_deg(joints[name], value)

    controls = {}
    for name, prim in joints.items():
        value = float(values[name])
        state_api = PhysxSchema.JointStateAPI.Get(prim, "angular")
        drive_api = UsdPhysics.DriveAPI.Get(prim, "angular")
        if not state_api or not drive_api:
            raise RuntimeError(
                f"{name} is missing prepared JointStateAPI or DriveAPI; rerun script 04"
            )

        position_attr = state_api.GetPositionAttr()
        velocity_attr = state_api.GetVelocityAttr()
        target_attr = drive_api.GetTargetPositionAttr()
        if not position_attr or not velocity_attr or not target_attr:
            raise RuntimeError(f"{name} has incomplete joint control attributes")
        controls[name] = (position_attr, velocity_attr, target_attr, value)

    trigger_attr = controls[trigger_name][0]
    trigger_prev = trigger_attr.Get()
    # ChangeBlock把104次普通属性写入合成一次USD变更通知；Inspector不会看到
    # 一个半更新姿态。trigger关节位置留给后面的唯一ChangeProperty命令。
    with Sdf.ChangeBlock():
        for name, (position_attr, velocity_attr, target_attr, value) in controls.items():
            # 清零速度并同步驱动目标，避免准静态步中被旧目标拉走。
            velocity_attr.Set(0.0)
            target_attr.Set(value)
            if name != trigger_name:
                position_attr.Set(value)

    # 唯一一次命令事件：让Physics Inspector刷新几何，但不连续运行时间线。
    omni.kit.commands.execute(
        "ChangeProperty",
        prop_path=trigger_attr.GetPath(),
        value=float(values[trigger_name]),
        prev=trigger_prev,
    )


def _build_editor(stage, joints):
    old_state = getattr(builtins, WINDOW_STATE_KEY, None)
    if old_state:
        if old_state.get("window"):
            old_state["window"].visible = False

    state = {
        "window": None,
        "models": {},
        "joints": joints,
        "status": None,
        "suppress_model_callbacks": False,
    }
    setattr(builtins, WINDOW_STATE_KEY, state)

    def set_status(message):
        state["status"].text = message
        print(message)

    def reload_values():
        state["suppress_model_callbacks"] = True
        try:
            for name, model in state["models"].items():
                model.set_value(_read_deg(joints[name]))
        finally:
            state["suppress_model_callbacks"] = False
        set_status("Reloaded current joint states from USD.")

    def apply_one(name):
        try:
            values = _collect_and_validate_model_values(joints, state["models"])
            _apply_full_static_pose(joints, values, name)
            set_status(
                f"Locked full 35-DOF pose; edited {name} = {values[name]:.3f} deg"
            )
        except Exception as exc:
            set_status(f"ERROR: {exc}")

    def apply_model_value(name, model):
        """同一模型同时服务滑块和数字框，任一改变都立即准静态应用。"""
        if state["suppress_model_callbacks"]:
            return
        try:
            values = _collect_and_validate_model_values(joints, state["models"])
            _apply_full_static_pose(joints, values, name)
            state["status"].text = (
                f"Locked full 35-DOF pose; edited {name} = {values[name]:.3f} deg"
            )
        except Exception as exc:
            state["status"].text = f"ERROR: {exc}"

    def apply_all():
        try:
            values = _collect_and_validate_model_values(joints, state["models"])
            trigger_name = next(iter(joints))
            _apply_full_static_pose(joints, values, trigger_name)
            set_status("Re-applied and locked the complete 35-DOF pose.")
        except Exception as exc:
            set_status(f"ERROR: {exc}")

    window = ui.Window("DGN2 Numeric Joint Angle Editor", width=690, height=850)
    state["window"] = window
    with window.frame:
        with ui.VStack(spacing=6):
            ui.Label(f"Articulation: {ARTICULATION_ROOT_PATH}")
            ui.Label("Unit: degree | Timeline must stay STOPPED")
            ui.Label("Use either slider or number field; both share one static model.")
            ui.Label("Every edit locks all 35 joints and clears all joint velocities.")
            ui.Label("Keep Physics Inspector open in Joint States Position + QuasiStatic mode.")
            with ui.HStack(height=30, spacing=6):
                ui.Button("Reload current", clicked_fn=reload_values)
                ui.Button("Re-apply/lock all 35", clicked_fn=apply_all)
            state["status"] = ui.Label("Ready. No upstream USD has been modified.", height=24)
            with ui.ScrollingFrame():
                with ui.VStack(spacing=4, height=0):
                    for group_title, names in JOINT_GROUPS:
                        with ui.CollapsableFrame(group_title, collapsed=False, height=0):
                            with ui.VStack(spacing=3, height=0):
                                for name in names:
                                    prim = joints[name]
                                    lower, upper = _limits_deg(prim)
                                    with ui.HStack(height=27, spacing=5):
                                        ui.Label(name, width=215)
                                        model = ui.SimpleFloatModel(_read_deg(prim))
                                        ui.FloatSlider(
                                            model=model,
                                            width=ui.Fraction(1),
                                            step=0.1,
                                            min=lower,
                                            max=upper,
                                            precision=3,
                                        )
                                        ui.FloatField(
                                            model=model,
                                            width=90,
                                            step=0.1,
                                            min_value=lower,
                                            max_value=upper,
                                        )
                                        state["models"][name] = model
                                        model.add_value_changed_fn(
                                            lambda changed_model, n=name: apply_model_value(
                                                n, changed_model
                                            )
                                        )
                                        ui.Label(
                                            f"[{lower:.1f}, {upper:.1f}] deg",
                                            width=145,
                                        )
                                        ui.Button(
                                            "Apply",
                                            width=70,
                                            clicked_fn=lambda n=name: apply_one(n),
                                        )
    window.visible = True
    print("\n[NUMERIC JOINT EDITOR READY]")
    print("Joint count: 35")
    print("Unit: degree")
    print("Every edit atomically locks all 35 Joint States and Drive Targets.")
    print("Changes are authored only into the current scene layer.")


_enable_physx_support_ui()
_stage = omni.usd.get_context().get_stage()
if _stage is None:
    raise RuntimeError("No active USD stage")
_joints = _find_revolute_joints(_stage)
_prepare_all_joint_controls(_joints)
_build_editor(_stage, _joints)
