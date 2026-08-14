"""Isaac Sim 5.0 Script Editor：启用并打开 Physics Inspector。

适用于 Window 菜单中没有 ``Physics -> Physics Authoring Toolbar`` 的情况。
本脚本不改场景、不运行物理，只启用本机已有的 PhysX SupportUI 扩展并显示窗口。
"""

import carb.settings
import omni.kit.app


EXTENSION_ID = "omni.physx.supportui"

manager = omni.kit.app.get_app().get_extension_manager()
manager.set_extension_enabled_immediate(EXTENSION_ID, True)

enabled_id = manager.get_enabled_extension_id(EXTENSION_ID)
if not enabled_id:
    raise RuntimeError(
        "Could not enable omni.physx.supportui. Open Window -> Extensions, "
        "search for PhysX SupportUI, and enable it manually."
    )

# 必须在扩展启用后导入，否则绑定模块可能还没有注册。
import omni.physxsupportui.bindings._physxSupportUi as pxsupportui

settings = carb.settings.get_settings()
settings.set_bool(pxsupportui.SETTINGS_ACTION_BAR_ENABLED, True)
settings.set_bool(pxsupportui.SETTINGS_PHYSICS_INSPECTOR_ENABLED, True)

print("\n[PHYSICS INSPECTOR OPEN REQUESTED]")
print("Enabled extension:", enabled_id)
print("If the inspector is docked, look below the Stage panel or along the window tabs.")
