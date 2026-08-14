"""Isaac Sim 5.0 Script Editor：隐藏Camera视口辅助图标，不影响相机拍摄。"""

import carb.settings


carb.settings.get_settings().set_bool("/app/viewport/show/camera", False)
print("[CAMERA VIEWPORT ICON HIDDEN]")
print("Camera Prim and RGB-D capture remain enabled.")
