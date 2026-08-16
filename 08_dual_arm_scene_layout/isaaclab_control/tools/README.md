# Offline and preparation tools

这些脚本只保留 RGB-D 捕获、机械臂 Cartesian target 构建和动态场景捕获。旧 CPU reachability、SciPy/Pinocchio waypoint IK、完整 mesh/table collision 与候选重采样编排已经归档到 `../history/legacy_route_c_cpu_mesh/`，不再属于 production path。

正式运行代码在 `../runtime/`，cuRobo V2 实现在 `../core/`，诊断入口在 `../diagnostics/`。
