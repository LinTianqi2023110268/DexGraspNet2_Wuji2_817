# Offline and preparation tools

这些脚本负责 RGB-D 捕获、机械臂目标构建、可达性筛选、完整 waypoint IK、关节路径碰撞审计和候选重采样。每个程序通过命令行显式接收 case；它们不是正式抓放入口。

正式运行代码在 `../runtime/`，诊断入口在 `../diagnostics/`。
