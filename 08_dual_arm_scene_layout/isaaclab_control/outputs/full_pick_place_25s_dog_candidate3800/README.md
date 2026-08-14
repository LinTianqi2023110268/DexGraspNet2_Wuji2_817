# Verified route-C baseline: dog candidate3800

这是双臂路线唯一正式完整业务案例。

- status：PASS；
- 动作：抓取、抬升、搬运、放置、松手、退让、回初始位姿；
- 仿真动作时间：24.85 s；
- 当前机器实际计算时间：约 272.98 s；
- 最大目标抬升：180.85 mm；
- 最终目标占地位于绿色区域。

核心证据：`report.json`、`trace.csv`、`physical_replay_30fps.npz`。输入 provenance、正式 config、launcher、arm IK 和路径审计通过本目录 `SHA256SUMS` 原位引用，不复制大文件。

稳定回放命令：

```bash
cd /home/lin/Projects/DexGraspNet2_Wuji2
./08_dual_arm_scene_layout/isaaclab_control/launch_replay_25s_visible.sh
```

回放不重新运行网络、IK 或物理。已否决的自动 MP4 实验已归档到 `../../rejected/mp4_export/`，缓存视频不属于本基线。
