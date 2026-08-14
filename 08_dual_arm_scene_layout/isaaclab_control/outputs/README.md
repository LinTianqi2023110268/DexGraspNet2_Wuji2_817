# 输出说明

正式业务结果只有 `full_pick_place_25s_dog_candidate3800/`。其余目录是诊断、历史或阶段性输出，不应按目录名推断为成功案例。

dog 17.78 s LIFT-only 只属于 `STAGE_VALIDATION / SUMMARY`；完整抓放结论只认 24.85 s 正式目录中的 `report.json` 和 `SHA256SUMS`。

`short_motion/trace.csv`保存每个物理步的法兰目标、实际位姿误差和右臂最大关节速度。

`short_motion/report.json`保存本次短程往返是否通过、最终回到起点的误差和控制上限。

运行程序会覆盖同名结果，不会修改标定场景、机械臂USD或Wuji2官方USD。
