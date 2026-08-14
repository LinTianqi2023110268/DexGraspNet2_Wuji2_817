# 已验证慢速抓取基线：scene_0000 / dog / candidate 3800

这是当前必须保留的物理成功证据，不属于 20 秒演示调速实验。

准确的数据来源是：正式测试集 `wuji2_test60_10upright_10view_v1` 的
`scene_0000`，在双臂大桌面中开启重力重新稳定，并由顶部相机重新拍摄后的
现场版本 `live_dynamic_scene0000`。

## 已通过的完整前置门

1. 顶部相机 RGB-D 与 40,000 点输入；
2. GroundingDINO + SAM 文本目标 `dog`；
3. 官方 DexGraspNet 2.0 checkpoint 推理；
4. 官方 PREGRASP 场景/桌面碰撞检查；
5. LEAP 到 Wuji2 官方 retargeting；
6. Wuji2 稠密 SQUEEZE 路径；
7. 右臂全部关键位姿 IK；
8. 完整关节插值路径碰撞检查；
9. Isaac Lab 2.2 + Isaac Sim 5.0 真实动力学抓取与抬升。

## 物理结果

- 目标：`sem-Dog-2526b1f358f773034203f59b4fda924b`，segmentation 33；
- 候选：3800；
- 实际最大抬升：122.10 mm；
- 空中保持：2 s；
- 空中持续接触：4/5 根手指；
- 最终法兰误差：约 4.24 mm / 1.25 deg；
- 判定：PASS（抬升硬门槛为 30 mm）。

关键输入和当时实际使用的公共执行器 SHA256 固定在 `baseline_manifest.json`，
物理结果摘要固定在 `verified_result.json`。

注意：旧版公共执行器曾让慢速和快速实验使用同一个输出目录，后来进行的 20 秒
实验覆盖了原始慢速 `report.json/trace.csv`。因此这里保留的是已经验收的结果摘要，
不把后来生成的快速 trace 冒充慢速证据。公共执行器现已修复为严格遵从配置中的
`output_directory`，后续实验不会再互相覆盖。
