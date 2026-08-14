# 配置合同

本目录保存全工程共享、需要人工审计的固定合同。

- `project.json`：项目内数据、官方资产、默认checkpoint、两个运行环境和输出目录；
- `wuji2_joint_order.json`：Wuji2右手20关节的唯一网络顺序；
- `wuji2_official_asset.json`：旧标签根到官方USD根的固定转换、资产路径和哈希。
- `external_dependencies.json`：GroundedSAM、Isaac Sim和Isaac Lab等项目外依赖登记。

当前默认推理checkpoint是力调整后100场景数据训练至50000步的模型。q_opt训练集
仍作为场景/相机输入基线和严格A/B对照保留。修改这里的路径后必须运行：

```bash
/home/lin/miniconda3/envs/graspnet2.0/bin/python tools_validate.py
```

不要在代码中另建第二套关节顺序、手根转换或默认checkpoint。
