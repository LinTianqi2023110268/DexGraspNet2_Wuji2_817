# scene0001_view0001_official_rank0 — VERIFIED PASS

这是2026-08-11在Isaac Sim 5.0中真实运行并抓起目标物体的冻结基线。

## 结果

- 目标：segmentation ID 60，Battery；
- DexGraspNet 2.0候选：49；
- 网络分数：56.14188385；
- 目标质心抬升：204.1405 mm；
- 横向位移：16.0856 mm；
- `target_specific_success=true`；
- `official_any_object_success=true`。

## 内容

```text
scripts/       可独立重放的Isaac Sim 01/02入口
task/          冻结的final_waypoints任务
result/        首次成功的final_result.json
parameters/    迁移、SQUEEZE和指尖方向参数
runtime/       成功时使用的公共导入器和执行器
reports/       案例、GRASP迁移、根对齐、SQUEEZE报告
manifest.json  关键参数和结果
SHA256SUMS     文件完整性校验
```

## 重放

在Isaac Sim 5.0 Script Editor中运行：

1. `scripts/01_import.py`；
2. 等待 `[01 IMPORT COMPLETE]`；
3. `scripts/02_execute.py`。

重放结果写入 `result/replay_result.json`，不会覆盖首次成功结果
`result/final_result.json`。
