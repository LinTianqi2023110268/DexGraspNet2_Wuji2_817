# 03_runtime

- `import_scene_with_3p3r.py`：加载官方 Wuji2 USD、场景和任务，在当前
  stage 临时创建 3P+3R 手根链；不修改源 USD。
- `execute_native_grasp.py`：执行旧原生 Wuji2 动作时序，通过 articulation
  位置目标同时驱动 6 个手根自由度和 20 个手指自由度。

这两个文件不是独立入口。请运行 `selected_native_case/05_isaacsim/` 下的
01/02 薄封装脚本。
