# 共享Python实现

`wuji2_dgn2/`只保存被数据、推理、可视化和仿真共同调用的实现：

- `project.py`：读取`config/project.json`并解析项目内路径；
- `adapter_common.py`：配置、JSON和公共数据工具；
- `collision.py`：Wuji2几何、虎口PREGRASP和指尖法向IK；
- `palm_path.py`：增强掌心路径过滤；
- `official_asset.py`：官方Wuji2 USD资产校验及旧根到`r_wrist`转换；
- `visual.py`：共享可视化工具。

这里不保存可直接运行的训练、推理或Isaac Sim入口。入口分别位于02、04、05、06
和`trimesh/`，避免共享库与实验脚本混在一起。
