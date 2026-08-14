# Runtime

Route C 正式运行实现。用户通常只调用上一级两个 `launch_*_visible.sh`，无需直接进入这里。

- `scripts/10_run_full_pick_place.py`：完整物理抓放状态机；
- `scripts/11_replay_physical_trajectory.py`：只按墙钟连续回放已记录状态；
- `config/full_pick_place_25s_dog_candidate3800.json`：正式 dog 配置；
- `launchers/`：根入口调用的内部启动器。

禁止把 `rejected/mp4_export` 的逐帧捕获逻辑重新并回正式 replay。
