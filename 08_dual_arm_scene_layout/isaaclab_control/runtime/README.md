# Runtime

Route C 正式运行实现。用户通常只调用上一级两个 `launch_*_visible.sh`，无需直接进入这里。

- `scripts/10_run_full_pick_place.py`：完整物理抓放状态机；settle 后读取 q_current，通过 persistent `curobo_v2` worker 执行 GPU 多解 IK 与 observed-scene ESDF 碰撞硬过滤；
- `scripts/11_replay_physical_trajectory.py`：只按墙钟连续回放已记录状态；
- `config/full_pick_place_25s_dog_candidate3800.json`：正式 dog 配置；
- `launchers/`：根入口调用的内部启动器。

正式 IK 不再导入 SciPy/Pinocchio。RGB-D 输入为 `depth_m.npy + intrinsics.npy + T_world_camera.npy`，GroundedSAM mask 保留 target/non-target 语义层。单视角 unknown 独立报告，不能将 observed-safe 表述为真实世界保证无碰撞；当前 `path_pass` 仍为 null，GRASP/SQUEEZE/LIFT 的 target contact allowance 仍是 phase-wide limitation。

禁止把 `rejected/mp4_export` 的逐帧捕获逻辑重新并回正式 replay。
