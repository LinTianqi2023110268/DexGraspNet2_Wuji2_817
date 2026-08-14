# 新案例模板

每个案例唯一对应：场景＋相机视角＋目标物体＋LEAP网络候选。

推荐直接使用根目录 `run_new_test_case.py`，它会建立：

```text
01_input/          单视角点云、1024候选、LEAP动作和场景输入
02_retargeting/    Wuji2 GRASP q20
03_root_alignment/ 四指尖约束下的Wuji2手根6D
04_squeeze/        Wuji2 SQUEEZE及41点路径
05_visualization/  四手Trimesh
06_isaacsim/       当前唯一成功方法的01/02脚本、任务和结果
```

所有新案例自动继承：官方Wuji2 USD、LEAP同构3P+3R手根驱动、K=800、D=20、
大拇指沿局部`-Y`方向做30 mm PREGRASP目标IK，以及SQUEEZE后恢复重力。

不要从 `99_archive` 复制脚本，也不要重新引入旧的逐步 `set_world_pose()` 手根分支。
